"""FFN-only W4A16 diagnostic. The backbone and its KV remain BF16."""
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def digest(tensor):
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(data).cast("B")).hexdigest()


class TensorStore:
    def __init__(self, path):
        from safetensors import safe_open
        self.path = Path(path)
        index = self.path / "model.safetensors.index.json"
        if index.exists():
            self.files = json.loads(index.read_text())["weight_map"]
        else:
            self.files = {}
            for shard in sorted(self.path.glob("*.safetensors")):
                with safe_open(shard, framework="pt", device="cpu") as stream:
                    self.files.update({key: shard.name for key in stream.keys()})
        if not self.files:
            raise ValueError(f"no tensors in {path}")

    def get(self, key):
        from safetensors import safe_open
        with safe_open(self.path / self.files[key], framework="pt", device="cpu") as stream:
            return stream.get_tensor(key)


def unpack_unsigned(packed, dim):
    """Unpack compressed-tensors little-endian INT4, including packed zeros."""
    if packed.dtype != torch.int32 or packed.ndim != 2 or dim not in (0, 1):
        raise ValueError("expected a rank-two int32 tensor and packing axis 0/1")
    shifts = torch.arange(8, device=packed.device) * 4
    if dim == 1:
        return ((packed[..., None].long() >> shifts) & 15).reshape(packed.shape[0], -1)
    return ((packed[:, None, :].long() >> shifts[None, :, None]) & 15).reshape(-1, packed.shape[1])


def dequantize(weight, scales, zeros, group_size=128):
    q, z = unpack_unsigned(weight, 1), unpack_unsigned(zeros, 0)
    if q.shape != (scales.shape[0], scales.shape[1] * group_size) or z.shape != scales.shape:
        raise ValueError("packed weight, scale or zero-point shape mismatch")
    # compressed-tensors adds 8 to both signed values before packing; it cancels.
    return ((q.float() - z.repeat_interleave(group_size, 1)) *
            scales.float().repeat_interleave(group_size, 1)).to(scales.dtype)


def check_quant_config(path):
    cfg = json.loads((Path(path) / "config.json").read_text())
    q = cfg["quantization_config"]
    groups = q["config_groups"]
    if q["quant_method"] != "compressed-tensors" or q["format"] != "pack-quantized" or len(groups) != 1:
        raise ValueError("expected the pinned compressed-tensors W4A16 checkpoint")
    group = next(iter(groups.values()))
    w = group["weights"]
    if (w["num_bits"], w["group_size"], w["strategy"], w["symmetric"], w["type"]) != (4, 128, "group", False, "int"):
        raise ValueError("expected asymmetric INT4, group size 128")
    if w.get("actorder") not in (None, "weight") or group.get("input_activations") is not None:
        raise ValueError("activation quantization or group reordering is unsupported")
    if cfg["model_type"] != "qwen3" or cfg["num_hidden_layers"] != 36:
        raise ValueError("this diagnostic is pinned to Qwen3-8B")
    return cfg


class PackedLinear(nn.Module):
    def __init__(self, store, prefix, workspace):
        super().__init__()
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_make_empty_g_idx, marlin_permute_scales, marlin_zero_points)

        self.raw = tuple(store.get(prefix + suffix) for suffix in
                         (".weight_packed", ".weight_scale", ".weight_zero_point"))
        weight, scale, zeros = self.raw
        self.n, self.k = weight.shape[0], weight.shape[1] * 8
        if tuple(store.get(prefix + ".weight_shape").tolist()) != (self.n, self.k):
            raise ValueError(f"shape mismatch: {prefix}")
        self.empty = marlin_make_empty_g_idx(torch.device("cuda"))
        self.workspace = workspace
        self.register_buffer("weight", ops.gptq_marlin_repack(
            weight.t().contiguous().cuda(), self.empty, self.k, self.n, 4, False))
        self.register_buffer("scale", marlin_permute_scales(
            scale.t().contiguous().cuda(), self.k, self.n, 128))
        self.register_buffer("zeros", marlin_zero_points(
            unpack_unsigned(zeros, 0).t().contiguous().cuda(), self.k // 128, self.n, 4))

    def forward(self, x):
        from vllm.model_executor.layers.quantization.utils.marlin_utils import apply_gptq_marlin_linear
        from vllm.scalar_type import scalar_types
        return apply_gptq_marlin_linear(
            input=x, weight=self.weight, weight_scale=self.scale, weight_zp=self.zeros,
            g_idx=self.empty, g_idx_sort_indices=self.empty, workspace=self.workspace,
            wtype=scalar_types.uint4, output_size_per_partition=self.n,
            input_size_per_partition=self.k, is_k_full=True, use_fp32_reduce=True)

    def reference(self, x):
        return F.linear(x, dequantize(*self.raw).to(x.device))


class PackedFFN(nn.Module):
    def __init__(self, store, layer, workspace):
        super().__init__()
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(self, name, PackedLinear(store, f"model.layers.{layer}.mlp.{name}", workspace))

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def reference(self, x):
        return self.down_proj.reference(F.silu(self.gate_proj.reference(x)) * self.up_proj.reference(x))


def relative_l2(actual, expected):
    return float((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12))


def accepted_prefix(proposed, verified):
    if len(proposed) != len(verified):
        raise ValueError("proposal and verification lengths differ")
    return next((i for i, (a, b) in enumerate(zip(proposed, verified)) if a != b), len(proposed))
