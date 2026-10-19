"""INT4 decoder projections with BF16 activations and one shared KV cache.

The existing Marlin FFN implementation is reused. Q/K/V and attention-output
weights are also packed; embedding, norms, LM head and indexer remain shared.
"""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from .ffn_draft import FFNDraftModel
from .ffn_quant import PackedLinear, TensorStore, dequantize, relative_l2
from .hierarchical_profile import HierarchicalProfileWorker


class QKVTensorStore:
    def __init__(self, store):
        self.store = store

    def get(self, key):
        if ".qkv_proj." not in key:
            return self.store.get(key)
        tensors = [self.store.get(key.replace("qkv_proj", name))
                   for name in ("q_proj", "k_proj", "v_proj")]
        if key.endswith(".weight_shape"):
            if len({int(t[1]) for t in tensors}) != 1:
                raise ValueError("Q/K/V input dimensions differ")
            return torch.tensor([sum(int(t[0]) for t in tensors), int(tensors[0][1])])
        return torch.cat(tensors, dim=0)


class PackedProjection(nn.Module):
    def __init__(self, store, prefix, workspace):
        super().__init__()
        self.linear = PackedLinear(store, prefix, workspace)
        del self.linear.raw

    def forward(self, hidden):
        # vLLM's QKVParallelLinear and RowParallelLinear return (output, bias).
        return self.linear(hidden), None


class WeightDraftModel(FFNDraftModel):
    def __init__(self, target, checkpoint):
        super().__init__(target, checkpoint)
        from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new

        self.checkpoint = checkpoint
        store = QKVTensorStore(TensorStore(checkpoint))
        workspace = marlin_make_workspace_new(torch.device("cuda"))
        for i, (layer, original) in enumerate(zip(self.layers, target.model.layers, strict=True)):
            attn = copy.copy(original.self_attn)
            attn._modules = original.self_attn._modules.copy()
            for name in ("qkv_proj", "o_proj"):
                packed = PackedProjection(store, f"model.layers.{i}.self_attn.{name}", workspace)
                setattr(attn, name, packed)
                self.extra_weight_bytes += sum(t.numel()*t.element_size() for t in packed.buffers())
            layer.self_attn = attn
            if attn.attn is not original.self_attn.attn or attn.indexer is not original.self_attn.indexer:
                raise AssertionError("draft must share backbone attention and indexer caches")

    @torch.inference_mode()
    def validate_attention_weights(self, target, batch):
        store = QKVTensorStore(TensorStore(self.checkpoint))
        generator = torch.Generator(device="cuda").manual_seed(42)
        records = []
        for i, (layer, original) in enumerate(zip(self.layers, target.model.layers, strict=True)):
            if (layer.self_attn.attn is not original.self_attn.attn or
                    layer.self_attn.indexer is not original.self_attn.indexer or
                    layer.mlp is original.mlp):
                raise AssertionError("draft lost isolation or shared cache")
            if original.mlp.gate_up_proj.weight.dtype != torch.bfloat16:
                raise AssertionError("target FFN weights changed precision")
            for name in ("qkv_proj", "o_proj"):
                native = getattr(original.self_attn, name)
                packed = getattr(layer.self_attn, name).linear
                if native.weight.dtype != torch.bfloat16:
                    raise AssertionError("target attention weights changed precision")
                prefix = f"model.layers.{i}.self_attn.{name}"
                raw = [store.get(prefix+s) for s in
                       (".weight_packed", ".weight_scale", ".weight_zero_point")]
                reference_weight = dequantize(*raw).cuda()
                x = torch.randn((batch, packed.k), device="cuda", dtype=torch.bfloat16,
                                generator=generator)
                expected, actual = F.linear(x, reference_weight), packed(x)
                error = relative_l2(actual, expected)
                if not torch.isfinite(actual).all() or error > .01:
                    raise AssertionError(f"{prefix}: packed/dequantized relative L2={error}")
                records.append(dict(layer=i, projection=name, relative_l2=error,
                                    max_abs=float((actual.float()-expected.float()).abs().max())))
                del raw, reference_weight, expected, actual, x
        if self.embed_tokens is not target.model.embed_tokens or self.norm is not target.model.norm:
            raise AssertionError("embedding and final norm must remain shared")
        return dict(status="passed", attention_projection_checks=records,
            extra_weight_bytes=self.extra_weight_bytes, batch=batch,
            decoder_projections_int4=7*len(self.layers), target_weights="BF16",
            shared_backbone_cache=True, shared_indexer=True,
            shared_bf16_components=["embedding", "norms", "LM head", "Highlight indexer"],
            kernel="Marlin W4A16; asymmetric INT4 groups of 128")


class HierarchicalWeightProfileWorker(HierarchicalProfileWorker):
    def hierarchical_weight_validate(self, batch=25):
        return self.model_runner.speculator.model.validate_attention_weights(self.model_runner.model, batch)
