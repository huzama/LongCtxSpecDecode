"""Native FFN-only draft stack; attention modules and backbone KV are shared.

The target is never mutated. CUDA graphs capture the same vLLM decoder layers
with either their original FFNs or packed Marlin FFNs. No second KV is created.
"""
import copy

import torch
from torch import nn

from .ffn_quant import PackedLinear, TensorStore, check_quant_config


class MergedTensorStore:
    """Present gate/up checkpoint tensors as one output-concatenated projection."""
    def __init__(self, store):
        self.store = store

    def get(self, key):
        if ".gate_up_proj." not in key:
            return self.store.get(key)
        tensors = [self.store.get(key.replace("gate_up_proj", n)) for n in ("gate_proj", "up_proj")]
        if key.endswith(".weight_shape"):
            if tensors[0][1] != tensors[1][1]:
                raise ValueError("gate/up input widths differ")
            return torch.tensor([int(tensors[0][0] + tensors[1][0]), int(tensors[0][1])])
        return torch.cat(tensors, dim=0)


class ServingFFN(nn.Module):
    def __init__(self, store, layer, workspace):
        super().__init__()
        prefix = f"model.layers.{layer}.mlp."
        self.gate_up_proj = PackedLinear(MergedTensorStore(store), prefix + "gate_up_proj", workspace)
        self.down_proj = PackedLinear(store, prefix + "down_proj", workspace)
        # CPU reference tensors belong to the diagnostic, not the serving path.
        del self.gate_up_proj.raw, self.down_proj.raw

    def forward(self, hidden):
        return self.down_proj(fused_silu(self.gate_up_proj(hidden)))


def fused_silu(gate_up):
    output = torch.empty((*gate_up.shape[:-1], gate_up.shape[-1] // 2),
                         device=gate_up.device, dtype=gate_up.dtype)
    torch.ops._C.silu_and_mul(output, gate_up)
    return output


class NativeBF16FFN(nn.Module):
    """Use the same fused activation as INT4, sharing target BF16 projections."""
    def __init__(self, original):
        super().__init__()
        self.gate_up_proj, self.down_proj = original.gate_up_proj, original.down_proj

    def forward(self, hidden):
        gate_up, _ = self.gate_up_proj(hidden)
        output, _ = self.down_proj(fused_silu(gate_up))
        return output


class FFNDraftModel(nn.Module):
    def __init__(self, target, checkpoint=None):
        super().__init__()
        from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
        from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new

        if get_tensor_model_parallel_world_size() != 1 or get_pp_group().world_size != 1:
            raise ValueError("FFN draft benchmark supports a single GPU only")
        cfg = target.config
        if cfg.model_type != "qwen3" or cfg.hidden_act != "silu":
            raise ValueError("FFN draft requires Qwen3 with SiLU")
        self.embed_tokens, self.norm = target.model.embed_tokens, target.model.norm
        self.compute_logits = target.compute_logits
        self.get_top_tokens = target.get_top_tokens
        self.layers = nn.ModuleList()
        self.extra_weight_bytes = 0
        store = None
        if checkpoint:
            quant = check_quant_config(checkpoint)
            for field in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                          "num_key_value_heads", "vocab_size"):
                if quant[field] != getattr(cfg, field):
                    raise ValueError(f"quantized FFN checkpoint differs in {field}")
            store = TensorStore(checkpoint)
        workspace = marlin_make_workspace_new(torch.device("cuda")) if store else None
        for i, original in enumerate(target.model.layers):
            layer = copy.copy(original)
            layer._modules = original._modules.copy()
            if store:
                layer.mlp = ServingFFN(store, i, workspace)
                self.extra_weight_bytes += sum(x.numel() * x.element_size() for x in layer.mlp.buffers())
            else:
                layer.mlp = NativeBF16FFN(original.mlp)
            self.layers.append(layer)
            if layer.self_attn is not original.self_attn:
                raise AssertionError("draft allocated separate attention")
            if store and layer.mlp is original.mlp:
                raise AssertionError("target FFN was replaced")

    def forward(self, input_ids, positions):
        hidden, residual = self.embed_tokens(input_ids), None
        for layer in self.layers:
            hidden, residual = layer(positions, hidden, residual)
        hidden, _ = self.norm(hidden, residual)
        return hidden


# Imported only by the serving worker; the stats RPC runs outside timed decode.
from spotattention.serve.stats import SelectionStatsWorker


class FFNStatsWorker(SelectionStatsWorker):
    @torch.inference_mode()
    def ffn_draft_info(self, validate=False, num_reqs=1):
        from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
        spec = self.model_runner.speculator
        if spec is None:
            return None
        result = dict(graph_scope="step",
            captured_graphs=sum(len(g.graphs) for g in spec.graphs),
            step_graph_replays=spec.step_graph_replays,
            extra_weight_bytes=getattr(spec.model, "extra_weight_bytes", 0),
            shared_attention=all(a.self_attn is b.self_attn for a, b in
                zip(spec.model.layers, self.model_runner.model.model.layers)))
        if not result["shared_attention"] or not result["captured_graphs"]:
            raise AssertionError("FFN draft lost shared attention or CUDA graphs")
        if validate:
            if not 1 <= num_reqs <= spec.max_num_reqs:
                raise ValueError("invalid draft validation batch")
            n = num_reqs
            buffers = spec.input_buffers
            names = ("input_ids", "positions", "seq_lens", "query_start_loc")
            saved = {n: getattr(buffers, n).clone() for n in names}
            step = spec.current_draft_step.clone()
            proposals = spec.draft_tokens.clone()
            def restore():
                for n, value in saved.items():
                    getattr(buffers, n).copy_(value)
                spec.current_draft_step.copy_(step)
                spec.draft_tokens.copy_(proposals)
            upper = buffers.seq_lens[:n].cpu()
            if (upper <= 0).any():
                raise ValueError("draft validation requires active requests")
            saved_max = spec.draft_max_seq_len
            spec.draft_max_seq_len = min(int(upper.max()) + spec.num_speculative_steps, spec.max_model_len)
            def sequence(graph):
                for i in range(spec.num_speculative_steps):
                    phase = int(spec.selection.layer_scores is not None and i > 0)
                    spec.selection.reuse_scores = bool(phase)
                    spec.current_draft_step.fill_(i)
                    mapping = spec.block_tables.compute_slot_mappings(
                        spec.idx_mapping[:n], buffers.query_start_loc[:n + 1], buffers.positions[:n], n)
                    slots = build_slot_mappings_by_layer(mapping, spec.kv_cache_config)
                    md = spec._build_draft_attn_metadata(num_reqs=n, num_reqs_padded=n,
                        num_tokens_padded=n, seq_lens_cpu_upper_bound=upper, step=i + 1)
                    if graph:
                        manager = spec.graphs[phase]
                        desc = next(d for d in manager.graphs if d.num_tokens == n)
                        manager.run_fullgraph(desc)
                    else:
                        spec._step(n, n, md, slots, None)
            try:
                sequence(True)
                graph_tokens = spec.draft_tokens.clone()
                restore()
                sequence(False)
                torch.testing.assert_close(spec.draft_tokens[:n], graph_tokens[:n], rtol=0, atol=0)
                result["graph_matches_eager"] = True
                result["validation_batch"] = n
                result["validation_drafts"] = graph_tokens[:n].cpu().tolist()
            finally:
                restore()
                spec.draft_max_seq_len = saved_max
                spec.selection.reuse_scores = False
        return result
