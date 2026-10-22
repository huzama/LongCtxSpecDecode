"""Opt-in vLLM adapter for the validated paged hierarchical cache.

The vLLM allocator owns the single packed backbone pool. Native BF16 attention
runs complete prefills; decode reads packed pages directly. Chunked/prefix-cached
prefill are rejected. Direct sparse decoding and sparse verification consume
the same logical block selections as the native Highlight path.
"""
import copy
from dataclasses import dataclass
from types import MethodType

import torch
import triton as tr
import triton.language as tl
from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backend import (
    AttentionBackend, AttentionCGSupport, AttentionImpl, AttentionMetadata,
    AttentionMetadataBuilder,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from spotattention.serve.model import MolleModelState, MolleQwen3ForCausalLM
from spotattention.serve.stats import SelectionStatsWorker

from .hierarchical_paged import PageLayout, PagedHierarchicalKV
from .hierarchical_paged_kernels import attention_workspace, attention, prefill


def _select_with_logical_ids(indexer, q, w, md, kv_cache):
    """Use MoLLE's same fused selector, enabling its logical-id output.

    Native FA consumes only physical ids and sets write_ids=False. Packed KV
    also needs absolute positions for key-group metadata and the BF16 ring.
    """
    from spotattention.serve import kernels
    sel, cfg = indexer.sel, indexer.cfg
    if cfg.selection.mode != "topk_fraction":
        raise ValueError("hierarchical adapter currently preserves top-K fraction selection only")
    kc = indexer._kernel_config(q.device)
    scores = sel.scores_of(indexer.layer_idx)
    if not sel.reuse_scores:
        indexer._score(q, w, md, kv_cache, kc, scores)
    if cfg.selection.topk_recency:
        kernels.keep_recent_blocks(scores, md.seq_lens, recent_tokens=cfg.selection.topk_recency,
                                    block_size=cfg.block_size)
    control = indexer.hierarchical_control
    fraction = (control.verify_fraction if control.speculative and not control.drafting
                and control.verify_fraction else cfg.selection.topk_fraction)
    chunk, warps, splits = kc.select_launch(sel.num_blocks, "topk", len(md.seq_lens))
    kernels.select_topk(scores, sel.fa_table, md.seq_lens, md.query_start_loc, md.is_prefilling,
        sel.prompt_len, sel.cmp_table, sel.seqused, sel.cnt, sel.ids, sel.select,
        stats=sel.stats[indexer.layer_idx], k_blocks=0, k_fraction=fraction,
        block_size=cfg.block_size, chunk=chunk, num_warps=warps, splits=splits, write_ids=True,
        verify_fraction=cfg.selection.verify_fraction,
        verify_k_min=-(-cfg.selection.topk_recency//cfg.block_size)+1)


class HierarchicalContext:
    def __init__(self, config, kv_heads, query_heads, selection):
        n = config.scheduler_config.max_num_seqs
        device = config.device_config.device
        self.selection = selection
        self.drafting = False
        self.benchmark_active = False
        self.audit_layer = None
        self.target_execution = []
        self.draft_execution = []
        self.speculative = config.speculative_config is not None
        hf = config.model_config.hf_config
        self.verify_fraction = float(getattr(hf, "molle_verify_fraction", 0))
        self.draft_fraction = float(getattr(hf, "molle_topk_fraction", .07))
        if self.verify_fraction not in (0., .25, .4, .5):
            raise ValueError("unsupported hierarchical verifier budget")
        if self.verify_fraction and (not self.speculative or selection is None):
            raise ValueError("sparse verification requires speculative Highlight selection")
        self.draft_bits = int(getattr(config.model_config.hf_config, "molle_hierarchical_draft_bits", 4))
        if self.draft_bits not in (4, 8):
            raise ValueError("hierarchical draft view must be four or eight bits")
        self.target_slots = torch.full((n,), -1, dtype=torch.int32, device=device)
        self.draft_slots = torch.full_like(self.target_slots, -1)
        self.draft_committed = torch.zeros_like(self.target_slots)
        self.max_length = config.model_config.max_model_len
        self.prefill_fa_version = config.attention_config.flash_attn_version or 4
        if self.prefill_fa_version not in (2, 4):
            raise ValueError("hierarchical prefill supports native FA2 or FA4")
        # Shared by every attention layer. Allocate before memory profiling so
        # neither the recent buffers nor attention scratch consume hidden KV capacity.
        # vLLM also warms short multi-query batches for an AR-only engine.
        from .attention_policy import attention_layouts
        layouts = attention_layouts(self.max_length, selection is not None, self.speculative,
                                    self.draft_bits, self.draft_fraction, self.verify_fraction)
        self.attention_backend = getattr(hf, 'molle_hierarchical_attention_backend', 'triton')
        self.workspaces = {key: attention_workspace(n, kv_heads, query_heads, key[1], width, device,
            backend=self.attention_backend, cache_format=f'hierarchical{bits}', sparse=key[2])
            for key, (bits, width) in layouts.items()}

    def sparse_decode(self, drafting):
        return self.selection is not None and (drafting or not self.speculative or self.verify_fraction > 0)

    def workspace(self, requests, queries, sparse=False, drafting=False):
        original = self.workspaces[drafting, queries, sparse]
        if hasattr(original, 'for_requests'):
            return original.for_requests(requests)
        ws = copy.copy(original)
        ws.shape = (requests, *ws.shape[1:])
        return ws


@dataclass
class HierarchicalMetadata(AttentionMetadata):
    num_reqs: int
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slots: torch.Tensor
    positions: torch.Tensor
    committed: torch.Tensor
    drafting: bool
    prefill_rows: list[tuple[int, int, int]] | None


@tr.jit
def _decode_positions(QSL, SEQ, POS, BASE, WRITE_BASE: tl.constexpr = True):
    req = tl.program_id(0)
    start, end = tl.load(QSL+req), tl.load(QSL+req+1)
    base = tl.load(SEQ+req)-(end-start)
    i = tl.arange(0, 8)
    tl.store(POS+start+i, base+i, start+i < end)
    if WRITE_BASE:
        tl.store(BASE+req, base)


class HierarchicalMetadataBuilder(AttentionMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH
    reorder_batch_threshold = None

    def __init__(self, spec, layer_names, vllm_config, device):
        super().__init__(spec, layer_names, vllm_config, device)
        layer = vllm_config.compilation_config.static_forward_context[layer_names[0]]
        self.control = layer.impl.control
        n = vllm_config.scheduler_config.max_num_seqs
        self._committed = torch.zeros(n, dtype=torch.int32, device=device)
        self._positions = torch.zeros(n*7, dtype=torch.int64, device=device)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        cm, c = common_attn_metadata, self.control
        if common_prefix_len:
            raise ValueError("hierarchical cache does not support shared prefix pages")
        n, drafting = cm.num_reqs, c.drafting
        if c.selection is not None:
            c.selection.ensure_width(cm.block_table_tensor.shape[1])
            c.selection.set_fa_table(cm.block_table_tensor)
        rows = None
        if cm.max_query_len > 7:
            bounds = cm.query_start_loc_cpu.tolist()
            lengths = cm.seq_lens.cpu().tolist()  # prefill only, outside decode timing
            rows = []
            for r in range(n):
                start, end = bounds[r:r+2]
                if end > start:
                    if lengths[r] != end-start:
                        raise ValueError("hierarchical prefill requires each complete prompt in one call")
                    rows.append((r, start, end))
            positions = cm.positions
            if positions is None:
                positions = self._positions  # prefill kernel does not consume this view
        else:
            # The draft builder does not pass CommonAttentionMetadata.positions.
            # Use one persistent buffer for both capture and replay, derived
            # from the runner's exact device lengths (never its CPU upper bound).
            positions = self._positions[:cm.num_actual_tokens]
        if drafting:
            committed = c.draft_committed[:n]
        else:
            committed = self._committed[:n]
        return HierarchicalMetadata(n, cm.num_actual_tokens, cm.max_query_len,
            cm.query_start_loc, cm.seq_lens, cm.block_table_tensor,
            (c.draft_slots if drafting else c.target_slots)[:n], positions, committed, drafting, rows)


@register_backend(AttentionBackendEnum.CUSTOM)
class HierarchicalBackend(AttentionBackend):
    forward_includes_kv_cache_update = True

    @staticmethod
    def get_name(): return "CUSTOM"

    @staticmethod
    def get_impl_cls(): return HierarchicalImpl

    @staticmethod
    def get_builder_cls(): return HierarchicalMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes(): return [16]

    @staticmethod
    def get_supported_head_sizes(): return [128]

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"):
        if block_size != 16 or head_size != 128:
            raise ValueError("hierarchical pages require 16 tokens and 128 channels")
        return (num_blocks, PageLayout(num_kv_heads).page_bytes)

    @staticmethod
    def get_kv_cache_stride_order(include_num_layers_dimension=False):
        return (0, 1, 2) if include_num_layers_dimension else (0, 1)


class HierarchicalImpl(AttentionImpl):
    def __init__(self, num_heads, head_size, scale, num_kv_heads=None,
                 alibi_slopes=None, sliding_window=None, kv_cache_dtype="auto",
                 logits_soft_cap=None, attn_type="decoder", kv_sharing_target_layer_name=None,
                 *, hierarchical_control, **kwargs):
        if (head_size != 128 or alibi_slopes is not None or sliding_window is not None or
                logits_soft_cap not in (None, 0) or kv_sharing_target_layer_name is not None):
            raise ValueError("unsupported hierarchical attention configuration")
        self.num_heads, self.num_kv_heads, self.head_size = num_heads, num_kv_heads, head_size
        self.scale, self.kv_cache_dtype = scale, kv_cache_dtype
        self.control = hierarchical_control
        if self.control.audit_layer is None:
            self.control.audit_layer = self
        device = self.control.target_slots.device
        # The empty page view is bound to vLLM's actual pool after profiling.
        self.cache = PagedHierarchicalKV.allocate(0, num_kv_heads, len(self.control.target_slots), device)

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        md, c = attn_metadata, self.control
        if md is None:  # native memory-profile call
            output.zero_()
            return output
        if self.cache.pages.data_ptr() != kv_cache.data_ptr():
            self.cache.bind_pages(kv_cache)
        k, v, q = key.contiguous(), value.contiguous(), query.contiguous()
        if md.prefill_rows is not None:
            output.zero_()
            slots = md.slots.cpu().tolist()
            for row, start, end in md.prefill_rows:
                slot = slots[row]
                if slot < 0: continue
                if c.prefill_fa_version == 4:
                    from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd
                    native = _flash_attn_fwd(q[start:end][None], k[start:end][None], v[start:end][None],
                        causal=True, softmax_scale=self.scale, num_splits=0)[0]
                    output[start:end].copy_(native[0])
                else:
                    from vllm.vllm_flash_attn import flash_attn_varlen_func
                    bounds = torch.tensor([0, end-start], device=q.device, dtype=torch.int32)
                    flash_attn_varlen_func(q[start:end], k[start:end], v[start:end],
                        end-start, bounds, end-start, cu_seqlens_k=bounds, causal=True,
                        softmax_scale=self.scale, fa_version=2, out=output[start:end])
                prefill(self.cache, slot, md.block_table[row], k[start:end], v[start:end])
            return output
        n = md.num_actual_tokens
        # Refresh once per model forward, inside capture/replay. All layers use
        # this metadata; deriving it in the CPU builder adds one off-graph
        # launch per draft step and forces six otherwise unnecessary rebuilds.
        if c.audit_layer is self:
            _decode_positions[(md.num_reqs,)](md.query_start_loc, md.seq_lens,
                                              md.positions, md.committed, not md.drafting)
        if c.benchmark_active and c.audit_layer is self and not md.drafting:
            c.target_execution.append(dict(kind="eager", requests=md.num_reqs,
                tokens=n, max_query_len=md.max_query_len))
        self.cache.update(k[:n], v[:n], md.positions[:n], md.query_start_loc,
                          md.slots, md.committed, md.block_table)
        sparse = c.sparse_decode(md.drafting)
        return attention(self.cache, q[:n], md.positions[:n], md.query_start_loc, md.slots,
            md.block_table, bits=c.draft_bits if md.drafting else 8,
            selected_ids=c.selection.ids[:md.num_reqs] if sparse else None,
            selected_counts=c.selection.cnt[:md.num_reqs] if sparse else None,
            # The selector already resolves physical pages in this same order.
            selected_page_table=c.selection.cmp_table[:md.num_reqs] if sparse else None,
            workspace=c.workspace(md.num_reqs, 1 if md.max_query_len == 1 else 7, sparse, md.drafting), output=output[:n])


class HierarchicalAttention(Attention):
    def get_kv_cache_spec(self, vllm_config):
        if vllm_config.cache_config.block_size != 16:
            raise ValueError("hierarchical cache requires native 16-token manager pages")
        return PageLayout(self.num_kv_heads).vllm_spec()


class HierarchicalModelState(MolleModelState):
    def prepare_inputs(self, input_batch, req_states):
        c = self.model.model.hierarchical_context
        c.target_slots.fill_(-1)
        n = len(input_batch.idx_mapping)
        c.target_slots[:n].copy_(input_batch.idx_mapping)
        return super().prepare_inputs(input_batch, req_states)


class HierarchicalQwen3ForCausalLM(MolleQwen3ForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        cfg = vllm_config
        hf = cfg.model_config.hf_config
        if (cfg.cache_config.enable_prefix_caching or cfg.scheduler_config.enable_chunked_prefill or
                cfg.parallel_config.tensor_parallel_size != 1 or
                cfg.parallel_config.pipeline_parallel_size != 1 or
                getattr(hf, "molle_ffn_draft_attention", None)):
            raise ValueError("hierarchical mode requires complete BF16 prefills, TP=PP=1 and shared BF16 weights")
        if cfg.speculative_config is not None and cfg.speculative_config.num_speculative_tokens != 6:
            raise ValueError("hierarchical speculative rounds keep exactly six drafts")
        super().__init__(vllm_config=cfg, prefix=prefix)
        c = HierarchicalContext(cfg, hf.num_key_value_heads, hf.num_attention_heads, self.model.sel)
        self.model.hierarchical_context = c
        context = cfg.compilation_config.static_forward_context
        for decoder in self.model.layers:
            old = decoder.self_attn.attn
            del context[old.layer_name]
            decoder.self_attn.attn = HierarchicalAttention(old.num_heads, old.head_size, old.impl.scale,
                num_kv_heads=old.num_kv_heads, cache_config=cfg.cache_config,
                prefix=old.layer_name, attn_backend=HierarchicalBackend, hierarchical_control=c)
            if c.selection is not None:
                indexer = decoder.self_attn.indexer
                indexer.hierarchical_control = c
                indexer._score_and_select = MethodType(_select_with_logical_ids, indexer)

    @classmethod
    def get_model_state_cls(cls): return HierarchicalModelState


class HierarchicalStatsWorker(SelectionStatsWorker):
    @torch.inference_mode()
    def hierarchical_graph_check(self, num_reqs=2):
        """Replay the last of the six draft steps from identical saved state.

        This compares the compiled model with and without graph replay. It
        does not compare separately compiled prefills or commit audit outputs.
        Only bounded recent buffers and the written indexer rows are saved.
        """
        from vllm.config.compilation import CUDAGraphMode
        from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
        runner, spec = self.model_runner, self.model_runner.speculator
        if spec is None or spec.num_speculative_steps != 6 or not 1 <= num_reqs <= spec.max_num_reqs:
            raise ValueError("graph audit requires an active fixed-six speculative batch")
        c = runner.model.model.hierarchical_context
        sel, buffers, n = spec.selection, spec.input_buffers, num_reqs
        layers = [d.self_attn for d in runner.model.model.layers]
        names = ("input_ids", "positions", "seq_lens", "query_start_loc")
        inputs = {name:getattr(buffers, name).clone() for name in names}
        proposals, step = spec.draft_tokens.clone(), spec.current_draft_step.clone()
        recent = [[x.clone() for x in layer.attn.impl.cache.tail] for layer in layers]
        states = [layer.attn.impl.cache.state.clone() for layer in layers]
        errors = [layer.attn.impl.cache.error.clone() for layer in layers]
        selected = {name:x.clone() for name, x in vars(sel).items() if isinstance(x, torch.Tensor)}
        previous_mode, previous_reuse, previous_max = c.drafting, sel.reuse_scores, spec.draft_max_seq_len
        slots = c.draft_slots[:n].long()
        positions = inputs["positions"][:n]-1
        if bool((slots < 0).any()) or bool((positions < 0).any()):
            raise ValueError("graph audit needs live request slots")
        for state in states:
            if not torch.equal(state[slots, 0].long()-1, positions):
                raise ValueError("audit must immediately follow the six-step proposal")
        # Slot mappings still describe the last executed draft step.
        mapping = spec.block_tables.slot_mappings[0, :n].long().clone()
        block, offset = mapping//16, mapping%16
        indexer_rows = ([layer.indexer.k_cache.kv_cache[block, offset].clone() for layer in layers]
                        if c.selection is not None else [None]*len(layers))

        def restore():
            for name, value in inputs.items(): getattr(buffers, name).copy_(value)
            spec.draft_tokens.copy_(proposals)
            spec.current_draft_step.copy_(step)
            for layer, tail, state, error, ik in zip(layers, recent, states, errors, indexer_rows):
                cache = layer.attn.impl.cache
                for dest, value in zip(cache.tail, tail): dest.copy_(value)
                cache.state.copy_(state)
                cache.error.copy_(error)
                if ik is not None:
                    layer.indexer.k_cache.kv_cache[block, offset] = ik
            for name, value in selected.items(): getattr(sel, name).copy_(value)

        def run(graph):
            restore()
            c.drafting, sel.reuse_scores = True, sel.layer_scores is not None
            buffers.positions[:n].sub_(1)
            buffers.seq_lens[:n].sub_(1)
            buffers.input_ids[:n].copy_(proposals[:n, 4])
            spec.current_draft_step.fill_(5)
            upper = buffers.seq_lens[:n].cpu()
            spec.draft_max_seq_len = int(upper.max())
            mappings = spec.block_tables.compute_slot_mappings(spec.idx_mapping[:n],
                buffers.query_start_loc[:n+1], buffers.positions[:n], n)
            md = spec._build_draft_attn_metadata(num_reqs=n, num_reqs_padded=n,
                num_tokens_padded=n, seq_lens_cpu_upper_bound=upper, step=1)
            if graph:
                # The graph now owns draft preparation. Deliberately leave the
                # shared indexer flags in their target-prefill state to prove
                # the first proposal cannot inherit stale prefill metadata.
                from spotattention.serve.kcache import IndexerKCacheMetadata
                flags = {id(x): x.is_prefilling for x in md.values()
                         if isinstance(x, IndexerKCacheMetadata)}
                for flag in flags.values(): flag.fill_(True)
                manager = spec.graphs[int(sel.layer_scores is not None)]
                desc = next(d for d in manager.graphs if d.num_tokens == n)
                manager.run_fullgraph(desc)
                if any(bool(flag.any()) for flag in flags.values()):
                    raise RuntimeError("draft graph retained stale target-prefill flags")
            else:
                spec._step(n, n, md, build_slot_mappings_by_layer(mappings, spec.kv_cache_config),
                           None, CUDAGraphMode.NONE)
            values = []
            for layer, state in zip(layers, states):
                cache = layer.attn.impl.cache
                if bool(cache.error.any()) or not torch.equal(cache.state, state):
                    raise RuntimeError("single-step audit changed a transaction boundary")
                values.extend(x[slots, positions%263].clone() for x in cache.tail)
            return spec.draft_tokens[:n, 5].clone(), values

        try:
            graph_tokens, graph_kv = run(True)
            direct_tokens, direct_kv = run(False)
            result = dict(graph_tokens=graph_tokens.tolist(), direct_tokens=direct_tokens.tolist(),
                tokens_equal=torch.equal(graph_tokens, direct_tokens),
                kv_bitwise_equal=all(torch.equal(a, b) for a, b in zip(graph_kv, direct_kv)),
                max_abs_kv_difference=max(float((a.float()-b.float()).abs().max())
                                          for a, b in zip(graph_kv, direct_kv)),
                stale_prefill_flags_cleared=c.selection is not None,
                scope="sixth draft step, same compiled model and identical cache/input state")
            if not result["tokens_equal"] or not result["kv_bitwise_equal"]:
                raise RuntimeError(f"graph/direct disagreement: {result}")
            return result
        finally:
            restore()
            c.drafting, sel.reuse_scores = previous_mode, previous_reuse
            spec.draft_max_seq_len = previous_max

    def hierarchical_benchmark_begin(self):
        """Count target graph replays without synchronizing inside decode."""
        runner = self.model_runner
        manager = runner.cudagraph_manager
        if manager is None or not manager.graphs:
            raise RuntimeError("target decode CUDA graphs were not captured")
        if not hasattr(self, "_hierarchical_graph_original"):
            self._hierarchical_graph_original = manager.run_fullgraph
            def counted(desc, *args, **kwargs):
                self._hierarchical_target_replays += 1
                runner.model.model.hierarchical_context.target_execution.append(
                    dict(kind="graph", requests=desc.num_reqs, tokens=desc.num_tokens))
                return self._hierarchical_graph_original(desc, *args, **kwargs)
            manager.run_fullgraph = counted
        self._hierarchical_target_replays = 0
        control = runner.model.model.hierarchical_context
        control.target_execution = []
        control.draft_execution = []
        control.benchmark_active = True
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        return self.hierarchical_benchmark_info()

    def hierarchical_benchmark_info(self):
        runner = self.model_runner
        return dict(target_execution=list(runner.model.model.hierarchical_context.target_execution),
            draft_execution=list(runner.model.model.hierarchical_context.draft_execution),
            target_graph_replays=getattr(self, "_hierarchical_target_replays", 0),
            draft_graph_replays=getattr(getattr(runner, "speculator", None), "step_graph_replays", 0),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
            device_total_bytes=torch.cuda.get_device_properties(0).total_memory)

    def hierarchical_cache_info(self):
        runner = self.model_runner
        layers = [m for m in runner.model.modules() if isinstance(m, HierarchicalAttention)]
        if not layers:
            raise RuntimeError("hierarchical layers were not installed")
        errors = {layer.layer_name: layer.impl.cache.error.cpu().tolist() for layer in layers}
        if any(any(values) for values in errors.values()):
            raise RuntimeError(f"hierarchical cache transaction failed: {errors}")
        c = runner.model.model.hierarchical_context
        selection_checked = False
        if c.selection is not None:
            sel = c.selection
            for row, count in enumerate(sel.cnt[:len(sel.fa_table)].cpu().tolist()):
                if count == 0: continue
                ids = sel.ids[row, :count].long()
                if (int(ids.min()) < 0 or int(ids.max()) >= sel.fa_table.shape[1] or
                        ids.unique().numel() != count or not bool((ids[1:] > ids[:-1]).all())):
                    raise RuntimeError("logical selection ids are missing or invalid")
                if not torch.equal(sel.fa_table[row, ids], sel.cmp_table[row, :count]):
                    raise RuntimeError("logical and physical selector outputs disagree")
                selection_checked = True
        return dict(layers=len(layers), errors=errors, attention_backend=c.attention_backend,
            attention_dispatch=[dict(drafting=key[0],queries=key[1],sparse=key[2],
                token_bound=ws.shape[-1],cache_format=getattr(ws,'cache_format',None),
                split_counts=getattr(ws,'split_counts',{ws.shape[0]:ws.splits}))
                for key,ws in c.workspaces.items()],
            attention_workspace_bytes=sum(ws.memory_bytes() if hasattr(ws, 'memory_bytes') else
                ws.partial.numel()*ws.partial.element_size()+ws.lse.numel()*ws.lse.element_size()
                for ws in c.workspaces.values()),
            logical_selection_matches_native_physical_ids=selection_checked,
            page_bytes=layers[0].impl.cache.layout.page_bytes,
            memory_bytes={key:sum(layer.impl.cache.memory_bytes()[key] for layer in layers)
                          for key in ("pages", "recent", "state")},
            backbone_dtype=str(layers[0].kv_cache.dtype),
            kv_cache_groups=len(runner.kv_cache_config.kv_cache_groups),
            draft_graph_replays=getattr(getattr(runner, "speculator", None), "step_graph_replays", 0))
