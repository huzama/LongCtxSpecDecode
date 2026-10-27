# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashAttention with shared four-bit coarse/refinement KV pages."""

from dataclasses import dataclass, field

import torch
import triton
import triton.language as tl

from vllm.config import get_current_vllm_config
from vllm.platforms import current_platform
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.ops.hierarchical.cache import PagedHierarchicalKV, PageLayout
from vllm.v1.attention.ops.hierarchical.cache_kernels import prefill, update
from vllm.v1.utils import CpuGpuBuffer


class HierarchicalContext:
    """Per-model request identities and persistent per-forward metadata."""

    def __init__(self, config, heads, query_heads):
        from vllm.v1.attention.ops.hierarchical.fa4 import NativeAttentionWorkspace

        batch = config.scheduler_config.max_num_seqs
        device = config.device_config.device
        tokens = max(config.scheduler_config.max_num_batched_tokens, batch * 7)
        self.slot_buffer = CpuGpuBuffer(
            batch, dtype=torch.int32, device=device, pin_memory=True, with_numpy=True
        )
        self.slot_buffer.cpu.fill_(-1)
        self.slot_buffer.copy_to_gpu()
        self.slots = self.slot_buffer.gpu
        self.active_slots = torch.full_like(self.slots, -1)
        self.decode_slots = torch.full_like(self.slots, -1)
        self.prefill_buffer = CpuGpuBuffer(
            batch, dtype=torch.bool, device=device, pin_memory=True, with_numpy=True
        )
        self.prefill_buffer.cpu.zero_()
        self.prefill_buffer.copy_to_gpu()
        self.positions = torch.zeros(tokens, dtype=torch.int32, device=device)
        self.verify_committed = torch.zeros_like(self.slots)
        self.draft_committed = torch.zeros_like(self.slots)
        self.request_slots = {}
        self.drafting = False
        self.overrider = None
        self.first_impl = None
        self.verify_query_len = 7 if config.speculative_config is not None else 1
        self.workspaces = {
            queries: NativeAttentionWorkspace(
                batch,
                heads,
                query_heads,
                queries,
                config.model_config.max_model_len,
                device,
            )
            for queries in {1, self.verify_query_len}
        }

    def update_requests(self, req_id_to_index, finished_req_ids):
        for req_id in finished_req_ids:
            self.request_slots.pop(req_id, None)
        free = sorted(set(range(len(self.slots))) - set(self.request_slots.values()))
        self.slot_buffer.cpu.fill_(-1)
        for req_id, index in req_id_to_index.items():
            if req_id not in self.request_slots:
                if not free:
                    raise RuntimeError(
                        "int4_4 recent-KV capacity exhausted by suspended requests; "
                        "increase max_num_seqs"
                    )
                self.request_slots[req_id] = free.pop(0)
            self.slot_buffer.np[index] = self.request_slots[req_id]
        self.slot_buffer.copy_to_gpu()


@dataclass
class HierarchicalMetadata(FlashAttentionMetadata):
    prefill_slots: dict[int, int] = field(default_factory=dict)
    query_rows: list[tuple[int, int, torch.Tensor]] = field(default_factory=list)


class HierarchicalMetadataBuilder(FlashAttentionMetadataBuilder):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        layer = vllm_config.compilation_config.static_forward_context[layer_names[0]]
        self.context = layer.impl.context

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        if common_prefix_len:
            raise ValueError("int4_4 does not support shared prefix pages")
        md = super().build(0, common_attn_metadata, fast_build)
        rows = {}
        c = self.context
        c.prefill_buffer.cpu.zero_()
        if not c.drafting and not fast_build:
            starts = common_attn_metadata.query_start_loc_cpu.tolist()
            computed = common_attn_metadata.num_computed_tokens_cpu.tolist()
            for row, (start, end) in enumerate(zip(starts, starts[1:])):
                if start < end and computed[row] == 0:
                    slot = int(c.slot_buffer.np[row])
                    if slot >= 0:
                        rows[row] = slot
                        c.prefill_buffer.np[row] = True
        c.prefill_buffer.copy_to_gpu()
        query_rows = []
        if rows:
            bounds = list(zip(starts, starts[1:]))
            cu = torch.tensor(
                [[0, end - start] for start, end in bounds],
                device=c.slots.device,
                dtype=torch.int32,
            )
            query_rows = [
                (start, end, cu[row]) for row, (start, end) in enumerate(bounds)
            ]
        return HierarchicalMetadata(
            **vars(md), prefill_slots=rows, query_rows=query_rows
        )

    def build_for_cudagraph_capture(self, common_attn_metadata):
        md = self.build(0, common_attn_metadata, fast_build=True)
        return md


class HierarchicalAttentionBackend(FlashAttentionBackend):
    forward_includes_kv_cache_update = True

    @staticmethod
    def validate_config(config, head_size, dtype):
        p, s = config.parallel_config, config.speculative_config
        cap = current_platform.get_device_capability()
        if (
            cap is None
            or cap.major != 10
            or config.attention_config.flash_attn_version != 4
            or head_size != 128
            or dtype != torch.bfloat16
            or config.cache_config.block_size not in (None, 16)
            or config.model_config.hf_config.model_type != "qwen3"
            or config.model_config.quantization is not None
        ):
            raise ValueError("int4_4 requires SM100, FA4, BF16 and head128/page16")
        if (
            p.tensor_parallel_size != 1
            or p.pipeline_parallel_size != 1
            or p.data_parallel_size != 1
            or p.decode_context_parallel_size != 1
            or getattr(p, "prefill_context_parallel_size", 1) != 1
            or config.cache_config.enable_prefix_caching
            or config.scheduler_config.enable_chunked_prefill
            or config.scheduler_config.async_scheduling
            or config.kv_transfer_config is not None
        ):
            raise ValueError(
                "int4_4 requires one GPU, complete prefills, synchronous scheduling "
                "and no prefix caching or KV transfer"
            )
        if s is not None and (
            s.method != "sparse_attn"
            or s.sparse_attn_algorithm != "vegas"
            or s.num_speculative_tokens != 6
        ):
            raise ValueError("int4_4 speculation requires Vegas with six draft tokens")

    @staticmethod
    def get_impl_cls():
        return HierarchicalAttentionImpl

    @staticmethod
    def get_builder_cls():
        return HierarchicalMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [16]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"
    ):
        if block_size != 16 or head_size != 128 or cache_dtype_str != "int4_4":
            raise ValueError("invalid hierarchical KV page layout")
        return (num_blocks, PageLayout(num_kv_heads).page_bytes)

    @staticmethod
    def get_kv_cache_stride_order(include_num_layers_dimension=False):
        return (0, 1, 2) if include_num_layers_dimension else (0, 1)


@triton.jit
def _prepare_metadata(CU, LENGTH, MAP, SLOTS, PREFILL, ACTIVE, DECODE, POS, BASE):
    b = tl.program_id(0)
    first, end = tl.load(CU + b), tl.load(CU + b + 1)
    slot = tl.load(SLOTS + b)
    valid = (end > first) & (tl.load(MAP + first, end > first, -1) >= 0)
    slot = tl.where(valid, slot, -1)
    is_prefill = tl.load(PREFILL + b)
    tl.store(ACTIVE + b, slot)
    tl.store(DECODE + b, tl.where(is_prefill, -1, slot))
    base = tl.load(LENGTH + b) - (end - first)
    tl.store(BASE + b, base)
    t = tl.arange(0, 8)
    tl.store(POS + first + t, base + t, (first + t < end) & ~is_prefill)


class HierarchicalAttentionImpl(FlashAttentionImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = get_current_vllm_config()
        if (
            self.alibi_slopes is not None
            or self.sliding_window != (-1, -1)
            or self.logits_soft_cap != 0
            or self.sinks is not None
            or self.attn_type != "decoder"
            or self.kv_sharing_target_layer_name
            or self.scale != 128**-0.5
        ):
            raise ValueError("int4_4 requires plain causal decoder attention")
        previous = [
            getattr(layer.impl, "context", None)
            for layer in config.compilation_config.static_forward_context.values()
            if hasattr(layer, "impl")
        ]
        self.context = next((c for c in previous if c is not None), None)
        if self.context is None:
            self.context = HierarchicalContext(
                config, self.num_kv_heads, self.num_heads
            )
            self.context.first_impl = self
        self.cache = PagedHierarchicalKV.allocate(
            0, self.num_kv_heads, len(self.context.slots), config.device_config.device
        )

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=None,
        output_scale=None,
        output_block_scale=None,
    ):
        if output_scale is not None or output_block_scale is not None:
            raise ValueError("int4_4 does not support output quantization")
        if attn_metadata is None:
            return output.zero_()
        md, c = attn_metadata, self.context
        if md.use_cascade or not md.causal:
            raise ValueError("int4_4 requires causal attention without cascade")
        if self.cache.pages.data_ptr() != kv_cache.data_ptr():
            self.cache.bind_pages(kv_cache)
        n, batch = md.num_actual_tokens, len(md.seq_lens)
        if c.first_impl is self:
            _prepare_metadata[(batch,)](
                md.query_start_loc,
                md.seq_lens,
                md.slot_mapping,
                c.slots,
                c.prefill_buffer.gpu,
                c.active_slots,
                c.decode_slots,
                c.positions,
                c.verify_committed,
            )
        committed = c.draft_committed if c.drafting else c.verify_committed
        update(
            self.cache,
            key[:n],
            value[:n],
            c.positions[:n],
            md.query_start_loc,
            c.decode_slots[:batch],
            committed[:batch],
            md.block_table,
        )
        kwargs = dict(
            q=query[:n],
            k=key[:n],
            v=value[:n],
            out=output[:n],
            cu_seqlens_q=md.query_start_loc,
            seqused_k=md.seq_lens,
            max_seqlen_q=md.max_query_len,
            max_seqlen_k=md.max_seq_len,
            block_table=md.block_table,
            softmax_scale=self.scale,
            causal=True,
            hierarchical_cache=self.cache,
            positions=c.positions[:n],
            request_slots=c.active_slots[:batch],
            prefill_mask=c.prefill_buffer.gpu[:batch],
            prefill_slots=md.prefill_slots,
            query_rows=md.query_rows,
            _attention_fn=self.attend,
        )
        if c.overrider is None:
            self.attend(**kwargs)
        else:
            c.overrider._attention(**kwargs)
        return output

    def attend(
        self,
        *,
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        block_table,
        request_slots,
        positions,
        prefill_slots,
        query_rows,
        return_softmax_lse=False,
        selected_ids=None,
        selected_counts=None,
        **kwargs,
    ):
        from vllm.v1.attention.ops.hierarchical.fa4 import attention

        c = self.context
        if not prefill_slots:
            queries = 1 if c.drafting else c.verify_query_len
            workspace = c.workspaces[queries].for_requests(len(request_slots))
            result, lse = attention(
                self.cache,
                q,
                positions,
                cu_seqlens_q,
                request_slots,
                block_table,
                bits=4 if c.drafting else 8,
                selected_ids=selected_ids,
                selected_counts=selected_counts,
                workspace=workspace,
                output=out,
            )
        else:
            # Full prompt rows use their original BF16 keys for both attention
            # and initial Vegas scoring. Packing never changes those inputs.
            lse = torch.empty(self.num_heads, len(q), device=q.device)
            for row, (start, end, cu) in enumerate(query_rows):
                if start == end:
                    continue
                if row in prefill_slots:
                    slot = prefill_slots[row]
                    _, normalizer = flash_attn_varlen_func(
                        q=q[start:end],
                        k=k[start:end],
                        v=v[start:end],
                        out=out[start:end],
                        cu_seqlens_q=cu,
                        cu_seqlens_k=cu,
                        max_seqlen_q=end - start,
                        max_seqlen_k=end - start,
                        causal=True,
                        softmax_scale=self.scale,
                        fa_version=4,
                        return_softmax_lse=True,
                    )
                    prefill(
                        self.cache,
                        slot,
                        block_table[row],
                        k[start:end],
                        v[start:end],
                    )
                else:
                    _, normalizer = attention(
                        self.cache,
                        q[start:end],
                        positions[start:end],
                        cu,
                        request_slots[row : row + 1],
                        block_table[row : row + 1],
                        bits=8,
                        workspace=c.workspaces[c.verify_query_len].for_requests(1),
                        output=out[start:end],
                    )
                lse[:, start:end].copy_(normalizer)
            result = out
        return (result, lse) if return_softmax_lse else result
