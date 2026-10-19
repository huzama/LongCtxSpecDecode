"""Self-speculative drafting: the target model drafts under Highlight block selection.

Every draft step runs the target at query length 1, so Spot's indexer replaces
each layer's KV with its selected blocks. Verification runs at query length
k + 1, which Spot attends densely. Drafts write their KV into the target cache
at their own slots and verification rewrites those slots before greedy
accept/reject. Spot's molle_verify_fraction makes verification select too,
changing the target computation. Greedy verification does not establish
bitwise equivalence to an independent serial run with different kernel shapes.

Spot's molle_draft_select sets when drafts score: "step" scores every draft
step; "round" scores a round's first step and selects the later steps from
those scores, which skips the side-cache read. Each phase has its own graphs.

The optional FFN benchmark shares attention and KV with the target while
replacing only draft FFNs. It also supports full-attention drafts. The standard
one-graph-per-step execution is retained; verification remains separate.
"""

from __future__ import annotations

from typing import Any
from types import SimpleNamespace

import torch
import triton as tr
import triton.language as tl

from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
    SpeculatorCudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator


class HighlightSpeculator(DraftModelSpeculator):
    supports_mm_inputs = False

    def __init__(self, vllm_config, device: torch.device):
        super().__init__(vllm_config, device)
        self.current_draft_step = torch.zeros((), dtype=torch.int64, device=device)
        # One graph manager per scoring phase: [scored] or [scored, reused].
        self.graphs: list[SpeculatorCudaGraphManager] = []
        self.step_graph_replays = 0

    def load_draft_model(self, target_model, target_attn_layer_names):
        self.hierarchical_context = getattr(target_model.model, "hierarchical_context", None)
        hf = self.vllm_config.model_config.hf_config
        attention = getattr(hf, "molle_ffn_draft_attention", None)
        self.selection = getattr(target_model.model, "sel", None)
        if self.selection is None:
            if attention != "full" and self.hierarchical_context is None:
                raise ValueError("Highlight drafting needs a SpotAttention model with use_dsa")
            self.selection = SimpleNamespace(layer_scores=None, reuse_scores=False)
        weight_checkpoint = getattr(hf, "molle_draft_weight_checkpoint", None)
        if weight_checkpoint is not None:
            if attention is not None or self.hierarchical_context is None:
                raise ValueError("decoder weight pilot requires hierarchical KV without FFN-only overrides")
            from highlight_spec.weight_draft import WeightDraftModel
            return WeightDraftModel(target_model, weight_checkpoint)
        if attention is not None:
            from highlight_spec.ffn_draft import FFNDraftModel
            return FFNDraftModel(target_model, getattr(hf, "molle_ffn_draft_checkpoint", None))
        return target_model

    def set_attn(
        self,
        model_state,
        kv_cache_config,
        block_tables,
        target_input_buffers,
        target_attn_groups,
    ) -> None:
        # Drafts run the target's own layers, so they build metadata through
        # the target's attention groups and share its KV cache.
        self.model_state = model_state
        self.kv_cache_config = kv_cache_config
        self.block_tables = block_tables
        self.attn_groups = target_attn_groups
        self.target_input_buffers = target_input_buffers
        self.target_attn_groups = target_attn_groups

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
            mode = CUDAGraphMode.FULL_DECODE_ONLY
        else:
            mode = CUDAGraphMode.NONE
        phases = 1 if self.selection.layer_scores is None else 2
        self.graphs = [
            SpeculatorCudaGraphManager(self.vllm_config, self.device, mode, decode_query_len=1)
            for _ in range(phases)
        ]

    def capture(self) -> None:
        hierarchical = getattr(self, "hierarchical_context", None)
        if hierarchical is not None:
            hierarchical.drafting = True
        try:
            for phase, graphs in enumerate(self.graphs):
                self.selection.reuse_scores = phase > 0
                graphs.capture(
                    self._step,
                    self.model_state,
                    self.input_buffers,
                    self.block_tables,
                    self.attn_groups,
                    self.kv_cache_config,
                    progress_bar_desc="Capturing draft CUDA graphs",
                )
        finally:
            self.selection.reuse_scores = False
            if hierarchical is not None:
                hierarchical.drafting = False

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        upper = input_batch.seq_lens_cpu_upper_bound
        self.draft_max_seq_len = min(
            int(upper[:num_reqs].max()) + self.num_speculative_steps, self.max_model_len
        )
        self._copy_request_inputs(num_reqs, input_batch.idx_mapping, temperature, seeds)
        prepare_first_step(
            self.input_buffers,
            input_batch.seq_lens[:num_reqs],
            input_batch.idx_mapping,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            self.max_model_len,
            self.max_num_reqs,
        )
        self.current_draft_step.zero_()
        # The graph managers exist from KV-cache init; the profile run precedes it.
        batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(
            self.graphs[0] if self.graphs else None,
            num_reqs,
            num_reqs,
            uniform_token_count=1,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=is_profile,
        )
        idx_mapping = self.idx_mapping[:num_reqs]
        positions = self.input_buffers.positions[:num_reqs]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
        skip_attn = dummy_run and skip_attn_for_dummy_run
        attn_metadata = slot_mappings_by_layer = None
        round_select = self.selection.layer_scores is not None
        hierarchical = getattr(self, "hierarchical_context", None)
        # One CPU record per proposal ties its graph count to the target call.
        # Aggregate counts also include prefill and post-window work and cannot
        # prove that each timed round used all six draft graphs.
        audit_start = None
        if hierarchical is not None and hierarchical.benchmark_active:
            audit_start = (len(hierarchical.target_execution)-1, self.step_graph_replays)
        if hierarchical is not None:
            hierarchical.draft_slots.fill_(-1)
            hierarchical.draft_slots[:num_reqs].copy_(input_batch.idx_mapping)
            hierarchical.draft_committed[:num_reqs].copy_(input_batch.seq_lens[:num_reqs]-num_rejected)
            hierarchical.drafting = True
        try:
            for step in range(self.num_speculative_steps):
                self.selection.reuse_scores = round_select and step > 0
                # Packed metadata uses persistent device buffers. Its positions
                # and native slot mapping are refreshed inside each draft graph.
                # Other backends retain their native per-step metadata builder.
                packed_graph = hierarchical is not None and batch_desc.cg_mode == CUDAGraphMode.FULL
                if not skip_attn and not packed_graph:
                    if hierarchical is None:
                        slot_mappings = self.block_tables.compute_slot_mappings(
                            idx_mapping, query_start_loc, positions, batch_desc.num_tokens
                        )
                        slot_mappings_by_layer = build_slot_mappings_by_layer(
                            slot_mappings, self.kv_cache_config
                        )
                    attn_metadata = self._build_draft_attn_metadata(
                        num_reqs=num_reqs,
                        num_reqs_padded=batch_desc.num_reqs or num_reqs,
                        num_tokens_padded=batch_desc.num_tokens,
                        seq_lens_cpu_upper_bound=upper,
                        step=step + 1,
                    )
                if batch_desc.cg_mode == CUDAGraphMode.FULL:
                    self.graphs[int(self.selection.reuse_scores)].run_fullgraph(batch_desc)
                    self.step_graph_replays += 1
                else:
                    self._step(
                        num_reqs,
                        batch_desc.num_tokens,
                        attn_metadata,
                        slot_mappings_by_layer,
                        num_tokens_across_dp,
                        batch_desc.cg_mode,
                    )
        finally:
            self.selection.reuse_scores = False
            if hierarchical is not None:
                hierarchical.drafting = False
        if audit_start is not None:
            target_event, graph_start = audit_start
            hierarchical.draft_execution.append(dict(
                target_event_index=target_event, requests=num_reqs,
                tokens=batch_desc.num_tokens, steps=self.num_speculative_steps,
                graph_replays=self.step_graph_replays-graph_start))
        return self.draft_tokens[:num_reqs]

    def _step(
        self,
        num_reqs: int,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        if self.hierarchical_context is not None and attn_metadata is not None:
            mappings = self.block_tables.compute_slot_mappings(
                self.idx_mapping[:num_reqs], self.input_buffers.query_start_loc[:num_reqs+1],
                self.input_buffers.positions[:num_reqs], num_tokens)
            slot_mappings = build_slot_mappings_by_layer(mappings, self.kv_cache_config)
            # The indexer builder shares this flag buffer with the target.
            # A proposal immediately after prefill must clear it in the graph,
            # even when no draft metadata builder runs on the host.
            from spotattention.serve.kcache import IndexerKCacheMetadata
            seen = set()
            for metadata in attn_metadata.values():
                if isinstance(metadata, IndexerKCacheMetadata) and id(metadata) not in seen:
                    metadata.is_prefilling.zero_()
                    seen.add(id(metadata))
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=BatchDescriptor(num_tokens=num_tokens),
        ):
            hidden_states = self.model(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.input_buffers.positions[:num_tokens],
            )
        positions = self.input_buffers.positions[:num_reqs]
        tokens = self.sample_draft(
            hidden_states[:num_reqs],
            positions,
            self.idx_mapping[:num_reqs],
            self.temperature,
            self.seeds,
            self.current_draft_step,
            self.draft_logits,
        )
        advance(
            self.draft_tokens[:num_reqs],
            self.current_draft_step,
            tokens,
            self.input_buffers,
            self.max_model_len,
        )


@tr.jit
def _prepare_first_step(SEQ, IDX, SAMPLED, REJECTED, LAST, NEXT, INPUT, POS, QSL, LENGTHS,
                        LAST_STRIDE: tl.constexpr, NEXT_STRIDE: tl.constexpr,
                        N: tl.constexpr, MAX_N: tl.constexpr, MAX_LEN: tl.constexpr,
                        BLOCK: tl.constexpr):
    row = tl.arange(0, BLOCK)
    active = row < N
    slot = tl.load(IDX+row, active, 0)
    sampled = tl.load(SAMPLED+row, active, 0)
    token = tl.where(sampled > 0,
                    tl.load(LAST+slot*LAST_STRIDE, active, 0),
                    tl.load(NEXT+slot*NEXT_STRIDE, active, 0))
    base = tl.load(SEQ+row, active, 0)-tl.load(REJECTED+row, active, 0)
    tl.store(INPUT+row, token, active)
    tl.store(POS+row, tl.minimum(base, MAX_LEN-1), active)
    tl.store(LENGTHS+row, tl.where(active, tl.minimum(base+1, MAX_LEN), 0), row < MAX_N)
    tl.store(QSL+row, tl.minimum(row, N), row <= MAX_N)


def prepare_first_step(
    input_buffers: InputBuffers,
    target_seq_lens: torch.Tensor,
    idx_mapping: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    last_sampled: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    max_model_len: int,
    max_num_reqs: int,
) -> None:
    """First draft input per row: the bonus token, right after the accepted prefix.

    A row mid chunked prefill samples nothing; it drafts from its next prompt
    token, whose KV the next chunk recomputes.
    """
    n = target_seq_lens.shape[0]
    _prepare_first_step[(1,)](target_seq_lens, idx_mapping, num_sampled, num_rejected,
        last_sampled, next_prefill_tokens, input_buffers.input_ids, input_buffers.positions,
        input_buffers.query_start_loc, input_buffers.seq_lens,
        last_sampled.stride(0), next_prefill_tokens.stride(1), n, max_num_reqs,
        max_model_len, tr.next_power_of_2(max_num_reqs+1))


@tr.jit
def _advance(DRAFTS, STEP, TOKENS, INPUT, POS, LENGTHS, N: tl.constexpr,
             STRIDE: tl.constexpr, STEPS: tl.constexpr, MAX_LEN: tl.constexpr,
             BLOCK: tl.constexpr):
    row = tl.arange(0, BLOCK)
    mask = row < N
    step = tl.load(STEP)
    token = tl.load(TOKENS+row, mask, 0)
    pos = tl.load(POS+row, mask, 0)
    length = tl.load(LENGTHS+row, mask, 0)
    tl.store(DRAFTS+row*STRIDE+step, token, mask)
    tl.store(INPUT+row, token, mask)
    tl.store(POS+row, tl.minimum(pos+1, MAX_LEN-1), mask)
    tl.store(LENGTHS+row, tl.where(length > 0, tl.minimum(length+1, MAX_LEN), 0), mask)
    # One CTA owns the scalar. Cycling also keeps capture warmups in bounds.
    tl.store(STEP, (step+1) % STEPS)


def advance(
    draft_tokens: torch.Tensor,
    step: torch.Tensor,
    tokens: torch.Tensor,
    input_buffers: InputBuffers,
    max_model_len: int,
) -> None:
    """Record this step's drafts and feed them to the next step (graph-safe)."""
    n = draft_tokens.shape[0]
    _advance[(1,)](draft_tokens, step, tokens, input_buffers.input_ids,
        input_buffers.positions, input_buffers.seq_lens, n, draft_tokens.stride(0),
        draft_tokens.shape[1], max_model_len, tr.next_power_of_2(n))
