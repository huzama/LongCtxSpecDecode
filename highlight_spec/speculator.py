"""Self-speculative drafting: the target model drafts under Highlight block selection.

Every draft step runs the target at query length 1, so Spot's indexer replaces
each layer's KV with its selected blocks. Verification runs at query length
k + 1, which Spot attends densely. Drafts write their KV into the target cache
at their own slots and verification rewrites those slots, so the output is the
target's greedy decoding. Spot's molle_verify_fraction makes verification
select too; the output is then the sparse verifier's greedy decoding.

Spot's molle_draft_select sets when drafts score: "step" scores every draft
step; "round" scores a round's first step and selects the later steps from
those scores, which skips the side-cache read. Each phase has its own graphs.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
    SpeculatorCudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    prepare_decode_inputs,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator


class HighlightSpeculator(DraftModelSpeculator):
    supports_mm_inputs = False

    def __init__(self, vllm_config, device: torch.device):
        super().__init__(vllm_config, device)
        self.current_draft_step = torch.zeros((), dtype=torch.int64, device=device)
        # One graph manager per scoring phase: [scored] or [scored, reused].
        self.graphs: list[SpeculatorCudaGraphManager] = []

    def load_draft_model(self, target_model, target_attn_layer_names):
        self.selection = getattr(target_model.model, "sel", None)
        if self.selection is None:
            raise ValueError("Highlight drafting needs a SpotAttention model with use_dsa")
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
        try:
            for step in range(self.num_speculative_steps):
                self.selection.reuse_scores = round_select and step > 0
                if not skip_attn:
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
                self.current_draft_step.fill_(step)
                if batch_desc.cg_mode == CUDAGraphMode.FULL:
                    self.graphs[int(self.selection.reuse_scores)].run_fullgraph(batch_desc)
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
    slots = idx_mapping.long()
    token = torch.where(
        num_sampled > 0, last_sampled[slots, 0].int(), next_prefill_tokens[0, slots]
    )
    n = target_seq_lens.shape[0]
    # prepare_decode_inputs advances the position by one.
    input_buffers.positions[:n] = target_seq_lens - num_rejected - 1
    prepare_decode_inputs(
        token.view(n, 1),
        target_seq_lens,
        num_rejected,
        input_buffers,
        max_model_len,
        max_num_reqs,
    )


def advance(
    draft_tokens: torch.Tensor,
    step: torch.Tensor,
    tokens: torch.Tensor,
    input_buffers: InputBuffers,
    max_model_len: int,
) -> None:
    """Record this step's drafts and feed them to the next step (graph-safe)."""
    n = draft_tokens.shape[0]
    draft_tokens.scatter_(1, step.expand(n, 1), tokens.view(n, 1))
    input_buffers.input_ids[:n] = tokens
    positions = input_buffers.positions[:n]
    positions.copy_((positions + 1).clamp(max=max_model_len - 1))
    # Padded rows keep seq_len 0, which Spot's selection treats as empty.
    seq_lens = input_buffers.seq_lens[:n]
    seq_lens.copy_(torch.where(seq_lens > 0, (seq_lens + 1).clamp(max=max_model_len), 0))
