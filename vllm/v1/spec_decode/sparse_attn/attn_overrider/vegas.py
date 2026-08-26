# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Vegas: https://arxiv.org/abs/2602.07223
import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.spec_decode.sparse_attn.attn_overrider import BaseAttnOverrider
from vllm.v1.spec_decode.sparse_attn.longspec.portable.draft_kv import (
    build_draft_kv,
)
from vllm.v1.spec_decode.sparse_attn.longspec.portable.score_collection import (
    build_score_collector,
)
from vllm.v1.spec_decode.sparse_attn.attn_overrider.utils import (
    varlen_topk,
    calc_topk_workspace_size,
    autotune_path,
)
from vllm.v1.spec_decode.sparse_attn.longspec.kernels.slot_table import (
    index_to_slots,
)
from vllm.v1.spec_decode.sparse_attn.longspec.portable.kernel_support import (
    flash_attn_version,
)

logger = init_logger(__name__)


class VegasAttnOverrider(BaseAttnOverrider):
    """Verification-guided sparse attention."""

    # Top-k ranking mode (class-level toggle):
    #   "logit"  -> rank by raw (pre-softmax) attention scores.
    #   "weight" -> rematerialize softmax attention weights, i.e.
    #               exp(scale * score - lse), and rank by those.
    SCORE_MODE = "weight"

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(vllm_config, device)
        parallel_config = vllm_config.parallel_config
        self.block_size = vllm_config.cache_config.block_size
        self.num_query_heads = \
            vllm_config.model_config.get_num_attention_heads(parallel_config)

        # Weight mode collects the attention log-sum-exp so the reduce kernel
        # can rematerialize softmax weights; see SCORE_MODE above.
        self._use_weight = self.SCORE_MODE == "weight"
        # Persistent placeholder passed for lse in logit mode (kernel ignores).
        self._lse_placeholder = torch.empty(
            1, device=self.device, dtype=torch.float32)

        self._reduced_score_buffer = torch.empty(
            self.max_batch_size, self.max_model_len,
            device=self.device, dtype=torch.bfloat16
        )

        # Width of the per-request top-k output row, i.e. the max_k passed to
        # varlen_topk (top-k slots + recent/draft slots). Drives both the page
        # table and the top-k workspace sizing, so they stay consistent.
        self._topk_width = self.max_tokens + 2 * self.num_spec_tokens + 1

        # Strategies for the two kernel features the method relies on, chosen
        # against the binary actually loaded (see longspec/portable/kernel_support.py).
        speculative_config = vllm_config.speculative_config
        fa_version = flash_attn_version()
        self._scores = build_score_collector(
            speculative_config.sparse_attn_score_source, fa_version,
            self.max_batch_size, self.num_query_heads, self.max_model_len,
            self.device,
        )
        model_config = vllm_config.model_config
        self._draft_kv = build_draft_kv(
            speculative_config.sparse_attn_draft_kv, fa_version,
            self.num_layers, self.max_batch_size, self._topk_width,
            self.block_size, self.device,
            model_config.get_num_kv_heads(parallel_config),
            model_config.get_head_size(), model_config.dtype,
        )
        logger.info(
            "Vegas on FA%d: scores via %s, draft KV via %s", fa_version,
            type(self._scores).__name__, type(self._draft_kv).__name__,
        )

        # Top-k workspace: the kernel collector's score buffer once reduced,
        # otherwise a dedicated buffer.
        topk_workspace_size = calc_topk_workspace_size(
            self.max_batch_size, self.max_model_len, self._topk_width
        )
        self._topk_workspace = self._scores.workspace(topk_workspace_size)
        if self._topk_workspace is None:
            self._topk_workspace = torch.empty(
                topk_workspace_size, device=self.device, dtype=torch.uint8
            )

        # The per-token page table for draft attention.
        self._page_table = torch.empty(
            self.num_layers, self.max_batch_size, self._topk_width,
            device=self.device, dtype=torch.int32
        )

        # Miscellaneous pre-request buffers.
        self._topk_budget = torch.empty(
            self.max_batch_size, device=self.device, dtype=torch.int32
        )
        self._reduce_entry = torch.empty(
            self.max_batch_size, device=self.device, dtype=torch.int32
        )
        self._topk_valid_lens = torch.empty(
            self.max_batch_size, device=self.device, dtype=torch.int32
        )

        # Whether sparsity metadata has been initialized.
        self._metadata_initialized = False

        # Learn the top-k single/multi-block crossover for this GPU/shape once,
        # here (outside any CUDA-graph capture) so the per-step call is a cheap
        # cached lookup. Best-effort: fall back to the built-in heuristic if it
        # fails for any reason.
        try:
            autotune_path(
                max_len=self.max_model_len,
                max_batch_size=self.max_batch_size,
                device=self.device,
                dtype=torch.bfloat16,
                sparse_ratio=self.sparse_ratio,
            )
        except Exception:
            pass

    def _init_metadata(self, *args, **kwargs):
        self._metadata_initialized = True
        seqlens_k: torch.Tensor = kwargs["seqused_k"]
        block_table: torch.Tensor = kwargs["block_table"]

        # We should be inside the first drafting step, and we need to
        # prepare the page table for all drafting steps.
        index_to_slots(
            self._page_table[:, :self.batch_size],
            block_table,
            self._topk_budget,
            self._topk_valid_lens,
            seqlens_k - 1 + self.num_spec_tokens,
            self.block_size,
        )

        # Repurpose topk_budget to store the number of total valid slots
        # for the draft attention (top-k slots + recent tokens).
        self._topk_budget[:self.batch_size] += \
            seqlens_k - self._topk_valid_lens[:self.batch_size]

    def enter_propose(self):
        super().enter_propose()
        # Mark that the draft page table/budgets must be (re)built on the first
        # draft step of this propose. This runs in eager Python on every
        # propose call (it is never captured into a CUDA graph), so it stays
        # correct even when the verify pass is replayed as a graph and its
        # Python body (which used to reset this flag) does not execute.
        self._metadata_initialized = False
        self._draft_kv.begin_propose()

    def _draft_attention(self, *args, **kwargs):
        # Derive the live batch size from this step's own metadata. Under CUDA
        # graphs the verify pass is replayed as a graph, so the Python
        # assignment of self.batch_size in _verify_attention does not re-run
        # and would be stale here. Draft attention runs eagerly each step (the
        # piecewise split point), so it must match its own inputs to avoid
        # "batch_size must be equal to batch_size_k" in FA3. Use the number of
        # sequences (cu_seqlens_q has batch_size + 1 entries), not q.shape[0],
        # which is the token count (batch*query_len).
        self.batch_size = kwargs["cu_seqlens_q"].numel() - 1

        # Will be executed only once, resolve sparse kv slots.
        if self.curr_layer == 0:
            if self._metadata_initialized:
                # Increase the used KV entries by one.
                self._topk_budget[:self.batch_size] += 1
            else:
                self._init_metadata(*args, **kwargs)

        self._draft_kv.attention_kwargs(
            kwargs, self.curr_layer,
            self._page_table[self.curr_layer, :self.batch_size],
            self._topk_budget[:self.batch_size],
        )
        return BaseAttnOverrider._original_attn_func(*args, **kwargs)

    def _verify_attention(self, *args, **kwargs):
        # Will be executed only once, compute metadata.
        if self.curr_layer == 0:
            seqlens_k: torch.Tensor = kwargs["seqused_k"]
            cu_seqlens_q: torch.Tensor = kwargs['cu_seqlens_q']
            seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
            self.batch_size = seqlens_k.shape[0]

            # Edge case handling when reducing attention scores.
            single_query = seqlens_q == 1
            in_prefill = seqlens_q > self.num_spec_tokens + 1
            self._reduce_entry.fill_(0)
            self._reduce_entry[:self.batch_size].masked_fill_(single_query, 1)
            self._reduce_entry[:self.batch_size].masked_fill_(in_prefill, 2)

            # The range for top-k selection: [0, seqlens_k - seqlens_q + 1)
            # In prefill: [0, seqlens_k)
            self._topk_valid_lens[:self.batch_size] = seqlens_k + 1 - \
                seqlens_q.masked_fill(in_prefill, 1)
            # Rows padded for a CUDA graph have no query and a block table
            # of -1; scoring them would read outside the cache.
            self._topk_valid_lens[:self.batch_size].masked_fill_(
                seqlens_q == 0, 0)

            # Calculate per-request budget.
            self._topk_budget[:self.batch_size] = \
                torch.ceil(seqlens_k * self.sparse_ratio).int()
            self._topk_budget.clamp_(min=self.min_tokens, max=self.max_tokens)
            # Ensure top-k budget does not exceed valid lengths.
            self._topk_budget[:self.batch_size].clamp_max_(
                self._topk_valid_lens[:self.batch_size])

            # Metadata has been reset.
            self._metadata_initialized = False

        # Collect the two query rows' scores, from the kernel or recomputed.
        self._scores.verify_kwargs(kwargs, self.batch_size)
        if self._use_weight:
            kwargs["return_softmax_lse"] = True
            out, softmax_lse = BaseAttnOverrider._original_attn_func(
                *args, **kwargs)
            softmax_scale = kwargs.get("softmax_scale")
            if softmax_scale is None:
                softmax_scale = kwargs["q"].shape[-1] ** -0.5
        else:
            out = BaseAttnOverrider._original_attn_func(*args, **kwargs)
            softmax_lse = self._lse_placeholder
            softmax_scale = 1.0

        # One metric per token, then top-k into this layer's page table.
        self._scores.reduce(
            kwargs=kwargs, lse=softmax_lse, softmax_scale=softmax_scale,
            valid_lens=self._topk_valid_lens[:self.batch_size],
            reduce_entry=self._reduce_entry[:self.batch_size],
            output=self._reduced_score_buffer[:self.batch_size],
            use_weight=self._use_weight,
        )
        varlen_topk(
            metric=self._reduced_score_buffer[:self.batch_size],
            topks=self._topk_budget[:self.batch_size],
            valid_lens=self._topk_valid_lens[:self.batch_size],
            output=self._page_table[self.curr_layer, :self.batch_size],
            buf=self._topk_workspace,
        )

        return out
