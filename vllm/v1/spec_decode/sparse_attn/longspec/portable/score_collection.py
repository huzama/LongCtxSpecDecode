# SPDX-License-Identifier: Apache-2.0
"""Where the verification-guided scores come from.

Both collectors end in the same place: one metric per KV token per request in
the reduced buffer that top-k consumes. The kernel collector needs the vegas
flash-attention fork's FA3 op, which writes the two rows' raw scores into a
buffer during the verify pass; the recompute collector needs nothing from the
kernel and works with any paged attention backend.
"""

import torch

from vllm.v1.spec_decode.sparse_attn.attn_overrider.utils import (
    varlen_reduce,
)
from ..kernels.c2q_scores import c2q_metric
from .kernel_support import kernel_collects_scores


class KernelScoreCollector:
    """Scores written by the patched FA3 op, reduced by varlen_reduce."""

    def __init__(self, max_batch_size: int, num_query_heads: int,
                 max_model_len: int, device: torch.device):
        self._buffer = torch.empty(
            max_batch_size, num_query_heads, 2, max_model_len,
            device=device, dtype=torch.bfloat16,
        )

    def workspace(self, min_bytes: int) -> torch.Tensor | None:
        """The score buffer doubles as top-k scratch once reduced."""
        if self._buffer.nbytes >= min_bytes:
            return self._buffer.view(torch.uint8).reshape(-1)
        return None

    def verify_kwargs(self, kwargs: dict, batch_size: int) -> None:
        kwargs["scores"] = self._buffer[:batch_size]

    def reduce(self, kwargs: dict, lse: torch.Tensor, softmax_scale: float,
               valid_lens: torch.Tensor, reduce_entry: torch.Tensor,
               output: torch.Tensor, use_weight: bool) -> None:
        varlen_reduce(
            x=self._buffer[:output.shape[0]],
            valid_lens=valid_lens,
            reduce_entry=reduce_entry,
            output=output,
            lse=lse,
            cu_seqlens_q=kwargs["cu_seqlens_q"],
            softmax_scale=softmax_scale,
            use_weight=use_weight,
        )


class RecomputedScoreCollector:
    """Scores recomputed from the paged K cache by the fused Triton kernel.

    Costs one extra read of K per verify pass and no score buffer at all.
    """

    def workspace(self, min_bytes: int) -> torch.Tensor | None:
        return None

    def verify_kwargs(self, kwargs: dict, batch_size: int) -> None:
        pass

    def reduce(self, kwargs: dict, lse: torch.Tensor, softmax_scale: float,
               valid_lens: torch.Tensor, reduce_entry: torch.Tensor,
               output: torch.Tensor, use_weight: bool) -> None:
        c2q_metric(
            q=kwargs["q"],
            k_cache=kwargs["k"],
            block_table=kwargs["block_table"],
            cu_seqlens_q=kwargs["cu_seqlens_q"],
            valid_lens=valid_lens,
            reduce_entry=reduce_entry,
            lse=lse,
            softmax_scale=softmax_scale,
            use_weight=use_weight,
            output=output,
        )


def build_score_collector(source: str, fa_version: int, max_batch_size: int,
                          num_query_heads: int, max_model_len: int,
                          device: torch.device):
    """Resolve the configured source against what the loaded kernel offers."""
    available = kernel_collects_scores(fa_version)
    if source == "kernel" and not available:
        raise ValueError(
            "sparse_attn_score_source='kernel' needs the vegas flash-attention "
            f"fork's FA3 op; the loaded kernel is FA{fa_version} without "
            "score collection. Use 'recompute' or 'auto'.")
    if source == "recompute" or (source == "auto" and not available):
        return RecomputedScoreCollector()
    return KernelScoreCollector(max_batch_size, num_query_heads,
                                max_model_len, device)
