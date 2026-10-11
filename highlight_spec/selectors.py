"""Selector primitives; no engine patches or implicit refresh policy.

Layouts: Q [B, Q, Hq, D], KV [B, T, Hkv, D], token indices [B, H, K].
Vegas shares indices across heads. TriForce preserves every query head,
including separate selections for query heads that share a GQA KV head.
References: arxiv.org/abs/2602.07223v2; Infini-AI-Lab/TriForce/models/cache.py.
"""

from dataclasses import dataclass
from math import ceil

import torch


@dataclass(frozen=True)
class SelectorBudget:
    protocol: str
    prefix_tokens: int
    highlight_tokens: int
    vegas_tokens: int
    triforce_tokens: int
    triforce_chunk: int

    @classmethod
    def matched(cls, prefix_tokens: int, tokens: int, *, highlight_block: int,
                triforce_chunk: int):
        """Require exact agreement; never silently round a matched budget."""
        if min(prefix_tokens, tokens, highlight_block, triforce_chunk) <= 0:
            raise ValueError("lengths and granularities must be positive")
        if tokens > prefix_tokens:
            raise ValueError("budget exceeds prefix")
        if tokens % highlight_block or tokens % triforce_chunk:
            raise ValueError("matched tokens must divide into both native granularities")
        return cls("matched", prefix_tokens, tokens, tokens, tokens, triforce_chunk)

    @classmethod
    def paper(cls, prefix_tokens: int, *, highlight_tokens: int):
        """Vegas 7%; TriForce released on-chip setup: 4096 tokens, chunks of 8.

        These are selector budgets, not full paper experiment configurations.
        Highlight's budget is supplied explicitly. No short-context clipping.
        """
        if prefix_tokens < 4096 or not 0 < highlight_tokens <= prefix_tokens:
            raise ValueError("paper budget requires a prefix >=4096 and valid Highlight budget")
        return cls("paper", prefix_tokens, highlight_tokens,
                   ceil(prefix_tokens * 0.07), 4096, 8)


def _check_qk(q: torch.Tensor, k: torch.Tensor):
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("Q and K must have layouts [B,Q,Hq,D] and [B,T,Hkv,D]")
    if q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
        raise ValueError("Q/K batch and head dimensions must match")
    if min(*q.shape, *k.shape) <= 0 or q.shape[2] % k.shape[2]:
        raise ValueError("positive dimensions and an integral GQA group are required")
    if q.device != k.device or q.dtype != k.dtype or not q.is_floating_point():
        raise ValueError("Q/K must share floating dtype and device")


def vegas_reference_scores(q: torch.Tensor, k: torch.Tensor, *,
                           prefix_tokens: int, phase: str) -> torch.Tensor:
    """FP32 oracle. Raw QK averaged over heads and boundary queries.

    Prefill uses its last query only. Verification uses first and last query,
    including rejected drafts. Callers specify the causally valid prefix.
    This oracle rescans K and must never be labelled fused collection timing.
    """
    _check_qk(q, k)
    if phase not in {"prefill", "verify"}:
        raise ValueError("phase must be prefill or verify")
    limit = k.shape[1] if phase == "prefill" else k.shape[1] - q.shape[1] + 1
    if not 0 < prefix_tokens <= limit:
        raise ValueError("prefix is outside the selected queries' causal visibility")
    rows = q[:, -1:] if phase == "prefill" or q.shape[1] == 1 else q[:, [0, -1]]
    b, r, h, d = rows.shape
    kh = k.shape[2]
    # Group Q instead of duplicating the full KV cache for GQA.
    logits = torch.einsum("brhgd,bthd->brhgt", rows.float().reshape(b, r, kh, h // kh, d),
                          k[:, :prefix_tokens].float())
    return logits.mean(dim=(1, 2, 3))


def vegas_select(scores: torch.Tensor, *, tokens: int) -> torch.Tensor:
    if scores.ndim != 2 or not 0 < tokens <= scores.shape[-1]:
        raise ValueError("expected [B,T] scores and 0 < tokens <= T")
    return scores.topk(tokens, dim=-1, sorted=True).indices[:, None, :]


def triforce_chunk_keys(k: torch.Tensor, *, chunk_size: int) -> torch.Tensor:
    """Released retrieval-cache initialization: mean K within each chunk."""
    if k.ndim != 4 or chunk_size <= 0 or k.shape[1] == 0 or k.shape[1] % chunk_size:
        raise ValueError("KV prefix must contain complete, nonempty chunks")
    b, t, h, d = k.shape
    return k.reshape(b, t // chunk_size, chunk_size, h, d).mean(dim=2)


def triforce_scores(q: torch.Tensor, chunk_keys: torch.Tensor) -> torch.Tensor:
    """Per-query-head dot products; Q is the last prefill/rebuild query."""
    _check_qk(q, chunk_keys)
    if q.shape[1] != 1:
        raise ValueError("TriForce requires one explicitly chosen retrieval query")
    b, _, h, d = q.shape
    kh = chunk_keys.shape[2]
    return torch.einsum("bhgd,bchd->bhgc", q[:, 0].reshape(b, kh, h // kh, d),
                        chunk_keys).reshape(b, h, chunk_keys.shape[1])


def triforce_select(scores: torch.Tensor, *, tokens: int, chunk_size: int) -> torch.Tensor:
    """Force the first chunk, then descending top chunks excluding chunk 0."""
    if scores.ndim != 3 or chunk_size <= 0 or tokens <= 0 or tokens % chunk_size:
        raise ValueError("expected [B,Hq,chunks] scores and a whole-chunk budget")
    count = tokens // chunk_size
    if count > scores.shape[-1]:
        raise ValueError("budget exceeds available chunks")
    first = torch.zeros((*scores.shape[:2], 1), dtype=torch.long, device=scores.device)
    rest = scores[..., 1:].topk(count - 1, dim=-1, sorted=True).indices + 1
    chunks = torch.cat((first, rest), dim=-1)
    offsets = torch.arange(chunk_size, device=scores.device)
    return (chunks[..., None] * chunk_size + offsets).flatten(-2)


def gather_draft_kv(k: torch.Tensor, v: torch.Tensor, indices: torch.Tensor, *,
                    query_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather only selected KV; per-head selection expands Hkv to Hq.

    Shared selection keeps Hkv. TriForce returns [B,K,Hq,D], explicitly
    accounting for the GQA duplication rather than averaging queries.
    """
    if k.ndim != 4 or k.shape != v.shape or k.device != v.device or k.dtype != v.dtype:
        raise ValueError("K/V must have identical [B,T,Hkv,D] shape, dtype and device")
    if query_heads <= 0 or k.shape[2] <= 0 or query_heads % k.shape[2]:
        raise ValueError("query_heads must be a positive multiple of KV heads")
    if (indices.ndim != 3 or indices.shape[0] != k.shape[0]
            or indices.shape[1] not in {1, query_heads}
            or indices.dtype != torch.long or indices.device != k.device):
        raise ValueError("indices must be int64 [B,1 or Hq,K] on the KV device")
    # Index bounds belong to capture validation, outside timed CUDA regions.
    heads = k.shape[2] if indices.shape[1] == 1 else query_heads
    head = torch.arange(heads, device=k.device)
    kv_head = head if indices.shape[1] == 1 else head // (query_heads // k.shape[2])
    batch = torch.arange(k.shape[0], device=k.device)[:, None, None]
    token = indices.expand(-1, heads, -1)
    return tuple(x[batch, token, kv_head[None, :, None]].transpose(1, 2).contiguous()
                 for x in (k, v))


def triforce_update_tail(draft_k: torch.Tensor, draft_v: torch.Tensor,
                         accepted_k: torch.Tensor, accepted_v: torch.Tensor, *,
                         query_heads: int) -> None:
    """Source-policy tail replacement, without scoring or an implicit rebuild.

    accepted_* is the entire accepted suffix since cache initialization,
    not just this round. The caller decides when to rebuild. Overflow fails.
    """
    if (draft_k.shape != draft_v.shape or accepted_k.shape != accepted_v.shape
            or draft_k.ndim != 4 or accepted_k.ndim != 4
            or draft_k.shape[0] != accepted_k.shape[0]
            or draft_k.shape[-1] != accepted_k.shape[-1]
            or draft_k.shape[2] != query_heads or query_heads <= 0
            or accepted_k.shape[2] <= 0 or query_heads % accepted_k.shape[2]):
        raise ValueError("invalid draft/accepted KV layout")
    if any(x.device != draft_k.device or x.dtype != draft_k.dtype
           for x in (draft_v, accepted_k, accepted_v)):
        raise ValueError("draft/accepted KV dtype and device must match")
    n = accepted_k.shape[1]
    if n > draft_k.shape[1]:
        raise ValueError("accepted suffix exceeds retrieval budget; explicit rebuild required")
    if n:
        head = torch.arange(query_heads, device=draft_k.device) // (query_heads // accepted_k.shape[2])
        draft_k[:, -n:].copy_(accepted_k[:, :, head])
        draft_v[:, -n:].copy_(accepted_v[:, :, head])


class TriForceRetrievalCache:
    """Initialize once at prefill, then replace the tail with accepted KV.

    This is the user-selected released-reference policy. There is no periodic
    or acceptance-triggered rebuild. Only committed tokens may be supplied;
    speculative/rejected KV must remain in the caller's temporary cache.
    """

    refresh_policy = "prefill-once-tail-update"

    def __init__(self, retrieval_q: torch.Tensor, prefix_k: torch.Tensor,
                 prefix_v: torch.Tensor, *, tokens: int, chunk_size: int):
        chunks = triforce_chunk_keys(prefix_k, chunk_size=chunk_size)
        scores = triforce_scores(retrieval_q, chunks)
        self.indices = triforce_select(scores, tokens=tokens, chunk_size=chunk_size)
        self.query_heads = retrieval_q.shape[2]
        self.k, self.v = gather_draft_kv(prefix_k, prefix_v, self.indices,
                                        query_heads=self.query_heads)
        self.prefix_tokens = prefix_k.shape[1]
        self.accepted_tokens = 0

    def update(self, accepted_k: torch.Tensor, accepted_v: torch.Tensor) -> None:
        """Supply the entire committed suffix since prefill, including prior rounds."""
        if accepted_k.ndim != 4 or accepted_k.shape[1] < self.accepted_tokens:
            raise ValueError("committed suffix cannot shrink; rejected drafts must not be committed")
        # All checks precede mutation. Overflow is explicit; never change policy.
        triforce_update_tail(self.k, self.v, accepted_k, accepted_v,
                              query_heads=self.query_heads)
        n = accepted_k.shape[1]
        if n:
            tail = torch.arange(self.prefix_tokens, self.prefix_tokens + n,
                                device=self.indices.device)
            self.indices[..., -n:] = tail
        self.accepted_tokens = n

    @property
    def kv_bytes(self) -> int:
        return self.k.numel() * self.k.element_size() + self.v.numel() * self.v.element_size()
