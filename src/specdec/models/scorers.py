"""Training-free block scorers for KV selection.

Each scorer turns a decode query and a block of cached keys into one score per
block, with no trained parameters and no offline preparation. Scores are used
for ranking only, so constant factors and the attention scale are dropped.

Head aggregation: scores are averaged over query heads, because every selector
compared here (trained or not) emits one block set shared by all heads, and the
sparse kernel consumes exactly that. GQA is handled by repeating kv-head stats
across their query group.
"""

import torch


@torch.no_grad()
def block_stats(keys: torch.Tensor, block_size: int) -> dict[str, torch.Tensor]:
    """Per-block key statistics from [B, H_kv, T, D] -> [B, H_kv, n_blocks, D].

    The tail block is reduced separately so padding never enters min/max/mean.
    """
    b, h, t, d = keys.shape
    n_full = t // block_size
    parts: dict[str, list[torch.Tensor]] = {k: [] for k in ("min", "max", "mean", "std")}
    if n_full:
        full = keys[:, :, : n_full * block_size, :].view(b, h, n_full, block_size, d).float()
        parts["min"].append(full.amin(dim=3))
        parts["max"].append(full.amax(dim=3))
        parts["mean"].append(full.mean(dim=3))
        parts["std"].append(full.std(dim=3, unbiased=False))
    tail = keys[:, :, n_full * block_size :, :].float()
    if tail.shape[2]:
        parts["min"].append(tail.amin(dim=2, keepdim=True))
        parts["max"].append(tail.amax(dim=2, keepdim=True))
        parts["mean"].append(tail.mean(dim=2, keepdim=True))
        parts["std"].append(tail.std(dim=2, unbiased=False, keepdim=True))
    return {k: torch.cat(v, dim=2) for k, v in parts.items()}


def _expand_to_query_heads(stat: torch.Tensor, n_query_heads: int) -> torch.Tensor:
    h_kv = stat.shape[1]
    if n_query_heads % h_kv:
        raise ValueError(f"GQA mismatch: {n_query_heads} query heads vs {h_kv} kv heads")
    return stat.repeat_interleave(n_query_heads // h_kv, dim=1)


@torch.no_grad()
def quest_scores(query: torch.Tensor, stats: dict[str, torch.Tensor]) -> torch.Tensor:
    """Upper bound on q·k within each block: sum_d max(q_d·min_d, q_d·max_d).

    query is [B, H_q, 1, D]. Returns [n_blocks], averaged over query heads.
    """
    q = query.squeeze(2).float()  # [B, H_q, D]
    lo = _expand_to_query_heads(stats["min"], q.shape[1])
    hi = _expand_to_query_heads(stats["max"], q.shape[1])
    qe = q.unsqueeze(2)  # [B, H_q, 1, D]
    bound = torch.maximum(qe * lo, qe * hi).sum(dim=-1)  # [B, H_q, n_blocks]
    return bound.mean(dim=1).squeeze(0)


@torch.no_grad()
def mean_std_scores(
    query: torch.Tensor, stats: dict[str, torch.Tensor], spread: float = 1.0
) -> torch.Tensor:
    """Mean-key score plus a spread term: q·mean + spread·sum_d |q_d|·std_d.

    The second term is the one-sigma envelope of q·k inside the block, a softer
    stand-in for the min/max bound that does not chase single outlier keys.
    """
    q = query.squeeze(2).float()
    mean = _expand_to_query_heads(stats["mean"], q.shape[1])
    std = _expand_to_query_heads(stats["std"], q.shape[1])
    qe = q.unsqueeze(2)
    score = (qe * mean).sum(dim=-1) + spread * (qe.abs() * std).sum(dim=-1)
    return score.mean(dim=1).squeeze(0)


@torch.no_grad()
def true_block_mass(
    query: torch.Tensor, keys: torch.Tensor, scaling: float, block_size: int,
    head_agg: str = "mean",
) -> torch.Tensor:
    """Oracle: true attention probability mass per block, aggregated over query heads.

    head_agg="mean" measures the average mass a shared block set preserves across
    heads; "max" measures whether any single head needs the block. Both are
    defensible for a set shared by all heads, so scorer rankings are reported
    against both to show the choice does not drive the result.

    query [B, H_q, 1, D], keys [B, H_kv, T, D]. Returns [n_blocks].
    """
    b, h_q, _, _ = query.shape
    k = _expand_to_query_heads(keys, h_q).float()
    probs = torch.softmax((query.float() @ k.transpose(-1, -2)) * scaling, dim=-1)
    probs = probs.squeeze(2)
    probs = probs.mean(dim=1) if head_agg == "mean" else probs.amax(dim=1)  # [B, T]
    t = probs.shape[-1]
    n_blocks = (t + block_size - 1) // block_size
    padded = torch.zeros(b, n_blocks * block_size, device=probs.device, dtype=probs.dtype)
    padded[:, :t] = probs
    return padded.view(b, n_blocks, block_size).sum(-1).squeeze(0)
