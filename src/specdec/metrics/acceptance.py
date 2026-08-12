"""Acceptance metrics between a draft and a target distribution.

Position-wise measurement: both models see the same prefix (the target's own
continuation, teacher-forced), so per-position acceptance is exact for a
one-token draft and an upper-bound proxy for chained drafts.

Two metrics per position:
- match:   greedy agreement (argmax equality), the acceptance of greedy decoding.
- overlap: sum_v min(p_v, q_v) = 1 - TV(p, q), the exact expected acceptance
  of lossless speculative sampling at temperature 1.
"""

import torch


@torch.no_grad()
def positionwise_alpha(
    target_logits: torch.Tensor, draft_logits: torch.Tensor, chunk: int = 1024
) -> dict:
    """Both inputs [T, V] on the same device. Returns CPU tensors [T]."""
    if target_logits.shape != draft_logits.shape:
        raise ValueError(
            f"shape mismatch: target {tuple(target_logits.shape)} vs draft {tuple(draft_logits.shape)}"
        )
    t_len = target_logits.shape[0]
    match = torch.empty(t_len, dtype=torch.bool)
    overlap = torch.empty(t_len, dtype=torch.float32)
    for lo in range(0, t_len, chunk):
        hi = min(lo + chunk, t_len)
        p = torch.softmax(target_logits[lo:hi].float(), dim=-1)
        q = torch.softmax(draft_logits[lo:hi].float(), dim=-1)
        match[lo:hi] = (p.argmax(-1) == q.argmax(-1)).cpu()
        overlap[lo:hi] = torch.minimum(p, q).sum(-1).cpu()
    return {"match": match, "overlap": overlap}


def summarize_alpha(metrics: dict) -> dict:
    return {
        "n_pos": int(metrics["match"].numel()),
        "alpha_greedy": float(metrics["match"].float().mean()),
        "alpha_overlap": float(metrics["overlap"].mean()),
    }
