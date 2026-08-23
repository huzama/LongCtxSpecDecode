"""Selection-quality metrics: how well a scorer's ranking matches the oracle.

All metrics take a candidate score vector and the oracle attention mass over the
same block set, and compare at a matched budget k so that scorers are never
credited for spending more.
"""

import torch


def _topk_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    return scores.topk(min(k, scores.numel())).indices


def recall_at_k(scores: torch.Tensor, oracle_mass: torch.Tensor, k: int) -> float:
    """Fraction of the oracle's top-k blocks that the scorer also puts in its top-k."""
    if k <= 0:
        return float("nan")
    got = set(_topk_indices(scores, k).tolist())
    want = set(_topk_indices(oracle_mass, k).tolist())
    return len(got & want) / len(want)


def captured_mass_at_k(scores: torch.Tensor, oracle_mass: torch.Tensor, k: int) -> float:
    """True attention mass the scorer's top-k blocks carry, as a fraction of all mass."""
    total = float(oracle_mass.sum())
    if total <= 0:
        return float("nan")
    return float(oracle_mass[_topk_indices(scores, k)].sum()) / total


def mass_efficiency_at_k(scores: torch.Tensor, oracle_mass: torch.Tensor, k: int) -> float:
    """Captured mass divided by the best any selector could capture at this k.

    1.0 means the ranking is as good as the oracle's own for this budget, so this
    is the number that isolates scoring quality from how much mass exists.
    """
    best = captured_mass_at_k(oracle_mass, oracle_mass, k)
    if not best or best != best:
        return float("nan")
    return captured_mass_at_k(scores, oracle_mass, k) / best
