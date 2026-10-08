# SPDX-License-Identifier: Apache-2.0
"""Offline sparse-verifier diagnostic: selection and decision accounting."""

import pytest
import torch

from benchmarks.longspec.sparse_verify import (
    accepted_prefix,
    compare_logits,
    select_prefix,
    summarize,
)


def test_selection_reserves_sinks_recent_and_unscored_tail():
    scores = torch.arange(80, dtype=torch.float32)
    indices = select_prefix(scores, 100, 0.1, sink=4, recent=8).tolist()
    assert indices == list(range(4)) + list(range(80, 100))
    # A requested fraction is a budget target; mandatory tokens can exceed it.
    assert len(indices) > 10


def test_selection_ranks_known_prefix_and_preserves_positions():
    scores = torch.arange(100, dtype=torch.float32)
    indices = select_prefix(scores, 100, 0.2, sink=4, recent=4).tolist()
    assert indices == list(range(4)) + list(range(84, 100))
    assert select_prefix(scores, 100, 1).tolist() == list(range(100))
    assert select_prefix(scores, 0, 0.5).numel() == 0


def test_selection_breaks_score_ties_by_earlier_position():
    indices = select_prefix(torch.ones(20), 20, 0.5, sink=0, recent=0)
    assert indices.tolist() == list(range(10))


def test_prefix_stops_at_first_rejection():
    assert accepted_prefix([1, 9, 3], [1, 2, 3]) == 1
    assert accepted_prefix([1, 2, 3, 4], [1, 2, 3]) == 3


def test_reachable_mismatches_exclude_after_reference_rejection():
    reference = torch.tensor([[0.0, 3.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    candidate = reference.clone()
    candidate[2] = torch.tensor([4.0, 0.0, 0.0])
    row = compare_logits(reference, candidate, [1, 2])
    assert row["accepted_reference"] == row["accepted_sparse"] == 1
    assert row["greedy_mismatches"] == row["strict_non_top"] == 1
    assert row["reachable_queries"] == 2
    assert row["reachable_mismatches"] == 0
    assert not row["emitted_changed"]


def test_tied_maximum_is_not_a_strict_non_top_token():
    reference = torch.tensor([[2.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    candidate = torch.tensor([[1.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    row = compare_logits(reference, candidate, [0])
    assert row["greedy_mismatches"] == row["reference_ties"] == 1
    assert row["strict_non_top"] == 0
    assert row["acceptance_changed"] and row["emitted_changed"]
    rows = [dict(fraction=0.5, kept_prefix_per_layer=[4, 6], prefix=10, **row)]
    result = summarize(rows)[0]
    assert result["blocks"] == 1
    assert result["mean_kept_prefix_fraction"] == 0.5


@pytest.mark.parametrize("fraction", [0, -0.1, 1.1])
def test_invalid_fraction(fraction):
    with pytest.raises(ValueError):
        select_prefix(torch.ones(10), 10, fraction)
