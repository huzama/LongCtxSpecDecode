# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression checks for benchmark timing, cell options and output comparisons."""

import json
from types import SimpleNamespace as NS

import pytest

from benchmarks.longspec import grid
from vllm.config.speculative import SpeculativeConfig


def test_mode_defaults():
    args = grid.parse_args(["--out", "/tmp/unused", "--mode", "coverage"])
    assert args.gen == 512 and args.samples == 8
    ours = SpeculativeConfig(**grid.speculative_config(args))
    assert (ours.sparse_attn_ratio, ours.sparse_attn_min_tokens) == (0.15, 0)
    assert ours.sparse_attn_draft_weights == "RedHatAI/Qwen3-4B-quantized.w4a16"
    args.mode, args.ratio, args.draft_weights = "vegas", 1, "unused"
    vegas = SpeculativeConfig(**grid.speculative_config(args))
    assert vegas.sparse_attn_ratio == 0.07 and vegas.sparse_attn_draft_weights is None
    args.mode = "dense"
    assert grid.speculative_config(args) is None


def output(first, last, tokens):
    return NS(
        metrics=NS(first_token_ts=first, last_token_ts=last, scheduled_ts=10),
        outputs=[NS(token_ids=tokens)],
    )


def test_decode_timing():
    # Prefill duration and host wall time must not enter decode throughput.
    result = grid.decode_timing(
        [output(100, 102, [1, 2, 3]), output(100, 104, [1, 2, 3, 4, 5])]
    )
    assert result["decode_seconds"] == 4 and result["decode_tok_s"] == 1.5
    assert grid.decode_timing([output(100, 100, [1])])["decode_tok_s"] is None
    with pytest.raises(RuntimeError):
        grid.decode_timing([NS(metrics=None)])


def test_comparison_handles_prefixes_and_lengths():
    report = grid.compare_tokens([[1, 2], [1], [4, 5]], [[1], [1, 2], [4, 3]])
    assert not report["identical"]
    assert [r["first_difference"] for r in report["per_prompt"]] == [1, 1, 1]
    assert all(r["mismatches"] == 1 for r in report["per_prompt"])
    with pytest.raises(ValueError):
        grid.compare_tokens([[1]], [])


def test_cells_use_parsed_options(tmp_path, monkeypatch):
    args = grid.parse_args(
        [
            "--out",
            str(tmp_path),
            "--ctx=999",
            "--batch=4",
            "--samples=1",
            "--cells=4096:2:dense,4096:2:coverage",
            "--draft-weights=target",
        ]
    )
    calls = []

    def run(cmd, check):
        child = grid.parse_args(cmd[2:])
        calls.append(child)
        assert check and child.ctx == 4096 and child.batch == 2 and child.cells is None
        (tmp_path / f"tokens-{child.mode}-4096-2.json").write_text(
            json.dumps([[1], [2]])
        )

    monkeypatch.setattr(grid.subprocess, "run", run)
    assert grid.run_cells(args, tmp_path) == 0
    assert [c.mode for c in calls] == ["dense", "coverage"]


def test_agreement_does_not_count_ties_as_exact_matches():
    from benchmarks.longspec.w4_agreement import summarize, token_agreement

    # A prompt token and a different top-k token may both have rank 1.
    tied = {3: NS(logprob=-1.0, rank=1), 7: NS(logprob=-1.0, rank=1)}
    assert token_agreement(tied, 3) is None
    scores = {3: NS(logprob=-2.0), 7: NS(logprob=-1.0)}
    assert token_agreement(scores, 3) is False
    assert token_agreement(scores, 7) is True
    report = summarize([[True, None], [False, False]])
    assert report["non_top_tokens"] == 2 and report["tied_top_tokens"] == 1
    assert report["disagreement_bounds_per_1000"] == [500, 750]
    assert summarize([])["disagreement_bounds_per_1000"] is None
