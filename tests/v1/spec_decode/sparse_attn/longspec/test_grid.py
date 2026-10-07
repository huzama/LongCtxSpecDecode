# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression checks for benchmark timing, cell options and output comparisons."""

import json
from collections import Counter
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


def output(first, last, tokens, request_id="a", finished=False):
    return NS(
        request_id=request_id,
        finished=finished,
        metrics=NS(first_token_ts=first, last_token_ts=last, scheduled_ts=10),
        outputs=[NS(token_ids=tokens)],
    )


def test_decode_timing():
    # Prefill duration and host wall time must not enter decode throughput.
    result = grid.decode_timing([output(100, 104, [1, 2, 3, 4, 5])])
    assert result["decode_seconds"] == 4 and result["decode_tok_s"] == 1
    with pytest.raises(ValueError, match="synchronized"):
        grid.decode_timing([output(100, 102, [1, 2]), output(100, 104, [1, 2])])
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
            "--measurement=batch",
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


def test_batch_window_excludes_prefill_and_partial_batch_tail():
    window = grid.BatchWindow(["a", "b"])
    window.update([output(100, 100, [1])])
    assert window.start is None
    window.update([output(110, 110, [1], "b")])
    assert window.start == 110
    window.update([output(100, 112, [1, 2, 3]), output(110, 112, [1, 2], "b")])
    window.update(
        [
            output(100, 114, [1, 2, 3, 4, 5], finished=True),
            output(110, 114, [1, 2, 3], "b"),
        ]
    )
    # The slow request drains later, outside the full-batch measurement.
    window.update([output(110, 120, [1, 2, 3, 4, 5], "b", finished=True)])
    report = window.report()
    assert report["decode_seconds"] == 4
    assert report["decode_tokens"] == 6
    assert report["decode_tok_s"] == 1.5
    assert report["decode_tokens_per_request"] == {"a": 4, "b": 2}


def test_batch_window_rejects_early_decode_and_missing_requests():
    window = grid.BatchWindow(["a", "b"])
    with pytest.raises(RuntimeError, match="before the prefill barrier"):
        window.update([output(100, 101, [1, 2])])
    window.update([output(100, 100, [1]), output(100, 100, [1], "b")])
    with pytest.raises(RuntimeError, match="whole batch"):
        window.update([output(100, 102, [1, 2])])


def test_batch_prompt_mix_is_balanced():
    prompts = [[i] for i in range(8)]
    for batch in (1, 3, 8):
        groups = grid.prompt_groups(
            NS(measurement="batch", samples=8, batch=batch), prompts
        )
        assert len(groups) == 8
        assert Counter(slot for _, slots in groups for slot in slots) == {
            i: batch for i in range(8)
        }
        for group, slots in groups:
            assert len(set(slots)) == batch
            assert group == [prompts[i] for i in slots]


def test_scheduler_holds_ready_requests_until_last_prefill(monkeypatch):
    scheduler = grid.BatchScheduler.__new__(grid.BatchScheduler)
    ready = NS(request_id="a", num_output_tokens=1)
    prefilling = NS(request_id="b", num_output_tokens=0)
    scheduler.running = [ready, prefilling]
    scheduler.waiting = []
    scheduler.requests = {"a": ready, "b": prefilling}
    scheduler.max_num_running_reqs = 2
    scheduler._released = False
    calls = []

    def schedule(self):
        ids = [r.request_id for r in self.running]
        calls.append(ids)
        return NS(
            num_scheduled_tokens=dict.fromkeys(ids, 1),
            total_num_scheduled_tokens=len(ids),
            preempted_req_ids=set(),
        )

    monkeypatch.setattr(grid.Scheduler, "schedule", schedule)
    scheduler.schedule()
    assert calls == [["b"]] and scheduler.running == [ready, prefilling]
    prefilling.num_output_tokens = 1
    scheduler.schedule()
    assert calls[-1] == ["a", "b"] and scheduler._released


@pytest.mark.parametrize("preempted,scheduled", [({"b"}, {"a": 1}), (set(), {"a": 1})])
def test_scheduler_rejects_partial_decode(monkeypatch, preempted, scheduled):
    scheduler = grid.BatchScheduler.__new__(grid.BatchScheduler)
    scheduler._released = True
    scheduler.running = [NS(request_id="a"), NS(request_id="b")]
    scheduler.waiting = []
    monkeypatch.setattr(
        grid.Scheduler,
        "schedule",
        lambda self: NS(preempted_req_ids=preempted, num_scheduled_tokens=scheduled),
    )
    with pytest.raises(RuntimeError, match="invalid batch measurement"):
        scheduler.schedule()


def test_capacity_reserves_generation_lookahead_and_null_block(monkeypatch):
    scheduler = grid.BatchScheduler.__new__(grid.BatchScheduler)
    scheduler.requests = {}
    scheduler.max_num_running_reqs = 2
    scheduler.block_size = 16
    scheduler.num_lookahead_tokens = 6
    scheduler.kv_cache_manager = NS(block_pool=NS(num_gpu_blocks=6))
    monkeypatch.setattr(
        grid.Scheduler,
        "add_request",
        lambda self, request: self.requests.update({request.request_id: request}),
    )
    scheduler.add_request(NS(request_id="a", num_prompt_tokens=16, max_tokens=16))
    # The prompts alone fit. Full output + lookahead needs six blocks, but
    # there are only five usable blocks after the reserved null block.
    with pytest.raises(ValueError, match="needs 6 KV blocks, has 5"):
        scheduler.add_request(NS(request_id="b", num_prompt_tokens=16, max_tokens=16))


def test_saved_batch_tokens_recover_original_prompt_order(monkeypatch):
    from benchmarks.longspec import w4_agreement

    args = NS(model="test", ctx=32, seed=42)
    payload = dict(
        model="test",
        ctx=32,
        seed=42,
        prompt_count=3,
        prompt_slots=[2, 0, 1, 2],
        tokens=[[7], [8], [9], [10]],
    )
    monkeypatch.setattr(
        w4_agreement.grid, "build_prompts", lambda args: [[1], [2], [3]]
    )
    prompts, tokens = w4_agreement.source_prompts(args, payload)
    assert prompts == [[3], [1], [2], [3]] and tokens == payload["tokens"]
    assert args.batch == 1 and args.samples == 3 and args.measurement == "latency"
    payload["seed"] = 43
    with pytest.raises(ValueError, match="--seed"):
        w4_agreement.source_prompts(args, payload)
