# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression checks for benchmark timing, cell options and output comparisons."""

import json
from collections import Counter
from types import SimpleNamespace as NS

import pytest

from benchmarks.longspec import grid
from vllm.config.speculative import SpeculativeConfig


@pytest.mark.parametrize("mode", ["dense", "vegas", "coverage"])
@pytest.mark.parametrize(
    "ctx,gen,expected_factor",
    [
        (16384, 16384, None),
        (32768, 1, 2.0),
        (32768, 512, 2.0),
        (65535, 1, 2.0),
        (65536, 512, 3.0),
        (130560, 512, 4.0),
        (32768, 16384, 2.0),
        (65536, 16384, 3.0),
    ],
)
def test_engine_yarn_covers_prompt_and_output(
    monkeypatch, mode, ctx, gen, expected_factor
):
    import transformers

    import vllm

    # Qwen3's configured limit must not replace its 32K trained context.
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda model, **kwargs: NS(max_position_embeddings=40960, rope_theta=1000000.0),
    )
    monkeypatch.setattr(vllm, "LLM", lambda **kwargs: kwargs)
    args = grid.parse_args(
        ["--out", "/tmp/unused", "--mode", mode, "--ctx", str(ctx), "--gen", str(gen)]
    )
    engine, factor = grid.build_engine(args)
    assert factor == expected_factor
    assert engine["max_model_len"] == ctx + gen
    if expected_factor is None:
        assert "hf_overrides" not in engine
    else:
        rope = engine["hf_overrides"]["rope_parameters"]
        assert rope == {
            "rope_type": "yarn",
            "factor": expected_factor,
            "original_max_position_embeddings": 32768,
            "rope_theta": 1000000.0,
        }
        assert rope["factor"] * 32768 >= engine["max_model_len"]


def test_mode_defaults():
    args = grid.parse_args(["--out", "/tmp/unused", "--mode", "coverage"])
    assert args.gen == 512 and args.samples == 8
    ours = SpeculativeConfig(**grid.speculative_config(args))
    assert (ours.sparse_attn_ratio, ours.sparse_attn_min_tokens) == (0.07, 0)
    assert ours.sparse_attn_fixed_budget and grid.draft_top_p(args) is None
    assert ours.sparse_attn_draft_weights is None
    assert ours.sparse_attn_verify_score_scope == "full"
    args.mode, args.ratio, args.draft_weights = "vegas", 1, "unused"
    vegas = SpeculativeConfig(**grid.speculative_config(args))
    assert vegas.sparse_attn_ratio == 0.07 and vegas.sparse_attn_draft_weights is None
    args.mode = "dense"
    assert grid.speculative_config(args) is None


@pytest.mark.parametrize("scope", ["ffn", "gate_up", "down"])
def test_ffn_only_draft_options(scope):
    args = grid.parse_args([
        "--out", "/tmp/unused", "--mode", "coverage",
        "--draft-weights", "/local/pinned-int4", "--draft-weights-scope", scope,
    ])
    spec = SpeculativeConfig(**grid.speculative_config(args))
    assert spec.sparse_attn_draft_weights_scope == scope
    assert spec.sparse_attn_verify_ratio == 1
    for extra in ([], ["--draft-weights", "target"],
                  ["--mode", "dense", "--draft-weights", "int4"]):
        with pytest.raises(SystemExit):
            grid.parse_args(["--out", "/tmp/unused",
                             "--draft-weights-scope", scope, *extra])


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


@pytest.mark.parametrize("scope", ["all", "ffn", "gate_up", "down"])
def test_cells_use_parsed_options(tmp_path, monkeypatch, scope):
    args = grid.parse_args(
        [
            "--out",
            str(tmp_path),
            "--ctx=999",
            "--batch=4",
            "--measurement=batch",
            "--samples=1",
            "--cells=4096:2:dense,4096:2:coverage",
            "--draft-weights=" + ("target" if scope == "all" else "int4"),
            "--draft-weights-scope=" + scope,
        ]
    )
    calls = []

    def run(cmd, check):
        child = grid.parse_args(cmd[2:])
        calls.append(child)
        assert check and child.ctx == 4096 and child.batch == 2 and child.cells is None
        expected_scope = scope if child.mode == "coverage" else "all"
        assert child.draft_weights_scope == expected_scope
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

    args = NS(model="test", ctx=1, seed=42)
    payload = dict(
        model="test",
        ctx=1,
        seed=42,
        prompt_count=3,
        prompt_slots=[2, 0, 1, 2],
        tokens=[[7], [8], [9], [10]],
        prompts=[[1], [2], [3]],
        revision="frozen",
        gen=16,
        yarn_factor=None,
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


def test_rejects_total_context_beyond_validated_window():
    with pytest.raises(SystemExit):
        grid.parse_args(["--out", "/tmp/unused", "--ctx", "131072", "--gen", "512"])


def test_sampling_is_explicit_greedy_without_penalties():
    for ignore_eos in (False, True):
        params = grid.sampling_params(512, ignore_eos=ignore_eos)
        assert params.temperature == 0 and params.top_p == 1 and params.top_k == 0
        assert params.min_p == params.min_tokens == 0
        assert params.presence_penalty == params.frequency_penalty == 0
        assert params.repetition_penalty == 1 and params.ignore_eos == ignore_eos
        assert params.max_tokens == 512 and not params.skip_special_tokens
        params.update_from_generation_config(
            {"eos_token_id": [151645, 151643]}, model_eos_token_id=151645
        )
        assert (151643 in params.stop_token_ids) is not ignore_eos


def test_cells_preserve_disabled_budget_and_apply_sparse_options_only_to_ours(
    tmp_path, monkeypatch
):
    args = grid.parse_args(
        [
            "--out",
            str(tmp_path),
            "--cells",
            "4096:1:dense,4096:1:coverage",
            "--no-fixed-budget",
            "--verify-ratio",
            ".5",
        ]
    )
    calls = []

    def run(cmd, check):
        child = grid.parse_args(cmd[2:])
        calls.append(child)
        (tmp_path / f"tokens-{child.mode}-4096-1.json").write_text("[[1]]")

    monkeypatch.setattr(grid.subprocess, "run", run)
    grid.run_cells(args, tmp_path)
    assert all(not a.fixed_budget for a in calls)
    assert [a.verify_ratio for a in calls] == [1, 0.5]


def test_prompt_identity_includes_thinking_template_revision_and_tokenizer():
    args = NS(model="Qwen/Qwen3-8B", revision="abc", seed=42)
    tokenizer = NS(chat_template="template", get_vocab=lambda: {"a": 1})
    old = grid.prompt_protocol(args, tokenizer)
    assert old["thinking"] is False and old["dataset_revision"] == grid.DATASET_REVISION
    args.revision = "def"
    assert grid.digest(old) != grid.digest(grid.prompt_protocol(args, tokenizer))


def test_fixed_budget_cli_and_explicit_attention_mass_mode():
    args = grid.parse_args(
        [
            "--out",
            "/tmp/unused",
            "--mode",
            "coverage",
            "--fixed-budget",
            "--ratio",
            "0.07",
            "--draft-weights",
            "target",
        ]
    )
    spec = SpeculativeConfig(**grid.speculative_config(args))
    assert spec.sparse_attn_fixed_budget and spec.sparse_attn_ratio == 0.07
    assert spec.sparse_attn_draft_weights is None
    args = grid.parse_args(
        [
            "--out",
            "/tmp/unused",
            "--mode",
            "coverage",
            "--no-fixed-budget",
            "--theta",
            "0.93",
        ]
    )
    spec = SpeculativeConfig(**grid.speculative_config(args))
    assert not spec.sparse_attn_fixed_budget and spec.sparse_attn_theta == 0.93
    assert spec.sparse_attn_ratio == 0.15 and grid.draft_top_p(args) == 0.93
