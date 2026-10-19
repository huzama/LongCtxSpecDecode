"""Audit useful throughput, censored speculative rounds, and queue publication."""
import argparse
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import eos_throughput as eos
from benchmarks import queue
from benchmarks.multilexsum import digest, output_audit


@pytest.fixture
def window_class():
    # The timing class has no runtime dependency. Load it without importing
    # a different local vLLM version from the B200's pinned serving runtime.
    path = Path(eos.__file__).with_name("decode.py")
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BatchWindow")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["BatchWindow"]


def output(key, count, timestamp, finished=False, first=1.):
    return SimpleNamespace(request_id=key, finished=finished,
        outputs=[SimpleNamespace(token_ids=[1] * count)],
        metrics=SimpleNamespace(first_token_ts=first, last_token_ts=timestamp))


def test_prefill_and_tail_are_outside_window(window_class):
    window = window_class(["a", "b"], trace=True)
    window.update([output("a", 1, 1.)])
    window.update([output("b", 1, 2., first=2.)])
    window.update([output("a", 5, 3.), output("b", 4, 3., first=2.)])
    window.update([output("a", 7, 4., True), output("b", 6, 4., first=2.)])
    window.update([output("b", 12, 9., True, first=2.)])
    record = window.report()
    assert record["decode_tokens"] == 11
    assert record["decode_seconds"] == 2
    assert record["decode_tok_s"] == 5.5
    assert len(record["decode_steps"]) == 2
    assert record["decode_steps"][-1]["after"] == [7, 6]


def test_eos_before_barrier_fails_without_hanging(window_class):
    window = window_class(["a", "b"], trace=True)
    with pytest.raises(RuntimeError, match="before the prefill barrier"):
        window.update([output("a", 1, 1., True)])


def test_missing_request_in_timed_step_is_rejected(window_class):
    window = window_class(["a", "b"], trace=True)
    window.update([output("a", 1, 1.), output("b", 1, 1.)])
    with pytest.raises(RuntimeError, match="whole batch"):
        window.update([output("a", 2, 2.)])


def test_terminal_eos_is_useful_but_not_an_acceptance_observation():
    steps = [dict(before=[1, 1], after=[5, 4], finished=[False, False]),
             dict(before=[5, 4], after=[7, 6], finished=[True, False])]
    value = eos.useful_speculation(steps, 6, 512)
    assert value["useful_tokens"] == 11 and value["verification_rounds"] == 4
    assert value["uncensored_rounds"] == 3
    assert value["accepted_in_uncensored_rounds"] == 6
    assert value["uncensored_acceptance"] == 6 / 18
    assert value["excluded_acceptance_rounds"] == 1


def test_cap_shortened_unfinished_round_is_not_counted_in_acceptance():
    value = eos.useful_speculation([dict(before=[509], after=[511], finished=[False]),
                                   dict(before=[511], after=[512], finished=[True])], 6, 512)
    assert value["useful_tokens_per_round"] == 1.5
    assert value["uncensored_acceptance"] is None
    assert value["excluded_acceptance_rounds"] == 2


@pytest.fixture
def evidence():
    prompts = [dict(id=str(i), prompt_token_ids=[i] * 8, prompt_sha256=digest([i] * 8)) for i in range(32)]
    job = dict(ctx=8, batch=2, mode="highlight", format="mxfp4", verify_fraction=.5)
    records = []
    for i in range(32):
        slots = [i, (i + 1) % 32]
        steps = [dict(before=[1, 1], after=[5, 4], finished=[False, False], elapsed_seconds=1.),
                 dict(before=[5, 4], after=[7, 6], finished=[True, False], elapsed_seconds=2.)]
        outputs = []
        for slot, ids in zip(slots, ([1] * 6 + [9], [2] * 511 + [3]), strict=True):
            reason = "stop" if ids[-1] == 9 else "length"
            outputs.append(dict(prompt_slot=slot, output_token_ids=ids, text="Summary.", finish_reason=reason,
                **output_audit(ids, [9], 512, reason)))
        records.append(dict(slot=i, prompt_slots=slots, prompt_sha256=[prompts[s]["prompt_sha256"] for s in slots],
            ctx=8, batch=2, gen=512, ignore_eos=False, thinking=False, temperature=0,
            mode="highlight", indexer_cache_format="mxfp4", flash_attn_version=4,
            draft_attention=None, draft_ffn_checkpoint=None, eos_token_ids=[9],
            selection=dict(molle_verify_fraction=.5, molle_topk_fraction=.07, molle_topk_recency=64, molle_draft_select="round"),
            kv_capacity=dict(fits=True, batch=2), request_outputs=outputs,
            decode_steps=steps, decode_tokens=11, decode_seconds=2., decode_tok_s=5.5,
            decode_window="all_prefills_complete_to_first_request_finish",
            speculation=eos.useful_speculation(steps, 6, 512),
            native_spec_counters=dict(drafts=4, accepted=7, draft_tokens=24)))
    return job, records, prompts


def test_audit_preserves_early_eos_and_excludes_drained_tail(evidence):
    result = eos.audit_records(*evidence)
    assert result["decode_tok_s"] == 5.5
    assert result["eos_stops"] == result["cap_stops"] == 32
    assert result["useful_tokens_per_round"] == 2.75


@pytest.mark.parametrize("change", ["post_eos", "trace_tail", "time", "missing", "slots", "forced", "budget", "tokens", "acceptance", "backend"])
def test_bad_evidence_cannot_reach_the_table(evidence, change):
    job, rows, prompts = copy.deepcopy(evidence)
    if change == "post_eos":
        rows[0]["request_outputs"][0]["output_token_ids"].append(5)
    elif change == "trace_tail":
        rows[0]["decode_steps"].append(dict(before=[7, 6], after=[8, 7], finished=[False, False], elapsed_seconds=3.))
    elif change == "time":
        rows[0]["decode_seconds"] = 1.
    elif change == "missing":
        rows.pop()
    elif change == "slots":
        rows[0]["prompt_slots"] = [1, 0]
    elif change == "forced":
        rows[0]["ignore_eos"] = True
    elif change == "budget":
        rows[0]["selection"]["molle_verify_fraction"] = .4
    elif change == "tokens":
        prompts[0]["prompt_token_ids"][0] = 55
    elif change == "acceptance":
        rows[0]["speculation"]["accepted_in_uncensored_rounds"] += 1
    else:
        rows[0]["flash_attn_version"] = 2
    with pytest.raises(ValueError):
        eos.audit_records(job, rows, prompts)


def test_full_grid_and_common_capacity_dependencies(tmp_path):
    manifests = {}
    for model in ("4b", "8b"):
        for ctx in eos.CONTEXTS:
            p = tmp_path / f"{model}-{ctx}.json"
            p.write_text("[]")
            manifests[model, ctx] = p
    jobs = eos.make_jobs(tmp_path, tmp_path, {"4b": "four", "8b": "eight"}, "quant", manifests)
    assert len(jobs) == 225  # 9 capacity probes and 216 fresh process runs.
    assert sum(j["kind"] == "eos-throughput" for j in jobs) == 216
    assert len({j["name"] for j in jobs}) == len(jobs)
    for job in jobs:
        assert "--eos-aware" in job["command"]
        if job["kind"] == "eos-throughput":
            assert "{batch}" in job["command"] and job["capacity_jobs"]
            assert all(d in {j["name"] for j in jobs} for d in job["depends_on"])


def test_batch_is_matched_to_smallest_capacity(tmp_path):
    for name, capacity in (("sv50", 25), ("ffn", 24)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "capacity.json").write_text(json.dumps(dict(maximum_batch=capacity)))
    job = dict(requested_batch=25, capacity_jobs=["sv50", "ffn"], command=["--batch", "{batch}"])
    assert eos.resolve_batch(job, tmp_path) == (24, ["--batch", "24"])


def test_single_batch_uses_first_distinct_prompts_without_cyclic_repeats():
    path = Path(eos.__file__).with_name("decode.py")
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "prompt_groups")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    args = SimpleNamespace(batch=6, samples=32, single_batch=True)
    prompts = [[i] for i in range(32)]
    assert namespace["prompt_groups"](args, prompts) == [(prompts[:6], list(range(6)))]


def test_single_batch_audit_keeps_pool_and_repetition_counts_separate(evidence):
    job, rows, prompts = copy.deepcopy(evidence)
    job.update(expected_rows=1, prompt_pool_size=32, single_batch=True)
    assert eos.audit_records(job, rows[:1], prompts)["batches"] == 1
    with pytest.raises(ValueError):
        eos.audit_records(job, rows, prompts)
    flags = eos.engine_flags(job | {"manifest": "prompts.json"}, "model", None)
    assert flags[flags.index("--samples") + 1] == "32" and "--single-batch" in flags


def test_ffn_priority_has_separate_dense_controls_and_blocks_other_dispatch(tmp_path):
    manifest = tmp_path / "prompts.json"
    manifest.write_text("[]")
    manifests = {(m, c): manifest for m in ("4b", "8b") for c in eos.CONTEXTS}
    original = eos.make_jobs(tmp_path, tmp_path, {"4b": "four", "8b": "eight"}, "quant", manifests)
    original = [dict(j, state="pending") for j in original]
    jobs = eos.ffn_priority_jobs(original, tmp_path / "new")
    priority = [j for j in jobs if j["kind"] == "eos-throughput" and j["method"] in (*eos.FFN, "ffn-dense")]
    assert len(priority) == 45
    assert all(j["expected_rows"] == 1 and j["prompt_pool_size"] == 32 for j in priority)
    assert all("--single-batch" in j["command"] for j in priority)
    visited = []
    while job := queue.eligible(jobs):
        visited.append(job)
        job["state"] = "complete"
    assert len(visited) == len(jobs)  # no dependency cycles
    last_priority = max(visited.index(j) for j in priority)
    assert all(visited.index(j) > last_priority for j in jobs
               if j["kind"] == "eos-throughput" and j["method"] not in (*eos.FFN, "ffn-dense"))
    for j in priority:
        if j["method"] != "ffn-dense":
            assert any("/ffn-dense-bf16/" in d for d in j["depends_on"])


@pytest.mark.parametrize("mismatched_control", [False, True])
def test_all_pending_runs_use_one_batch_and_share_only_identical_dense_controls(tmp_path, mismatched_control):
    manifest = tmp_path / "prompts.json"
    manifest.write_text("[]")
    manifests = {(m, c): manifest for m in ("4b", "8b") for c in eos.CONTEXTS}
    jobs = eos.make_jobs(tmp_path, tmp_path, {"4b": "four", "8b": "eight"}, "quant", manifests)
    jobs = eos.ffn_priority_jobs([dict(j, state="pending") for j in jobs], tmp_path)
    retained = next(j for j in jobs if j["model"] == "4b" and j["kind"] == "eos-throughput")
    retained["state"] = "complete"
    if mismatched_control:
        control = next(j for j in jobs if j["method"] == "ffn-dense")
        control["command"] += ["--seed", "123"]
        with pytest.raises(ValueError, match="identical workloads"):
            eos.single_batch_jobs(jobs)
        return
    revised = eos.single_batch_jobs(jobs)
    assert len(revised) == 225 and sum(j["kind"] == "eos-throughput" for j in revised) == 216
    assert next(j for j in revised if j["name"] == retained["name"]) == retained
    pending = [j for j in revised if j["state"] == "pending" and j["kind"] == "eos-throughput"]
    assert all(j["expected_rows"] == 1 and "--single-batch" in j["command"] for j in pending)
    assert not any(j["model"] == "8b" and j["method"] == "dense" for j in revised)
    while job := queue.eligible(revised):
        job["state"] = "complete"
    assert all(j["state"] == "complete" for j in revised)


def test_only_authorized_idle_gpus_are_returned(monkeypatch):
    def query(command, **kwargs):
        if "--query-compute-apps=gpu_uuid" in command:
            return "gpu2\n"
        return "0, gpu0, 0, 0\n1, gpu1, 0, 0\n2, gpu2, 0, 0\n3, gpu3, 0, 0\n"
    monkeypatch.setattr(queue.subprocess, "check_output", query)
    assert queue.idle_gpus((2, 3)) == {3}


def test_unstarted_report_has_empty_cells(tmp_path):
    (tmp_path / "plan.json").write_text(json.dumps(dict(jobs=[dict(kind="eos-throughput")])))
    eos.report(argparse.Namespace(run=tmp_path))
    text = (tmp_path / "report.md").read_text()
    assert "prepared_not_launched" in text and "0/1 process runs audited" in text
    assert "| 32,768 |  | BF16 |  |  |  |  |  |  |" in text
    assert json.loads((tmp_path / "summary.json").read_text())["cells"] == {}


def test_queue_dependencies_block_methods_until_dense_and_capacity_complete():
    jobs = [dict(name="capacity", state="pending"), dict(name="dense", state="pending", depends_on=["capacity"]),
            dict(name="spec", state="pending", depends_on=["capacity", "dense"])]
    assert queue.eligible(jobs)["name"] == "capacity"
    jobs[0]["state"] = "complete"
    assert queue.eligible(jobs)["name"] == "dense"
    jobs[1]["state"] = "failed"
    assert queue.eligible(jobs) is None


@pytest.mark.parametrize("change", ["missing", "failed", "source", "plan", "partial"])
def test_queue_rejects_missing_or_stale_gpu_preflight(tmp_path, change):
    import hashlib
    (tmp_path / "plan.json").write_text("{}")
    proof = dict(passed=True, source_sha256="source", plan_sha256=hashlib.sha256(b"{}").hexdigest(),
        checks=[dict(name=name, passed=True) for name in
            ("8b-dense-bf16", "4b-direct-bf16", "8b-sv50-mxfp4", "8b-int4-highlight-bf16")])
    if change == "failed":
        proof["checks"][0]["passed"] = False
    elif change == "source":
        proof["source_sha256"] = "stale"
    elif change == "plan":
        proof["plan_sha256"] = "stale"
    elif change == "partial":
        proof["checks"].pop()
    if change != "missing":
        (tmp_path / "preflight.json").write_text(json.dumps(proof))
    with pytest.raises((ValueError, FileNotFoundError)):
        queue.require_preflight(tmp_path, "source")


def test_report_requires_all_three_processes_and_reaudits_changed_evidence(tmp_path, evidence):
    job, records, prompts = copy.deepcopy(evidence)
    manifest = tmp_path / "prompts.json"
    manifest.write_text(json.dumps(prompts))
    base = dict(job, model="4b", method="sv50", name="unused", kind="eos-throughput",
        manifest=str(manifest), manifest_sha256=eos.sha(manifest), state="complete")
    summary = eos.audit_records(job, records, prompts)
    jobs = []
    for rep in range(1, 4):
        current = dict(base, rep=rep, name=f"rep{rep}")
        jobs.append(current)
        folder = tmp_path / current["name"]
        folder.mkdir()
        (folder / "results.jsonl").write_text("\n".join(json.dumps(r) for r in records))
        (folder / "summary.json").write_text(json.dumps(summary))
    (tmp_path / "plan.json").write_text(json.dumps(dict(jobs=jobs)))
    state = tmp_path / "status.json"
    state.write_text(json.dumps(dict(state="running", jobs=jobs[:2])))
    eos.report(argparse.Namespace(run=tmp_path))
    assert json.loads((tmp_path / "summary.json").read_text())["cells"] == {}
    state.write_text(json.dumps(dict(state="complete", jobs=jobs)))
    eos.report(argparse.Namespace(run=tmp_path))
    cell = json.loads((tmp_path / "summary.json").read_text())["cells"]["4b/8/sv50/mxfp4"]
    assert cell["median"] == 5.5 and len(cell["repetitions"]) == 3
    records[0]["decode_tokens"] += 1
    (tmp_path / "rep1/results.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    with pytest.raises(ValueError, match="totals"):
        eos.report(argparse.Namespace(run=tmp_path))
