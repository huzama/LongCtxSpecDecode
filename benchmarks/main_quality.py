"""Frozen Main Results quality runs, ordered 4B before remaining 8B work."""
import argparse
import collections
import datetime
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

from benchmarks.main_results import METHODS, ROOT, flags, load, save, sha
from benchmarks.native_fp8_throughput import METHOD as FP8
from benchmarks.queue import source_digest

QUALITY_METHODS = (*METHODS, FP8)


def make_jobs(runtime, run, model, methods=None, longbench_format="cot"):
    if longbench_format not in ("cot", "answer-tags"):
        raise ValueError("invalid LongBench format")
    methods = QUALITY_METHODS if methods is None else methods
    if (not isinstance(methods, (list, tuple)) or not methods
            or any(not isinstance(method, str) or method not in QUALITY_METHODS for method in methods)
            or len(set(methods)) != len(methods)):
        raise ValueError("invalid quality methods")
    jobs = []
    for stage, ctx in (("ruler", 65536), ("ruler", 129024), ("longbench", None)):
        for method in methods:
            name = f"{stage}/{model}/{ctx or longbench_format}/{method}"
            jobs.append(dict(name=name, kind="main-quality", stage=stage, model=model,
                ctx=ctx, method=method, expected_rows=192 if stage == "ruler" else 60,
                command=[str(runtime/".venv/bin/python"), "-P", "-m", "benchmarks.main_quality",
                         "--run", str(run), "--job", name]))
    return jobs


def prepare(parent, native, out):
    """Reuse reviewed cohorts and checkpoint hashes; freeze a separate source."""
    if out.exists():
        raise ValueError("use a new campaign directory")
    previous, native_plan = load(parent/"plan.json"), load(native/"plan.json")
    runtime = Path(previous["runtime_root"])
    source = out/"source"
    source.mkdir(parents=True)
    for name in ("benchmarks", "highlight_spec", "tests"):
        shutil.copytree(ROOT/name, source/name, ignore=shutil.ignore_patterns("__pycache__"))
    import spotattention
    shutil.copytree(Path(spotattention.__file__).parent, source/"spotattention",
                    ignore=shutil.ignore_patterns("__pycache__"))
    # The quality launch must not change serving code from the measured methods.
    for package, reference in (("highlight_spec", Path(native_plan["source_root"])),
                               ("spotattention", Path(native_plan["source_root"]))):
        current = {str(p.relative_to(source/package)): sha(p) for p in (source/package).rglob("*.py")}
        prior = {str(p.relative_to(reference/package)): sha(p) for p in (reference/package).rglob("*.py")}
        if current != prior:
            raise ValueError(f"serving implementation changed: {package}")
    plans = {}
    for model in ("4b", "8b"):
        run = out/model
        checkpoint = previous["inputs"]["checkpoints"][model]
        inputs = dict(checkpoint=checkpoint, questions=previous["inputs"]["questions"][model],
            ruler={str(ctx): previous["inputs"]["ruler"][f"{model}/{ctx}"] for ctx in (65536, 129024)})
        paths = {inputs["questions"], *inputs["ruler"].values()}
        hashes = {p: h for p, h in previous["input_sha256"].items()
                  if p in paths or Path(p).parent == Path(checkpoint)}
        if not paths <= hashes.keys():
            raise ValueError("missing pinned quality input")
        jobs = make_jobs(runtime, run, model)
        plan = dict(schema="main-quality-v1", host=previous["host"], model=model,
            source_root=str(source), runtime_root=str(runtime), source_sha256=source_digest(source),
            input_sha256=hashes, inputs=inputs, runtime=previous["runtime"],
            allowed_gpus=[2, 3], idle_seconds=60, report_module="benchmarks.main_quality",
            jobs=jobs, protocol=dict(batch=1, repetitions=1, thinking=False, greedy=True,
                eos_respected=True, ruler_tokens=1024, longbench="CoT only; 1024 reasoning + 128 answer",
                weights="bf16", indexer="bf16", selection_fraction=.07, recent_tokens=64,
                speculative_tokens=6), counts=dict(result_runs=len(jobs),
                by_stage=dict(collections.Counter(j["stage"] for j in jobs))))
        save(run/"plan.json", plan)
        plans[model] = str(run)
    save(out/"pipeline.json", dict(schema="main-quality-pipeline-v1", stages=["4b-quality", "8b-throughput", "8b-quality"],
        quality_runs=plans, throughput_run=str(parent), native_run=str(native),
        throughput_plan_sha256=sha(parent/"plan.json"), native_plan_sha256=sha(native/"plan.json"),
        state="prepared", source_root=str(source), source_sha256=source_digest(source)))
    return plans


def _verify_payload(plan):
    if source_digest(Path(plan["source_root"])) != plan["source_sha256"]:
        raise ValueError("frozen quality source changed")
    for name, version in plan["runtime"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("runtime changed: " + name)
    for path, digest in plan["input_sha256"].items():
        if sha(path) != digest:
            raise ValueError("frozen input changed: " + path)


def _unchanged_source(source, parent):
    def files(root):
        return {str(p.relative_to(root)): sha(p) for p in root.rglob("*.py")
                if str(p.relative_to(root)) != "benchmarks/main_quality.py"}
    if files(source) != files(parent):
        raise ValueError("recovery changed experiment implementation")


def prepare_recovery(parent, out, gpus):
    """Preserve the original campaign and rerun only its unfinished cohorts."""
    if out.exists():
        raise ValueError("use a new recovery directory")
    if not gpus or len(set(gpus)) != len(gpus) or any(g not in range(4) for g in gpus):
        raise ValueError("invalid recovery GPU list")
    with (parent/"controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous, state = load(parent/"plan.json"), load(parent/"status.json")
        if previous["schema"] != "main-quality-v1":
            raise ValueError("unexpected parent quality scope")
        if "recovery" in previous:
            verify_plan(parent)
        elif previous["jobs"] != make_jobs(Path(previous["runtime_root"]), parent, previous["model"],
                                           previous["protocol"].get("methods")):
            raise ValueError("unexpected parent quality scope")
        _verify_payload(previous)
        states = {j["name"]: j for j in state["jobs"]}
        if len(states) != len(previous["jobs"]) or set(states) != {j["name"] for j in previous["jobs"]}:
            raise ValueError("parent status scope differs")
        completed = {j["name"]: validate(j, parent/j["name"]) for j in previous["jobs"]
                     if states[j["name"]]["state"] == "complete"}
        parent_names = {j["name"] for j in previous["jobs"]}
        jobs = [j for j in make_jobs(Path(previous["runtime_root"]), out, previous["model"],
                                    previous["protocol"].get("methods"))
                if j["name"] in parent_names and j["name"] not in completed]
        if not jobs:
            raise ValueError("quality campaign already complete")
        source = out/"source"
        shutil.copytree(previous["source_root"], source, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copyfile(__file__, source/"benchmarks/main_quality.py")
        _unchanged_source(source, Path(previous["source_root"]))
        plan = {**previous, "host": socket.gethostname(), "source_root": str(source),
                "source_sha256": source_digest(source), "allowed_gpus": list(gpus), "jobs": jobs,
                "counts": dict(result_runs=len(jobs), by_stage=dict(collections.Counter(j["stage"] for j in jobs))),
                "recovery": dict(parent_run=str(parent), parent_plan_sha256=sha(parent/"plan.json"),
                    parent_status_sha256=sha(parent/"status.json"), completed_audits=completed)}
        save(out/"plan.json", plan)
        return dict(preserved=len(completed), remaining=len(jobs), allowed_gpus=list(gpus), run=str(out))


def verify_plan(run):
    plan = load(run/"plan.json")
    expected = make_jobs(Path(plan["runtime_root"]), run, plan["model"],
                         plan["protocol"].get("methods"), plan["protocol"].get("longbench_format", "cot"))
    allowed = plan["allowed_gpus"]
    if plan["schema"] == "main-quality-v2":
        authorization = plan.get("gpu_authorization", {})
        gpu_valid = (allowed == authorization.get("allowed_gpus") and bool(allowed)
                     and len(set(allowed)) == len(allowed) and all(g in range(4) for g in allowed))
        if (plan["protocol"].get("longbench_format") != "answer-tags"
                or plan["protocol"].get("longbench_tokens") != 8192
                or not authorization.get("user_instruction") or "recovery" in plan):
            raise ValueError("quality answer-tag protocol differs")
    elif "recovery" in plan:
        recovery = plan["recovery"]
        parent = Path(recovery["parent_run"])
        if (sha(parent/"plan.json") != recovery["parent_plan_sha256"]
                or sha(parent/"status.json") != recovery["parent_status_sha256"]):
            raise ValueError("parent campaign changed")
        previous = load(parent/"plan.json")
        if "recovery" in previous:
            verify_plan(parent)
        for field in ("model", "runtime_root", "runtime", "inputs", "input_sha256", "protocol"):
            if plan[field] != previous[field]:
                raise ValueError("recovery changed protocol: " + field)
        completed = recovery["completed_audits"]
        names = {j["name"] for j in load(parent/"status.json")["jobs"] if j["state"] == "complete"}
        if set(completed) != names or not names <= {j["name"] for j in expected}:
            raise ValueError("recovery completion set differs")
        for name, audit in completed.items():
            for filename, digest in audit["raw_sha256"].items():
                if sha(parent/name/filename) != digest:
                    raise ValueError("preserved result changed: " + name)
        _unchanged_source(Path(plan["source_root"]), Path(previous["source_root"]))
        parent_names = {j["name"] for j in previous["jobs"]}
        expected = [j for j in expected if j["name"] in parent_names and j["name"] not in completed]
        gpu_valid = bool(allowed) and len(set(allowed)) == len(allowed) and all(g in range(4) for g in allowed)
    else:
        gpu_valid = allowed == [2, 3]
    if (plan["schema"] not in ("main-quality-v1", "main-quality-v2") or plan["host"] != socket.gethostname()
            or not gpu_valid or plan["jobs"] != expected):
        raise ValueError("quality scope or GPU authorization changed")
    _verify_payload(plan)
    return plan


def validate(job, dest):
    """Recompute quality from saved outputs; reject changed cohorts or protocols."""
    from benchmarks import decode
    from benchmarks.multilexsum import output_audit
    from benchmarks.accuracy import ANSWER_TAG_FORMAT, extract_answer as extract_choice, score_answer_tags
    from spotattention.data._scorers import extract_answer, score_row
    run = dest.parents[3]
    plan = load(run/"plan.json")
    config, finished, cache = (load(dest/name) for name in ("config.json", "finished.json", "cache.json"))
    rows = [json.loads(line) for line in (dest/"results.jsonl").read_text().splitlines()]
    native = job["method"] == FP8
    method = None if native else METHODS[job["method"]]
    ruler = job["stage"] == "ruler"
    answer_tags = not ruler and plan["protocol"].get("longbench_format") == "answer-tags"
    gen = 8192 if answer_tags else 1024
    expected = dict(model=plan["inputs"]["checkpoint"], batch=1, thinking=False, ignore_eos=False,
        async_scheduling=False, prefix_caching=False, dtype="torch.bfloat16", flash_attn_version=4,
        indexer_cache_format="bf16", gen=gen, seed=42, mode="direct" if native else method.mode,
        hierarchical_method=None if native else job["method"],
        backbone_cache="native_fp8_e4m3" if native else "hierarchical_4plus4",
        kv_cache_dtype="fp8_e4m3" if native else "auto", spec_tokens=6 if method and method.speculative else 0)
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("quality engine configuration differs")
    if config["sampling"]["temperature"] != 0 or config["eos_audit_ids"] != decode.configured_eos_ids(expected["model"]):
        raise ValueError("sampling or EOS configuration differs")
    if config["mode"] != "dense":
        selection = config["selection"]
        for key, value in dict(molle_topk_fraction=.07, molle_topk_recency=64,
                molle_verify_fraction=0. if native else method.verify_fraction,
                molle_selection_mode="topk_fraction",
                molle_draft_select="round" if method and method.speculative else "step").items():
            if selection.get(key) != value:
                raise ValueError("selection configuration differs: " + key)
    if native:
        if (not config["calculate_kv_scales"] or cache["backbone_cache"] != "native_fp8_e4m3"
                or cache["kv_cache_groups"] != 1 or cache["indexer_cache"] != "bf16" or not cache["layers"]):
            raise ValueError("native FP8 cache differs")
        if any(layer["element_bytes"] != 1 or any(not math.isfinite(v) or v <= 0 for v in layer["scales"].values()) for layer in cache["layers"]):
            raise ValueError("native FP8 scale or storage failure")
    else:
        if cache.get("attention_backend", "triton") != plan["protocol"].get("attention_backend", "triton"):
            raise ValueError("quality attention backend differs from the frozen plan")
        if not cache.get("errors") or any(any(values) for values in cache["errors"].values()):
            raise ValueError("hierarchical cache transaction failure")
    path = plan["inputs"]["ruler"][str(job["ctx"])] if ruler else plan["inputs"]["questions"]
    manifest = load(path)
    cohort = manifest["rows" if ruler else "questions"]
    ids = [r["id"] for r in cohort]
    if (len(rows) != job["expected_rows"] or [r["id"] for r in rows] != ids
            or config["rows" if ruler else "questions"] != ids
            or config["rows_sha256" if ruler else "manifest_sha256"] != decode.digest(manifest)):
        raise ValueError("quality cohort differs")
    if answer_tags and (config["cot"] is not None or config.get("answer_format") != ANSWER_TAG_FORMAT
                        or manifest["protocol"].get("prompt_format") != ANSWER_TAG_FORMAT
                        or manifest["protocol"].get("generation_tokens") != 8192):
        raise ValueError("LongBench must use the agreed answer-tag protocol")
    if not ruler and not answer_tags and (not config["cot"] or config["cot"]["reasoning_tokens"] != 1024 or config["cot"]["answer_tokens"] != 128):
        raise ValueError("LongBench must use the agreed CoT stages")
    scores = []
    for row, original in zip(rows, cohort, strict=True):
        output_audit(row["output_token_ids"], config["eos_audit_ids"], gen if ruler or answer_tags else 128, row["finish_reason"])
        if ruler:
            extracted, tagged = extract_answer(row["output_text"], original)
            score = score_row(extracted, tagged, original)
            if (row["score"], row["extracted"], row["had_tag"]) != (score, extracted, tagged):
                raise ValueError("RULER saved score differs from rescoring")
        elif answer_tags:
            if (row["prompt_sha256"], row["prompt_tokens"], row["answer"], row["choices"], row["question"]) != (
                    original["prompt_sha256"], original["prompt_tokens"], original["answer"],
                    original["choices"], original["question"]):
                raise ValueError("LongBench prompt or answer differs")
            prediction, score = score_answer_tags(row["output_text"], original)
            if (row["prediction"], row["correct"]) != (prediction, score):
                raise ValueError("LongBench saved score differs from rescoring")
        else:
            output_audit(row["reasoning_token_ids"], config["eos_audit_ids"], 1024, row["reasoning_finish_reason"])
            prediction = extract_choice(row["output_text"])
            score = row["finish_reason"] == "stop" and prediction == original["answer"]
            if (row["prediction"], row["correct"]) != (prediction, score):
                raise ValueError("LongBench saved score differs from rescoring")
        scores.append(score)
    if finished["rows" if ruler else "questions"] != len(rows):
        raise ValueError("incomplete quality summary")
    value = sum(scores)/len(scores)
    if ruler and not math.isclose(finished["score"], value, abs_tol=1e-12):
        raise ValueError("RULER aggregate differs")
    if not ruler and finished["correct"] != sum(scores):
        raise ValueError("LongBench aggregate differs")
    return dict(model=job["model"], stage=job["stage"], ctx=job["ctx"], method=job["method"],
        examples=len(rows), score=value, percent=100*value, source_sha256=plan["source_sha256"],
        protocol="answer-tags" if answer_tags else "ruler" if ruler else "cot",
        raw_sha256={n: sha(dest/n) for n in ("config.json", "results.jsonl", "finished.json", "cache.json")})


def run_job(run, name):
    plan = load(run/"plan.json")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {str(g) for g in plan["allowed_gpus"]}:
        raise ValueError("job requires one authorized physical GPU")
    job = next(j for j in plan["jobs"] if j["name"] == name)
    dest = run/name
    base = (["--checkpoint", plan["inputs"]["checkpoint"], "--mode", "direct", "--native-fp8-highlight"]
            if job["method"] == FP8 else flags(job["method"], plan["inputs"]["checkpoint"]))
    if job["method"] != FP8:
        base += ["--hierarchical-attention-backend", plan["protocol"].get("attention_backend", "triton")]
    started = time.monotonic()
    if job["stage"] == "ruler":
        from benchmarks.ruler import run as execute
        argv = base + ["--rows", plan["inputs"]["ruler"][str(job["ctx"])]]
    else:
        from benchmarks.accuracy import main as execute
        argv = base + ["--questions", plan["inputs"]["questions"],
                       "--answer-tags" if plan["protocol"].get("longbench_format") == "answer-tags" else "--cot"]
    gen = 8192 if job["stage"] == "longbench" and plan["protocol"].get("longbench_format") == "answer-tags" else 1024
    execute(argv + ["--gen", str(gen), "--out", str(dest)])
    save(dest/"duration.json", dict(wall_seconds=time.monotonic()-started))
    save(dest/"audit.json", validate(job, dest))


def report(run):
    state = load(run/"status.json")
    rows = [validate(j, run/j["name"]) for j in state["jobs"] if j["state"] == "complete"]
    save(run/"report.json", dict(state=state["state"], measurements=rows))


def pipeline(out):
    """A completed 4B quality stage is required before restarting 8B dispatch."""
    with (out/"pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = load(out/"pipeline.json")
        recovery = Path(plan["throughput_run"])
        if sha(recovery/"plan.json") != plan["throughput_plan_sha256"]:
            raise ValueError("throughput plan changed")
        stages = [("4b-quality", Path(plan["quality_runs"]["4b"])),
                  ("8b-throughput", recovery), ("8b-quality", Path(plan["quality_runs"]["8b"]))]
        state = dict(controller_pid=os.getpid(), state="running", completed=[])
        for name, run in stages:
            config = load(run/"plan.json")
            state.update(stage=name, updated=datetime.datetime.now(datetime.timezone.utc).isoformat())
            save(out/"pipeline-status.json", state)
            if (run/"status.json").exists() and load(run/"status.json")["state"] == "complete":
                state["completed"].append(name)
                continue
            env = {**os.environ, "PYTHONPATH": config["source_root"], "CUDA_VISIBLE_DEVICES": ""}
            with (run/"controller-quality-priority.log").open("a") as log:
                proc = subprocess.Popen([str(Path(config["runtime_root"])/".venv/bin/python"),
                    "-P", "-m", "benchmarks.queue", "--run", str(run)], env=env,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                state["child_pid"] = proc.pid
                save(out/"pipeline-status.json", state)
                code = proc.wait()
            terminal = load(run/"status.json") if (run/"status.json").exists() else {}
            if code or terminal.get("state") != "complete" or terminal.get("report_exit_code") != 0:
                state.update(state="failed", exit_code=code)
                save(out/"pipeline-status.json", state)
                raise RuntimeError("pipeline stopped: " + name)
            state["completed"].append(name)
        state.update(state="complete", updated=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save(out/"pipeline-status.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--recover", type=Path)
    parser.add_argument("--gpus", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--pipeline", action="store_true")
    parser.add_argument("--job")
    args = parser.parse_args()
    if args.recover:
        print(json.dumps(prepare_recovery(args.recover.resolve(), args.run.resolve(), args.gpus)))
    elif args.prepare:
        if not args.parent or not args.native:
            parser.error("--prepare requires --parent and --native")
        print(json.dumps(prepare(args.parent.resolve(), args.native.resolve(), args.run.resolve())))
    elif args.check:
        verify_plan(args.run)
    elif args.pipeline:
        pipeline(args.run)
    elif args.job:
        run_job(args.run, args.job)
    else:
        report(args.run)


if __name__ == "__main__":
    main()
