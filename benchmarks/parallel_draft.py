# SPDX-License-Identifier: Apache-2.0
"""One Jacobi update followed by dense greedy verification; see notes/experiments.html#jacobi-study.

Uses the frozen dense engine without editing its source or adding to its
jakobi*.py source-hash set. No new persistent history cache is allocated.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import time

import torch

import jakobi as base

ROOT = base.ROOT


def sources():
    result = base.sources()
    for name in ("parallel_draft.py", "parallel_draft_test.py"):
        path = ROOT / "benchmarks" / name
        result[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def check_plan(run):
    plan = json.loads((run / "plan.json").read_text())
    if sources() != plan["source_sha256"]:
        raise RuntimeError("Parallel-draft source changed after submission")
    baseline = Path(plan["baseline_run"])
    original = json.loads((baseline / "plan.json").read_text())
    if base.sources() != original["source_sha256"]:
        raise RuntimeError("Original Jacobi campaign source changed")
    for name, expected in plan["manifests"].items():
        if base.digest(json.loads((baseline / name).read_text())) != expected:
            raise RuntimeError(f"Prompt manifest changed: {name}")
    return plan, baseline


@torch.inference_mode()
def generate(engine, prompt, eos, vocabulary, *, seed=42, cap=256):
    if cap < 1:
        raise ValueError("Generation cap must be positive")
    torch.cuda.synchronize()
    start = time.perf_counter()
    first = engine.prefill(prompt)
    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - start
    output = [base.best(first)[0]]
    margins = list(engine.last_margins) if engine.capture_margins else []
    rng = random.Random(seed)
    traces = []
    accepted_total = 0
    start = time.perf_counter()
    while len(output) < cap and output[-1] not in eos:
        anchor = output[-1]
        size = min(6, cap - len(output))
        guesses = [rng.choice(vocabulary) for _ in range(size)]
        # One update only. Keep the committed prefix unchanged between calls.
        proposals = base.best(engine.forward([anchor] + guesses[:-1]))
        # Overwrite all speculative KV using the actual proposed predecessors.
        verified = base.best(engine.forward([anchor] + proposals))
        accepted = 0
        for proposal, target in zip(proposals, verified):
            if proposal != target:
                break
            accepted += 1
        emitted = base.truncate(proposals[:accepted] + [verified[accepted]],
                                eos, cap - len(output))
        # Commit anchor and accepted predecessors; the final output stays pending.
        engine.commit(list(range(len(emitted))))
        accepted_emitted = min(accepted, len(emitted))
        accepted_total += accepted_emitted
        traces.append({"output_offset": len(output), "initial": guesses,
                       "proposals": proposals, "verified": verified,
                       "matching_prefix": accepted, "accepted_emitted": accepted_emitted,
                       "emitted": emitted, "bonus_emitted": accepted == size and len(emitted) > size})
        output.extend(emitted)
        if engine.capture_margins:
            margins.extend(engine.last_margins[:len(emitted)])
    torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - start
    assert engine.calls == 2 * len(traces)
    return {"tokens": output, "finish_reason": "eos" if output[-1] in eos else "cap",
            "prefill_seconds": prefill_seconds, "decode_seconds": decode_seconds,
            "forward_calls": engine.calls, "query_tokens": engine.query_tokens,
            "rounds": len(traces), "draft_calls": len(traces), "verify_calls": len(traces),
            "accepted_draft_tokens": accepted_total, "traces": traces,
            "margins": margins, "pool_payload_bytes": 0,
            "proposal_kv_peak_bytes": engine.peak_query_tokens * engine.bytes_per_position}


def warm(engine):
    engine.prefix = 8
    for size in range(1, 8):
        engine.forward([1] * size)
    engine.prefix = 0


@torch.inference_mode()
def validate(args):
    base.require_slurm()
    plan, _ = check_plan(args.run)
    model, tokenizer, eos, vocabulary = base.load_model(args, 4096)
    engine = base.DenseEngine(model, 2048)
    warm(engine)
    records = []
    for text in ("Reply with the number only: what is 17 plus 25?",
                 "Continue this list to ten: one, two, three,",
                 "The password is amber-river-491. Return only the password."):
        ids = tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                           tokenize=True, add_generation_prompt=True, enable_thinking=False)
        tensor = torch.tensor([ids], device="cuda")
        stock = model.generate(tensor, attention_mask=torch.ones_like(tensor), max_new_tokens=24,
                               do_sample=False, repetition_penalty=1.0, pad_token_id=tokenizer.pad_token_id)
        reference = stock[0, len(ids):].tolist()
        result = generate(engine, ids, eos, vocabulary, cap=24)
        records.append({"prompt": text, "matches": reference == result["tokens"],
                        "reference": reference, "actual": result["tokens"]})
    check_plan(args.run)
    payload = {"state": "passed" if all(r["matches"] for r in records) else "failed",
               "model": args.model, "source_sha256": plan["source_sha256"], "checks": records,
               "gpu": torch.cuda.get_device_name(), "job": os.environ["SLURM_JOB_ID"]}
    base.write_json(args.run / f"validation-{args.model}.json", payload)
    print("GPU validation", args.model, payload["state"], flush=True)
    if payload["state"] != "passed":
        raise RuntimeError("One-update speculative generation differs from stock Qwen3")


def run_method(args):
    base.require_slurm()
    plan, baseline = check_plan(args.run)
    validation = json.loads((args.run / f"validation-{args.model}.json").read_text())
    if validation["state"] != "passed" or validation["source_sha256"] != plan["source_sha256"]:
        raise RuntimeError("Matching GPU validation required")
    if args.length != 4096:
        for model_name in ("4B", "8B"):
            if json.loads((args.run / f"{model_name}-4096/audit.json").read_text())["state"] != "passed":
                raise RuntimeError("Both one-update 4K audits must pass first")
    manifest = json.loads((baseline / f"prompts-{args.model}-{args.length}.json").read_text())
    rows = manifest["rows"]
    assert manifest["model"] == args.model
    total = max(len(r["prompt_token_ids"]) for r in rows) + 288
    assert total <= 131072
    dest = args.run / f"{args.model}-{args.length}"
    dest.mkdir(exist_ok=True)
    if (dest / "config.json").exists():
        raise RuntimeError("Measurement directory already used")
    model, tokenizer, eos, vocabulary = base.load_model(args, total)
    engine = base.DenseEngine(model, math.ceil((total + 64) / 1024) * 1024)
    args.method = "jacobi_spec"
    config = {**base.metadata(args, model), "source_sha256": sources(),
              "base_source_sha256": base.sources(), "manifest_sha256": base.digest(manifest),
              "haystack": args.length, "kv_capacity": engine.capacity,
              "kv_reserved_bytes": engine.keys.numel() * 4,
              "kv_bytes_per_position": engine.bytes_per_position,
              "draft_updates": 1, "draft_tokens": 6, "verification": "dense",
              "baseline_run": str(baseline)}
    base.write_json(dest / "config.json", config)
    extract_answer, score_row = base.molle_imports(str(ROOT.parent / "MoLLE"))
    warm(engine)
    for _ in range(2):
        generate(engine, rows[0]["prompt_token_ids"], eos, vocabulary, cap=32)
    results = []
    with (dest / "results.jsonl").open("x") as stream:
        for i, row in enumerate(rows):
            torch.cuda.reset_peak_memory_stats()
            from torch._dynamo.utils import counters
            before = counters["stats"]["unique_graphs"]
            result = generate(engine, row["prompt_token_ids"], eos, vocabulary, seed=42 + i)
            result.update(id=row["id"], task=row["task"],
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          output_text=tokenizer.decode(result["tokens"], skip_special_tokens=True),
                          new_compiled_graphs=counters["stats"]["unique_graphs"] - before,
                          committed_kv_bytes=engine.prefix * engine.bytes_per_position)
            extracted, tag = extract_answer(result["output_text"], row)
            result["score"] = score_row(extracted, tag, row)
            results.append(result)
            stream.write(json.dumps(result) + "\n")
            stream.flush()
            print(f"{args.model} {args.length} jacobi_spec: {i + 1}/{len(rows)}", flush=True)
    check_plan(args.run)
    timing_valid = all(r["new_compiled_graphs"] == 0 for r in results)
    base.write_json(dest / "finished.json", {"state": "complete", "rows": len(results),
                                           "timing_valid": timing_valid})
    if not timing_valid:
        raise RuntimeError("Compilation during measurement invalidates timing")


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def update_notes(run, baseline):
    from study_report import publish_parallel
    with (baseline / "report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        publish_parallel(ROOT / "notes/experiments.html", run)


def audit(run, model, length):
    plan, baseline = check_plan(run)
    dest = run / f"{model}-{length}"
    reference_dir = baseline / f"{model}-{length}/ar"
    if not (dest / "finished.json").exists():
        return {"state": "awaiting_measurement"}
    if not (reference_dir / "finished.json").exists():
        report = {"state": "awaiting_ar", "model": model, "haystack": length}
        base.write_json(dest / "audit.json", report)
        return report
    manifest = json.loads((baseline / f"prompts-{model}-{length}.json").read_text())
    candidate, reference = read_rows(dest / "results.jsonl"), read_rows(reference_dir / "results.jsonl")
    expected = [r["id"] for r in manifest["rows"]]
    config = json.loads((dest / "config.json").read_text())
    original = json.loads((reference_dir / "config.json").read_text())
    for folder, rows in ((dest, candidate), (reference_dir, reference)):
        finished = json.loads((folder / "finished.json").read_text())
        if finished["state"] != "complete" or finished["rows"] != len(expected) or [r["id"] for r in rows] != expected:
            raise RuntimeError("Incomplete or differently ordered results")
    for key in ("model_revision", "versions", "gpu", "host", "rope_scaling", "dtype", "batch", "seed", "cap", "kv_capacity"):
        if config[key] != original[key]:
            raise RuntimeError(f"Unmatched AR setting: {key}")
    assert config["source_sha256"] == plan["source_sha256"]
    assert config["base_source_sha256"] == original["source_sha256"] == base.sources()
    assert config["manifest_sha256"] == original["manifest_sha256"] == base.digest(manifest)
    mismatches = []
    for row, ref in zip(candidate, reference):
        if row["tokens"] != ref["tokens"] or row["finish_reason"] != ref["finish_reason"]:
            common = min(len(row["tokens"]), len(ref["tokens"]))
            position = next((i for i in range(common) if row["tokens"][i] != ref["tokens"][i]), common)
            mismatches.append({"row": row["id"], "position": position})
    seconds = sum(r["decode_seconds"] for r in candidate)
    rounds = sum(r["rounds"] for r in candidate)
    calls = sum(r["forward_calls"] for r in candidate)
    assert calls == 2 * rounds
    timing_valid = all(r["new_compiled_graphs"] == 0 for r in candidate + reference)
    report = {"state": "passed" if not mismatches and timing_valid else "failed",
              "model": model, "haystack": length, "rows": len(expected),
              "matching": len(expected) - len(mismatches), "mismatches": mismatches,
              "timing_valid": timing_valid, "score": sum(r["score"] for r in candidate) / len(candidate),
              "cap_hits": sum(r["finish_reason"] == "cap" for r in candidate),
              "prefill_seconds": sum(r["prefill_seconds"] for r in candidate),
              "decode_seconds": seconds, "ar_decode_seconds": sum(r["decode_seconds"] for r in reference),
              "speedup": sum(r["decode_seconds"] for r in reference) / seconds,
              "rounds": rounds, "forward_calls": calls,
              "query_tokens": sum(r["query_tokens"] for r in candidate),
              "accepted_per_round": sum(r["accepted_draft_tokens"] for r in candidate) / rounds if rounds else 0,
              "emitted_per_round": sum(len(r["tokens"]) - 1 for r in candidate) / rounds if rounds else 0,
              "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in candidate),
              "source_sha256": plan["source_sha256"], "manifest_sha256": base.digest(manifest),
              "ar_results_sha256": hashlib.sha256((reference_dir / "results.jsonl").read_bytes()).hexdigest(),
              "results_sha256": hashlib.sha256((dest / "results.jsonl").read_bytes()).hexdigest()}
    base.write_json(dest / "audit.json", report)
    update_notes(run, baseline)
    return report


def replay(args, mismatch):
    _, baseline = check_plan(args.run)
    manifest = json.loads((baseline / f"prompts-{args.model}-{args.length}.json").read_text())
    index = next(i for i, row in enumerate(manifest["rows"]) if row["id"] == mismatch["row"])
    total = max(len(r["prompt_token_ids"]) for r in manifest["rows"]) + 288
    model, _, eos, vocabulary = base.load_model(args, total)
    engine = base.DenseEngine(model, math.ceil((total + 64) / 1024) * 1024)
    engine.capture_margins = True
    prompt = manifest["rows"][index]["prompt_token_ids"]
    results = {"ar": base.generate(engine, prompt, "ar", eos, vocabulary, seed=42 + index),
               "jacobi_spec": generate(engine, prompt, eos, vocabulary, seed=42 + index)}
    base.write_json(args.run / f"{args.model}-{args.length}/mismatch-replay.json",
                    {"row": mismatch["row"], "source_sha256": sources(), "results": results})


def cell(args):
    check_plan(args.run)
    command = [sys.executable, str(Path(__file__).resolve())]
    common = ["--run", str(args.run), "--model", args.model, "--length", str(args.length)]
    if args.length == 4096:
        subprocess.run(command + ["validate"] + common, check=True)
    subprocess.run(command + ["run"] + common, check=True)
    report = audit(args.run, args.model, args.length)
    print("Audit", args.model, args.length, report["state"], flush=True)
    if report.get("mismatches"):
        base.require_slurm()
        replay(args, report["mismatches"][0])
    if report["state"] not in ("passed", "awaiting_ar"):
        raise RuntimeError("Audit failed; dependent jobs must not run")


def resume_baseline(args):
    """Finish one original method after its four-hour Slurm allocation ends."""
    _, baseline = check_plan(args.run)
    dest = baseline / f"{args.model}-{args.length}" / args.method
    manifest_path = baseline / f"prompts-{args.model}-{args.length}.json"
    manifest = json.loads(manifest_path.read_text())
    if (dest / "finished.json").exists():
        completed = json.loads((dest / "finished.json").read_text())
        config = json.loads((dest / "config.json").read_text())
        assert completed["state"] == "complete" and completed["rows"] == len(manifest["rows"])
        assert config["source_sha256"] == base.sources()
        assert config["manifest_sha256"] == base.digest(manifest)
        print("Retained complete baseline", args.model, args.method, flush=True)
    else:
        if dest.exists():
            archived = baseline / "interrupted" / f"{args.model}-{args.length}" / f"{args.method}-{os.environ['SLURM_JOB_ID']}"
            archived.parent.mkdir(parents=True, exist_ok=True)
            dest.rename(archived)
            print("Preserved interrupted measurement", archived, flush=True)
        subprocess.run([sys.executable, str(ROOT / "benchmarks/jakobi.py"), "run",
                        "--model", args.model, "--method", args.method,
                        "--rows", str(manifest_path), "--out", str(dest)], check=True)
    if args.method == "ar":
        comparison = audit(args.run, args.model, args.length)
        if comparison.get("mismatches"):
            base.require_slurm()
            replay(args, comparison["mismatches"][0])
    if args.audit_baseline:
        from jakobi_campaign import audit as original_audit
        report = original_audit(baseline, args.model, args.length)
        if report["mismatches"]:
            first = report["mismatches"][0]
            subprocess.run([sys.executable, str(ROOT / "benchmarks/jakobi.py"), "replay",
                            "--model", args.model, "--method", first["method"],
                            "--rows", str(manifest_path), "--row-id", first["row"],
                            "--out", str(dest.parent / "mismatch-replay.json")], check=True)
        if report["state"] != "passed":
            raise RuntimeError("Original long-context audit failed")


def submit(args):
    run = args.run.resolve()
    baseline = args.baseline.resolve()
    if (run / "plan.json").exists():
        raise RuntimeError("Campaign already submitted")
    previous = json.loads((baseline / "plan.json").read_text())
    assert previous["source_sha256"] == base.sources()
    plan = {"source_sha256": sources(), "baseline_run": str(baseline),
            "manifests": previous["manifests"], "jobs": {},
            "settings": {"draft_updates": 1, "draft_tokens": 6, "verification": "dense",
                         "partition": "srv09", "gpu": "a6000", "qos": "normal"}}
    base.write_json(run / "plan.json", plan)
    dependency = None
    for length in (4096, 129024):
        for model in ("4B", "8B"):
            name = f"{model}-{length}"
            command = [sys.executable, str(Path(__file__).resolve()), "cell", "--run", str(run),
                       "--model", model, "--length", str(length)]
            script = run / f"{name}.sh"
            script.write_text("#!/bin/bash\nset -euo pipefail\ncd " + shlex.quote(str(ROOT)) +
                              "\nexport TOKENIZERS_PARALLELISM=false\nexport OMP_NUM_THREADS=8\n" + shlex.join(command) + "\n")
            argv = ["sbatch", "--parsable", "--partition=srv09", "--nodelist=srv09", "--gres=gpu:a6000:1",
                    "--qos=normal", "--cpus-per-task=8", "--mem=64G",
                    "--time=" + ("01:00:00" if length == 4096 else "08:00:00"),
                    "--kill-on-invalid-dep=yes", "--job-name=jacobi-spec-" + name,
                    "--output=" + str(run / (name + "-%j.log"))]
            if dependency:
                argv.append("--dependency=afterok:" + dependency)
            job = subprocess.check_output(argv + [str(script)], text=True).strip()
            plan["jobs"][name] = job
            dependency = job
            base.write_json(run / "plan.json", plan)
            print(name, job, flush=True)
    # The old multi-method allocations cannot be lengthened by this user.
    # After they end, retain completed outputs and rerun only partial methods.
    plan["baseline_continuations"] = {}
    for model in ("4B", "8B"):
        dependency = previous["jobs"][f"{model}-129024"]
        jobs = []
        for index, method in enumerate(base.METHODS):
            command = [sys.executable, str(Path(__file__).resolve()), "resume", "--run", str(run),
                       "--model", model, "--length", "129024", "--method", method]
            if index == len(base.METHODS) - 1:
                command.append("--audit-baseline")
            name = f"resume-{model}-{method}"
            script = run / f"{name}.sh"
            script.write_text("#!/bin/bash\nset -euo pipefail\ncd " + shlex.quote(str(ROOT)) +
                              "\nexport TOKENIZERS_PARALLELISM=false\nexport OMP_NUM_THREADS=8\n" + shlex.join(command) + "\n")
            condition = "afterany:" if index == 0 else "afterok:"
            job = subprocess.check_output(
                ["sbatch", "--parsable", "--partition=srv09", "--nodelist=srv09", "--gres=gpu:a6000:1",
                 "--qos=normal", "--cpus-per-task=8", "--mem=64G", "--time=08:00:00", "--kill-on-invalid-dep=yes",
                 "--dependency=" + condition + dependency, "--job-name=jakobi-" + name,
                 "--output=" + str(run / (name + "-%j.log")), str(script)], text=True).strip()
            jobs.append(job)
            dependency = job
            plan["baseline_continuations"][model] = jobs
            base.write_json(run / "plan.json", plan)
            print(name, job, flush=True)
    command = [sys.executable, str(Path(__file__).resolve()), "report", "--run", str(run)]
    script = run / "report.sh"
    script.write_text("#!/bin/bash\nset -euo pipefail\ncd " + shlex.quote(str(ROOT)) + "\n" + shlex.join(command) + "\n")
    waits = list(plan["jobs"].values()) + [job for jobs in plan["baseline_continuations"].values() for job in jobs]
    job = subprocess.check_output(["sbatch", "--parsable", "--partition=srv09", "--nodelist=srv09",
                                   "--qos=normal", "--cpus-per-task=1", "--mem=2G", "--time=00:10:00",
                                   "--dependency=afterany:" + ":".join(waits), "--job-name=jacobi-spec-report",
                                   "--output=" + str(run / "report-%j.log"), str(script)], text=True).strip()
    plan["report_job"] = job
    base.write_json(run / "plan.json", plan)
    base.write_json(ROOT / "outputs/runs/parallel-draft-current.json", {"run": str(run)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("submit", "validate", "run", "cell", "resume", "report"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--model", choices=("4B", "8B"))
    parser.add_argument("--length", type=int)
    parser.add_argument("--method", choices=base.METHODS)
    parser.add_argument("--audit-baseline", action="store_true")
    args = parser.parse_args()
    args.run = args.run.resolve()
    torch.set_num_threads(8)
    if args.command == "report":
        reports = [audit(args.run, model, length) for model in ("4B", "8B") for length in (4096, 129024)]
        base.write_json(args.run / "status.json", {"state": "complete" if all(r["state"] == "passed" for r in reports) else "incomplete_or_failed",
                                                 "cells": reports})
        print(json.dumps([{k: r[k] for k in ("model", "haystack", "state") if k in r} for r in reports]))
    else:
        {"submit": submit, "validate": validate, "run": run_method, "cell": cell,
         "resume": resume_baseline}[args.command](args)


if __name__ == "__main__":
    main()
