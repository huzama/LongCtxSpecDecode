# SPDX-License-Identifier: Apache-2.0
"""Submit, gate and report the approved local Slurm Jacobi experiments."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys

from jakobi import METHODS, ROOT, digest, sources, write_json


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def check_sources(plan):
    if sources() != plan["source_sha256"]:
        raise RuntimeError("Source changed after submission; do not run this frozen campaign")


def update_notes(run):
    from study_report import publish_jacobi
    with (run / "report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        publish_jacobi(ROOT / "notes/experiments.html", run)


def audit(run, model, length):
    cell = run / f"{model}-{length}"
    plan = json.loads((run / "plan.json").read_text())
    check_sources(plan)
    configs = {m: json.loads((cell / m / "config.json").read_text()) for m in METHODS}
    manifest = json.loads((run / f"prompts-{model}-{length}.json").read_text())
    rows = {m: read_rows(cell / m / "results.jsonl") for m in METHODS}
    ids = [r["id"] for r in manifest["rows"]]
    for method in METHODS:
        finished = json.loads((cell / method / "finished.json").read_text())
        if finished["state"] != "complete" or [r["id"] for r in rows[method]] != ids:
            raise RuntimeError("Incomplete result set")
        if configs[method]["source_sha256"] != plan["source_sha256"]:
            raise RuntimeError("Measured source differs from frozen source")
        if configs[method]["manifest_sha256"] != digest(manifest):
            raise RuntimeError("Measured prompt manifest differs")
        for key in ("model_revision", "versions", "gpu", "host", "rope_scaling", "dtype", "batch", "seed", "cap"):
            if configs[method][key] != configs["ar"][key]:
                raise RuntimeError(f"Unmatched setting: {key}")
    reference = rows["ar"]
    baseline_time = sum(r["decode_seconds"] for r in reference)
    methods, mismatches = {}, []
    for method in METHODS:
        matches = [r["tokens"] == a["tokens"] and r["finish_reason"] == a["finish_reason"]
                   for r, a in zip(rows[method], reference)]
        seconds = sum(r["decode_seconds"] for r in rows[method])
        calls = sum(r["forward_calls"] for r in rows[method])
        methods[method] = {
            "rows": len(ids), "matching": sum(matches),
            "score": sum(r["score"] for r in rows[method]) / len(ids),
            "cap_hits": sum(r["finish_reason"] == "cap" for r in rows[method]),
            "prefill_seconds": sum(r["prefill_seconds"] for r in rows[method]),
            "decode_seconds": seconds, "speedup": baseline_time / seconds,
            "tokens_per_call": sum(len(r["tokens"]) - 1 for r in rows[method]) / calls if calls else 0,
            "query_tokens": sum(r["query_tokens"] for r in rows[method]),
            "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in rows[method]),
            "timing_valid": all(r["new_compiled_graphs"] == 0 for r in rows[method] + reference),
        }
        for row, original, matches_row in zip(rows[method], reference, matches):
            if not matches_row:
                common = min(len(row["tokens"]), len(original["tokens"]))
                position = next((i for i in range(common) if row["tokens"][i] != original["tokens"][i]), common)
                mismatches.append({"method": method, "row": row["id"], "position": position})
    count = prefix_sum = full = 0
    for row, original in zip(rows["jacobi"], reference):
        if row["tokens"] != original["tokens"]:
            continue
        stop = len(original["tokens"]) - (original["finish_reason"] == "eos")
        for trace in row["traces"]:
            offset = trace["output_offset"]
            if offset + 6 > stop or len(trace["first_update"]) < 6:
                continue
            expected = original["tokens"][offset:offset + 6]
            matching = 0
            for a, b in zip(expected, trace["first_update"]):
                if a != b:
                    break
                matching += 1
            count += 1
            prefix_sum += matching
            full += matching == 6
    state = "passed" if not mismatches and all(m["timing_valid"] for m in methods.values()) else "failed"
    report = {"state": state, "model": model, "haystack": length,
              "source_sha256": plan["source_sha256"], "manifest_sha256": digest(manifest),
              "methods": methods, "mismatches": mismatches,
              "first_update": {"blocks": count, "prefix_sum": prefix_sum, "full_matches": full}}
    write_json(cell / "audit.json", report)
    update_notes(run)
    return report


def cell(args):
    run = args.run.resolve()
    plan = json.loads((run / "plan.json").read_text())
    check_sources(plan)
    validation = json.loads((run / f"validation-{args.model}.json").read_text())
    if validation["state"] != "passed" or validation["source_sha256"] != plan["source_sha256"]:
        raise RuntimeError("Matching GPU validation is required")
    if args.length != 4096:
        for model in ("4B", "8B"):
            short = json.loads((run / f"{model}-4096/audit.json").read_text())
            if short["state"] != "passed":
                raise RuntimeError("Both short-context audits must pass before long-context runs")
    rows = run / f"prompts-{args.model}-{args.length}.json"
    if digest(json.loads(rows.read_text())) != plan["manifests"][rows.name]:
        raise RuntimeError("Prompt manifest changed after submission")
    rotation = ((args.model == "8B") + (args.length != 4096)) % 3
    order = METHODS[rotation:] + METHODS[:rotation]
    for method in order:
        check_sources(plan)
        subprocess.run([sys.executable, str(ROOT / "benchmarks/jakobi.py"), "run",
                        "--model", args.model, "--method", method, "--rows", str(rows),
                        "--out", str(run / f"{args.model}-{args.length}" / method)], check=True)
    report = audit(run, args.model, args.length)
    if report["mismatches"]:
        first = report["mismatches"][0]
        subprocess.run([sys.executable, str(ROOT / "benchmarks/jakobi.py"), "replay",
                        "--model", args.model, "--method", first["method"], "--rows", str(rows),
                        "--row-id", first["row"], "--out", str(run / f"{args.model}-{args.length}/mismatch-replay.json")],
                       check=True)
    if report["state"] != "passed":
        raise RuntimeError("Audit failed; dependent long-context jobs must not run")


def submit(args):
    run = args.run.resolve()
    if (run / "plan.json").exists():
        raise RuntimeError("Campaign already submitted")
    manifests = {}
    for model in ("4B", "8B"):
        for length in (4096, 129024):
            path = run / f"prompts-{model}-{length}.json"
            manifests[path.name] = digest(json.loads(path.read_text()))
    plan = {"source_sha256": sources(), "manifests": manifests, "jobs": {}}
    write_json(run / "plan.json", plan)
    for length in (4096, 129024):
        for model in ("4B", "8B"):
            name = f"{model}-{length}"
            dependencies = [(run / f"validate-{model}.job").read_text().strip()]
            if length != 4096:
                dependencies.extend(plan["jobs"][f"{m}-4096"] for m in ("4B", "8B"))
            command = [sys.executable, str(Path(__file__).resolve()), "cell", "--run", str(run),
                       "--model", model, "--length", str(length)]
            script = run / f"{name}.sh"
            script.write_text("#!/bin/bash\nset -euo pipefail\ncd " + shlex.quote(str(ROOT)) +
                              "\nexport TOKENIZERS_PARALLELISM=false\nexport OMP_NUM_THREADS=8\n" +
                              shlex.join(command) + "\n")
            job = subprocess.check_output(
                ["sbatch", "--parsable", "--partition=a100", "--gres=gpu:a100_80gb:1", "--qos=normal",
                 "--cpus-per-task=8", "--mem=64G", "--time=04:00:00", "--kill-on-invalid-dep=yes",
                 "--dependency=afterok:" + ":".join(dependencies), "--job-name=jakobi-" + name,
                 "--output=" + str(run / (name + "-%j.log")), str(script)], text=True).strip()
            plan["jobs"][name] = job
            write_json(run / "plan.json", plan)
            print(name, job, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("submit", "cell", "report"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", choices=("4B", "8B"))
    parser.add_argument("--length", type=int)
    args = parser.parse_args()
    if args.command == "report":
        update_notes(args.run.resolve())
    else:
        {"submit": submit, "cell": cell}[args.command](args)
