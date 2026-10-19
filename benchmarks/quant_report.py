"""CPU-only audit and compact progress tables for the MXFP4 experiment grid."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics

from benchmarks.queue import atomic_json


def report(run, *, notes=None):
    state = json.loads((run / "status.json").read_text())
    plan = json.loads((run / "plan.json").read_text())
    groups = collections.defaultdict(list)
    selectors = []
    hashes = {}

    def read(path):
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    for job in state["jobs"]:
        if job["state"] != "complete" or job["kind"] == "validation":
            continue
        dest = run / job["name"]
        kind = job["kind"]
        if kind == "selector":
            read(dest / "results/config.json")
            records = [read(p) for p in sorted((dest / "results").glob("slot*-layer*.json"))]
            assert len(records) == 18
            for protocol in ("matched", "paper"):
                selected = [r for record in records for r in record["results"] if r["budget"]["protocol"] == protocol]
                assert len(selected) == 18
                assert all(r["indexer_cache_format"] == "mxfp4" and r["validation"]["highlight"]["passed"] for r in selected)
                median = lambda key: statistics.median(r["timings"][key]["median_ms"] for r in selected)
                score, rank = median("highlight_score") * 1000, median("highlight_rank") * 1000
                selectors.append(dict(model=job["model"], ctx=job["ctx"], protocol=protocol,
                    tokens=selected[0]["budget"]["highlight_tokens"], score_us=score, rank_us=rank,
                    total_us=score+rank, measured_selection_us=median("highlight_selection")*1000,
                    initialization_ms=median("highlight_initialize"),
                    extra_kv_mib=selected[0]["workspace_bytes"]["highlight_indexer_cache"]/2**20))
        else:
            data = read(dest / ("summary.json" if kind == "throughput" else "finished.json"))
            key = (kind, job["model"], str(job.get("ctx", job.get("protocol"))), job["method"])
            if kind == "throughput":
                value = data["decode_tok_s"]
                assert math.isfinite(value) and value > 0
            else:
                read(dest / "config.json")
                path = dest / "results.jsonl"
                hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                records = [json.loads(line) for line in path.read_text().splitlines()]
                assert len(records) == job["expected_rows"]
                if kind == "ruler":
                    value = statistics.mean(r["score"] for r in records) * 100
                    assert abs(value / 100 - data["score"]) < 1e-10
                else:
                    assert sum(r["correct"] for r in records) == data["correct"]
                    value = data["correct"] / data["questions"] * 100
            groups[key].append(value)
    counts = dict(collections.Counter(j["state"] for j in state["jobs"]))
    result = dict(state=state["state"], counts=counts, settings=plan["settings"], selectors=selectors,
        measurements=[dict(kind=k[0], model=k[1], setting=k[2], method=k[3], values=v,
                           median=statistics.median(v)) for k,v in groups.items()], source_sha256=hashes)
    atomic_json(run / "summary.json", result)
    methods = ["direct-highlight", "ours-full", "ours-sv50", "ours-sv40"]

    def cell(kind, model, setting, method):
        values = groups.get((kind, model, str(setting), method), [])
        needed = 3 if kind == "throughput" else 1
        return f"{statistics.median(values):.{1 if kind == 'throughput' else 2}f}" if len(values) == needed else ""

    lines = ["### MXFP4 Highlight", "", "MXFP4 indexer keys and native Blackwell scoring; model weights and backbone KV",
        "remain BF16. Existing matched throughput batches and frozen accuracy prompts",
        "are reused. Quantization adds capacity headroom; these batches are not retuned",
        "to the MXFP4 saturation limit. Dense and external-selector baselines above are reused.", "",
        f"Queue: {counts.get('complete', 0)}/{len(state['jobs'])} jobs complete, "
        f"{counts.get('running', 0)} running, {counts.get('failed', 0)} failed. "
        "This includes one native-kernel validation job and 110 measurement jobs.", "",
        "**Decode throughput (tokens/s)**. Same prefill barrier, 512 outputs, asynchronous",
        "scheduling off, decode-only window and three repetitions as BF16. Cells remain",
        "blank until all three repetitions pass validation.", "",
        "| Model | Context | Batch | Direct Highlight | Dense verifier | SV50 | SV40 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for model, batches in [("4b", [26,13,6]), ("8b", [25,12,6])]:
        for ctx,batch in zip([32768,65536,130560],batches):
            lines.append(f"| {model.upper()} | {ctx} | {batch} | " + " | ".join(cell("throughput",model,ctx,m) for m in methods) + " |")
    lines += ["", "**RULER (%)**. Same six needle tasks × 32 rows, batch 1, cap 1024.", "",
        "| Model | Haystack tokens | Direct Highlight | Dense verifier | SV50 | SV40 |", "|---|---:|---:|---:|---:|---:|"]
    for model in ("4b", "8b"):
        for ctx in (65536,129024):
            lines.append(f"| {model.upper()} | {ctx} | " + " | ".join(cell("ruler",model,ctx,m) for m in methods) + " |")
    lines += ["", "**LongBench v2 (%)**. Same 60 questions/model and answer-only/CoT token caps.", "",
        "| Model | Protocol | Direct Highlight | Dense verifier | SV50 | SV40 |", "|---|---|---:|---:|---:|---:|"]
    for model in ("4b", "8b"):
        for protocol in ("answer-only", "cot"):
            label = "Answer-only" if protocol == "answer-only" else "CoT"
            lines.append(f"| {model.upper()} | {label} | " + " | ".join(cell("longbench",model,protocol,m) for m in methods) + " |")
    lines += ["", "**Highlight selector timing**. Same batch-1 inputs, layers, budgets and CUDA-graph",
        "aggregation as the BF16 selector table. Total sums score and rank medians;",
        "native paged attention needs no KV gather. Extra KV includes the MXFP4 scales.", "",
        "| Model | Context | Budget | Tokens/head | Score (µs) | Rank (µs) | Total (µs) | Initialization (ms) | Extra KV (MiB) |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for model in ("4b", "8b"):
        for ctx in (32768,65536,130560):
            for protocol in ("matched", "paper"):
                row = next((r for r in selectors if (r["model"],r["ctx"],r["protocol"]) == (model,ctx,protocol)), None)
                cells = [model.upper(),str(ctx),protocol.capitalize()]
                cells += ([str(row['tokens']),f"{row['score_us']:.2f}",f"{row['rank_us']:.2f}",f"{row['total_us']:.2f}",
                           f"{row['initialization_ms']:.3f}",f"{row['extra_kv_mib']:.2f}"] if row else [""]*6)
                lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "[MXFP4 results and source hashes](../outputs/evidence/mxfp4.json).", ""]
    if notes is not None:
        text = notes.read_text()
        # Completed reports integrate both formats and are edited as a whole.
        # Only replace the original progress section while it still exists.
        if notes.suffix == ".md" and "### MXFP4 Highlight" in text and "### Findings" in text:
            start, end = text.index("### MXFP4 Highlight"), text.index("### Findings")
            assert start < end
            notes.write_text(text[:start] + "\n".join(lines) + "\n" + text[end:])
        else:
            print("Preserved integrated report; refreshed MXFP4 evidence only.")
    evidence_root = notes.parent.parent if notes is not None else run.resolve().parents[2]
    evidence = evidence_root / "outputs/evidence/mxfp4.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(evidence, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--notes", type=Path, help="Optional integrated report; HTML prose is preserved")
    args = parser.parse_args()
    report(args.run, notes=args.notes)


if __name__ == "__main__":
    main()
