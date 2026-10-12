"""Audit independent selector processes and publish their median and range."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics

from benchmarks.queue import atomic_json


def aggregate(processes, repetitions):
    """Keep process medians as the units of replication; never pool samples."""
    groups = collections.defaultdict(list)
    for row in processes:
        groups[(row["model"], row["ctx"], row["protocol"], row["selector"], row["format"])].append(row)
    rows = []
    for key, values in sorted(groups.items()):
        assert len({v["rep"] for v in values}) == len(values), "duplicate process repetition"
        if len(values) != repetitions:
            continue
        assert {v["rep"] for v in values} == set(range(1, repetitions + 1))
        row = dict(zip(("model", "ctx", "protocol", "selector", "format"), key))
        row["processes"] = repetitions
        for field in ("tokens", "extra_kv_mib"):
            assert len({v[field] for v in values}) == 1
            row[field] = values[0][field]
        row["ranges"] = {}
        for field in ("score_us", "rank_us", "gather_us", "total_us", "initialization_ms"):
            samples = [v[field] for v in values]
            if all(v is None for v in samples):
                row[field] = None
                continue
            assert all(v is not None and math.isfinite(v) for v in samples)
            row[field] = statistics.median(samples)
            row["ranges"][field] = [min(samples), max(samples)]
        rows.append(row)
    return rows


def markdown_table(headers, rows, group_columns=()):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(
        "---" if h in {"Model", "Selector", "Indexer", "Budget"} else "---:" for h in headers) + "|"]
    previous = None
    for row in rows:
        group = tuple(row[i] for i in group_columns)
        if previous is not None and group != previous:
            lines.append("| " + " | ".join([""] * len(headers)) + " |")
        lines.append("| " + " | ".join(map(str, row)) + " |")
        previous = group
    return "\n".join(lines) + "\n"


def update_notes(text, rows):
    def replace(key, content):
        nonlocal text
        start = f"<!-- selector-repeatability:{key}:start -->"
        end = f"<!-- selector-repeatability:{key}:end -->"
        assert text.count(start) == text.count(end) == 1
        a, z = text.index(start) + len(start), text.index(end)
        assert a < z
        text = text[:a] + "\n" + content.strip() + "\n\n" + text[z:]

    lookup = {(r["model"], r["ctx"], r["protocol"], r["selector"], r["format"]): r for r in rows}
    order = [("highlight", "bf16"), ("highlight", "mxfp4"), ("vegas", "n/a"), ("triforce", "n/a")]
    labels = dict(highlight="Highlight", vegas="Vegas", triforce="TriForce")
    for protocol in ("matched", "paper"):
        blocks = []
        for model in ("4b", "8b"):
            cells = []
            for ctx in (32768, 65536, 130560):
                for selector, fmt in order:
                    r = lookup[model, ctx, protocol, selector, fmt]
                    cells.append([f"{ctx:,}", labels[selector], fmt.upper() if fmt != "n/a" else fmt,
                        f"{r['tokens']:,}", f"{r['score_us']:,.2f}", f"{r['rank_us']:,.2f}",
                        "n/a" if r["gather_us"] is None else f"{r['gather_us']:,.2f}",
                        f"{r['total_us']:,.2f}", f"{r['extra_kv_mib']:,.2f}"])
            blocks += [f"#### Qwen3-{model.upper()}\n", markdown_table(
                ["Context", "Selector", "Indexer", "Tokens/head", "Score (µs)", "Rank/index (µs)",
                 "Gather (µs)", "Sum of stages (µs)", "Extra KV (MiB)"], cells, (0,))]
        replace(protocol, "\n".join(blocks))
    cells = []
    spreads = []
    reductions = []
    for model in ("4b", "8b"):
        for ctx in (32768, 65536, 130560):
            for protocol in ("matched", "paper"):
                b = lookup[model, ctx, protocol, "highlight", "bf16"]
                m = lookup[model, ctx, protocol, "highlight", "mxfp4"]
                tri = lookup[model, ctx, protocol, "triforce", "n/a"]
                cells.append([model.upper(), f"{ctx:,}", protocol.capitalize(),
                    f"{b['initialization_ms']:.3f}", f"{m['initialization_ms']:.3f}", f"{tri['initialization_ms']:.3f}"])
                intervals = []
                for selector, fmt in order:
                    low, high = lookup[model, ctx, protocol, selector, fmt]["ranges"]["total_us"]
                    intervals.append(f"{low:,.2f}–{high:,.2f}")
                spreads.append([model.upper(), f"{ctx:,}", protocol.capitalize(), *intervals])
                if protocol == "matched":
                    reductions.append(100 * (1 - m["total_us"] / b["total_us"]))
    replace("initialization", markdown_table(["Model", "Context", "Budget", "Highlight BF16 (ms)",
        "Highlight MXFP4 (ms)", "TriForce (ms)"], cells, (0, 1)))
    replace("spread", "### Selector variation across processes\n\n"
        "Minimum–maximum summed stages (µs) across three fresh processes. These\n"
        "are observed ranges, not confidence intervals. Each process contributes\n"
        "one sum of capture-level stage medians; raw timing samples are not pooled.\n\n" +
        markdown_table(["Model", "Context", "Budget", "Highlight BF16", "Highlight MXFP4", "Vegas", "TriForce"],
                       spreads, (0, 1)) + "\n[Repetition evidence and source hashes](../outputs/evidence/selector-repeatability.json).")
    replace("protocol", """Batch 1; the first six existing prompts per model/context; layers 2, 18 and 35.
Three fresh process repetitions per model/context and indexer format. Within
each process, each stage is the median of 18 capture medians. Each capture
uses five warmups and 20 CUDA-graph samples of 20 calls; initialization uses
one call per sample. Loading, capture, transfers, compilation and warm-up are
excluded. Each displayed stage is the median of the three process medians.

**Sum of stages** is computed within each process from its score, rank/index
and gather medians, then summarized by the median across processes. It may
differ from the sum of the separately aggregated displayed stages. It
excludes initialization and is not an independently measured end-to-end
latency. Highlight requires no KV gather; `n/a` contributes zero to the sum.
The appendix reports the observed ranges across processes. The original
single-process runs are retained as prior evidence and are not pooled into
these three fresh repetitions.""")
    changes = [-value for value in reductions]
    replace("comparison", f"""At matched budgets, MXFP4 changes Highlight's summed stages by
{min(changes):+.1f}% to {max(changes):+.1f}% relative to BF16 across the six settings.
The appendix gives the observed process ranges. The indexer cache uses 26.56%
of BF16 storage, including scales: 8.50 / 17.00 / 33.87 MiB versus
32.00 / 64.00 / 127.50 MiB. This is a 73.44% reduction in indexer storage,
not in total model or backbone KV memory.""")
    text = text.replace("- **Repeatability:** throughput has three repetitions; selector captures have\n"
        "  no independent process repetitions. Cross-host baseline reuse and timing\n"
        "  variation limit interpretation of small differences.",
        "- **Repeatability:** throughput and selector timing each have three process\n"
        "  repetitions. Selector ranges describe observed variation, not confidence\n"
        "  intervals. Cross-host baseline reuse limits small throughput comparisons.")
    return text


def report(run):
    state = json.loads((run / "status.json").read_text())
    plan = json.loads((run / "plan.json").read_text())
    repetitions = plan["settings"]["process_repetitions"]
    root = run.parents[2]
    hashes, processes, provenance = {}, [], collections.defaultdict(set)
    inputs, source_sets = {}, set()

    def read(path):
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    for job in state["jobs"]:
        if job["state"] != "complete":
            continue
        dest = run / job["name"] / "results"
        config = read(dest / "config.json")
        # Controller/report repairs do not change the measured kernels or inputs.
        # Every original source hash remains in the saved config; require all
        # measurement code to match across the independent processes.
        administrative = ("/benchmarks/queue.py", "/benchmarks/quant_report.py", "/benchmarks/selector_report.py")
        measurement_sources = {p: h for p, h in config["source_sha256"].items()
                               if not p.endswith(administrative)}
        source_sets.add(hashlib.sha256(json.dumps(measurement_sources, sort_keys=True).encode()).hexdigest())
        finished = read(dest / "finished.json")
        paths = sorted(dest.glob("slot*-layer*.json"))
        assert len(paths) == finished["records"] == 18 and finished["budget_records"] == 36
        records = [read(p) for p in paths]
        for path, record in zip(paths, records):
            key = (job["model"], job["ctx"], path.name)
            p = record["provenance"]
            identity = (p["prompt_file_sha256"], p["checkpoint_config_sha256"], p["prefix_tokens"])
            assert inputs.setdefault(key, identity) == identity, "inputs differ between processes"
            provenance[key].add(hashlib.sha256(json.dumps(p["tensor_sha256"], sort_keys=True).encode()).hexdigest())
        for protocol in ("matched", "paper"):
            values = [r for record in records for r in record["results"] if r["budget"]["protocol"] == protocol]
            assert len(values) == 18
            def median_stage(name, scale=1000):
                return statistics.median(v["timings"][name]["median_ms"] for v in values) * scale
            for selector in job["selectors"]:
                assert all(v["validation"][selector]["passed"] for v in values)
                if selector == "vegas":
                    paired = []
                    for v in values:
                        a = v["timings"]["vegas_collection_delta"]["samples_ms"]
                        b = v["timings"]["vegas_reduce"]["samples_ms"]
                        assert len(a) == len(b) == 20
                        paired.append(statistics.median(x + y for x, y in zip(a, b)))
                    score, init = statistics.median(paired) * 1000, None
                else:
                    score = median_stage(selector + "_score")
                    init = median_stage("highlight_initialize" if selector == "highlight" else
                                        "triforce_initialize_chunk_keys", scale=1)
                rank = median_stage(selector + "_rank")
                gather = None if selector == "highlight" else median_stage(selector + "_gather")
                memory_key = "highlight_indexer_cache" if selector == "highlight" else selector + "_gathered_kv"
                memory = {v["workspace_bytes"][memory_key] / 2**20 for v in values}
                tokens = {v["budget"][selector + "_tokens"] for v in values}
                assert len(memory) == len(tokens) == 1
                processes.append(dict(model=job["model"], ctx=job["ctx"], protocol=protocol,
                    selector=selector, format=job["indexer_cache_format"] if selector == "highlight" else "n/a",
                    rep=job["rep"], pid=job["pid"], gpu=job["gpu"], job=job["name"],
                    tokens=tokens.pop(), score_us=score, rank_us=rank, gather_us=gather,
                    total_us=score + rank + (gather or 0), initialization_ms=init, extra_kv_mib=memory.pop()))
    assert len(source_sets) <= 1, "source versions differ between processes"
    rows = aggregate(processes, repetitions)
    result = dict(state=state["state"], counts=dict(collections.Counter(j["state"] for j in state["jobs"])),
        settings=plan["settings"], rows=rows, process_results=processes, source_sha256=hashes,
        measurement_source_sha256=sorted(source_sets),
        capture_tensor_variants={"/".join(map(str, k)): len(v) for k, v in provenance.items()},
        aggregation="Within process: median of 18 capture medians for each stage, then sum stages. "
                    "Across fresh processes: median and min/max of each stage and of the process sum; no pooled samples.")
    if state["state"] == "complete":
        assert len(rows) == 48 and len(processes) == 48 * repetitions
        assert len({j["pid"] for j in state["jobs"]}) == len(state["jobs"]), "process IDs were reused"
        notes = root / "notes/experiments.md"
        content = notes.read_text()
        if hashlib.sha256(content.encode()).hexdigest() == plan["notes_sha256"]:
            updated = update_notes(content, rows)
            notes.write_text(updated)
            result["notes_updated"] = True
        else:
            result["notes_updated"] = False
            print("Preserved edited notes; complete repetition evidence is ready.", flush=True)
    atomic_json(run / "summary.json", result)
    atomic_json(root / "outputs/evidence/selector-repeatability.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    report(args.run.resolve())


if __name__ == "__main__":
    main()
