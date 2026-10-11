"""Audit saved chapter runs without starting an engine.

Usage: python -m benchmarks.report CHAPTER_RUN --out REPORT.json
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

METHODS = ("dense", "dense-thesis", "vegas", "ours-full", "ours-sv50", "ours-sv40")


def read_json(path):
    value = json.loads(path.read_text())
    if "tokens" in value and "prompt_sha256" not in value:
        value["prompt_sha256"] = [hashlib.sha256(json.dumps(p, separators=(",", ":")).encode()).hexdigest()
                                    for p in value["prompts"]]
    return value


def agreement(reference, candidate, eos_token_id=None):
    for key in ("prompt_slots", "prompt_sha256"):
        if reference[key] != candidate[key]:
            raise ValueError(f"output comparison has different {key}")
    a, b = reference["tokens"], candidate["tokens"]
    if len(a) != len(b):
        raise ValueError("output comparison has different sequence counts")
    mismatches = []
    for row, (x, y) in enumerate(zip(a, b)):
        if x != y:
            pos = next((i for i, pair in enumerate(zip(x, y)) if pair[0] != pair[1]), min(len(x), len(y)))
            mismatches.append({"row": row, "slot": reference["prompt_slots"][row], "position": pos})
    result = {"sequences": len(a), "identical": len(a) - len(mismatches), "mismatches": mismatches}
    if eos_token_id is not None:
        def stopped(tokens):
            return tokens[:tokens.index(eos_token_id) + 1] if eos_token_id in tokens else tokens
        result["identical_through_eos"] = sum(stopped(x) == stopped(y) for x, y in zip(a, b))
    return result


def summarize(runs):
    """Pool each repetition, then divide median rates (not median ratios)."""
    rates = [sum(r["decode_tokens"] for r in rows) / sum(r["decode_seconds"] for r in rows) for rows in runs]
    drafted = sum(r.get("draft_tokens", 0) or 0 for rows in runs for r in rows)
    accepted = sum(r.get("accepted", 0) or 0 for rows in runs for r in rows)
    return {"tok_s": statistics.median(rates), "range": [min(rates), max(rates)],
            "repetitions": rates, "acceptance": accepted / drafted if drafted else None}


def report(root, eos_token_id=None):
    cells = []
    for model in ("4b", "8b"):
        for ctx in (32768, 65536, 130560):
            cell = root / model / str(ctx)
            reps = sorted(cell.glob("rep*"))
            if len(reps) != 3:
                raise ValueError(f"expected three repetitions: {cell}")
            records, tokens = {}, {}
            for method in METHODS:
                records[method] = [[json.loads(line) for line in (rep / method / "results.jsonl").read_text().splitlines()] for rep in reps]
                tokens[method] = [read_json(next((rep / method).glob("tokens-*.json"))) for rep in reps]
            for method in METHODS:
                for baseline, rows in zip(records["dense"], records[method]):
                    for key in ("prompt_sha256", "batch", "ctx", "gen"):
                        if [r[key] for r in baseline] != [r[key] for r in rows]:
                            raise ValueError(f"mismatched {key}: {cell}/{method}")
            stats = {method: summarize(records[method]) for method in METHODS}
            comparisons = {}
            for method in ("vegas", "ours-full", "ours-sv50", "ours-sv40"):
                base = "dense-thesis" if method == "vegas" else "dense"
                stats[method]["vs_own_dense"] = stats[method]["tok_s"] / stats[base]["tok_s"]
                stats[method]["vs_vegas"] = stats[method]["tok_s"] / stats["vegas"]["tok_s"]
                comparisons[method] = [agreement(a, b, eos_token_id) for a, b in zip(tokens[base], tokens[method])]
            cells.append({"model": model, "ctx": ctx, "batch": records["dense"][0][0]["batch"],
                          "methods": stats, "agreement": comparisons,
                          "dense_repeat": agreement(tokens["dense"][0], tokens["dense"][1])})
    sources = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(root.glob("*/*/rep*/*/*.json*")) if p.name == "results.jsonl" or p.name.startswith("tokens-")}
    return {"source": str(root.resolve()), "aggregation": "ratio of median pooled throughput",
            "eos_token_id": eos_token_id, "cells": cells, "sha256": sources}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--eos-token", type=int, help="also compare sequences through the first EOS")
    args = parser.parse_args()
    result = report(args.root, args.eos_token)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for cell in result["cells"]:
        s = cell["methods"]["ours-full"]
        print(f"{cell['model']} {cell['ctx']}: {s['vs_own_dense']:.3f}x dense, {s['vs_vegas']:.3f}x Vegas")


if __name__ == "__main__":
    main()
