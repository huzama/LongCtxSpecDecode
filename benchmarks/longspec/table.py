# SPDX-License-Identifier: Apache-2.0
"""Markdown table of a grid run: references per cell, coverage per theta.

Usage: python benchmarks/longspec/table.py [run dir]
"""
import glob, json, sys
run = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("outputs/grid-longctx-2*/"))[-1]
rows = [json.loads(l) for l in open(run + "/results.jsonl")]
cells = sorted({(r["ctx"], r["batch"]) for r in rows})
def key(r): return (r["ctx"], r["batch"])
def kv(r):
    b = r.get("budget")
    return None if not b else 100 * sum(b["mean_ratio_per_layer"]) / len(b["mean_ratio_per_layer"])
print("| ctx | batch | mode | decode tok/s | vs dense | alpha | tau (of 7) | KV read per draft step |")
print("|---|---|---|---|---|---|---|---|")
for c in cells:
    dense = [r for r in rows if key(r) == c and r["mode"] == "dense"]
    d = dense[-1]["decode_tok_s"] if dense else None
    for r in [r for r in rows if key(r) == c]:
        mode = r["mode"] if r["mode"] != "coverage" else "coverage θ %.2f" % r["theta"]
        ratio = "" if d is None or r["mode"] == "dense" else "%.2fx" % (r["decode_tok_s"] / d)
        alpha = "" if r.get("alpha") is None else "%.3f" % r["alpha"]
        tau = "" if r.get("tau") is None else "%.2f" % r["tau"]
        k = kv(r)
        kvs = "" if r["mode"] == "dense" else ("%.1f%%" % (100 * r["ratio"]) if k is None else "%.1f%% mean" % k)
        print("| %dK | %d | %s | %.1f | %s | %s | %s | %s |" % (c[0] // 1024, c[1], mode, r["decode_tok_s"], ratio, alpha, tau, kvs))
print()
print("Per-layer KV ratio (%), coverage cells:")
for r in rows:
    if r["mode"] == "coverage":
        prof = r["budget"]["mean_ratio_per_layer"]
        print("%dK b%d θ %.2f: " % (r["ctx"] // 1024, r["batch"], r["theta"]) + " ".join("%.1f" % (100 * x) for x in prof))
