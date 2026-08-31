# SPDX-License-Identifier: Apache-2.0
"""Where a decode round spends its time: verify, sample, draft, idle.

Runs one cell of the grid under vLLM's torch profiler and reduces the trace
to GPU time per phase and per kernel, plus the idle time no kernel covers.
Phases are vLLM's own scopes (``gpu_model_runner: forward|sample|draft``,
enabled by ``VLLM_CUSTOM_SCOPES_FOR_PROFILING``) and one scope of ours
around the drafter's sampler. Kernels are attributed to the scope whose CPU
range launched them (correlation id), falling back to launch time.

    round_phases.py --ctx 32768 --batch 1 --mode coverage --theta 0.98 ...
    round_phases.py --trace outputs/<run>/trace/<file>.pt.trace.json.gz

Grid arguments are ``grid.py``'s. Reports go to the run directory as JSON
and to stdout as a table; ``--trace`` re-runs the reduction on a saved trace.
"""

import argparse
import gzip
import json
import os
import re
import sys
import traceback
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid  # noqa: E402

SCOPE_PREFIXES = ("gpu_model_runner: ", "longspec: ")
STEP_SCOPE = "gpu_model_runner: preprocess"
FORWARD_SCOPE = "gpu_model_runner: forward"
DRAFT_SCOPE = "gpu_model_runner: draft"
DECODE_FORWARD_FACTOR = 2.5  # a prefill chunk's forward is far above this
FULL_BATCH_FRACTION = 0.9  # rounds shorter than this share of the longest lost sequences
DRAFT_SAMPLE_SCOPE = "longspec: draft sample"
GPU_CATS = ("kernel", "gpu_memcpy", "gpu_memset")
LAUNCH_CATS = ("cuda_runtime", "cuda_driver")

CATEGORIES = (
    ("attention", re.compile(r"flash|fmha|attn|mha", re.I)),
    ("longspec", re.compile(r"mass_select|longspec|_c2q_|_gather_kernel|"
                            r"_index_to_slot", re.I)),
    ("gemm", re.compile(r"gemm|cutlass|xmma|cublas|nvjet|matmul", re.I)),
    ("norm_rope_act", re.compile(r"rms_norm|rotary|silu|act_and_mul|"
                                 r"layer_norm", re.I)),
    ("memcpy_memset", re.compile(r"memcpy|memset", re.I)),
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, add_help=False)
    p.add_argument("--rounds", type=int, default=12,
                   help="decode rounds to profile after warmup")
    p.add_argument("--trace", help="reduce a saved trace; no engine")
    p.add_argument("--help", action="store_true")
    own, rest = p.parse_known_args(argv)
    if own.help:
        p.print_help()
        print("\ngrid arguments:")
        grid.parse_args(["--help"])
    return own, grid.parse_args(rest)


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

def _install_markers(worker) -> None:
    """Worker side: one scope around the drafter's per-step sampler, so the
    draft splits into forward plus metadata versus sampling."""
    from torch.profiler import record_function
    drafter = getattr(worker.model_runner, "drafter", None)
    if drafter is None or not hasattr(drafter, "_save_hidden_states_and_sample"):
        return
    original = drafter._save_hidden_states_and_sample

    def wrapped(*args, **kwargs):
        with record_function(DRAFT_SAMPLE_SCOPE):
            return original(*args, **kwargs)

    drafter._save_hidden_states_and_sample = wrapped


def capture(own, args, run_dir: Path) -> Path:
    trace_dir = run_dir / f"trace-{args.ctx}-{args.batch}-{args.mode}"
    trace_dir.mkdir(parents=True, exist_ok=True)
    llm, _ = grid.build_engine(args, profiler_config={
        "profiler": "torch",
        "torch_profiler_dir": str(trace_dir),
        "torch_profiler_with_stack": False,
        "torch_profiler_use_gzip": True,
    })
    llm.collective_rpc(_install_markers)
    prompts = grid.build_prompts(args)
    grid.generate(llm, prompts, 1)  # graphs, JIT kernels, allocator
    grid.generate(llm, prompts, 8)
    per_round = 1 if args.mode == "dense" else args.spec_tokens
    tokens = own.rounds * per_round + 2
    llm.start_profile()
    elapsed, _ = grid.generate(llm, prompts, tokens)
    llm.stop_profile()
    print(f"profiled {tokens} tokens per sequence in {elapsed:.1f}s")
    traces = sorted(trace_dir.glob("*.pt.trace.json*"),
                    key=lambda f: f.stat().st_mtime)
    assert traces, f"no trace written under {trace_dir}"
    return traces[-1]


# ---------------------------------------------------------------------------
# Reduction
# ---------------------------------------------------------------------------

def load_events(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        doc = json.load(f)
    return [e for e in doc["traceEvents"] if e.get("ph") == "X"]


def category(name: str, cat: str) -> str:
    if cat in ("gpu_memcpy", "gpu_memset"):
        return "memcpy_memset"
    for label, pattern in CATEGORIES:
        if pattern.search(name):
            return label
    return "other"


def reduce_trace(events: list[dict], skip_first: int = 2) -> dict:
    scopes = [e for e in events
              if e["name"].startswith(SCOPE_PREFIXES) and "dur" in e]
    launches = {e["args"]["correlation"]: e for e in events
                if e.get("cat") in LAUNCH_CATS
                and "correlation" in e.get("args", {})}
    gpu = [e for e in events if e.get("cat") in GPU_CATS]
    assert scopes, "no vLLM scopes in the trace; set " \
                   "VLLM_CUSTOM_SCOPES_FOR_PROFILING=1 before the engine starts"

    # Steps: consecutive preprocess scopes on the main thread.
    starts = sorted(e["ts"] for e in scopes if e["name"] == STEP_SCOPE)
    assert len(starts) >= skip_first + 2, f"only {len(starts)} steps traced"

    def step_of(ts: float) -> int:
        lo, hi = 0, len(starts)
        while lo < hi:
            mid = (lo + hi) // 2
            if starts[mid] <= ts:
                lo = mid + 1
            else:
                hi = mid
        return lo - 1

    by_thread = defaultdict(list)
    for e in scopes:
        by_thread[(e["pid"], e["tid"])].append(e)
    for lst in by_thread.values():
        lst.sort(key=lambda e: e["ts"])

    def innermost(pid, tid, ts):
        best = None
        for e in by_thread.get((pid, tid), ()):
            if e["ts"] <= ts <= e["ts"] + e["dur"]:
                if best is None or e["dur"] < best["dur"]:
                    best = e
            elif e["ts"] > ts:
                break
        return best["name"] if best else None

    main = max(by_thread, key=lambda k: len(by_thread[k]))
    steps = defaultdict(lambda: {"gpu_us": defaultdict(float),
                                 "kernels": defaultdict(lambda: [0.0, 0]),
                                 "cats": defaultdict(float), "spans": []})
    fallback = 0
    for k in gpu:
        corr = k.get("args", {}).get("correlation")
        launch = launches.get(corr)
        if launch is not None:
            step = step_of(launch["ts"])
            phase = innermost(launch["pid"], launch["tid"], launch["ts"])
        else:
            fallback += 1
            step = step_of(k["ts"])
            phase = innermost(main[0], main[1], k["ts"])
        phase = phase or "unscoped"
        s = steps[step]
        s["spans"].append((k["ts"], k["ts"] + k["dur"]))
        s["gpu_us"][phase] += k["dur"]
        s["cats"][(phase, category(k["name"], k.get("cat", "")))] += k["dur"]
        entry = s["kernels"][(phase, k["name"])]
        entry[0] += k["dur"]
        entry[1] += 1
    for e in scopes:
        if (e["pid"], e["tid"]) != main:
            continue
        steps[step_of(e["ts"])].setdefault("cpu_us", defaultdict(float))
        steps[step_of(e["ts"])]["cpu_us"][e["name"]] += e["dur"]

    # Decode rounds: the forward of a prefill chunk dwarfs a decode forward,
    # so steps within a small factor of the cheapest forward are decode. Drop
    # the first decode steps and the last step, whose wall time is unknown.
    forward = {i: steps[i]["gpu_us"].get(FORWARD_SCOPE, 0.0) for i in steps}
    floor = min(v for v in forward.values() if v > 0)
    decode = [i for i in sorted(steps)
              if 0 < forward[i] <= DECODE_FORWARD_FACTOR * floor
              and i + 1 < len(starts)]
    rounds = decode[skip_first:]
    assert rounds, "no steady-state rounds after skipping warmup"
    # Sequences finish staggered, so late rounds carry fewer of them; keep
    # the full-batch cluster, the longest rounds.
    longest = max(starts[i + 1] - starts[i] for i in rounds)
    rounds = [i for i in rounds
              if starts[i + 1] - starts[i] >= FULL_BATCH_FRACTION * longest]
    excluded = len(steps) - len(rounds)

    def busy_union(i: int) -> float:
        spans = sorted(steps[i]["spans"])
        total, end = 0.0, -1.0
        for s, e in spans:
            if s > end:
                total += e - s
                end = e
            elif e > end:
                total += e - end
                end = e
        return total

    per_round = [{
        "step": i,
        "wall_ms": (starts[i + 1] - starts[i]) / 1e3,
        "gpu_ms": busy_union(i) / 1e3,
        "forward_gpu_ms": forward[i] / 1e3,
        "draft_gpu_ms": steps[i]["gpu_us"].get(DRAFT_SCOPE, 0.0) / 1e3,
        "draft_cpu_ms": steps[i].get("cpu_us", {}).get(DRAFT_SCOPE, 0.0) / 1e3,
    } for i in rounds]

    n = len(rounds)
    wall = sum(starts[i + 1] - starts[i] for i in rounds) / n
    phases = defaultdict(float)
    cpu = defaultdict(float)
    cats = defaultdict(float)
    kernels = defaultdict(lambda: [0.0, 0])
    for i in rounds:
        for phase, us in steps[i]["gpu_us"].items():
            phases[phase] += us / n
        for name, us in steps[i].get("cpu_us", {}).items():
            cpu[name] += us / n
        for key, us in steps[i]["cats"].items():
            cats[key] += us / n
        for key, (us, count) in steps[i]["kernels"].items():
            kernels[key][0] += us / n
            kernels[key][1] += count / n
    busy = sum(r["gpu_ms"] for r in per_round) / n * 1e3
    return {
        "rounds": n,
        "prefill_steps_excluded": excluded,
        "round_wall_ms": wall / 1e3,
        "gpu_busy_ms": busy / 1e3,
        "idle_ms": (wall - busy) / 1e3,
        "per_round": per_round,
        "gpu_ms_by_phase": {k: v / 1e3 for k, v in
                            sorted(phases.items(), key=lambda kv: -kv[1])},
        "cpu_ms_by_scope": {k: v / 1e3 for k, v in
                            sorted(cpu.items(), key=lambda kv: -kv[1])},
        "gpu_ms_by_phase_category": [
            {"phase": p, "category": c, "ms": us / 1e3}
            for (p, c), us in sorted(cats.items(), key=lambda kv: -kv[1])],
        "top_kernels": [
            {"phase": p, "kernel": k[:90], "ms": us / 1e3, "calls": count}
            for (p, k), (us, count) in sorted(
                kernels.items(), key=lambda kv: -kv[1][0])[:40]],
        "kernels_attributed_by_time": fallback,
    }


def render(report: dict) -> str:
    walls = " ".join(f"{r['wall_ms']:.0f}" for r in report["per_round"])
    lines = [
        f"rounds {report['rounds']} ({report['prefill_steps_excluded']} "
        f"prefill, warm or partial-batch steps excluded): wall "
        f"{report['round_wall_ms']:.2f} "
        f"ms, GPU busy {report['gpu_busy_ms']:.2f} ms, idle "
        f"{report['idle_ms']:.2f} ms",
        f"per-round wall ms: {walls}",
        "", "| phase | GPU ms | CPU scope ms |", "|---|---|---|"]
    cpu = report["cpu_ms_by_scope"]
    for phase, ms in report["gpu_ms_by_phase"].items():
        lines.append(f"| {phase} | {ms:.2f} | {cpu.get(phase, 0.0):.2f} |")
    lines += ["", "| phase | category | GPU ms |", "|---|---|---|"]
    for row in report["gpu_ms_by_phase_category"][:16]:
        lines.append(f"| {row['phase']} | {row['category']} | {row['ms']:.2f} |")
    lines += ["", "| phase | kernel | GPU ms | calls |", "|---|---|---|---|"]
    for row in report["top_kernels"][:20]:
        lines.append(f"| {row['phase']} | `{row['kernel']}` | {row['ms']:.2f} "
                     f"| {row['calls']:.0f} |")
    lines.append(f"\nkernels attributed by launch time (no correlation): "
                 f"{report['kernels_attributed_by_time']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    own, args = parse_args(argv)
    if own.trace:
        trace = Path(own.trace)
        report = reduce_trace(load_events(trace))
        cell = re.fullmatch(r"trace-(\d+)-(\d+)-(\w+)", trace.parent.name)
        if cell:
            ctx, batch, mode = int(cell[1]), int(cell[2]), cell[3]
            report.update(ctx=ctx, batch=batch, mode=mode, trace=str(trace))
            out = trace.parent.parent / f"profile-{ctx}-{batch}-{mode}.json"
            out.write_text(json.dumps(report, indent=1))
            print(f"report {out}")
        print(render(report))
        return 0
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    os.environ["VLLM_CUSTOM_SCOPES_FOR_PROFILING"] = "1"
    run_dir = Path(args.out) if args.out else grid.create_run_dir("profile")
    trace = capture(own, args, run_dir)
    print(f"trace {trace}")
    try:
        report = reduce_trace(load_events(trace))
    except Exception:  # keep the trace; the reduction can be rerun offline
        traceback.print_exc()
        return 0
    report.update(ctx=args.ctx, batch=args.batch, mode=args.mode,
                  theta=args.theta, ratio=args.ratio, node=os.uname().nodename,
                  git_sha=grid.git_sha(), trace=str(trace))
    out = run_dir / f"profile-{args.ctx}-{args.batch}-{args.mode}.json"
    out.write_text(json.dumps(report, indent=1))
    print(render(report))
    print(f"report {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
