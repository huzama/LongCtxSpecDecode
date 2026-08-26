# SPDX-License-Identifier: Apache-2.0
"""Long-context decode grid: dense, vegas, coverage on pg19 prompts.

One cell = one context length, one batch size, one mode, one fresh engine in
its own process. Decode throughput is the difference between a gen=1 and a
gen=N run at identical prefill; acceptance comes from vLLM's spec-decode
counters; per-layer budgets from the coverage overrider. Every cell appends
one JSON line to <run dir>/results.jsonl and saves its output token ids, so
a `--parity` run can compare spec modes with dense token for token.

Rules from the baseline grid: serial cells on an idle GPU, a drain wait
between cells, power-of-two batches, prefix caching off, greedy, ignore_eos.
Spec modes need nvcc and ninja on PATH (the fork's JIT top-k kernel).

Examples (inside a slurm allocation on the node holding the venv):
  python benchmarks/longspec/grid.py --ctx 32768 --batch 1 --mode coverage
  python benchmarks/longspec/grid.py --cells 32768:1:dense,32768:1:vegas,32768:1:coverage
  python benchmarks/longspec/grid.py --cells 4096:1:dense,4096:1:coverage --parity --theta 1 --ratio 1
"""

import argparse
import json
import os
import random
import shlex
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

MODEL = "Qwen/Qwen3-4B"
NATIVE_WINDOW = 40960  # Qwen3 max_position_embeddings; beyond it, YaRN
SPEC_MODES = ("vegas", "coverage")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells", help="comma list of ctx:batch:mode; runs each "
                   "in a subprocess, serially")
    p.add_argument("--ctx", type=int, default=32768)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--mode", choices=("dense",) + SPEC_MODES, default="dense")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--gen", type=int, default=256, help="generated tokens")
    p.add_argument("--spec-tokens", type=int, default=6)
    p.add_argument("--ratio", type=float, default=0.07,
                   help="vegas budget ratio; coverage cap")
    p.add_argument("--theta", type=float, default=0.9)
    p.add_argument("--sink", type=int, default=4)
    p.add_argument("--recent", type=int, default=64)
    p.add_argument("--min-tokens", type=int, default=None,
                   help="default: 256 for vegas, 0 for coverage")
    p.add_argument("--skip-attn-layers", default="",
                   help="comma list of layers whose attention the draft skips")
    p.add_argument("--skip-layers", default="",
                   help="comma list of layers the draft bypasses (eager)")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--gpu-mem-util", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt-source", choices=("pg19", "synthetic"),
                   default="pg19")
    p.add_argument("--prompts-dir", default="outputs/prompts",
                   help="cache of tokenized prompts, one file per ctx and slot")
    p.add_argument("--out", help="run directory; default outputs/<slug>-<stamp>")
    p.add_argument("--slug", default="grid")
    p.add_argument("--drain", type=float, default=20.0,
                   help="seconds to wait after a cell releases the GPU")
    p.add_argument("--parity", action="store_true",
                   help="after --cells, compare each spec cell's tokens with "
                   "the dense cell at the same ctx and batch")
    return p.parse_args(argv)


def _int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Run directory
# ---------------------------------------------------------------------------

def repo_root() -> Path:
    return Path(subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True).strip())


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                   text=True).strip()


def create_run_dir(slug: str) -> Path:
    stamp = time.strftime("%y%m%d-%H%M%S")
    run_dir = repo_root() / "outputs" / f"{slug}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "command.txt").write_text(
        shlex.join([sys.executable, *sys.argv]) + "\n")
    return run_dir


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def _long_text(seed: int, min_chars: int) -> str:
    """Concatenate pg19 books from the parquet mirror until min_chars."""
    from datasets import load_dataset
    stream = load_dataset("emozilla/pg19", split="train", streaming=True)
    rng = random.Random(seed)
    skip = rng.randrange(0, 50)
    pieces, chars = [], 0
    for i, row in enumerate(stream):
        if i < skip:
            continue
        pieces.append(row["text"])
        chars += len(row["text"]) + 2
        if chars >= min_chars:
            break
    return "\n\n".join(pieces)


def _pg19_ids(tokenizer, n_tokens: int, seed: int) -> list[int]:
    text = _long_text(seed, n_tokens * 5)
    ids = tokenizer(text, truncation=True, max_length=n_tokens).input_ids
    if len(ids) < n_tokens:
        raise RuntimeError(f"pg19 sample too short: {len(ids)} < {n_tokens}")
    return ids


def _synthetic_ids(tokenizer, n_tokens: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    vocab = tokenizer.vocab_size
    return [rng.randrange(1000, vocab - 1000) for _ in range(n_tokens)]


def build_prompts(args) -> list[list[int]]:
    from transformers import AutoTokenizer
    cache = repo_root() / args.prompts_dir
    cache.mkdir(parents=True, exist_ok=True)
    tokenizer = None
    prompts = []
    for slot in range(args.batch):
        path = cache / f"{args.prompt_source}-{args.ctx}-{slot}.json"
        if path.exists():
            prompts.append(json.loads(path.read_text()))
            continue
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(args.model)
        sample = _pg19_ids if args.prompt_source == "pg19" else _synthetic_ids
        ids = sample(tokenizer, args.ctx, args.seed + slot)
        path.write_text(json.dumps(ids))
        prompts.append(ids)
    return prompts


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def yarn_factor(ctx: int, gen: int) -> float | None:
    need = ctx + gen
    if need <= NATIVE_WINDOW:
        return None
    return float(-(-need // NATIVE_WINDOW))  # ceil, as a float


def speculative_config(args) -> dict | None:
    if args.mode == "dense":
        return None
    min_tokens = args.min_tokens
    if min_tokens is None:
        min_tokens = 0 if args.mode == "coverage" else 256
    cfg = {
        "method": "sparse_attn",
        "num_speculative_tokens": args.spec_tokens,
        "sparse_attn_algorithm": args.mode,
        "sparse_attn_ratio": args.ratio,
        "sparse_attn_min_tokens": min_tokens,
    }
    if args.mode == "coverage":
        cfg.update({
            "sparse_attn_coverage": args.theta,
            "sparse_attn_sink": args.sink,
            "sparse_attn_recent": args.recent,
            "sparse_attn_skip_attn_layers": _int_list(args.skip_attn_layers),
            "sparse_attn_skip_layers": _int_list(args.skip_layers),
        })
    return cfg


def build_engine(args):
    from vllm import LLM
    factor = yarn_factor(args.ctx, args.gen)
    kwargs = dict(
        model=args.model,
        dtype="bfloat16",
        max_num_seqs=args.batch,
        max_model_len=args.ctx + args.gen,
        enable_prefix_caching=False,
        gpu_memory_utilization=args.gpu_mem_util,
        seed=args.seed,
        disable_log_stats=False,
        enforce_eager=args.enforce_eager or bool(_int_list(args.skip_layers)),
        speculative_config=speculative_config(args),
    )
    if factor is not None:
        kwargs["hf_overrides"] = {"rope_scaling": {
            "rope_type": "yarn", "factor": factor,
            "original_max_position_embeddings": NATIVE_WINDOW}}
    return LLM(**kwargs), factor


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def generate(llm, prompts, max_tokens: int):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens,
                            ignore_eos=True)
    inputs = [TokensPrompt(prompt_token_ids=ids) for ids in prompts]
    start = time.perf_counter()
    outputs = llm.generate(inputs, params, use_tqdm=False)
    elapsed = time.perf_counter() - start
    return elapsed, [list(o.outputs[0].token_ids) for o in outputs]


def spec_counters(llm, spec_tokens: int) -> dict:
    from vllm.v1.metrics.reader import Counter, Vector
    counts = {"drafts": 0, "draft_tokens": 0, "accepted": 0,
              "per_pos": [0] * spec_tokens}
    for metric in llm.get_metrics():
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            counts["drafts"] += metric.value
        elif metric.name == "vllm:spec_decode_num_draft_tokens":
            assert isinstance(metric, Counter)
            counts["draft_tokens"] += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            counts["accepted"] += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos, value in enumerate(metric.values[:spec_tokens]):
                counts["per_pos"][pos] += value
    return counts


def _diff(after: dict, before: dict) -> dict:
    return {
        "drafts": after["drafts"] - before["drafts"],
        "draft_tokens": after["draft_tokens"] - before["draft_tokens"],
        "accepted": after["accepted"] - before["accepted"],
        "per_pos": [a - b for a, b in zip(after["per_pos"], before["per_pos"])],
    }


# Module-level so cloudpickle ships them to the worker process.
def _overrider(worker):
    return worker.model_runner.drafter.attn_overrider


def _overrider_stats(worker) -> dict:
    return _overrider(worker).stats()


def _overrider_reset(worker) -> None:
    _overrider(worker).reset_stats()


def run_cell(args, run_dir: Path) -> dict:
    if args.mode in SPEC_MODES and shutil.which("nvcc") is None:
        raise SystemExit("spec modes need nvcc on PATH: export "
                         'PATH="$PWD/.venv/bin:/usr/local/cuda/bin:$PATH" '
                         "CUDA_HOME=/usr/local/cuda")
    import torch
    prompts = build_prompts(args)
    llm, factor = build_engine(args)
    spec = args.mode in SPEC_MODES

    generate(llm, prompts, 1)  # warmup: graphs, JIT kernels, allocator
    t_1, _ = generate(llm, prompts, 1)
    if args.mode == "coverage":
        llm.collective_rpc(_overrider_reset)
    before = spec_counters(llm, args.spec_tokens) if spec else None
    t_gen, tokens = generate(llm, prompts, args.gen)
    counters = _diff(spec_counters(llm, args.spec_tokens), before) if spec else None
    budget = llm.collective_rpc(_overrider_stats)[0] if args.mode == "coverage" else None

    decode_tok_s = args.batch * (args.gen - 1) / (t_gen - t_1)
    record = {
        "ctx": args.ctx, "batch": args.batch, "mode": args.mode,
        "model": args.model, "gen": args.gen, "spec_tokens": args.spec_tokens,
        "ratio": args.ratio, "theta": args.theta, "sink": args.sink,
        "recent": args.recent,
        "min_tokens": speculative_config(args)["sparse_attn_min_tokens"] if spec else None,
        "skip_attn_layers": _int_list(args.skip_attn_layers),
        "skip_layers": _int_list(args.skip_layers),
        "enforce_eager": args.enforce_eager or bool(_int_list(args.skip_layers)),
        "yarn_factor": factor, "prompt_source": args.prompt_source,
        "seed": args.seed,
        "t_1": t_1, "t_gen": t_gen, "decode_tok_s": decode_tok_s,
        "prefill_tok_s": args.batch * args.ctx / t_1,
        "node": socket.gethostname(), "gpu": torch.cuda.get_device_name(),
        "git_sha": git_sha(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if spec:
        drafts, draft_tokens = counters["drafts"], counters["draft_tokens"]
        record.update({
            "drafts": drafts, "draft_tokens": draft_tokens,
            "accepted": counters["accepted"],
            "alpha": counters["accepted"] / draft_tokens if draft_tokens else None,
            "tau": 1 + counters["accepted"] / drafts if drafts else None,
            "accept_per_pos": [c / drafts if drafts else None
                               for c in counters["per_pos"]],
        })
    if budget is not None:
        record["budget"] = budget
    with (run_dir / "results.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    (run_dir / f"tokens-{args.mode}-{args.ctx}-{args.batch}.json").write_text(
        json.dumps(tokens))
    print(json.dumps(record))
    return record


# ---------------------------------------------------------------------------
# Serial cells
# ---------------------------------------------------------------------------

def _gpu_memory_used_mb() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return max(int(x) for x in out.split())


def drain(seconds: float, timeout: float = 180.0) -> None:
    """Wait until the previous engine has released the GPU, then settle."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        used = _gpu_memory_used_mb()
        if used is None or used < 2048:
            break
        time.sleep(2)
    time.sleep(seconds)


def run_cells(args, run_dir: Path) -> int:
    cells = [c.split(":") for c in args.cells.split(",") if c.strip()]
    passthrough = [a for a in sys.argv[1:]]
    # Strip the cell selection and any per-cell overrides from the child args.
    skip = {"--cells", "--ctx", "--batch", "--mode", "--out", "--parity"}
    child_args, i = [], 0
    while i < len(passthrough):
        if passthrough[i] in skip:
            i += 1 if passthrough[i] == "--parity" else 2
            continue
        child_args.append(passthrough[i])
        i += 1
    for ctx, batch, mode in cells:
        cmd = [sys.executable, __file__, "--ctx", ctx, "--batch", batch,
               "--mode", mode, "--out", str(run_dir), *child_args]
        print("+", shlex.join(cmd), flush=True)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"cell {ctx}:{batch}:{mode} failed ({result.returncode})",
                  flush=True)
            return result.returncode
        drain(args.drain)
    return check_parity(run_dir, cells) if args.parity else 0


def check_parity(run_dir: Path, cells) -> int:
    failures = []
    for ctx, batch, mode in cells:
        if mode == "dense":
            continue
        dense = run_dir / f"tokens-dense-{ctx}-{batch}.json"
        spec = run_dir / f"tokens-{mode}-{ctx}-{batch}.json"
        if not dense.exists():
            failures.append(f"{ctx}:{batch}:{mode}: no dense cell to compare")
            continue
        a, b = json.loads(dense.read_text()), json.loads(spec.read_text())
        for slot, (x, y) in enumerate(zip(a, b)):
            if x != y:
                first = next(i for i, (p, q) in enumerate(zip(x, y)) if p != q)
                failures.append(f"{ctx}:{batch}:{mode} slot {slot}: first "
                                f"mismatch at token {first}")
    (run_dir / "parity.json").write_text(json.dumps(
        {"ok": not failures, "failures": failures}, indent=2))
    print("parity:", "ok" if not failures else failures)
    return 0 if not failures else 1


def main(argv=None) -> int:
    args = parse_args(argv)
    os.chdir(repo_root())
    run_dir = Path(args.out) if args.out else create_run_dir(args.slug)
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.cells:
        return run_cells(args, run_dir)
    # Callables cross into the engine-core process for the overrider stats.
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    run_cell(args, run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
