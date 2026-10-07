# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LongBench v2 decode measurements: dense, Vegas, and our method.

One fresh engine per cell, serially on one GPU. Decode time comes from the
engine's token timestamps. Latency runs stop at EOS or the token limit.
Batch runs hold completed prefills, then time full-batch decode to the first
finished request; EOS is ignored in this throughput measurement. Output equality
is reported separately from speed; independent trajectories can diverge numerically.

Example (inside Slurm):
  .venv/bin/python benchmarks/longspec/grid.py --out outputs/runs/<run> \
      --cells 32768:1:dense,32768:1:vegas,32768:1:coverage
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

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import FullAttentionSpec

MODEL = "Qwen/Qwen3-4B"
NATIVE_WINDOW = 40960  # Qwen3 max_position_embeddings; beyond it, YaRN
SPEC_MODES = ("vegas", "coverage")
LONGBENCH2 = "THUDM/LongBench-v2"
# prompts/0shot_cot.txt of github.com/THUDM/LongBench, verbatim.
LONGBENCH2_COT = """Please read the following text and answer the questions below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Let’s think step by step:"""


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument(
        "--cells", help="comma-separated ctx:batch:mode; serial subprocesses"
    )
    p.add_argument("--ctx", type=int, default=32768)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument(
        "--measurement",
        choices=("latency", "batch"),
        default="latency",
        help="single-request EOS run, or synchronized full-batch decode",
    )
    p.add_argument("--mode", choices=("dense",) + SPEC_MODES, default="dense")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--gen", type=int, default=512, help="maximum generated tokens")
    p.add_argument("--spec-tokens", type=int, default=6)
    p.add_argument("--ratio", type=float, help="override our cap; Vegas stays at 0.07")
    p.add_argument("--theta", type=float, help="override the draft top-p default")
    p.add_argument("--min-tokens", type=int, help="override our selection floor")
    p.add_argument(
        "--draft-weights", help="our draft checkpoint; 'target' uses target weights"
    )
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--gpu-mem-util", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompts-dir", default="outputs/prompts")
    p.add_argument(
        "--samples",
        type=int,
        default=8,
        help="prompt pool size; 1 selects a single batch for checks",
    )
    p.add_argument("--out", required=True, help="run directory")
    args = p.parse_args(argv)
    if min(args.ctx, args.batch, args.gen, args.samples, args.spec_tokens) < 1:
        p.error("context, batch, generation, samples and draft length must be positive")
    if not args.cells:
        if args.measurement == "latency" and args.batch != 1:
            p.error("batched runs require --measurement batch")
        if args.measurement == "batch" and args.gen < 2:
            p.error("batch measurement needs at least two output tokens")
        if args.measurement == "batch" and 1 < args.samples < args.batch:
            p.error("the prompt pool must contain at least --batch questions")
    return args


class BatchScheduler(Scheduler):
    """Benchmark-only scheduler: retain completed prefills until all are ready."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.scheduler_config.async_scheduling:
            raise ValueError("batch measurement requires synchronous scheduling")
        groups = self.kv_cache_config.kv_cache_groups
        if (
            len(groups) != 1
            or type(groups[0].kv_cache_spec) is not FullAttentionSpec
            or groups[0].kv_cache_spec.sliding_window is not None
            or groups[0].kv_cache_spec.attention_chunk_size is not None
        ):
            raise ValueError("batch measurement supports one full-attention KV group")
        if self.cache_config.enable_prefix_caching:
            raise ValueError("batch measurement requires prefix caching disabled")
        self._released = False

    def add_request(self, request):
        if not self.requests:
            self._released = False
        if self._released or len(self.requests) >= self.max_num_running_reqs:
            raise ValueError("submit exactly one batch, then drain it before the next")
        super().add_request(request)
        # Reserve for the complete requested generation, not just the prefill.
        required = sum(
            (
                r.num_prompt_tokens
                + r.max_tokens
                + self.num_lookahead_tokens
                + self.block_size
                - 1
            )
            // self.block_size
            for r in self.requests.values()
        )
        available = self.kv_cache_manager.block_pool.num_gpu_blocks - 1
        if required > available:
            raise ValueError(
                f"batch does not fit: needs {required} KV blocks, has {available} "
                "after reserving the null block; reduce --batch"
            )

    def schedule(self):
        held = []
        if not self._released:
            held = [r for r in self.running if r.num_output_tokens > 0]
            if len(held) == self.max_num_running_reqs:
                self._released = True
                held = []
            else:
                # These requests retain their KV blocks and pending draft tokens.
                self.running = [r for r in self.running if r.num_output_tokens == 0]
        expected = {r.request_id for r in self.running} if self._released else None
        try:
            output = super().schedule()
        finally:
            self.running = held + self.running
        if output.preempted_req_ids:
            raise RuntimeError("invalid batch measurement: KV preemption occurred")
        if self._released:
            if self.waiting or set(output.num_scheduled_tokens) != expected:
                raise RuntimeError(
                    "invalid batch measurement: not every active request ran"
                )
        elif (
            len(self.requests) == self.max_num_running_reqs
            and output.total_num_scheduled_tokens == 0
        ):
            raise RuntimeError("prefill barrier cannot make progress; reduce --batch")
        return output


class BatchWindow:
    """Count emitted tokens from the prefill barrier through the first finish."""

    def __init__(self, request_ids):
        self.request_ids = set(request_ids)
        self.counts = {}
        self.first = {}
        self.start = None
        self.end = None
        self.tokens = 0

    def update(self, outputs):
        if not outputs or self.end is not None:
            return
        if self.start is not None and (
            len(outputs) != len(self.request_ids)
            or {o.request_id for o in outputs} != self.request_ids
        ):
            raise RuntimeError("decode step did not contain the whole batch")
        timestamps = []
        for output in outputs:
            if (
                output.request_id not in self.request_ids
                or output.metrics is None
                or output.metrics.first_token_ts <= 0
            ):
                raise RuntimeError("missing request or engine timestamps")
            count = len(output.outputs[0].token_ids)
            if self.start is None and (count != 1 or output.finished):
                raise RuntimeError("generation advanced before the prefill barrier")
            if count < self.counts.get(output.request_id, 0):
                raise RuntimeError("batch output must contain cumulative token IDs")
            self.counts[output.request_id] = count
            self.first[output.request_id] = output.metrics.first_token_ts
            timestamps.append(output.metrics.last_token_ts)
        if self.start is None:
            if self.counts.keys() == self.request_ids:
                self.start = max(self.first.values())
            return
        if len(set(timestamps)) != 1 or timestamps[0] <= self.start:
            raise RuntimeError("batch outputs do not share a valid engine timestamp")
        if any(o.finished for o in outputs):
            self.end = timestamps[0]
            self.tokens = sum(self.counts.values()) - len(self.request_ids)

    def report(self):
        if self.end is None or self.start is None:
            raise RuntimeError("no complete full-batch decode window")
        seconds = self.end - self.start
        return {
            "decode_seconds": seconds,
            "decode_tok_s": self.tokens / seconds,
            "decode_tokens": self.tokens,
            "decode_tokens_per_request": {
                key: count - 1 for key, count in self.counts.items()
            },
            "decode_window": "all_prefills_complete_to_first_request_finish",
        }


def prompt_groups(args, prompts):
    if args.measurement == "latency":
        return [([prompt], [slot]) for slot, prompt in enumerate(prompts)]
    if args.samples == 1:
        return [(prompts, list(range(len(prompts))))]
    # Each question appears equally often at every tested batch size.
    return [
        (
            [prompts[(slot + j) % len(prompts)] for j in range(args.batch)],
            [(slot + j) % len(prompts) for j in range(args.batch)],
        )
        for slot in range(len(prompts))
    ]


def repo_root() -> Path:
    return Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
    )


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True
    ).strip()


def _longbench2_slots(
    tokenizer, n_tokens: int, count: int, seed: int, cache: Path
) -> None:
    """Write `count` LongBench v2 prompts of exactly n_tokens ids, plus an
    index naming the question behind each slot.

    The official chain-of-thought prompt goes into the model's chat template
    with the default (thinking) mode; the document is cut in the middle, as
    the official pred.py does. Questions are drawn, seeded, from those whose
    document fills the budget at most twice over, so the cut stays mild;
    longer documents fill any shortfall, shortest first."""
    from huggingface_hub import hf_hub_download

    items = json.loads(
        Path(hf_hub_download(LONGBENCH2, "data.json", repo_type="dataset")).read_text()
    )
    rows = []
    for item in items:
        text = LONGBENCH2_COT
        for key, field in (
            ("$Q$", "question"),
            ("$C_A$", "choice_A"),
            ("$C_B$", "choice_B"),
            ("$C_C$", "choice_C"),
            ("$C_D$", "choice_D"),
        ):
            text = text.replace(key, item[field].strip())
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        head, tail = chat.split("$DOC$")
        head_ids = tokenizer(head, add_special_tokens=False).input_ids
        tail_ids = tokenizer(tail, add_special_tokens=False).input_ids
        budget = n_tokens - len(head_ids) - len(tail_ids)
        if budget <= 0:
            raise ValueError("context length is too short for the LongBench question")
        doc_ids = tokenizer(item["context"].strip(), add_special_tokens=False).input_ids
        if len(doc_ids) >= budget:
            rows.append((item, head_ids, doc_ids, tail_ids, budget))
    mild = [r for r in rows if len(r[2]) <= 2 * r[4]]
    random.Random(seed).shuffle(mild)
    rest = sorted((r for r in rows if len(r[2]) > 2 * r[4]), key=lambda r: len(r[2]))
    chosen = (mild + rest)[:count]
    if len(chosen) < count:
        raise RuntimeError(
            f"LongBench v2 has {len(chosen)} documents of at "
            f"least {n_tokens} tokens, {count} requested"
        )
    index = []
    for slot, (item, head_ids, doc_ids, tail_ids, budget) in enumerate(chosen):
        doc = doc_ids[: budget // 2] + doc_ids[len(doc_ids) - (budget - budget // 2) :]
        ids = head_ids + doc + tail_ids
        assert len(ids) == n_tokens
        (cache / f"longbench2-{n_tokens}-{slot}.json").write_text(json.dumps(ids))
        index.append(
            {
                "slot": slot,
                "id": item["_id"],
                "domain": item["domain"],
                "sub_domain": item["sub_domain"],
                "difficulty": item["difficulty"],
                "length": item["length"],
                "doc_tokens": len(doc_ids),
                "kept_tokens": budget,
            }
        )
    (cache / f"longbench2-{n_tokens}-index.json").write_text(
        json.dumps(index, indent=1)
    )


def prompt_count(args) -> int:
    return args.samples if args.samples > 1 else args.batch


def build_prompts(args) -> list[list[int]]:
    from transformers import AutoTokenizer

    cache = (
        repo_root() / args.prompts_dir / args.model.replace("/", "--") / str(args.seed)
    )
    cache.mkdir(parents=True, exist_ok=True)
    paths = [
        cache / f"longbench2-{args.ctx}-{slot}.json"
        for slot in range(prompt_count(args))
    ]
    if not all(path.exists() for path in paths):
        _longbench2_slots(
            AutoTokenizer.from_pretrained(args.model),
            args.ctx,
            len(paths),
            args.seed,
            cache,
        )
    return [json.loads(path.read_text()) for path in paths]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def yarn_factor(ctx: int, gen: int) -> float | None:
    need = ctx + gen
    if need <= NATIVE_WINDOW:
        return None
    return float(-(-need // NATIVE_WINDOW))  # ceil, as a float


def draft_top_p(args) -> float:
    """--theta, else the engine's default sparse_attn_theta."""
    if args.theta is not None:
        return args.theta
    from vllm.config.speculative import SpeculativeConfig

    return SpeculativeConfig.__pydantic_fields__["sparse_attn_theta"].default


def speculative_config(args) -> dict | None:
    if args.mode == "dense":
        return None
    cfg = {
        "method": "sparse_attn",
        "num_speculative_tokens": args.spec_tokens,
        "sparse_attn_algorithm": args.mode,
    }
    if args.mode == "vegas":
        return {**cfg, "sparse_attn_ratio": 0.07, "sparse_attn_min_tokens": 256}
    for key, value in (
        ("ratio", args.ratio),
        ("theta", args.theta),
        ("min_tokens", args.min_tokens),
    ):
        if value is not None:
            cfg[f"sparse_attn_{key}"] = value
    checkpoint = args.draft_weights
    if checkpoint is None:
        if args.model not in ("Qwen/Qwen3-4B", "Qwen/Qwen3-8B"):
            raise ValueError("set --draft-weights for this model, or use 'target'")
        checkpoint = f"RedHatAI/{args.model.split('/')[-1]}-quantized.w4a16"
    if checkpoint != "target":
        cfg["sparse_attn_draft_weights"] = checkpoint
    cfg["sparse_attn_collect_stats"] = True
    return cfg


def build_engine(args, **overrides):
    """``overrides`` are extra ``LLM`` keyword arguments, applied last."""
    from vllm import LLM

    factor = yarn_factor(args.ctx, args.gen)
    kwargs = dict(
        model=args.model,
        dtype="bfloat16",
        max_num_seqs=args.batch,
        max_model_len=args.ctx + args.gen,
        enable_prefix_caching=False,
        async_scheduling=False,
        enable_chunked_prefill=True,
        max_num_batched_tokens=max(2048, args.batch * (args.spec_tokens + 1)),
        gpu_memory_utilization=args.gpu_mem_util,
        seed=args.seed,
        disable_log_stats=False,
        enforce_eager=args.enforce_eager,
        speculative_config=speculative_config(args),
    )
    if args.measurement == "batch":
        # The spawned engine must be able to import this benchmark module.
        root = str(repo_root())
        if root not in sys.path:
            sys.path.insert(0, root)
        kwargs["scheduler_cls"] = "benchmarks.longspec.grid.BatchScheduler"
    if factor is not None:
        kwargs["hf_overrides"] = {
            "rope_parameters": yarn_parameters(args.model, factor)
        }
    kwargs.update(overrides)
    return LLM(**kwargs), factor


def yarn_parameters(model: str, factor: float) -> dict:
    """vLLM derives the context limit from ``rope_parameters``; the legacy
    ``rope_scaling`` key is converted before overrides apply and is ignored."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model)
    theta = getattr(config, "rope_theta", None)
    if theta is None:
        theta = (getattr(config, "rope_parameters", None) or {})["rope_theta"]
    return {
        "rope_type": "yarn",
        "factor": factor,
        "original_max_position_embeddings": NATIVE_WINDOW,
        "rope_theta": float(theta),
    }


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


def generate(llm, prompts, max_tokens: int):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=False)
    inputs = [TokensPrompt(prompt_token_ids=ids) for ids in prompts]
    start = time.perf_counter()
    outputs = llm.generate(inputs, params, use_tqdm=False)
    elapsed = time.perf_counter() - start
    return elapsed, outputs


def generate_batch(llm, prompts, max_tokens, spec_tokens, speculative):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import RequestOutputKind

    if max_tokens < 2:
        raise ValueError("batch measurement needs at least two output tokens")
    params = SamplingParams(
        temperature=0,
        max_tokens=max_tokens,
        ignore_eos=True,
        output_kind=RequestOutputKind.CUMULATIVE,
    )
    ids = [str(next(llm.request_counter)) for _ in prompts]
    window = BatchWindow(ids)
    before = after = None
    finished = {}
    start = time.perf_counter()
    try:
        for request_id, prompt in zip(ids, prompts):
            llm.llm_engine.add_request(
                request_id, TokensPrompt(prompt_token_ids=prompt), params
            )
        while llm.llm_engine.has_unfinished_requests():
            outputs = llm.llm_engine.step()
            started, ended = window.start is not None, window.end is not None
            window.update(outputs)
            if speculative and not started and window.start is not None:
                before = spec_counters(llm, spec_tokens)
            if speculative and not ended and window.end is not None:
                after = spec_counters(llm, spec_tokens)
            for output in outputs:
                if output.finished:
                    finished[output.request_id] = output
    except Exception:
        llm.llm_engine.abort_request(ids)
        raise
    elapsed = time.perf_counter() - start
    timing = window.report()
    outputs = [finished[key] for key in ids]
    if any(len(o.outputs[0].token_ids) != max_tokens for o in outputs):
        raise RuntimeError("a batch request ended before the requested output length")
    counters = _diff(after, before) if speculative else None
    return elapsed, outputs, timing, counters


def _cache_capacity(worker):
    config = worker.model_runner.kv_cache_config
    return {
        "kv_cache_blocks": config.num_blocks,
        "kv_block_size": worker.cache_config.block_size,
        "kv_cache_bytes": sum(t.size for t in config.kv_cache_tensors),
    }


def _round_graph_stats(worker):
    drafter = getattr(worker.model_runner, "drafter", None)
    graphs = getattr(drafter, "round_graphs", None)
    if graphs is None:
        return None
    return {
        "batches": sorted(graphs.graphs),
        "verify": graphs.verify_replays,
        "draft": graphs.draft_replays,
    }


def load_tokens(path):
    payload = json.loads(Path(path).read_text())
    return payload["tokens"] if isinstance(payload, dict) else payload


def spec_counters(llm, spec_tokens: int) -> dict:
    from vllm.v1.metrics.reader import Counter, Vector

    counts = {
        "drafts": 0,
        "draft_tokens": 0,
        "accepted": 0,
        "per_pos": [0] * spec_tokens,
    }
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


def run_cell(args, run_dir: Path) -> list[dict]:
    if args.mode in SPEC_MODES and shutil.which("nvcc") is None:
        raise SystemExit(
            "spec modes need nvcc on PATH: export "
            'PATH="$PWD/.venv/bin:/usr/local/cuda/bin:$PATH" '
            "CUDA_HOME=/usr/local/cuda"
        )
    prompts = build_prompts(args)
    llm, factor = build_engine(args)
    groups = prompt_groups(args, prompts)
    capacity = llm.collective_rpc(_cache_capacity)[0]
    print(json.dumps({"cache_capacity": capacity}), flush=True)
    if args.measurement == "batch":
        generate_batch(
            llm,
            groups[0][0],
            min(args.gen, 16),
            args.spec_tokens,
            args.mode in SPEC_MODES,
        )
    else:
        generate(llm, groups[0][0], min(args.gen, 16))
    records, tokens, token_slots = [], [], []
    for slot, (group, prompt_slots) in enumerate(groups):
        record, group_tokens = measure(llm, args, group, factor, slot)
        record.update(capacity)
        record["prompt_slots"] = prompt_slots
        with (run_dir / "results.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        records.append(record)
        tokens.extend(group_tokens)
        token_slots.extend(prompt_slots)
    payload = (
        tokens
        if args.measurement == "latency"
        else {
            "tokens": tokens,
            "prompt_slots": token_slots,
            "prompt_count": len(prompts),
            "ctx": args.ctx,
            "model": args.model,
            "seed": args.seed,
        }
    )
    (run_dir / f"tokens-{args.mode}-{args.ctx}-{args.batch}.json").write_text(
        json.dumps(payload)
    )
    return records


def measure(llm, args, prompts, factor, slot) -> tuple[dict, list]:
    """Measure the selected decode window and retain complete output tokens."""
    import torch

    spec = args.mode in SPEC_MODES
    cfg = speculative_config(args)
    if cfg is not None:
        effective = llm.llm_engine.vllm_config.speculative_config
        cfg.update(
            {
                key: getattr(effective, key)
                for key in (
                    "sparse_attn_ratio",
                    "sparse_attn_min_tokens",
                    "sparse_attn_theta",
                    "sparse_attn_sink",
                    "sparse_attn_recent",
                    "sparse_attn_draft_weights",
                )
            }
        )
    if args.mode == "coverage":
        llm.collective_rpc(_overrider_reset)
    graph_before = llm.collective_rpc(_round_graph_stats)[0] if spec else None
    if args.measurement == "batch":
        elapsed, outputs, timing, counters = generate_batch(
            llm, prompts, args.gen, args.spec_tokens, spec
        )
    else:
        before = spec_counters(llm, args.spec_tokens) if spec else None
        elapsed, outputs = generate(llm, prompts, args.gen)
        counters = _diff(spec_counters(llm, args.spec_tokens), before) if spec else None
        timing = decode_timing(outputs)
    budget = (
        llm.collective_rpc(_overrider_stats)[0] if args.mode == "coverage" else None
    )
    tokens = [list(o.outputs[0].token_ids) for o in outputs]
    record = {
        "ctx": args.ctx,
        "batch": args.batch,
        "mode": args.mode,
        "model": args.model,
        "gen": args.gen,
        "spec_tokens": args.spec_tokens,
        "speculative_config": cfg,
        "theta": draft_top_p(args) if args.mode == "coverage" else None,
        "enforce_eager": args.enforce_eager,
        "measurement": args.measurement,
        "async_scheduling": False,
        "gpu_memory_utilization": args.gpu_mem_util,
        "ignore_eos": args.measurement == "batch",
        "acceptance_window": "full_batch_decode"
        if args.measurement == "batch"
        else "generation",
        "yarn_factor": factor,
        "seed": args.seed,
        "samples": args.samples,
        "slot": slot,
        "gen_tokens": [len(t) for t in tokens],
        "finish_reasons": [o.outputs[0].finish_reason for o in outputs],
        "elapsed_seconds": elapsed,
        **timing,
        "node": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(),
        "git_sha": git_sha(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if spec:
        drafts, draft_tokens = counters["drafts"], counters["draft_tokens"]
        record.update(
            {
                "drafts": drafts,
                "draft_tokens": draft_tokens,
                "accepted": counters["accepted"],
                "alpha": counters["accepted"] / draft_tokens if draft_tokens else None,
                "tau": 1 + counters["accepted"] / drafts if drafts else None,
                "accept_per_pos": [
                    c / drafts if drafts else None for c in counters["per_pos"]
                ],
            }
        )
    if graph_before is not None:
        graph_after = llm.collective_rpc(_round_graph_stats)[0]
        record["round_graphs"] = {
            "captured_batches": graph_after["batches"],
            "verify_replays": graph_after["verify"] - graph_before["verify"],
            "draft_replays": graph_after["draft"] - graph_before["draft"],
            "window": "whole_generation",
        }
    if budget is not None:
        record["budget"] = budget
        record["budget_window"] = "whole_generation"
    return record, tokens


def decode_timing(outputs) -> dict:
    if len(outputs) != 1:
        raise ValueError("batched decode requires the synchronized measurement path")
    metrics = [o.metrics for o in outputs]
    if not metrics or any(m is None or m.first_token_ts <= 0 for m in metrics):
        raise RuntimeError("engine token timestamps are required for decode timing")
    seconds = max(m.last_token_ts for m in metrics) - min(
        m.first_token_ts for m in metrics
    )
    count = sum(max(0, len(o.outputs[0].token_ids) - 1) for o in outputs)
    return {
        "decode_seconds": seconds,
        "decode_tok_s": count / seconds if count and seconds > 0 else None,
        "decode_tokens": count,
        "prefill_seconds": [m.first_token_ts - m.scheduled_ts for m in metrics],
    }


def compare_tokens(dense, candidate) -> dict:
    if len(dense) != len(candidate):
        raise ValueError("different numbers of prompts")
    rows = []
    for a, b in zip(dense, candidate):
        mismatches = sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        if first is None and len(a) != len(b):
            first = min(len(a), len(b))
        rows.append(
            {
                "dense_tokens": len(a),
                "candidate_tokens": len(b),
                "mismatches": mismatches,
                "first_difference": first,
            }
        )
    return {"identical": all(r["mismatches"] == 0 for r in rows), "per_prompt": rows}


def run_cells(args, run_dir: Path) -> int:
    cells = [c.split(":") for c in args.cells.split(",")]
    for cell in cells:
        if len(cell) != 3 or cell[2] not in ("dense",) + SPEC_MODES:
            raise ValueError(f"invalid cell: {cell}")
        ctx, batch, mode = cell
        child = vars(args) | {
            "ctx": int(ctx),
            "batch": int(batch),
            "mode": mode,
            "out": str(run_dir),
        }
        cmd = [sys.executable, __file__]
        for key, value in child.items():
            if key == "cells" or value is None or value is False:
                continue
            cmd.append("--" + key.replace("_", "-"))
            if value is not True:
                cmd.append(str(value))
        print("+", shlex.join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    comparisons = {}
    for ctx, batch, mode in cells:
        dense = run_dir / f"tokens-dense-{ctx}-{batch}.json"
        if mode != "dense" and dense.exists():
            candidate = run_dir / f"tokens-{mode}-{ctx}-{batch}.json"
            comparisons[f"{ctx}:{batch}:{mode}"] = compare_tokens(
                load_tokens(dense), load_tokens(candidate)
            )
    (run_dir / "output_comparison.json").write_text(json.dumps(comparisons, indent=2))
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    os.chdir(repo_root())
    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.cells:
        (run_dir / "command.txt").write_text(
            shlex.join([sys.executable, *sys.argv]) + "\n"
        )
        return run_cells(args, run_dir)
    # Callables cross into the engine-core process for the overrider stats.
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    run_cell(args, run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
