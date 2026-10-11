"""Full-batch decode: dense, direct Highlight, or Highlight-drafted speculation.

The batch measurement of branch thesis's benchmarks/longspec/grid.py: one
fresh engine per run on one GPU, synchronous scheduling, completed prefills
held until the whole batch has prefilled, and a window from the last first
token to the first finished request; EOS is ignored. Prompts are the token
files grid.py builds (longbench2-<ctx>-<slot>.json), so both branches decode
identical inputs.

  .venv/bin/python -m benchmarks.decode --checkpoint <spot ckpt> \
      --mode highlight --prompts <dir> --batch 1 --samples 2 --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from vllm.v1.core.sched.scheduler import Scheduler

from highlight_spec import SPECULATOR

ROOT = Path(__file__).resolve().parents[1]
NATIVE_WINDOW = 32768  # Qwen3's trained context; the config's 40,960 is not the YaRN base
MAX_WINDOW = 131072


def add_engine_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--checkpoint", required=True, help="Spot (Highlight) checkpoint")
    p.add_argument("--mode", choices=("dense", "direct", "highlight"), required=True,
                   help="dense target, direct sparse generation, or sparse-drafted speculation")
    p.add_argument("--spec-tokens", type=int, default=6)
    p.add_argument("--topk-fraction", type=float, default=0.07)
    p.add_argument("--topp", type=float, help="draft by top-p block mass instead of top-K")
    p.add_argument("--recent", type=int, default=64, help="trailing tokens always kept")
    p.add_argument("--sink", type=int, default=16, help="top-p: leading tokens always kept")
    p.add_argument("--topp-min-fraction", type=float, default=0.01,
                   help="top-p: floor on kept tokens, as a fraction of --ctx")
    p.add_argument("--topp-max-fraction", type=float,
                   help="top-p: cap on kept tokens, as a fraction of --ctx (best blocks kept)")
    p.add_argument("--draft-select", choices=("step", "round"), default="step",
                   help="score every draft step, or once per round")
    p.add_argument("--verify-fraction", type=float, default=0.0,
                   help="verification keeps this fraction of the prompt; 0 verifies densely (lossless)")
    p.add_argument("--flash-attn-version", type=int, choices=(2, 3, 4), default=4)
    p.add_argument("--indexer-cache-format", choices=("bf16", "mxfp4"), default="bf16",
                   help="Highlight side-cache format; model weights and backbone KV stay BF16")
    p.add_argument("--gpu-mem-util", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_engine_args(p)
    p.add_argument("--prompts", required=True, help="directory of grid.py prompt files")
    p.add_argument("--ctx", type=int, default=32640)
    p.add_argument("--gen", type=int, default=128)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--samples", type=int, default=1,
                   help="prompt pool N > 1: N batches of --batch consecutive prompts, wrapping around")
    p.add_argument("--reference", help="tokens file of a dense run on the same prompts")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if args.gen < 2:
        p.error("batch measurement needs at least two output tokens")
    if 1 < args.samples < args.batch:
        p.error("the prompt pool must hold at least --batch prompts")
    return args


def blocks_needed(prompt_lens, max_tokens: int, lookahead: int, block_size: int) -> int:
    """KV blocks a batch holds at its longest: prompt, every output and the draft lookahead."""
    return sum(-(-(n + max_tokens + lookahead) // block_size) for n in prompt_lens)


class BatchScheduler(Scheduler):
    """Benchmark-only scheduler: retain completed prefills until all are ready."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.scheduler_config.async_scheduling:
            raise ValueError("batch measurement requires synchronous scheduling")
        if len(self.kv_cache_config.kv_cache_groups) != 1:
            raise ValueError("batch measurement supports one KV cache group")
        if self.cache_config.enable_prefix_caching:
            raise ValueError("batch measurement requires prefix caching disabled")
        if self.num_spec_tokens and self.num_lookahead_tokens != self.num_spec_tokens:
            raise ValueError("draft lookahead slots are not reserved")
        self._released = False

    def add_request(self, request):
        if not self.requests:
            self._released = False
        if self._released or len(self.requests) >= self.max_num_running_reqs:
            raise ValueError("submit exactly one batch, then drain it before the next")
        super().add_request(request)
        # Reserve for the complete requested generation, not just the prefill. The client
        # checks first (generate_batch); a failure here kills the engine core.
        required = sum(
            blocks_needed([r.num_prompt_tokens], r.max_tokens, self.num_lookahead_tokens, self.block_size)
            for r in self.requests.values()
        )
        available = self.kv_cache_manager.block_pool.num_gpu_blocks - 1
        if required > available:
            raise ValueError(
                f"batch does not fit: needs {required} KV blocks, has {available} "
                "after reserving the null block; reduce --batch"
            )

    def schedule(self, *args, **kwargs):
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
            output = super().schedule(*args, **kwargs)
        finally:
            self.running = held + self.running
        if output.preempted_req_ids:
            raise RuntimeError("invalid batch measurement: KV preemption occurred")
        if self._released:
            if self.waiting or set(output.num_scheduled_tokens) != expected:
                raise RuntimeError("invalid batch measurement: not every active request ran")
        elif len(self.requests) == self.max_num_running_reqs and output.total_num_scheduled_tokens == 0:
            raise RuntimeError("prefill barrier cannot make progress; reduce --batch")
        return output


class BatchWindow:
    """Count emitted tokens from the prefill barrier through the first finish."""

    def __init__(self, request_ids):
        self.request_ids = set(request_ids)
        self.counts, self.first = {}, {}
        self.start = self.end = None
        self.tokens = 0

    def update(self, outputs):
        if not outputs or self.end is not None:
            return
        if self.start is not None and {o.request_id for o in outputs} != self.request_ids:
            raise RuntimeError("decode step did not contain the whole batch")
        stamps = []
        for o in outputs:
            if o.request_id not in self.request_ids or o.metrics is None or o.metrics.first_token_ts <= 0:
                raise RuntimeError("missing request or engine timestamps")
            count = len(o.outputs[0].token_ids)
            if self.start is None and (count != 1 or o.finished):
                raise RuntimeError("generation advanced before the prefill barrier")
            if count < self.counts.get(o.request_id, 0):
                raise RuntimeError("batch output must contain cumulative token IDs")
            self.counts[o.request_id] = count
            self.first[o.request_id] = o.metrics.first_token_ts
            stamps.append(o.metrics.last_token_ts)
        if self.start is None:
            if self.counts.keys() == self.request_ids:
                self.start = max(self.first.values())
            return
        if len(set(stamps)) != 1 or stamps[0] <= self.start:
            raise RuntimeError("batch outputs do not share a valid engine timestamp")
        if any(o.finished for o in outputs):
            self.end = stamps[0]
            self.tokens = sum(self.counts.values()) - len(self.request_ids)

    def report(self):
        if self.end is None:
            raise RuntimeError("no complete full-batch decode window")
        seconds = self.end - self.start
        return {
            "decode_seconds": seconds,
            "decode_tok_s": self.tokens / seconds,
            "decode_tokens": self.tokens,
            "decode_window": "all_prefills_complete_to_first_request_finish",
        }


def load_prompts(args) -> list[list[int]]:
    count = args.samples if args.samples > 1 else args.batch
    prompts = []
    for slot in range(count):
        paths = sorted(Path(args.prompts).rglob(f"longbench2-{args.ctx}-{slot}.json"))
        if len(paths) != 1:
            raise ValueError(f"need exactly one prompt file for slot {slot}, found {len(paths)}")
        ids = json.loads(paths[0].read_text())
        if len(ids) != args.ctx:
            raise ValueError(f"{paths[0]} holds {len(ids)} tokens, not {args.ctx}")
        prompts.append(ids)
    return prompts


def prompt_groups(args, prompts):
    """Branch thesis's rule: every prompt appears equally often at every batch size."""
    if args.samples == 1:
        return [(prompts, list(range(len(prompts))))]
    n = len(prompts)
    return [
        ([prompts[(start + j) % n] for j in range(args.batch)], [(start + j) % n for j in range(args.batch)])
        for start in range(n)
    ]


def summarize_run(records: list[dict]) -> dict:
    """One repetition: emitted tokens over timed seconds across its batches, acceptance pooled."""
    tokens = sum(r["decode_tokens"] for r in records)
    seconds = sum(r["decode_seconds"] for r in records)
    summary = {"batches": len(records), "decode_tokens": tokens, "decode_seconds": seconds,
               "decode_tok_s": tokens / seconds, "kv_cache_blocks": records[0]["kv_cache_blocks"]}
    if "drafts" in records[0]:
        spec = {k: sum(r[k] for r in records) for k in ("drafts", "draft_tokens", "accepted")}
        spec["alpha"] = spec["accepted"] / spec["draft_tokens"]
        spec["tau"] = 1 + spec["accepted"] / spec["drafts"]
        summary.update(spec)
    return summary


def yarn_factor(total: int) -> float | None:
    """Static YaRN: the smallest integer multiple of the native window covering prompt + output."""
    if total > MAX_WINDOW:
        raise ValueError(f"prompt + output {total} exceeds the validated {MAX_WINDOW}-token window")
    return None if total <= NATIVE_WINDOW else float(-(-total // NATIVE_WINDOW))


def rope_parameters(checkpoint: str, factor: float) -> dict:
    """YaRN over the native window; Spot's indexer mirrors the backbone's RoPE."""
    config = json.loads((Path(checkpoint) / "config.json").read_text())
    theta = config.get("rope_theta") or config["rope_parameters"]["rope_theta"]
    return {
        "rope_type": "yarn",
        "factor": factor,
        "original_max_position_embeddings": NATIVE_WINDOW,
        "rope_theta": float(theta),
    }


def sampling_params(max_tokens: int, *, ignore_eos: bool):
    """Greedy, every knob explicit: generation_config="vllm" inherits nothing from the checkpoint."""
    from vllm import SamplingParams

    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=0,  # disabled
        min_p=0.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        max_tokens=max_tokens,
        min_tokens=0,
        ignore_eos=ignore_eos,
        skip_special_tokens=False,
    )


def selection_overrides(args) -> dict:
    """Spot's draft selection knobs: top-K fraction of the prompt, or top-p block mass."""
    if args.mode == "direct" and (args.draft_select != "step" or args.verify_fraction != 0):
        raise ValueError("direct Highlight requires per-step selection and no verifier")
    if args.topp is None:
        selection = {
            "molle_selection_mode": "topk_fraction",
            "molle_topk_fraction": args.topk_fraction,
            "molle_topk_recency": args.recent,
        }
    else:
        selection = {
            "molle_selection_mode": "topp",
            "molle_topp": args.topp,
            "molle_recency_window": args.recent,
            "molle_sink_boundary": args.sink,
            "molle_topp_kmin": math.ceil(round(args.topp_min_fraction * args.ctx, 6)),
        }
        if args.topp_max_fraction is not None:
            selection["molle_topp_kmax"] = math.floor(round(args.topp_max_fraction * args.ctx, 6))
    if args.indexer_cache_format != "bf16":
        selection["molle_kcache_format"] = args.indexer_cache_format
    return {**selection, "molle_draft_select": args.draft_select, "molle_verify_fraction": args.verify_fraction}


def build_engine(args, scheduler_cls: str | None = "benchmarks.decode.BatchScheduler"):
    """max_model_len is args.ctx + args.gen; above the native window RoPE becomes static YaRN."""
    from vllm import LLM
    from spotattention.serve import stats

    if args.mode == "dense" and args.indexer_cache_format != "bf16":
        raise ValueError("dense has no Highlight indexer to quantize")

    factor = yarn_factor(args.ctx + args.gen)
    if args.mode == "highlight":
        hf_overrides = {**selection_overrides(args), "molle_spec_decode": True}
        speculative_config = {
            "method": "custom_class",
            "model": SPECULATOR,
            "num_speculative_tokens": args.spec_tokens,
        }
        sizes = {1, args.batch, args.batch * (args.spec_tokens + 1)}
    elif args.mode == "direct":
        hf_overrides = {**selection_overrides(args), "use_dsa": True, "molle_spec_decode": False}
        speculative_config = None
        sizes = {1, args.batch}
    else:
        hf_overrides, speculative_config = {"use_dsa": False}, None
        sizes = {1, args.batch}
    if factor is not None:
        hf_overrides["rope_parameters"] = rope_parameters(args.checkpoint, factor)
    extra = {} if scheduler_cls is None else {"scheduler_cls": scheduler_cls}
    return LLM(
        model=args.checkpoint,
        generation_config="vllm",
        dtype="bfloat16",
        kv_cache_dtype="auto",
        block_size=16,
        max_num_seqs=args.batch,
        max_model_len=args.ctx + args.gen,
        enable_prefix_caching=False,
        async_scheduling=False,
        enable_chunked_prefill=True,
        max_num_batched_tokens=max(2048, args.batch * (args.spec_tokens + 1)),
        gpu_memory_utilization=args.gpu_mem_util,
        seed=args.seed,
        disable_log_stats=False,
        hf_overrides=hf_overrides,
        speculative_config=speculative_config,
        compilation_config={"cudagraph_capture_sizes": sorted(sizes)},
        attention_config={"backend": "FLASH_ATTN", "flash_attn_version": args.flash_attn_version},
        worker_extension_cls=stats.WORKER_EXTENSION,
        **extra,
    )


def spec_counters(llm, spec_tokens: int) -> dict:
    from vllm.v1.metrics.reader import Counter, Vector

    counts = {"drafts": 0, "draft_tokens": 0, "accepted": 0, "per_pos": [0] * spec_tokens}
    names = {
        "vllm:spec_decode_num_drafts": "drafts",
        "vllm:spec_decode_num_draft_tokens": "draft_tokens",
        "vllm:spec_decode_num_accepted_tokens": "accepted",
    }
    for metric in llm.get_metrics():
        if metric.name in names:
            assert isinstance(metric, Counter)
            counts[names[metric.name]] += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos, value in enumerate(metric.values[:spec_tokens]):
                counts["per_pos"][pos] += value
    return counts


def selection_snapshot(llm) -> dict:
    from spotattention.serve import stats

    return llm.collective_rpc(stats.RPC_METHOD, kwargs={"reset": False})[0]


def generate_batch(llm, args, prompts, max_tokens):
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import RequestOutputKind

    engine = llm.llm_engine.vllm_config
    lookahead = args.spec_tokens if args.mode == "highlight" else 0
    required = blocks_needed([len(p) for p in prompts], max_tokens, lookahead, engine.cache_config.block_size)
    available = engine.cache_config.num_gpu_blocks - 1  # the null block
    if required > available:
        raise ValueError(f"batch does not fit: needs {required} KV blocks, has {available}; reduce --batch")
    params = sampling_params(max_tokens, ignore_eos=True)
    params.output_kind = RequestOutputKind.CUMULATIVE
    ids = [str(next(llm.request_counter)) for _ in prompts]
    window = BatchWindow(ids)
    # Selection counters need an engine RPC, which must not fall inside the
    # timed window: snapshot after every prefill step, keep the last one.
    track = args.mode != "dense"
    selection = selection_snapshot(llm) if track else None
    before = after = None
    finished = {}
    try:
        for request_id, prompt in zip(ids, prompts):
            llm.llm_engine.add_request(request_id, TokensPrompt(prompt_token_ids=prompt), params)
        while llm.llm_engine.has_unfinished_requests():
            outputs = llm.llm_engine.step()
            started, ended = window.start is not None, window.end is not None
            window.update(outputs)
            if not started and window.start is not None:
                before = {"spec": spec_counters(llm, args.spec_tokens), "selection": selection}
            elif window.start is None and track:
                selection = selection_snapshot(llm)
            if not ended and window.end is not None:
                after = {
                    "spec": spec_counters(llm, args.spec_tokens),
                    "selection": selection_snapshot(llm) if track else None,
                }
            for o in outputs:
                if o.finished:
                    finished[o.request_id] = o
    except Exception:
        llm.llm_engine.abort_request(ids)
        raise
    outputs = [finished[key] for key in ids]
    if any(len(o.outputs[0].token_ids) != max_tokens for o in outputs):
        raise RuntimeError("a batch request ended before the requested output length")
    return outputs, window.report(), before, after


def spec_record(before: dict, after: dict) -> dict:
    d = {k: after[k] - before[k] for k in ("drafts", "draft_tokens", "accepted")}
    per_pos = [a - b for a, b in zip(after["per_pos"], before["per_pos"])]
    return {
        **d,
        "alpha": d["accepted"] / d["draft_tokens"] if d["draft_tokens"] else None,
        "tau": 1 + d["accepted"] / d["drafts"] if d["drafts"] else None,
        "accept_per_pos": [c / d["drafts"] if d["drafts"] else None for c in per_pos],
    }


def selection_record(before: dict, after: dict) -> dict | None:
    from spotattention.serve import stats

    diff = {**after, **{f: [a - b for a, b in zip(after[f], before[f])] for f in stats.FIELDS}}
    summary = stats.summarize(diff)
    if summary is None:
        return None
    fractions = summary["per_layer_fraction"].values()
    return {
        "retained_fraction": summary["retained_fraction"],
        "layer_fraction_range": [min(fractions), max(fractions)],
        "draft_rows_per_layer": summary["decode_rows"],
    }


def compare_tokens(reference: list[list[int]], candidate: list[list[int]]) -> list[dict]:
    rows = []
    for a, b in zip(reference, candidate, strict=True):
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        rows.append({"first_mismatch": first, "mismatches": sum(x != y for x, y in zip(a, b))})
    return rows


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_sha256() -> str:
    """Digest of this package's source, so results identify code that is not a git checkout."""
    h = hashlib.sha256()
    for path in sorted([*ROOT.glob("highlight_spec/*.py"), *ROOT.glob("benchmarks/*.py")]):
        h.update(path.relative_to(ROOT).as_posix().encode() + b"\0" + path.read_bytes())
    return h.hexdigest()


def git_sha(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv=None) -> None:
    import torch
    import spotattention

    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    token_path = out / f"tokens-{args.mode}-{args.ctx}-{args.batch}.json"
    if token_path.exists():
        raise ValueError("run already has output; use a fresh directory")
    # The spawned engine imports this module for the scheduler.
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))
    prompts = load_prompts(args)
    groups = prompt_groups(args, prompts)
    llm = build_engine(args)
    engine = llm.llm_engine.vllm_config
    generate_batch(llm, args, groups[0][0], min(args.gen, 16))  # warm-up
    reference = None
    if args.reference:
        reference = json.loads(Path(args.reference).read_text())
    records, tokens = [], []
    for slot, (group, slots) in enumerate(groups):
        outputs, timing, before, after = generate_batch(llm, args, group, args.gen)
        group_tokens = [list(o.outputs[0].token_ids) for o in outputs]
        record = {
            "mode": args.mode,
            "checkpoint": args.checkpoint,
            "ctx": args.ctx,
            "gen": args.gen,
            "batch": args.batch,
            "slot": slot,
            "prompt_slots": slots,
            "prompt_sha256": [digest(p) for p in group],
            "spec_tokens": args.spec_tokens if args.mode == "highlight" else 0,
            "selection": selection_overrides(args) if args.mode != "dense" else None,
            "indexer_cache_format": getattr(engine.model_config.hf_config, "molle_kcache_format", "bf16"),
            "measurement": "batch",
            "flash_attn_version": engine.attention_config.flash_attn_version,
            "gpu_memory_utilization": args.gpu_mem_util,
            "kv_cache_blocks": engine.cache_config.num_gpu_blocks,
            "max_model_len": args.ctx + args.gen,
            "yarn_factor": yarn_factor(args.ctx + args.gen),
            "thinking": False,
            "temperature": 0.0,
            "ignore_eos": True,
            "seed": args.seed,
            **timing,
            "versions": {n: importlib.metadata.version(n) for n in ("torch", "vllm")},
            "spotattention_commit": git_sha(Path(spotattention.__file__).parent),
            "git_sha": git_sha(ROOT),
            "source_sha256": source_sha256(),
            "node": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if args.mode == "highlight":
            record.update(spec_record(before["spec"], after["spec"]))
            record["draft_kv"] = selection_record(before["selection"], after["selection"])
        elif args.mode == "direct":
            record["decode_kv"] = selection_record(before["selection"], after["selection"])
        if reference is not None:
            ref = [reference["tokens"][reference["prompt_slots"].index(s)] for s in slots]
            record["match"] = compare_tokens(ref, group_tokens)
        with (out / "results.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        records.append(record)
        tokens.extend(group_tokens)
    token_path.write_text(json.dumps({
        "tokens": tokens,
        "prompt_slots": [s for _, slots in groups for s in slots],
        "prompt_sha256": [digest(p) for p in prompts],
        "mode": args.mode,
        "ctx": args.ctx,
        "gen": args.gen,
    }))
    summary = {"mode": args.mode, "ctx": args.ctx, "gen": args.gen, "batch": args.batch,
               "selection": records[0]["selection"], **summarize_run(records)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    sys.exit(main())
