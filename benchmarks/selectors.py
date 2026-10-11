"""Time selector stages on a saved, real-model layer capture.

Capture (torch.save dict, weights_only-readable): q [B,Q,Hq,D], k/v
[B,T,Hkv,D], prefix_tokens, phase ('prefill' or 'verify'), model, layer,
prompt_sha256, and retrieval_q [B,1,Hq,D]. retrieval_q must be the chosen
TriForce prefill/rebuild query. Its choice and refresh schedule are never
inferred from Vegas's verification queries. Q/K are the actual model
attention inputs after QK normalization and rotary position encoding.

This is a per-selection-event microbenchmark, not decode throughput or
amortized per-token cost. No random/synthetic capture is generated here.
The campaign supplies captured hidden states and indexer weights as well,
so Highlight uses its unchanged production kernels on the same prefix.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from importlib.metadata import version
import os
from pathlib import Path
import statistics

import torch

from highlight_spec.selectors import (
    SelectorBudget, TriForceRetrievalCache, _check_qk, gather_draft_kv, triforce_chunk_keys,
    triforce_scores, triforce_select, vegas_reference_scores, vegas_select,
)
from highlight_spec.vegas_fa4 import FusedVegasAttention


def load_capture(path: Path):
    capture = torch.load(path, map_location="cpu", weights_only=True)
    required = {"q", "k", "v", "retrieval_q", "prefix_tokens", "phase",
                "model", "layer", "prompt_sha256"}
    if not isinstance(capture, dict) or required - capture.keys():
        raise ValueError(f"capture requires {sorted(required)}")
    q, k, v = (capture[name] for name in ("q", "k", "v"))
    _check_qk(q, k)
    _check_qk(capture["retrieval_q"], k)
    if capture["retrieval_q"].shape != (q.shape[0], 1, q.shape[2], q.shape[3]):
        raise ValueError("retrieval_q must explicitly supply one query per head/request")
    if k.shape != v.shape or k.dtype != v.dtype:
        raise ValueError("K/V mismatch")
    if capture["phase"] not in {"prefill", "verify"}:
        raise ValueError("unknown capture phase")
    prefix = capture["prefix_tokens"]
    limit = k.shape[1] if capture["phase"] == "prefill" else k.shape[1] - q.shape[1] + 1
    if not isinstance(prefix, int) or not 0 < prefix <= limit:
        raise ValueError("prefix outside causal visibility")
    hashes = capture["prompt_sha256"]
    if (not isinstance(hashes, list) or len(hashes) != q.shape[0]
            or any(not isinstance(h, str) or len(h) != 64
                   or any(c not in "0123456789abcdef" for c in h) for h in hashes)):
        raise ValueError("one SHA256 prompt hash is required per request")
    if not isinstance(capture["model"], str) or not capture["model"] or not isinstance(capture["layer"], int):
        raise ValueError("model and layer provenance are required")
    if any(not torch.isfinite(capture[name]).all() for name in ("q", "k", "v", "retrieval_q")):
        raise ValueError("nonfinite capture")
    return capture


def time_stages(stages, *, warmup: int, repetitions: int, execution="eager", iterations=20):
    """CUDA event samples; rotate order to reduce systematic order effects."""
    if warmup < 1 or repetitions < 2:
        raise ValueError("warmup >=1 and repetitions >=2 required")
    for fn in stages.values():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    functions, graphs, divisors = {}, {}, {}
    for name, fn in stages.items():
        count = 1 if "initialize" in name else iterations
        if execution == "graph":
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(count):
                    value = fn()
            graphs[name] = graph
            functions[name], divisors[name] = graph.replay, count
        else:
            functions[name], divisors[name] = fn, 1
    samples = {name: [] for name in stages}
    names = list(stages)
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    for rep in range(repetitions):
        order = names[rep % len(names):] + names[:rep % len(names)]
        for name in order:
            start.record()
            value = functions[name]()
            stop.record()
            stop.synchronize()
            samples[name].append(start.elapsed_time(stop) / divisors[name])
            del value
    return {name: {"median_ms": statistics.median(times), "samples_ms": times}
            for name, times in samples.items()}


@torch.inference_mode()
def measure(capture, budget, *, warmup, repetitions, execution="eager"):
    from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd
    q, k, v, retrieval_q = (capture[name].to("cuda").contiguous()
                             for name in ("q", "k", "v", "retrieval_q"))
    prefix = budget.prefix_tokens
    prefix_k, prefix_v = k[:, :prefix], v[:, :prefix]
    collector = FusedVegasAttention(q, k, v, prefix_tokens=prefix, phase=capture["phase"])
    collector.scores.fill_(float("nan"))
    output = collector.attention().clone()
    dense_out = torch.empty_like(q)

    def dense():
        return _flash_attn_fwd(q, k, v, causal=True, num_splits=1, out=dense_out)

    dense()
    torch.testing.assert_close(output, dense_out, atol=0.02, rtol=0.02)
    oracle = vegas_reference_scores(q, k, prefix_tokens=prefix, phase=capture["phase"])
    torch.testing.assert_close(collector.reduce(), oracle, atol=0.03, rtol=0.01)
    scores = collector.reduce()
    vegas_ids = vegas_select(scores, tokens=budget.vegas_tokens)
    chunks = triforce_chunk_keys(prefix_k, chunk_size=budget.triforce_chunk)
    tri_scores = triforce_scores(retrieval_q, chunks)
    tri_ids = triforce_select(tri_scores, tokens=budget.triforce_tokens, chunk_size=budget.triforce_chunk)
    from highlight_spec.selector_timing import HighlightTiming, validate_triforce
    validation = {"vegas": {"passed": True},
        "triforce": validate_triforce(retrieval_q, prefix_k, prefix_v, chunks, tri_scores,
                                     tri_ids, budget.triforce_chunk)}
    stages = {
        "dense_attention": dense,
        "vegas_attention_with_fused_collection": collector.attention,
        "vegas_reduce": collector.reduce,
        "vegas_rank": lambda: vegas_select(scores, tokens=budget.vegas_tokens),
        "vegas_gather": lambda: gather_draft_kv(prefix_k, prefix_v, vegas_ids, query_heads=q.shape[2]),
        "triforce_initialize_chunk_keys": lambda: triforce_chunk_keys(prefix_k, chunk_size=budget.triforce_chunk),
        "triforce_score": lambda: triforce_scores(retrieval_q, chunks),
        "triforce_rank": lambda: triforce_select(tri_scores, tokens=budget.triforce_tokens, chunk_size=budget.triforce_chunk),
        "triforce_gather": lambda: gather_draft_kv(prefix_k, prefix_v, tri_ids, query_heads=q.shape[2]),
    }
    highlight = None
    if "hidden_all" in capture:
        highlight = HighlightTiming(capture, budget.highlight_tokens)
        validation["highlight"] = highlight.validate()
        stages.update(highlight_score=highlight.score, highlight_rank=highlight.rank,
            highlight_selection=highlight.run, highlight_initialize=highlight.initialize)
        # Native Highlight consumes its compacted page table directly. It does
        # not materialize selected KV; a fabricated gather is not charged.
    try:
        timings = time_stages(stages, warmup=warmup, repetitions=repetitions, execution=execution)
    finally:
        if highlight is not None:
            highlight.close()
    # Preserve individual paired observations, including negative noise.
    extra = [a - b for a, b in zip(timings["vegas_attention_with_fused_collection"]["samples_ms"],
                                  timings["dense_attention"]["samples_ms"])]
    timings["vegas_collection_delta"] = {"median_ms": statistics.median(extra), "samples_ms": extra}
    bytes_per_head_token = q.shape[0] * q.shape[-1] * k.element_size() * 2
    return {
        "budget": asdict(budget), "timings": timings, "execution": execution,
        "graph_iterations": 20 if execution == "graph" else 1,
        "workspace_bytes": {
            "vegas_collection": collector.workspace_bytes,
            "triforce_chunk_keys": chunks.numel() * chunks.element_size(),
            "vegas_gathered_kv": bytes_per_head_token * k.shape[2] * budget.vegas_tokens,
            "triforce_gathered_kv": bytes_per_head_token * q.shape[2] * budget.triforce_tokens,
            "highlight_indexer_cache": highlight.cache_bytes if highlight else None,
            "highlight_gathered_kv": 0 if highlight else None,
        },
        "validation": validation,
        "refresh": {"highlight": "once per draft round", "vegas": "after verification",
                    "triforce": "once at prefill; no repeated selection"},
        "highlight_gather": "not required; native paged attention reads compacted block table",
    }


@torch.inference_mode()
def measure_highlight(capture, budget, *, cache_format, warmup, repetitions):
    """Only the changed Highlight variant; do not rerun unchanged baselines."""
    from highlight_spec.selector_timing import HighlightTiming
    selector = HighlightTiming(capture, budget.highlight_tokens, cache_format=cache_format)
    try:
        validation = selector.validate()
        timings = time_stages(dict(highlight_score=selector.score, highlight_rank=selector.rank,
            highlight_selection=selector.run, highlight_initialize=selector.initialize),
            warmup=warmup, repetitions=repetitions, execution="graph")
        return dict(budget=asdict(budget), indexer_cache_format=cache_format, timings=timings,
            execution="graph", graph_iterations=20, validation={"highlight": validation},
            workspace_bytes={"highlight_indexer_cache": selector.cache_bytes, "highlight_gathered_kv": 0})
    finally:
        selector.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--protocol", choices=("matched", "paper", "both"), required=True)
    parser.add_argument("--matched-tokens", type=int)
    parser.add_argument("--highlight-tokens", type=int, required=True)
    parser.add_argument("--highlight-block", type=int, required=True)
    parser.add_argument("--triforce-chunk", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--repetitions", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"2", "3"}:
        parser.error("requires exactly one idle authorized physical GPU: CUDA_VISIBLE_DEVICES=2 or 3")
    if args.out.exists():
        parser.error("output already exists")
    capture = load_capture(args.capture)
    prefix = capture["prefix_tokens"]
    budgets = []
    if args.protocol in {"matched", "both"}:
        if args.matched_tokens is None or args.highlight_tokens != args.matched_tokens:
            parser.error("matched protocol requires matching --matched-tokens and --highlight-tokens")
        budgets.append(SelectorBudget.matched(prefix, args.matched_tokens,
                       highlight_block=args.highlight_block, triforce_chunk=args.triforce_chunk))
    if args.protocol in {"paper", "both"}:
        budgets.append(SelectorBudget.paper(prefix, highlight_tokens=args.highlight_tokens))
    results = [measure(capture, b, warmup=args.warmup, repetitions=args.repetitions) for b in budgets]
    payload = {
        "scope": "per-selection-event baseline microbenchmark; no amortization or quality claim",
        "implementation": "FA4 fused Vegas adaptation; PyTorch TriForce retrieval reference",
        "highlight": "not measured here; learned-indexer acquisition requires the model harness",
        "triforce_refresh_policy": TriForceRetrievalCache.refresh_policy,
        "capture_sha256": hashlib.sha256(args.capture.read_bytes()).hexdigest(),
        "provenance": {key: capture[key] for key in ("model", "layer", "phase", "prompt_sha256")},
        "torch": torch.__version__, "vllm": version("vllm"),
        "gpu": torch.cuda.get_device_name(),
        "source_sha256": {str(path.relative_to(Path(__file__).resolve().parents[1])):
                           hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in (Path(__file__).resolve(),
                           Path(__file__).resolve().parents[1] / "highlight_spec/selectors.py",
                           Path(__file__).resolve().parents[1] / "highlight_spec/vegas_fa4.py")},
        "warmup": args.warmup, "repetitions": args.repetitions, "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against concurrent benchmark writers.
    with args.out.open("x") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
