"""Attention-only B200 diagnostic for the hierarchical-cache implementation.

Synthetic tensors at the actual Qwen3-8B 32K shape. This excludes model layers,
selection, prefill and cache updates and must never be labeled serving throughput.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics

from benchmarks.hierarchical_cache import save


def run(out, profile_kernel=False):
    import torch
    from highlight_spec.hierarchical_cache import HierarchicalKV
    from highlight_spec.hierarchical_kernels import attention, AttentionWorkspace
    from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd

    torch.manual_seed(42)
    length, heads, query_heads = 32768, 8, 32
    k, v = [torch.randn((length, heads, 128), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    cache = HierarchicalKV(length+128, heads, 128, device="cuda", encoder="triton")
    cache.prefill(k, v)
    if profile_kernel:
        q = torch.randn((7, query_heads, 128), device="cuda", dtype=torch.bfloat16)
        qp = torch.arange(length-7, length, device="cuda")
        positions = torch.arange(length, device="cuda")
        ws = AttentionWorkspace(heads, query_heads, 7, length, "cuda")
        attention(cache, q, qp, positions, 8, ws)
        torch.cuda.synchronize()
        return

    def measure(fn):
        for _ in range(3): fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): fn()
        for _ in range(20): graph.replay()
        rates = []
        for _ in range(5):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(200): graph.replay()
            end.record()
            end.synchronize()
            rates.append(start.elapsed_time(end)/200)
        return dict(median_ms=statistics.median(rates), five_graph_replay_means_ms=rates)

    rows = []
    for nq, sparse in ((1, True), (7, False)):
        # 144 selected blocks: 140 early blocks plus the four most recent.
        positions = (torch.cat((torch.arange(2240, device="cuda"),
                                torch.arange(length-64, length, device="cuda"))) if sparse else
                     torch.arange(length, device="cuda"))
        q = torch.randn((nq, query_heads, 128), device="cuda", dtype=torch.bfloat16)
        qp = torch.arange(length-nq, length, device="cuda")
        ws = AttentionWorkspace(heads, query_heads, nq, len(positions), "cuda")
        native_k, native_v = (a.index_select(0, positions)[None] for a in (k, v))
        # Gather is outside timing, just as selected block addressing avoids
        # materializing gathers in the serving backend. This is a kernel bound.
        native_unsplit = measure(lambda: _flash_attn_fwd(q[None], native_k, native_v,
                                 causal=True, softmax_scale=128**-.5, num_splits=1)[0])
        # Give the native implementation its occupancy heuristic as well. A
        # custom split-KV kernel versus forced unsplit FA4 is not a fair speedup.
        native = measure(lambda: _flash_attn_fwd(q[None], native_k, native_v,
                         causal=True, softmax_scale=128**-.5, num_splits=0)[0])
        row = dict(query_tokens=nq, positions=len(positions), sparse=sparse,
                   native_bf16_fa4=native, native_bf16_fa4_unsplit=native_unsplit)
        for bits in ((8, 4) if sparse else (8,)):
            result = measure(lambda: attention(cache, q, qp, positions, bits, ws))
            result["compiled"] = {name:dict(registers=getattr(kernel, "n_regs", None),
                spills=getattr(kernel, "n_spills", None), shared_bytes=kernel.metadata.shared,
                warps=kernel.metadata.num_warps)
                for name, kernel in (("attention", ws.compiled_attention), ("merge", ws.compiled_merge))}
            result["splits"] = ws.splits
            result["relative_to_bf16_fa4"] = native["median_ms"]/result["median_ms"]
            row[f"hierarchical_{bits}"] = result
        rows.append(row)
        print(json.dumps(row), flush=True)
        save(out/"kernel-timing.json", dict(status="running", rows=rows, serving_throughput=False))
    save(out/"kernel-timing.json", dict(status="complete", rows=rows,
        scope="attention only, synthetic 32K Qwen3-8B tensors, one CUDA graph per attention call",
        native_baseline="FA4 automatic num_splits=0; unsplit control retained; selected KV gathered before timing",
        measurements="five means of 200 replays in one process; no process repeatability claim",
        serving_throughput=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--profile-kernel", action="store_true", help="one dense verifier call for Nsight Compute")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    runtime_root = args.runtime_root or root
    validation = json.loads((args.validation/"validation.json").read_text())
    hashes = json.loads((args.validation/"status.json").read_text())["source_sha256"]
    if validation["status"] != "passed": raise ValueError("kernel validation failed")
    for name in ("highlight_spec/hierarchical_cache.py", "highlight_spec/hierarchical_kernels.py"):
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != hashes[name]:
            raise ValueError("kernel changed since validation")
    args.out.mkdir(parents=True, exist_ok=False)
    save(args.out/"plan.json", dict(source_sha256=hashes, gpu=args.gpu, serving_throughput=False,
                                   timing_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    from benchmarks.ffn_agreement import gpu_lock
    lock = gpu_lock(runtime_root, args.gpu)
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        for name, sub in (("TMPDIR", "tmp"), ("TRITON_CACHE_DIR", "triton")):
            path = runtime_root/"outputs"/"hierarchical-cache"/sub
            path.mkdir(parents=True, exist_ok=True)
            os.environ[name] = str(path)
        run(args.out, args.profile_kernel)
    finally:
        lock.close()


if __name__ == "__main__":
    main()
