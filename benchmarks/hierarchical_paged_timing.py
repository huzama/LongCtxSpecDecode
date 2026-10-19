"""Reproducible attention-only checks of the actual paged serving kernel.

Run inside a Slurm GPU allocation on the local cluster. Native FA2 timings
exclude gathering selected KV and use original BF16 tensors; they are a kernel
reference, not an accuracy comparison or an end-to-end throughput measurement.
The optional baseline is an immutable copy of the previous kernel module.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import statistics
import sys

import torch
import triton

from highlight_spec.hierarchical_cache import reference_attention
from highlight_spec.hierarchical_paged import PagedHierarchicalKV
from highlight_spec import hierarchical_paged_kernels as current


CASES = {"draft4": (1, .07, 4), "draft8": (1, .07, 8),
         "ar": (1, 1., 8), "verify": (7, 1., 8),
         "sv50": (7, .5, 8), "sv40": (7, .4, 8), "sv25": (7, .25, 8)}


def load_baseline(path):
    spec = importlib.util.spec_from_file_location("hierarchical_paged_baseline", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fixture(batch, length, query_heads=32, kv_heads=8):
    torch.manual_seed(42)
    width = triton.cdiv(length, 16)
    table = torch.randperm(batch*width, device="cuda", dtype=torch.int32).reshape(batch, width)
    cache = PagedHierarchicalKV.allocate(batch*width, kv_heads, batch, "cuda")
    k, v = [torch.randn((batch, length, kv_heads, 128), device="cuda", dtype=torch.bfloat16)
            for _ in range(2)]
    for slot in range(batch):
        current.prefill(cache, slot, table[slot], k[slot], v[slot])
    return dict(batch=batch, length=length, h=kv_heads, hq=query_heads,
                cache=cache, table=table, k=k, v=v,
                slots=torch.arange(batch, device="cuda", dtype=torch.int32))


def case_inputs(data, name):
    nq, fraction, bits = CASES[name]
    b, length, hq = data["batch"], data["length"], data["hq"]
    blocks = triton.cdiv(length, 16)
    count = min(blocks, max(4, triton.cdiv(int(length*fraction), 16)))
    if fraction < 1:
        # Scattered history plus the full recent window, including all queries.
        recent = min(4, blocks)
        history = torch.linspace(0, blocks-recent-1, count-recent, device="cuda").long()
        chosen = torch.cat((history, torch.arange(blocks-recent, blocks, device="cuda"))).int()
    else:
        chosen = torch.arange(blocks, device="cuda", dtype=torch.int32)
    ids = chosen[None].expand(b, -1).contiguous()
    positions = (chosen.long()[:, None]*16+torch.arange(16, device="cuda")).flatten()
    positions = positions[positions < length]
    q = torch.randn((b*nq, hq, 128), device="cuda", dtype=torch.bfloat16)
    qp = torch.arange(length-nq, length, device="cuda").repeat(b)
    qsl = torch.arange(0, (b+1)*nq, nq, device="cuda", dtype=torch.int32)
    return dict(nq=nq, bits=bits, fraction=fraction, q=q, qp=qp, qsl=qsl,
                ids=ids, positions=positions, max_positions=count*16,
                counts=torch.full((b,), count, device="cuda", dtype=torch.int32))


def kernel_call(module, data, inputs, workspace=None):
    ws = workspace or module.PagedAttentionWorkspace(data["batch"], data["h"], data["hq"],
        inputs["nq"], inputs["max_positions"], "cuda")
    out = torch.empty_like(inputs["q"])
    kwargs = dict(bits=inputs["bits"], workspace=ws, output=out)
    if inputs["fraction"] < 1:
        kwargs.update(selected_ids=inputs["ids"], selected_counts=inputs["counts"])
    def call():
        return module.attention(data["cache"], inputs["q"], inputs["qp"], inputs["qsl"],
                                data["slots"], data["table"], **kwargs)
    return call, ws


def measure(fn, replays=100, repeats=3):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): result = fn()
    eager = result.clone()
    graph.replay()
    torch.testing.assert_close(result, eager, rtol=0, atol=0)
    for _ in range(10): graph.replay()
    samples = []
    for _ in range(repeats):
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start.record()
        for _ in range(replays): graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end)/replays)
    return dict(median_ms=statistics.median(samples), replay_means_ms=samples,
                graph_matches_eager=True)


def native_call(data, inputs):
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    b, nq = data["batch"], inputs["nq"]
    k, v = [x.index_select(1, inputs["positions"]).flatten(0, 1) for x in (data["k"], data["v"])]
    length = len(inputs["positions"])
    ksl = torch.arange(0, (b+1)*length, length, device="cuda", dtype=torch.int32)
    out = torch.empty_like(inputs["q"])
    return lambda: flash_attn_varlen_func(inputs["q"], k, v, nq, inputs["qsl"], length,
        cu_seqlens_k=ksl, causal=True, out=out, fa_version=2, num_splits=0)


def check_reference(data, inputs, actual):
    # Independent FP32 attention on decoded storage; check the first request.
    pos = inputs["positions"]
    kv = data["cache"].read(0, data["table"][0], inputs["bits"], pos)
    expected = reference_attention(inputs["q"][:inputs["nq"]], *kv,
        inputs["qp"][:inputs["nq"]], pos)
    got = actual[:inputs["nq"]]
    torch.testing.assert_close(got, expected, rtol=.025, atol=.004)
    return float((got.float()-expected.float()).abs().max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 25])
    parser.add_argument("--lengths", type=int, nargs="+", default=[32768])
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--replays", type=int, default=100)
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    if args.out.exists(): raise ValueError("refusing to overwrite timing evidence")
    if min(*args.batches, *args.lengths, args.replays) < 1 or min(args.lengths) < 7:
        raise ValueError("positive batches/replays and at least seven context positions required")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    source = Path(current.__file__).read_bytes()
    args.out.with_suffix(".kernel.py").write_bytes(source)
    baseline = load_baseline(args.baseline) if args.baseline else None
    result = dict(status="running", host=socket.gethostname(),
        slurm_job_id=os.getenv("SLURM_JOB_ID"), gpu=torch.cuda.get_device_name(),
        capability=torch.cuda.get_device_capability(), torch=torch.__version__, triton=triton.__version__,
        source_sha256=hashlib.sha256(source).hexdigest(),
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest() if baseline else None,
        scope="paged attention only; excludes prefill, selection, KV updates and model execution",
        native_scope=("FA2 BF16; selected KV gathered before timing; not B200 FA4"
                      if args.native else None),
        serving_throughput=False, process_repetitions=1, rows=[])
    for length in args.lengths:
        for batch in args.batches:
            data = fixture(batch, length, args.query_heads)
            for name in args.cases:
                inputs = case_inputs(data, name)
                row = dict(case=name, batch=batch, length=length, query_heads=args.query_heads)
                for label, module in (("baseline", baseline), ("current", current)):
                    if module is None: continue
                    call, ws = kernel_call(module, data, inputs)
                    row[label] = dict(**measure(call, args.replays), splits=ws.splits,
                        tile=getattr(ws, 'tile', 32), warps=getattr(ws, 'warps', 4),
                        contiguous_codes=getattr(ws, 'contiguous_codes', None),
                        scratch_bytes=ws.partial.numel()*4+ws.lse.numel()*4,
                        reference_max_abs_error=check_reference(data, inputs, call()))
                if args.native: row["native_fa2"] = measure(native_call(data, inputs), args.replays)
                if baseline: row["kernel_speedup"] = row["baseline"]["median_ms"]/row["current"]["median_ms"]
                result["rows"].append(row)
                args.out.write_text(json.dumps(result, indent=2)+"\n")
                print(json.dumps(row), flush=True)
            del data, inputs, call, ws
            torch.cuda.empty_cache()
    if Path(current.__file__).read_bytes() != source:
        raise RuntimeError("kernel source changed during measurement")
    result["status"] = "passed"
    args.out.write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
