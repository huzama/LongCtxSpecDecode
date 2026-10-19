"""Matched B200 attention timings, with BF16 inputs/outputs and no selection.

Stored KV is prepared before timing. Each timed call includes query conversion,
attention, split reduction and output conversion. A 512 MiB eviction precedes
each sample outside its CUDA-event interval. This is an attention microbenchmark,
not model throughput or a generation-quality experiment.
"""
import argparse
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time
import traceback

import torch


METHODS = ("hierarchical4", "hierarchical4+4", "native-fp8-fa4",
           "native-fp8-trtllm", "native-nvfp4-trtllm")


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(q, k, v):
    # Independent FP32 oracle, one request at a time, outside all timing.
    ratio = q.shape[0] // k.shape[1]
    q = q.float().reshape(k.shape[1], ratio, 128)
    scores = torch.einsum("hrd,thd->hrt", q, k.float()) * 128**-.5
    return torch.einsum("hrt,thd->hrd", scores.softmax(-1), v.float()).reshape(-1, 128)


def errors(actual, expected):
    delta = actual.float() - expected.float()
    return dict(max_abs=float(delta.abs().max()),
                relative_rms=float(delta.square().mean().sqrt() /
                                   expected.float().square().mean().sqrt().clamp_min(1e-8)))


def decode_nvfp4(packed, scales, global_scale, value=False):
    # Independent E2M1 decoding, including the native V-scale swizzle inverse.
    if value:
        p, h, t, s = scales.shape
        scales = scales.reshape(p, h, t//4, 4, s//4, 4).permute(0, 1, 2, 5, 3, 4)
        scales = scales.reshape(p, h, t, s)
    table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                          -0., -.5, -1., -1.5, -2., -3., -4., -6.], device=packed.device)
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
    return table[codes] * scales.float().repeat_interleave(16, -1) * global_scale


def fixture(batch, length):
    from highlight_spec.hierarchical_paged import PagedHierarchicalKV
    from highlight_spec import hierarchical_paged_kernels as hk
    from flashinfer.fp4_quantization import nvfp4_quantize_paged_kv_cache
    torch.manual_seed(42)
    npage = length // 16
    table = torch.randperm(batch*npage, device="cuda", dtype=torch.int32).reshape(batch, npage)
    k, v = [torch.randn(batch, length, 8, 128, dtype=torch.bfloat16, device="cuda") for _ in range(2)]
    q = torch.randn(batch, 32, 128, dtype=torch.bfloat16, device="cuda")
    cache = PagedHierarchicalKV.allocate(batch*npage, 8, batch, "cuda")
    for slot in range(batch):
        hk.prefill(cache, slot, table[slot], k[slot], v[slot])
    assert not cache.error.any().item()
    paged = []
    for x in (k, v):
        dst = torch.empty(batch*npage, 8, 16, 128, dtype=x.dtype, device="cuda")
        dst[table.flatten().long()] = x.reshape(batch*npage, 16, 8, 128).transpose(1, 2)
        paged.append(dst)
    qs = float(q.float().abs().max())/448
    ks, vs = [float(x.float().abs().max())/448 for x in paged]
    fp8 = tuple((x.float()/s).to(torch.float8_e4m3fn) for x, s in zip(paged, (ks, vs)))
    # FA4 consumes NHD; layout construction is prefill work, never timed.
    fp8_fa = [x.transpose(1, 2).contiguous() for x in fp8]
    fp4, sf4, k4s, v4s = nvfp4_quantize_paged_kv_cache(*paged, kv_layout="HND")
    del paged
    return dict(batch=batch, length=length, table=table, k=k, v=v, q=q, cache=cache,
                slots=torch.arange(batch, dtype=torch.int32, device="cuda"),
                qp=torch.full((batch,), length-1, dtype=torch.int64, device="cuda"),
                qsl=torch.arange(batch+1, dtype=torch.int32, device="cuda"),
                seq=torch.full((batch,), length, dtype=torch.int32, device="cuda"),
                qs=qs, ks=ks, vs=vs, fp8=fp8, fp8_fa=fp8_fa,
                fp4=fp4, sf4=sf4, k4s=k4s, v4s=v4s)


def make_call(data, method):
    from highlight_spec import hierarchical_paged_kernels as hk
    from vllm import _custom_ops as ops
    b, length, q = data["batch"], data["length"], data["q"]
    out = torch.empty_like(q)
    if method.startswith("hierarchical"):
        bits = 4 if method == "hierarchical4" else 8
        ws = hk.PagedAttentionWorkspace(b, 8, 32, 1, length, "cuda")
        def call():
            return hk.attention(data["cache"], q, data["qp"], data["qsl"], data["slots"],
                                data["table"], bits=bits, workspace=ws, output=out)
        meta = dict(bits=bits, arithmetic="BF16 dot / FP32 accumulation",
                    stored_cache_bytes=data["cache"].memory_bytes(),
                    scratch_bytes=ws.partial.numel()*4 + ws.lse.numel()*4,
                    splits=ws.splits, tile=ws.tile)
        return call, meta
    q8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    qscale = torch.tensor([data["qs"]], device="cuda", dtype=torch.float32)
    def quant_q():
        ops.scaled_fp8_quant(q.view(b, -1), qscale, output=q8.view(b, -1))
    if method == "native-fp8-fa4":
        from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd
        descales = [torch.full((b, 8), data[s], device="cuda", dtype=torch.float32)
                    for s in ("qs", "ks", "vs")]
        def call():
            quant_q()
            _flash_attn_fwd(q=q8, k=data["fp8_fa"][0], v=data["fp8_fa"][1],
                cu_seqlens_q=data["qsl"], seqused_k=data["seq"], max_seqlen_q=1,
                max_seqlen_k=length, page_table=data["table"], softmax_scale=128**-.5,
                causal=True, num_splits=0, out=out,
                q_descale=descales[0], k_descale=descales[1], v_descale=descales[2])
            return out
        return call, dict(cache_format="E4M3", layout="NHD", query_conversion_timed=True,
                          stored_cache_bytes=sum(x.numel() for x in data["fp8_fa"]))
    from flashinfer.decode import (trtllm_batch_decode_with_kv_cache,
                                  _resolve_trtllm_gen_multi_ctas_kv_counter_buffer)
    workspace = torch.empty(256*1024*1024, dtype=torch.uint8, device="cuda")
    counters = _resolve_trtllm_gen_multi_ctas_kv_counter_buffer(
        None, b, 32, torch.cuda.get_device_properties(0).multi_processor_count, q.device)
    fp4 = method == "native-nvfp4-trtllm"
    native_out = torch.empty_like(q, dtype=torch.float8_e4m3fn) if fp4 else out
    ks, vs = (data["k4s"], data["v4s"]) if fp4 else (data["ks"], data["vs"])
    def call():
        quant_q()
        trtllm_batch_decode_with_kv_cache(query=q8, kv_cache=data["fp4"] if fp4 else data["fp8"],
            workspace_buffer=workspace, block_tables=data["table"], seq_lens=data["seq"],
            max_seq_len=length, bmm1_scale=data["qs"]*ks*128**-.5,
            bmm2_scale=1.0 if fp4 else vs,
            out=native_out, kv_layout="HND", backend="trtllm-gen", q_len_per_req=1,
            kv_cache_sf=data["sf4"] if fp4 else None,
            multi_ctas_kv_counter_buffer=counters)
        if fp4:
            out.copy_(native_out)
            out.mul_(vs)
        return out
    return call, dict(cache_format="NVFP4" if fp4 else "E4M3", layout="HND",
        query_conversion_timed=True, output_conversion_timed=fp4,
        fp8_output_descale=vs if fp4 else None,
        stored_cache_bytes=sum(x.numel() for x in data["fp4"]+data["sf4"])
                           if fp4 else sum(x.numel() for x in data["fp8"]),
        scratch_bytes=workspace.numel()+counters.numel()*4)


def validate(data, method, actual):
    checks = []
    q = data["q"]
    q_native = (q.float()/data["qs"]).to(torch.float8_e4m3fn).float()*data["qs"]
    for slot in sorted({0, data["batch"]-1}):
        pages = data["table"][slot].long()
        if method.startswith("hierarchical"):
            bits = 4 if method == "hierarchical4" else 8
            k, v = data["cache"].read(slot, data["table"][slot], bits)
            expected = reference(q[slot], k, v)
        else:
            if method == "native-nvfp4-trtllm":
                k, v = [decode_nvfp4(data["fp4"][i][pages], data["sf4"][i][pages], s, i == 1)
                        for i, s in enumerate((data["k4s"], data["v4s"]))]
                # Native NVFP4 attention converts each reconstructed block to
                # E4M3 before tensor-core math, with global scales applied later.
                k, v = [(x/s).to(torch.float8_e4m3fn).float()*s
                        for x, s in zip((k, v), (data["k4s"], data["v4s"]))]
            else:
                k, v = [data["fp8"][i][pages].float()*s
                        for i, s in enumerate((data["ks"], data["vs"]))]
            k, v = [x.transpose(1, 2).reshape(data["length"], 8, 128) for x in (k, v)]
            expected = reference(q_native[slot], k, v)
            if method == "native-nvfp4-trtllm":
                # This native kernel emits E4M3 before the BF16 conversion. Its
                # output rounding is part of the API, not an implementation error.
                # Error versus the original BF16 attention is recorded separately.
                expected = (expected/data["v4s"]).to(torch.float8_e4m3fn).float()*data["v4s"]
        error = errors(actual[slot], expected)
        limit = .015 if method.startswith("hierarchical") else .08
        if not torch.isfinite(actual).all() or error["relative_rms"] > limit:
            raise AssertionError(f"{method}: packed-storage reference failed: {error}, limit={limit}")
        original = reference(q[slot], data["k"][slot], data["v"][slot])
        checks.append(dict(request=slot, packed_reference=error, relative_rms_limit=limit,
                           original_bf16_reference=errors(actual[slot], original)))
    return checks


def measure(call, evict, samples=20):
    for _ in range(3):
        actual = call()
    torch.cuda.synchronize()
    eager = actual.clone()
    pairs = [(torch.cuda.Event(enable_timing=True, external=True),
              torch.cuda.Event(enable_timing=True, external=True)) for _ in range(samples)]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for start, end in pairs:
            evict.add_(1)
            start.record()
            actual = call()
            end.record()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, eager, rtol=0, atol=0)
    rounds = []
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        rounds.append([start.elapsed_time(end) for start, end in pairs])
    return dict(median_ms=statistics.median(x for row in rounds for x in row),
                sample_ms=rounds, graph_matches_eager=True,
                eviction_bytes=evict.numel(), samples_per_round=samples), actual.clone()


def run_worker(run, rep):
    plan = json.loads((run/"plan.json").read_text())
    dest = run/f"r{rep}.json"
    if dest.exists():
        raise FileExistsError(dest)
    from highlight_spec import hierarchical_paged_kernels as hk
    if digest(Path(hk.__file__)) != plan["hierarchical_kernel_sha256"]:
        raise RuntimeError("hierarchical kernel changed")
    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("requires B200 SM100")
    result = dict(status="running", host=socket.gethostname(), gpu=os.getenv("CUDA_VISIBLE_DEVICES"),
        gpu_name=torch.cuda.get_device_name(), repetition=rep, seed=42,
        plan_sha256=digest(run/"plan.json"), benchmark_sha256=digest(Path(__file__)),
        versions={x:importlib.metadata.version(x) for x in ("torch", "vllm", "flashinfer-python", "triton")},
        scope=plan["scope"], synthetic_inputs=True, rows=[])
    save(dest, result)
    try:
        evict = torch.zeros(512*1024*1024, dtype=torch.uint8, device="cuda")
        for shape in plan["shapes"]:
            print("PREPARE",shape,flush=True)
            data = fixture(shape["batch"], shape["length"])
            # Rotate method order between fresh processes to reduce thermal/order bias.
            order = METHODS[rep-1:] + METHODS[:rep-1]
            for method in order:
                print("CHECK",method,shape,flush=True)
                call, metadata = make_call(data, method)
                check = validate(data, method, call())
                timing, output = measure(call, evict, plan["samples"])
                validate(data, method, output)
                row = dict(**shape, method=method, **metadata, **timing, correctness=check)
                result["rows"].append(row);save(dest,result)
                print("RESULT",method,shape,"ms",timing["median_ms"],flush=True)
                del call, output
                gc.collect();torch.cuda.empty_cache()
            del data
            gc.collect();torch.cuda.empty_cache()
        result["status"] = "passed"
    except Exception:
        result["status"] = "failed";result["error"] = traceback.format_exc()
        raise
    finally:
        save(dest, result)


def run_queue(run):
    from benchmarks.queue import idle_gpus
    plan = json.loads((run/"plan.json").read_text())
    if socket.gethostname() != plan["host"]:
        raise RuntimeError("wrong host")
    owner = (run/"controller.lock").open("a")
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = dict(state="waiting_for_gpu", controller_pid=os.getpid(), allowed_gpus=[2,3])
    save(run/"status.json",status)
    gpu, lock = None, None
    while lock is None:
        for candidate in (2, 3):
            trial = (Path(plan["runtime_root"])/f"outputs/gpu-{candidate}-benchmark.lock").open("a")
            try:
                fcntl.flock(trial, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                trial.close();continue
            if candidate not in idle_gpus((candidate,)):
                trial.close();continue
            gpu, lock = candidate, trial
            break
        if lock is None:time.sleep(5)
    status.update(state="waiting_for_idle",gpu=gpu);save(run/"status.json",status)
    idle_since = None
    while idle_since is None or time.monotonic()-idle_since < 60:
        idle_since = (idle_since or time.monotonic()) if gpu in idle_gpus((gpu,)) else None
        time.sleep(5)
    try:
        for rep in range(1,4):
            command = [sys.executable,"-P","-m","benchmarks.attention_compare","--run",str(run),"--rep",str(rep)]
            with (run/f"r{rep}.log").open("x") as log:
                proc = subprocess.Popen(command,env={**os.environ,"CUDA_VISIBLE_DEVICES":str(gpu)},
                    stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,pass_fds=(lock.fileno(),))
                status.update(state="running",repetition=rep,worker_pid=proc.pid);save(run/"status.json",status)
                code = proc.wait()
            if code:
                raise RuntimeError(f"repetition {rep} failed: exit {code}; inspect r{rep}.log")
        status.update(state="complete")
    except Exception:
        status.update(state="failed",error=traceback.format_exc())
        raise
    finally:
        save(run/"status.json",status);lock.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--rep",type=int,choices=(1,2,3))
    args=p.parse_args()
    if args.rep is None:run_queue(args.run)
    else:run_worker(args.run,args.rep)


if __name__ == "__main__":
    main()
