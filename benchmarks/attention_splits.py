"""Controlled one/two-split FA4 comparison at the two recorded 8B batch shapes.

One-layer attention only. The production heuristic is overridden only while
allocating each diagnostic workspace; serving code and cache values are unchanged.
"""
import argparse
import fcntl
import importlib.metadata
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

from benchmarks.main_results import load, save
from benchmarks.queue import idle_gpus, source_digest


def verify(run):
    plan = load(run / 'plan.json')
    if socket.gethostname() != plan['host'] or plan['allowed_gpus'] != [2, 3]:
        raise ValueError('host or GPU authorization differs')
    root = Path(__file__).resolve().parents[1]
    if str(root) != plan['source_root'] or source_digest(root) != plan['source_sha256']:
        raise ValueError('frozen source differs')
    for name, version in plan['runtime'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError('runtime differs: ' + name)
    return plan


def telemetry(gpu):
    return subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=index,uuid,pstate,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu',
        '--format=csv,noheader,nounits'], text=True).strip()


def wait_after_worker(gpu):
    """Allow utilization telemetry to settle without waiting out another job."""
    idle_since = None
    deadline = time.monotonic() + 120
    while idle_since is None or time.monotonic() - idle_since < 60:
        apps = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
            '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if apps:
            raise RuntimeError('GPU acquired by another process; no further jobs dispatched')
        if time.monotonic() > deadline:
            raise RuntimeError('GPU did not settle; no further jobs dispatched')
        idle_since = (idle_since or time.monotonic()) if gpu in idle_gpus([gpu]) else None
        time.sleep(5)


def worker(run, shape_id, repetition):
    import torch
    from benchmarks.attention_compare import errors, measure
    from highlight_spec.hierarchical_cache import reference_attention
    from highlight_spec.hierarchical_paged import PagedHierarchicalKV
    from highlight_spec import hierarchical_native as native
    from highlight_spec import hierarchical_paged_kernels as storage

    plan = verify(run)
    shape = plan['shapes'][shape_id]
    gpu = shape['gpu']
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu):
        raise ValueError('worker must use its assigned physical GPU')
    device = torch.cuda.get_device_properties(0)
    if (device.major, device.minor) != (10, 0):
        raise ValueError('SM100 required')
    dest = run / shape_id / f'r{repetition}.json'
    if dest.exists():
        raise FileExistsError(dest)
    b, ctx, pages = shape['batch'], shape['context'], shape['physical_pages']
    length = ctx + 1
    width, blocks = (ctx + 512 + 6 + 15) // 16, (length + 15) // 16
    torch.manual_seed(plan['seed'])
    table = torch.zeros((b, width), device='cuda', dtype=torch.int32)
    table[:, :blocks] = (torch.randperm(pages - 1, device='cuda', dtype=torch.int32)[:b*blocks] + 1).reshape(b, blocks)
    cache = PagedHierarchicalKV.allocate(pages, 8, b, 'cuda')
    for slot in range(b):
        kv = [torch.randn(length, 8, 128, device='cuda', dtype=torch.bfloat16) for _ in range(2)]
        storage.prefill(cache, slot, table[slot], *kv)
    del kv
    if cache.error.any().item():
        raise ValueError('cache initialization failed')
    q = torch.randn(b, 32, 128, device='cuda', dtype=torch.bfloat16)
    positions = torch.full((b,), ctx, device='cuda', dtype=torch.int64)
    starts = torch.arange(b + 1, device='cuda', dtype=torch.int32)
    slots = torch.arange(b, device='cuda', dtype=torch.int32)
    selected = torch.arange(length, device='cuda', dtype=torch.int64)
    reference = {}
    for slot in sorted({0, b // 2, b - 1}):
        kv = cache.read(slot, table[slot], 8, selected)
        reference[slot] = reference_attention(q[slot:slot+1], *kv, positions[slot:slot+1], selected)
    del kv
    evict = torch.zeros(512*1024*1024, device='cuda', dtype=torch.uint8)
    from vllm.vllm_flash_attn.cute.interface import num_splits_heuristic
    automatic = max(1, num_splits_heuristic(b*8, device.multi_processor_count, (ctx+512+127)//128, 128))
    result = dict(status='running', host=socket.gethostname(), gpu=gpu, repetition=repetition,
        shape=shape, actual_kv_length=length, seed=plan['seed'], device=device.name,
        multiprocessors=device.multi_processor_count, automatic_splits=automatic,
        source_sha256=plan['source_sha256'], runtime=plan['runtime'], rows=[],
        synthetic_inputs=True, one_layer=True, serving_throughput=False,
        scope='Dense 4+4-bit KV attention; BF16 Q/output; includes query preparation, attention, split reduction and scatter',
        timing='CUDA graph; 512 MiB eviction outside each timed interval; 3 replays of 10 samples; warm-up and compilation excluded',
        telemetry_before=telemetry(gpu))
    save(dest, result)
    outputs = {}
    try:
        for splits in ([1, 2] if repetition % 2 else [2, 1]):
            print('COMPILE', shape_id, 'repetition', repetition, 'splits', splits, flush=True)
            with patch('vllm.vllm_flash_attn.cute.interface.num_splits_heuristic', return_value=splits):
                ws = native.NativeAttentionWorkspace(b, 8, 32, 1, ctx + 512, 'cuda')
            out = torch.empty_like(q)
            def call():
                return native.attention(cache, q, positions, starts, slots, table,
                                        bits=8, workspace=ws, output=out)
            call()
            checks = []
            for slot, expected in reference.items():
                actual = out[slot:slot+1]
                error = errors(actual, expected)
                if not torch.isfinite(actual).all() or error['relative_rms'] > .015:
                    raise ValueError('packed-storage reference failed: ' + str(error))
                checks.append(dict(request=slot, relative_rms_limit=.015, **error))
            measured, output = measure(call, evict, samples=10)
            if not all(math.isfinite(x) and x > 0 for row in measured['sample_ms'] for x in row):
                raise ValueError('invalid measured interval')
            outputs[splits] = output
            result['rows'].append(dict(splits=splits, workspace_bytes=ws.memory_bytes(),
                                       correctness=checks, **measured))
            save(dest, result)
            print('RESULT', splits, measured['median_ms'], flush=True)
        torch.testing.assert_close(outputs[1], outputs[2], rtol=.025, atol=.004)
        result['split_output_agreement'] = dict(**errors(outputs[2], outputs[1]),
            bitwise_equal=torch.equal(outputs[1], outputs[2]), rtol=.025, atol=.004, passed=True)
        verify(run)
        result.update(status='passed', telemetry_after=telemetry(gpu))
    except Exception:
        result.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        save(dest, result)


def queue(run, shape_id):
    plan = verify(run)
    gpu = plan['shapes'][shape_id]['gpu']
    folder = run / shape_id
    folder.mkdir(exist_ok=True)
    status_path = folder / 'status.json'
    owner = (folder / 'controller.lock').open('a')
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if status_path.exists():
        raise ValueError('inspect existing results; this diagnostic never resumes automatically')
    lock = (Path(plan['runtime_root']) / f'outputs/gpu-{gpu}-benchmark.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = dict(state='waiting_for_idle', gpu=gpu, controller_pid=os.getpid())
    save(status_path, status)
    try:
        idle_since = None
        while idle_since is None or time.monotonic() - idle_since < 60:
            idle_since = (idle_since or time.monotonic()) if gpu in idle_gpus([gpu]) else None
            time.sleep(5)
        for rep in (1, 2, 3):
            if rep > 1:
                wait_after_worker(gpu)
            verify(run)
            if gpu not in idle_gpus([gpu]):
                raise RuntimeError('GPU became occupied; no further jobs dispatched')
            command = [sys.executable, '-P', '-m', 'benchmarks.attention_splits',
                       '--run', str(run), '--shape', shape_id, '--rep', str(rep)]
            with (folder / f'r{rep}.log').open('x') as log:
                proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)},
                    pass_fds=(lock.fileno(),))
                status.update(state='running', repetition=rep, worker_pid=proc.pid)
                save(status_path, status)
                code = proc.wait()
            if code or load(folder / f'r{rep}.json')['status'] != 'passed':
                raise RuntimeError(f'repetition {rep} failed; no automatic retry')
        status['state'] = 'complete'
    except Exception:
        status.update(state='failed', error=traceback.format_exc())
        raise
    finally:
        save(status_path, status)
        lock.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--shape', choices=('64k', '128k'), required=True)
    p.add_argument('--rep', type=int, choices=(1, 2, 3))
    args = p.parse_args()
    if args.rep is None:
        queue(args.run, args.shape)
    else:
        worker(args.run, args.shape, args.rep)
