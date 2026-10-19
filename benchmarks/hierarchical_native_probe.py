"""Correctness and matched attention timing for the native hierarchical loader."""
import argparse
import json
import hashlib
import importlib.metadata
from pathlib import Path
import socket
import time
import torch
import triton

from highlight_spec.hierarchical_paged import PagedHierarchicalKV
from highlight_spec import hierarchical_paged_kernels as old
from highlight_spec import hierarchical_native as native
from benchmarks.attention_compare import measure, errors
from highlight_spec.hierarchical_cache import reference_attention


def run(args):
    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError('requires B200')
    torch.manual_seed(42)
    b, length, nq = args.batch, args.length, args.queries
    blocks = triton.cdiv(length, 16)
    table = torch.randperm(b*blocks, device='cuda', dtype=torch.int32).reshape(b, blocks)
    cache = PagedHierarchicalKV.allocate(b*blocks, 8, b, 'cuda')
    for slot in range(b):
        kv = [torch.randn(length, 8, 128, device='cuda', dtype=torch.bfloat16) for _ in range(2)]
        old.prefill(cache, slot, table[slot], *kv)
    del kv
    q = torch.randn(b*nq, 32, 128, device='cuda', dtype=torch.bfloat16)
    qp = torch.arange(length-nq, length, device='cuda').repeat(b)
    qsl = torch.arange(0, (b+1)*nq, nq, device='cuda', dtype=torch.int32)
    slots = torch.arange(b, device='cuda', dtype=torch.int32)
    # Production budgets include the recent 64 tokens inside the selection.
    count = min(blocks, max(5, (int(length*args.fraction)+15)//16))
    recent = min(5, blocks)
    selected = torch.cat((torch.randperm(max(0, blocks-recent), device='cuda')[:count-recent],
                          torch.arange(blocks-recent, blocks, device='cuda'))).sort().values.int()
    ids = selected[None].repeat(b, 1) if args.sparse else None
    counts = torch.full((b,), len(selected), device='cuda', dtype=torch.int32) if args.sparse else None
    maxpos = len(selected)*16 if args.sparse else length
    ws = native.NativeAttentionWorkspace(b, 8, 32, nq, maxpos, 'cuda')
    legacy = old.PagedAttentionWorkspace(b, 8, 32, nq, maxpos, 'cuda')
    output = torch.empty_like(q)
    def call():
        return native.attention(cache, q, qp, qsl, slots, table, bits=args.bits,
                                selected_ids=ids, selected_counts=counts, workspace=ws, output=output)
    print('COMPILE', vars(args), 'splits', ws.splits, flush=True)
    call();torch.cuda.synchronize()
    before = old.attention(cache, q, qp, qsl, slots, table, bits=args.bits,
                           selected_ids=ids, selected_counts=counts, workspace=legacy)
    print('LEGACY_ERROR',errors(output,before),flush=True)
    torch.testing.assert_close(output,before,rtol=.025,atol=.004)
    positions = ((selected.long()[:, None]*16+torch.arange(16,device='cuda')).flatten()
                 if args.sparse else torch.arange(length,device='cuda'))
    positions = positions[positions < length]
    checks=[]
    for slot in sorted({0,b-1}):
        kv=cache.read(slot,table[slot],args.bits,positions)
        expected=reference_attention(q[slot*nq:(slot+1)*nq],*kv,qp[slot*nq:(slot+1)*nq],positions)
        torch.testing.assert_close(output[slot*nq:(slot+1)*nq],expected,rtol=.025,atol=.004)
        checks.append(errors(output[slot*nq:(slot+1)*nq],expected))
    print('REFERENCE_PASS',checks,flush=True)
    evict=torch.zeros(512*1024*1024,device='cuda',dtype=torch.uint8)
    result,_=measure(call,evict,samples=10)
    oldout=torch.empty_like(q)
    def oldcall():return old.attention(cache,q,qp,qsl,slots,table,bits=args.bits,
        selected_ids=ids,selected_counts=counts,workspace=legacy,output=oldout)
    oldtime,_=measure(oldcall,evict,samples=10)
    shape = {k:v for k,v in vars(args).items() if k != 'out'}
    root = Path(__file__).resolve().parents[1]
    sources = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in [Path(__file__).resolve(), root/'benchmarks/attention_compare.py',
                         *sorted((root/'highlight_spec').glob('hierarchical*.py'))]}
    result.update(shape=shape,reference=checks,old_median_ms=oldtime['median_ms'],
                  speedup=oldtime['median_ms']/result['median_ms'],workspace_bytes=ws.memory_bytes())
    result.update(status='passed', source_sha256=sources, host=socket.gethostname(),
        finished=time.time(), runtime={k:importlib.metadata.version(k) for k in ('torch','vllm','triton')},
        device=torch.cuda.get_device_name(), selected_tokens=count*16 if args.sparse else length,
        attention_only=True, process_repetitions=1, old_sample_ms=oldtime['sample_ms'])
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('x') as f:
            json.dump(result, f, indent=2)
            f.write('\n')
    print('RESULT',json.dumps({k:v for k,v in result.items()
        if k not in ('sample_ms','old_sample_ms','source_sha256')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--length',type=int,default=4096)
    p.add_argument('--batch',type=int,default=1)
    p.add_argument('--queries',type=int,choices=(1,7),default=1)
    p.add_argument('--bits',type=int,choices=(4,8),default=4)
    p.add_argument('--sparse',action='store_true')
    p.add_argument('--fraction',type=float,default=.07,help='selected token fraction, including recent tokens')
    p.add_argument('--out',type=Path)
    args = p.parse_args()
    if not 0 < args.fraction <= 1 or args.length < args.queries or args.batch < 1:
        p.error('invalid shape or selection fraction')
    run(args)
