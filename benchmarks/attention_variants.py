"""Matched, cold-cache attention diagnostics for the Main Results kernel variants.

One layer at the 4B/32K/batch-42 shape. Synthetic Q/K/V and fixed nested
selection sets isolate attention cost. Production modules come from the pinned
completed campaign; model execution, selection and cache writes are not timed.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import socket
import statistics
import sys
import time

CASES = (
    ('full-ar-4plus4', 1, 1., 8),
    ('full-draft-4', 1, 1., 4),
    ('highlight-draft-4', 1, .07, 4),
    ('highlight-ar-draft-4plus4', 1, .07, 8),
    ('full-verify-4plus4', 7, 1., 8),
    ('sv50-4plus4', 7, .5, 8),
    ('sv40-4plus4', 7, .4, 8),
    ('sv25-4plus4', 7, .25, 8),
    ('highlight-native-fp8', 1, .07, 'fp8'),
)


def run(args):
    parent=args.parent.resolve()
    plan=json.loads((parent/'plan.json').read_text())
    sys.path.insert(0,plan['source_root'])
    import torch
    from benchmarks.main_results import verify_plan,load,save,sha
    from benchmarks.attention_compare import measure,errors
    from highlight_spec.hierarchical_cache import reference_attention
    from highlight_spec.hierarchical_paged import PagedHierarchicalKV
    from highlight_spec import hierarchical_paged_kernels as storage,hierarchical_native as native
    from highlight_spec.native_fp8_attention import flash_attn_varlen_func
    from vllm import _custom_ops as ops
    verify_plan(parent)
    if torch.cuda.get_device_capability()!=(10,0):raise RuntimeError('B200 SM100 required')
    gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
    if gpu!=plan['repetition_gpus'][str(args.repetition)]:raise ValueError('repetition GPU differs')
    original=load(parent/'throughput/4b/32768/dense8/r1/throughput.json')
    b,length,h,hq,dim=42,32768,8,32,128
    if original['batch']!=b:raise ValueError('completed common batch differs')
    if args.out.exists():raise FileExistsError(args.out)
    torch.manual_seed(42)
    started=time.monotonic()
    pages=original['capacity']['available_blocks']+1
    logical=length//16; width=(length+512+15)//16
    table=torch.zeros((b,width),device='cuda',dtype=torch.int32)
    table[:,:logical]=(torch.randperm(pages-1,device='cuda',dtype=torch.int32)[:b*logical]+1).reshape(b,logical)
    cache=PagedHierarchicalKV.allocate(pages,h,b,'cuda')
    kv=[torch.randn(b,length,h,dim,device='cuda',dtype=torch.bfloat16) for _ in range(2)]
    q7=torch.randn(b,7,hq,dim,device='cuda',dtype=torch.bfloat16)
    q1=q7[:,-1].contiguous()
    for slot in range(b):storage.prefill(cache,slot,table[slot],kv[0][slot],kv[1][slot])
    if cache.error.any().item():raise ValueError('prefill cache error')
    # Same physical pages and values; FP8 stores the entire context before timing.
    scales=[float(x.abs().max())/448 for x in (q1,*kv)]
    fp8=[torch.empty((pages,16,h,dim),device='cuda',dtype=torch.float8_e4m3fn) for _ in range(2)]
    for kind in range(2):
        for slot in range(b):
            encoded=(kv[kind][slot].float()/scales[kind+1]).to(torch.float8_e4m3fn)
            # Indexing uses the uint8 view because torch FP8 index_put is unsupported in some builds.
            fp8[kind].view(torch.uint8)[table[slot,:logical].long()]=encoded.view(torch.uint8).reshape(logical,16,h,dim)
    del kv
    # Five recent blocks cover the 64-token recency rule at any within-block offset.
    rankings=[torch.randperm(logical-5,device='cuda') for _ in range(b)]
    recent=torch.arange(logical-5,logical,device='cuda')
    slots=torch.arange(b,device='cuda',dtype=torch.int32)
    evict=torch.zeros(512*1024*1024,device='cuda',dtype=torch.uint8)
    record=dict(status='running',host=socket.gethostname(),gpu=gpu,repetition=args.repetition,
        model_shape='Qwen3-4B',batch=b,context=length,query_heads=hq,kv_heads=h,head_dim=dim,
        physical_pages=pages,seed=42,synthetic_inputs=True,one_layer=True,
        source_root=plan['source_root'],source_sha256=plan['source_sha256'],benchmark_sha256=sha(__file__),
        runtime={k:importlib.metadata.version(k) for k in ('torch','vllm','triton')},
        scope='Attention call including query preparation/conversion, attention, reduction and output scatter; excludes selection/block-table compaction, KV writes, projections, FFN and model execution',
        timing='CUDA graph; 512 MiB cache eviction before each sample outside CUDA events; 3 replays of 10 samples; compilation and warm-up excluded',
        serving_throughput=False,rows=[])
    save(args.out,record)
    cases=list(CASES);shift=3*(args.repetition-1);cases=cases[shift:]+cases[:shift]
    for name,nq,fraction,bits in cases:
        print('CASE',name,flush=True)
        count=logical if fraction==1 else min(logical,max(5,(int(length*fraction)+15)//16))
        chosen=(torch.arange(logical,device='cuda').expand(b,-1) if fraction==1 else
            torch.stack([torch.cat((rank[:count-5],recent)).sort().values for rank in rankings]))
        ids=torch.full((b,width),-1,device='cuda',dtype=torch.int32);ids[:,:count]=chosen.int()
        counts=torch.full((b,),count,device='cuda',dtype=torch.int32)
        q=q1 if nq==1 else q7.flatten(0,1)
        qp=torch.arange(length-nq,length,device='cuda').repeat(b)
        qsl=torch.arange(0,(b+1)*nq,nq,device='cuda',dtype=torch.int32)
        out=torch.empty_like(q)
        extra={}
        if bits=='fp8':
            q8=torch.empty_like(q,dtype=torch.float8_e4m3fn)
            qscale=torch.tensor([scales[0]],device='cuda',dtype=torch.float32)
            descales=[torch.full((b,h),v,device='cuda',dtype=torch.float32) for v in scales]
            selected_table=torch.zeros_like(table)
            selected_table[:,:count]=torch.gather(table,1,chosen.long())
            lengths=torch.full((b,),count*16,device='cuda',dtype=torch.int32)
            def call():
                ops.scaled_fp8_quant(q.view(b,-1),qscale,output=q8.view(b,-1))
                flash_attn_varlen_func(q=q8,k=fp8[0],v=fp8[1],out=out,cu_seqlens_q=qsl,
                    max_seqlen_q=1,seqused_k=lengths,max_seqlen_k=length,
                    softmax_scale=dim**-.5,causal=True,block_table=selected_table,fa_version=4,
                    q_descale=descales[0],k_descale=descales[1],v_descale=descales[2],num_splits=0)
                return out
            extra=dict(arithmetic='Native FP8 FA4; BF16 input/output; query quantization timed',scales=scales)
        else:
            maxpos=length+512 if fraction==1 else max(80,int((length+512)*fraction)+96)
            ws=native.NativeAttentionWorkspace(b,h,hq,nq,maxpos,'cuda')
            def call():
                return native.attention(cache,q,qp,qsl,slots,table,bits=bits,
                    selected_ids=ids if fraction<1 else None,selected_counts=counts if fraction<1 else None,
                    workspace=ws,output=out)
            extra=dict(arithmetic='Packed KV reconstructed to BF16; BF16 tensor-core attention',
                       splits=ws.splits,workspace_bytes=ws.memory_bytes(),workspace_positions=maxpos)
        call();torch.cuda.synchronize()
        checks=[]
        for slot in (0,b//2,b-1):
            pos=(chosen[slot,:,None]*16+torch.arange(16,device='cuda')).flatten().long()
            if bits=='fp8':
                physical=selected_table[slot,:count].long()
                decoded=[x.view(torch.uint8)[physical].view(torch.float8_e4m3fn).float().reshape(-1,h,dim)*scale
                         for x,scale in zip(fp8,scales[1:])]
                query=q8[slot:slot+1].float()*scales[0]
                limit=.08
            else:
                decoded=cache.read(slot,table[slot],bits,pos)
                query=q[slot*nq:(slot+1)*nq]
                limit=.015
            expected=reference_attention(query,*decoded,qp[slot*nq:(slot+1)*nq],pos)
            actual=out[slot*nq:(slot+1)*nq]
            error=errors(actual,expected)
            if not torch.isfinite(actual).all() or error['relative_rms']>limit:
                raise AssertionError(f'{name}: storage reference failed: {error}')
            checks.append(dict(request=slot,**error,relative_rms_limit=limit))
        measured,_=measure(call,evict,samples=10)
        samples=[x for row in measured['sample_ms'] for x in row]
        if not all(math.isfinite(x) and x>0 for x in samples):raise ValueError('invalid timing')
        row=dict(case=name,queries=nq,fraction=fraction,kv_bits=bits,selected_tokens=count*16,
            reference=checks,**extra,**measured)
        record['rows'].append(row);save(args.out,record)
        print(json.dumps({k:v for k,v in row.items() if k not in ('reference','sample_ms')}),flush=True)
    verify_plan(parent)
    record.update(status='passed',wall_seconds=time.monotonic()-started)
    save(args.out,record)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--repetition',type=int,choices=(1,2,3),required=True)
    run(p.parse_args())
