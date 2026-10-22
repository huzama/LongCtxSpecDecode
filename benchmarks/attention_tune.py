"""Offline split calibration over identical one-layer attention workloads.

Only diagnostic workspaces override splits. Production dispatch is untouched.
Every candidate includes query preparation, reduction and output conversion.
"""
import argparse
import fcntl
import importlib.metadata
import hashlib
import statistics
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

from benchmarks.main_results import load, save
from benchmarks.queue import source_digest
from benchmarks.attention_splits import wait_after_worker
from benchmarks.attention_variants import CASES


def verify(run):
    p = load(run/'plan.json')
    root = Path(__file__).resolve().parents[1]
    if (p['host'] != socket.gethostname() or p['allowed_gpus'] != [2, 3]
            or p['source_root'] != str(root) or p['source_sha256'] != source_digest(root)):
        raise ValueError('frozen calibration source or authorization differs')
    for name, version in p['runtime'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError('runtime differs: '+name)
    return p


def worker(run, shape_id, rep):
    import torch
    from benchmarks.attention_compare import errors, measure
    from highlight_spec.hierarchical_cache import reference_attention
    from highlight_spec.hierarchical_paged import PagedHierarchicalKV
    from highlight_spec import hierarchical_native as native
    from highlight_spec.hierarchical_paged_kernels import prefill
    from highlight_spec.native_fp8_attention import flash_attn_varlen_func
    from vllm import _custom_ops as ops
    p = verify(run)
    shape = p['shapes'][shape_id]
    gpu = int(os.environ['CUDA_VISIBLE_DEVICES'])
    if gpu != shape['gpu'] or torch.cuda.get_device_capability() != (10, 0):
        raise ValueError('assigned B200 required')
    dest = run/shape_id/f'r{rep}.json'
    if dest.exists(): raise FileExistsError(dest)
    b, ctx, pages = shape['batch'], shape['context'], shape['physical_pages']
    length = ctx+7
    blocks, width = (length+15)//16, (ctx+512+6+15)//16
    torch.manual_seed(42)
    table = torch.zeros((b, width), device='cuda', dtype=torch.int32)
    table[:, :blocks] = (torch.randperm(pages-1, device='cuda', dtype=torch.int32)[:b*blocks]+1).reshape(b, blocks)
    cache = PagedHierarchicalKV.allocate(pages, 8, b, 'cuda')
    fp8 = [torch.empty((pages, 16, 8, 128), device='cuda', dtype=torch.float8_e4m3fn) for _ in range(2)]
    # Deterministic non-unit scales. Gaussian fixture is far inside E4M3 range.
    scales = [.025, .025, .025]
    for slot in range(b):
        kv = [torch.randn(length, 8, 128, device='cuda', dtype=torch.bfloat16) for _ in range(2)]
        prefill(cache, slot, table[slot], *kv)
        for kind in range(2):
            padded = torch.zeros((blocks*16, 8, 128), device='cuda', dtype=torch.bfloat16)
            padded[:length] = kv[kind]
            encoded = (padded.float()/scales[kind+1]).to(torch.float8_e4m3fn)
            fp8[kind].view(torch.uint8)[table[slot, :blocks].long()] = encoded.view(torch.uint8).reshape(blocks,16,8,128)
    del kv, padded, encoded
    if cache.error.any().item(): raise ValueError('cache prefill failed')
    q7 = torch.randn(b,7,32,128,device='cuda',dtype=torch.bfloat16)
    q1 = q7[:,-1].contiguous()
    rankings = [torch.randperm(blocks-5,device='cuda') for _ in range(b)]
    recent = torch.arange(blocks-5,blocks,device='cuda')
    slots = torch.arange(b,device='cuda',dtype=torch.int32)
    evict = torch.zeros(512*1024*1024,device='cuda',dtype=torch.uint8)
    result = dict(status='running',shape=shape,host=socket.gethostname(),gpu=gpu,
        repetition=rep,source_sha256=p['source_sha256'],runtime=p['runtime'],rows=[],
        one_layer=True,synthetic_inputs=True,serving_throughput=False,
        timing='Three graph replays of ten cold-cache CUDA-event samples; preparation and reduction included',
        device=torch.cuda.get_device_name(),multiprocessors=torch.cuda.get_device_properties(0).multi_processor_count)
    save(dest,result)
    cases = list(CASES); shift = 3*(rep-1); cases = cases[shift:]+cases[:shift]
    try:
        for name,nq,fraction,bits in cases:
            count = blocks if fraction==1 else max(5,int(ctx*fraction)//16)
            chosen = (torch.arange(blocks,device='cuda').expand(b,-1) if fraction==1 else
                torch.stack([torch.cat((rank[:count-5],recent)).sort().values for rank in rankings]))
            ids = torch.full_like(table,-1); ids[:,:count] = chosen.int()
            counts = torch.full((b,),count,device='cuda',dtype=torch.int32)
            physical = torch.zeros_like(table); physical[:,:count] = torch.gather(table,1,chosen.long())
            q = q1 if nq==1 else q7.flatten(0,1)
            qp = torch.arange(length-nq,length,device='cuda').repeat(b)
            starts = torch.arange(0,(b+1)*nq,nq,device='cuda',dtype=torch.int32)
            out = torch.empty_like(q)
            bound = ctx+512 if fraction==1 else max(80,int((ctx+512)*fraction)//16*16)
            refs = {}
            if bits=='fp8':
                q8 = torch.empty_like(q,dtype=torch.float8_e4m3fn)
                qscale = torch.tensor([scales[0]],device='cuda',dtype=torch.float32)
                descales = [torch.full((b,8),x,device='cuda',dtype=torch.float32) for x in scales]
                lengths = torch.full((b,),count*16-(blocks*16-length),device='cuda',dtype=torch.int32)
                ops.scaled_fp8_quant(q.view(b,-1),qscale,output=q8.view(b,-1))
            for slot in sorted({0,b//2,b-1}):
                positions = (chosen[slot,:,None]*16+torch.arange(16,device='cuda')).flatten().long()
                positions = positions[positions<length]
                if bits=='fp8':
                    decoded = [x.view(torch.uint8)[physical[slot,:count].long()].view(torch.float8_e4m3fn)
                        .float().reshape(-1,8,128)[:len(positions)]*scale for x,scale in zip(fp8,scales[1:])]
                    query = q8[slot:slot+1].float()*scales[0]
                else:
                    decoded = cache.read(slot,table[slot],bits,positions)
                    query = q[slot*nq:(slot+1)*nq]
                refs[slot] = reference_attention(query,*decoded,qp[slot*nq:(slot+1)*nq],positions)
            del decoded
            limit = .08 if bits=='fp8' else .015
            if p.get('validate_policy'):
                from highlight_spec.attention_policy import choose_splits
                candidates = [choose_splits(b,8,32,nq,bound,
                    'fp8' if bits=='fp8' else f'hierarchical{bits}',fraction<1,result['multiprocessors'])]
            else:
                candidates = [s for s in (1,2,4,8,16,32,64,128) if s <= (bound+127)//128]
            if rep%2==0: candidates.reverse()
            for splits in candidates:
                print('COMPILE',shape_id,rep,name,splits,flush=True)
                if bits=='fp8':
                    def call():
                        ops.scaled_fp8_quant(q.view(b,-1),qscale,output=q8.view(b,-1))
                        flash_attn_varlen_func(q=q8,k=fp8[0],v=fp8[1],out=out,cu_seqlens_q=starts,
                            max_seqlen_q=1,seqused_k=lengths,
                            max_seqlen_k=ctx+512 if p.get('validate_policy') else bound,
                            causal=True,block_table=physical,fa_version=4,softmax_scale=128**-.5,
                            q_descale=descales[0],k_descale=descales[1],v_descale=descales[2],
                            num_splits=0 if p.get('validate_policy') else splits,
                            split_config=(bound,result['multiprocessors']) if p.get('validate_policy') else None)
                        return out
                    workspace_bytes = None
                else:
                    ws = native.NativeAttentionWorkspace(b,8,32,nq,bound,'cuda',
                        cache_format=f'hierarchical{bits}',sparse=fraction<1,
                        num_splits=None if p.get('validate_policy') else splits)
                    if ws.splits!=splits: raise ValueError('production split policy differs from calibration')
                    def call():
                        return native.attention(cache,q,qp,starts,slots,table,bits=bits,
                            selected_ids=ids if fraction<1 else None,selected_counts=counts if fraction<1 else None,
                            selected_page_table=physical if fraction<1 else None,workspace=ws,output=out)
                    workspace_bytes = ws.memory_bytes()
                call()
                checks = []
                for slot,expected in refs.items():
                    actual = out[slot*nq:(slot+1)*nq]
                    error = errors(actual,expected)
                    if not torch.isfinite(actual).all() or error['relative_rms']>limit:
                        raise ValueError(f'{name}/{splits}: reference failure {error}')
                    checks.append(dict(request=slot,**error,relative_rms_limit=limit))
                timing,_ = measure(call,evict,samples=10)
                if not all(math.isfinite(x) and x>0 for row in timing['sample_ms'] for x in row):
                    raise ValueError('invalid timing')
                result['rows'].append(dict(case=name,queries=nq,fraction=fraction,kv_bits=bits,
                    token_bound=bound,selected_blocks=count,splits=splits,reference=checks,
                    workspace_bytes=workspace_bytes,**timing))
                save(dest,result)
                print('RESULT',name,splits,timing['median_ms'],flush=True)
        verify(run)
        result['status']='passed'
    except Exception:
        result.update(status='failed',error=traceback.format_exc())
        raise
    finally:
        save(dest,result)


def queue(run,gpu):
    p = verify(run)
    if gpu not in p['allowed_gpus']: raise ValueError('unauthorized GPU')
    folder = run/f'gpu-{gpu}';folder.mkdir(exist_ok=True)
    status_path = folder/'status.json'
    owner = (folder/'controller.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if status_path.exists(): raise ValueError('automatic restart is disabled')
    lock = (Path(p['runtime_root'])/f'outputs/gpu-{gpu}-benchmark.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status = dict(state='waiting_for_idle',controller_pid=os.getpid(),gpu=gpu)
    try:
        for shape_id,shape in p['shapes'].items():
            if shape['gpu']!=gpu: continue
            (run/shape_id).mkdir(exist_ok=True)
            for rep in range(1, p.get('repetitions',3)+1):
                status.update(state='waiting_for_idle',shape=shape_id,repetition=rep);save(status_path,status)
                wait_after_worker(gpu)
                verify(run)
                with (run/shape_id/f'r{rep}.log').open('x') as log:
                    child = subprocess.Popen([sys.executable,'-P','-m','benchmarks.attention_tune',
                        '--run',str(run),'--shape',shape_id,'--rep',str(rep)],stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu)},
                        pass_fds=(lock.fileno(),))
                    status.update(state='running',worker_pid=child.pid);save(status_path,status)
                    rc=child.wait()
                if rc or load(run/shape_id/f'r{rep}.json')['status']!='passed':
                    raise RuntimeError('calibration failed; no automatic retry')
        status['state']='complete'
    except Exception:
        status.update(state='failed',error=traceback.format_exc());raise
    finally:
        save(status_path,status);lock.close()


def summarize(run):
    """Audit every candidate and emit data for review, never edit serving code."""
    from highlight_spec.attention_policy import workload
    p=load(run/'plan.json')
    if source_digest(run/'source') != p['source_sha256']:
        raise ValueError('calibration source changed')
    summary=dict(status='passed',source_sha256=p['source_sha256'],runtime=p['runtime'],
        scope='Offline kernel dispatch calibration, not model throughput',rows=[])
    choices={}
    for name,shape in p['shapes'].items():
        if load(run/f"gpu-{shape['gpu']}"/'status.json')['state']!='complete':
            raise ValueError('calibration queue is incomplete')
        processes=[load(run/name/f'r{rep}.json') for rep in (1,2,3)]
        for rep,d in enumerate(processes,1):
            if (d['status']!='passed' or d['shape']!=shape or d['repetition']!=rep
                    or d['source_sha256']!=p['source_sha256']):
                raise ValueError('process provenance differs')
        for case,nq,fraction,bits in CASES:
            bound=shape['context']+512 if fraction==1 else max(80,int((shape['context']+512)*fraction)//16*16)
            splits=[s for s in (1,2,4,8,16,32,64,128) if s <= (bound+127)//128]
            samples={s:[] for s in splits}
            for d in processes:
                rows=[r for r in d['rows'] if r['case']==case]
                if sorted(r['splits'] for r in rows)!=splits:
                    raise ValueError('missing or duplicate candidate')
                for row in rows:
                    raw=[v for rr in row['sample_ms'] for v in rr]
                    if (len(raw)!=30 or not all(math.isfinite(x) and x>0 for x in raw)
                            or row['median_ms']!=statistics.median(raw) or not row['graph_matches_eager']
                            or row['token_bound']!=bound or len(row['reference'])!=3
                            or any(c['relative_rms']>c['relative_rms_limit'] for c in row['reference'])):
                        raise ValueError('candidate validation failed')
                    samples[row['splits']].append(row['median_ms'])
            medians={s:statistics.median(v) for s,v in samples.items()}
            best=min(medians,key=lambda s:(medians[s],s))
            key=workload(shape['batch'],8,32,nq,bound,'fp8' if bits=='fp8' else f'hierarchical{bits}',fraction<1,148)
            if key in choices:raise ValueError('duplicate dispatch key')
            choices[key]=best
            prior=max(1,min(148//(shape['batch']*8),128,(bound+127)//128))
            summary['rows'].append(dict(shape=name,case=case,key=key,splits=best,
                original_splits=prior,median_ms=medians[best],original_ms=medians[prior],
                attention_speedup=medians[prior]/medians[best],candidate_medians_ms=medians,
                candidate_process_ms=samples,raw=[dict(path=str(Path(name)/f'r{rep}.json'),
                    sha256=hashlib.sha256((run/name/f'r{rep}.json').read_bytes()).hexdigest()) for rep in (1,2,3)]))
    save(run/'summary.json',summary)
    # Bind the dispatch table to the candidate wrappers that consume it.
    root=Path(__file__).resolve().parents[1]/'highlight_spec'
    names=('hierarchical_native.py','hierarchical_native_pipeline.py','hierarchical_native_loader.py',
           'hierarchical_paged_kernels.py','native_fp8_attention.py')
    hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
    text='"""SM100 split calibration; see outputs/evidence/attention-tuning.json."""\n'
    text+='KERNEL_SHA256 = '+repr(hashes)+'\nRUNTIME = '+repr(p['runtime'])+'\nSPLITS = {\n'
    text+=''.join(f'    {key!r}: {value},\n' for key,value in sorted(choices.items()))+'}\n'
    (run/'attention_tuning.py').write_text(text)
    print('Validated',len(choices),'dispatch workloads from',len(processes)*len(p['shapes']),'fresh processes')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--gpu',type=int,choices=(2,3))
    ap.add_argument('--shape')
    ap.add_argument('--rep',type=int,choices=(1,2,3))
    ap.add_argument('--summarize',action='store_true')
    a=ap.parse_args()
    if a.summarize: summarize(a.run)
    elif a.gpu is not None: queue(a.run,a.gpu)
    else: worker(a.run,a.shape,a.rep)
