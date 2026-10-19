"""One assigned Main Results job. Invoked by the explicit campaign queue."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from benchmarks.main_results import (METHODS, ROOT, capacity_candidate, common_batch, campaign_gpus,
                                     flags, load, save, sha, audit_execution)


def engine_args(plan,job,batch,cap=512):
    from benchmarks import decode
    p=argparse.ArgumentParser();decode.add_engine_args(p)
    args=p.parse_args(flags(job['method'],plan['inputs']['checkpoints'][job['model']]))
    args.hierarchical_attention_backend=plan['protocol'].get('attention_backend','triton')
    args.batch=batch;args.ctx=job['ctx'];args.gen=cap
    args.eos_aware=True
    return args


def capacity_info(llm,args):
    from benchmarks.decode import blocks_needed
    cfg=llm.llm_engine.vllm_config
    available=cfg.cache_config.num_gpu_blocks-1
    # Identical output+six-slot reservation for every method, including AR.
    per=blocks_needed([args.ctx],args.gen,6,16)
    return dict(batch=args.batch,available_blocks=available,blocks_per_request=per,
        maximum_batch=available//per,fits=args.batch*per<=available,
        reserved_fraction=args.batch*per/available)


def capacity_probe(plan,job,batch,out):
    from benchmarks import decode
    args=engine_args(plan,job,batch)
    started=time.monotonic()
    try:
        llm=decode.build_engine(args)
        result=capacity_info(llm,args)
    except Exception as error:
        # Other failures must abort calibration. Do not relabel broken kernels
        # or unsupported configurations as insufficient capacity.
        text=str(error).lower()
        if not any(s in text for s in ('cuda out of memory','no available memory for the cache blocks',
                                      'no available memory for the cache','not enough memory to hold')):
            raise
        result=dict(batch=batch,fits=False,maximum_batch=0,oom=str(error))
    result['wall_seconds']=time.monotonic()-started
    save(out,result)


def calibrate(plan,job,dest):
    history=[]
    while (candidate:=capacity_candidate(history)) is not None:
        output=dest/f'probe-{candidate}.json'
        command=[sys.executable,'-P','-m','benchmarks.hierarchical_run','--run',str(dest.parents[3]),
                 '--job',job['name'],'--probe-batch',str(candidate),'--probe-out',str(output)]
        with (dest/f'probe-{candidate}.log').open('x') as log:
            subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=3600)
        history.append(load(output))
        save(dest/'capacity.json',dict(status='searching',probes=history))
    maximum=max(row['batch'] for row in history if row['fits'])
    save(dest/'capacity.json',dict(status='complete',maximum_batch=maximum,probes=history,
                                  engine_processes=len(history)))


def prepared_prompts(run,plan,model,ctx,batch):
    """Expand the distinct prompt pool only to the calibrated need, once."""
    from benchmarks.multilexsum import prepare_suite
    folder=run/'prompts'/model
    folder.parent.mkdir(parents=True,exist_ok=True)
    with (folder.parent/f'{model}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not (folder/'suite.json').exists():
            if folder.exists():
                raise ValueError('incomplete frozen prompt suite; inspect before retry')
            contexts=plan.get('scope',{}).get('contexts',(32768,65536,130560))
            count=max(common_batch(run,model,length) for length in contexts)
            prior=plan.get('prompt_source')
            if prior:
                import shutil
                prior=Path(prior)/model
                if any(len(load(prior/f'ctx-{length}/prompts.json'))<count for length in contexts):
                    # Extend the same dataset-ordered cohort if the native backend
                    # fits more requests. Never cap the batch to the old pool.
                    prepare_suite(SimpleNamespace(checkpoint=plan['inputs']['checkpoints'][model],
                        data=plan['inputs']['data'],out=folder,count=count))
                    for length in contexts:
                        old=load(prior/f'ctx-{length}/prompts.json')
                        new=load(folder/f'ctx-{length}/prompts.json')
                        if any(a['id']!=b['id'] or a['prompt_token_ids']!=b['prompt_token_ids']
                               for a,b in zip(old,new)):
                            (folder/'suite.json').unlink()
                            raise ValueError('expanded prompt cohort changed its frozen prefix')
                else:
                    folder.mkdir()
                    for length in contexts:
                        dest=folder/f'ctx-{length}';dest.mkdir()
                        shutil.copyfile(prior/f'ctx-{length}/prompts.json',dest/'prompts.json')
                    save(folder/'suite.json',dict(contexts=list(contexts),count=count,
                        reused_from=str(prior),cohort='same ordered frozen prompt pool'))
            else:
                prepare_suite(SimpleNamespace(checkpoint=plan['inputs']['checkpoints'][model],
                    data=plan['inputs']['data'],out=folder,count=count))
            save(folder/'main-protocol.json',dict(count=count,allowed_gpus=plan.get('allowed_gpus',[2,3]),
                data_sha256=plan['input_sha256'][str(Path(plan['inputs']['data']).resolve())],
                purpose='Main Results: identical ordered cohort across all methods and repetitions'))
        path=folder/f'ctx-{ctx}'/'prompts.json'
        rows=load(path)
        if len(rows)<batch:
            raise ValueError('distinct prompt pool cannot fill the calibrated batch')
        return rows[:batch],str(path),sha(path)


def run_decode(plan,job,run,dest,smoke=False):
    from benchmarks import decode
    from benchmarks.eos_throughput import useful_speculation
    batch=2 if smoke else common_batch(run,job['model'],job['ctx'])
    args=engine_args(plan,job,batch)
    args.out=str(dest)
    if smoke:
        path=Path(plan['inputs']['smoke_prompts'][job['model']]);rows=load(path)[:batch]
        fingerprint=sha(path)
    else:
        rows,path,fingerprint=prepared_prompts(run,plan,job['model'],job['ctx'],batch)
    prompts=[row['prompt_token_ids'] for row in rows]
    if len({row['id'] for row in rows})!=batch or any(len(ids)!=args.ctx for ids in prompts):
        raise ValueError('prompt cohort or context changed')
    llm=decode.build_engine(args)
    capacity=capacity_info(llm,args)
    save(dest/'capacity.json',capacity)
    if not capacity['fits']:
        raise ValueError('calibrated batch no longer fits; no automatic batch reduction')
    graph_check=None
    def check():
        nonlocal graph_check
        graph_check=llm.collective_rpc('hierarchical_graph_check',kwargs={'num_reqs':batch})[0]
        save(dest/'graph-check.json',graph_check)
    speculative=METHODS[job['method']].speculative
    decode.generate_batch(llm,args,prompts,16,check if speculative else None)
    if speculative and graph_check is None:
        raise ValueError('warm-up ended before same-state draft graph validation')
    llm.collective_rpc('hierarchical_cache_info')
    before=llm.collective_rpc('hierarchical_benchmark_begin')[0]
    cap=512
    outputs,timing,counters_before,counters_after=decode.generate_batch(llm,args,prompts,cap)
    execution=llm.collective_rpc('hierarchical_benchmark_info')[0]
    execution['draft_graph_replays']-=before['draft_graph_replays']
    cache=llm.collective_rpc('hierarchical_cache_info')[0]
    result=dict(status='recorded',method=job['method'],model=job['model'],
        ctx=job['ctx'],batch=batch,batches=1,execution_audit_version=2,
        manifest_path=str(path),manifest_sha256=fingerprint,
        prompt_ids=[r['id'] for r in rows],prompt_sha256=[decode.digest(ids) for ids in prompts],
        eos_ids=decode.configured_eos_ids(args.checkpoint),timing=timing,execution=execution,
        graph_check=graph_check,cache=cache,capacity=capacity,
        outputs=[dict(token_ids=list(o.outputs[0].token_ids),text=o.outputs[0].text,
                      finish_reason=o.outputs[0].finish_reason) for o in outputs],
        speculation=useful_speculation(timing['decode_steps'],6,cap) if speculative else None,
        counters_before=counters_before,counters_after=counters_after,
        graph_scope='full graphs per draft step and steady target call; native cap-shortened fallback shape audited and timed',
        timing_scope='native engine timestamps; all prefills complete through first request finish')
    result_path=dest/('smoke.json' if smoke else 'throughput.json')
    # Preserve the raw evidence even when a post-run audit rejects a result.
    save(result_path,result)
    audit_execution(execution,timing['decode_steps'],batch,speculative,require_draft_trace=True)
    result['status']='passed' if smoke else 'complete'
    save(result_path,result)


def quality(plan,job,dest):
    base=flags(job['method'],plan['inputs']['checkpoints'][job['model']])
    if job['stage']=='ruler':
        from benchmarks.ruler import run
        argv=base+['--rows',plan['inputs']['ruler'][f"{job['model']}/{job['ctx']}"],
                   '--gen','1024','--out',str(dest)]
    else:
        from benchmarks.accuracy import main as run
        argv=base+['--questions',plan['inputs']['questions'][job['model']],
                   '--cot','--gen','1024','--out',str(dest)]
    run(argv)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--job',required=True)
    p.add_argument('--probe-batch',type=int);p.add_argument('--probe-out',type=Path)
    args=p.parse_args();run=args.run.resolve();plan=load(run/'plan.json')
    job=next(j for j in plan['jobs'] if j['name']==args.job);dest=run/job['name']
    allowed,_=campaign_gpus(plan)
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    if (visible not in {str(g) for g in allowed}
            or ('required_gpu' in job and visible!=str(job['required_gpu']))):
        raise ValueError('each job must have exactly one authorized physical GPU')
    started=time.monotonic()
    if args.probe_batch:
        if job['stage']!='capacity' or args.probe_out is None:
            raise ValueError('capacity probe arguments require a capacity job')
        capacity_probe(plan,job,args.probe_batch,args.probe_out)
        return
    if job['stage']=='kernels':
        from benchmarks.hierarchical_cache import validate_paged
        validate_paged(dest,attention_backend=plan['protocol'].get('attention_backend','triton'))
    elif job['stage']=='capacity': calibrate(plan,job,dest)
    elif job['stage'] in ('smoke','throughput'): run_decode(plan,job,run,dest,job['stage']=='smoke')
    else: quality(plan,job,dest)
    save(dest/'duration.json',dict(wall_seconds=time.monotonic()-started,stage=job['stage']))


if __name__=='__main__': main()
