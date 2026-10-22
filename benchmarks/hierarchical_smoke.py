"""Small full-model integration and graph audit, never a throughput result.

Use a Slurm GPU locally. Native prefill may use FA2 on Ampere; packed decode,
selection and six-step speculation use the production implementations.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
from types import SimpleNamespace


def main():
    from highlight_spec.hierarchical_config import METHODS
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--method',choices=list(METHODS),required=True)
    parser.add_argument('--context',type=int,default=4093)
    parser.add_argument('--batch',type=int,default=2)
    parser.add_argument('--gpu-mem-util',type=float,default=.6)
    parser.add_argument('--prefill-fa-version',type=int,choices=[2,4],default=4)
    parser.add_argument('--attention-backend', choices=['triton', 'fa4'], default='triton')
    parser.add_argument('--trace',action='store_true')
    args=parser.parse_args()
    if args.out.exists():raise ValueError('use a new output directory')
    args.out.mkdir(parents=True)
    from benchmarks.main_results import audit_execution, save, sha
    from benchmarks.decode import generate_batch, rope_parameters, yarn_factor
    from highlight_spec.hierarchical_config import engine_options
    from vllm import LLM
    import torch

    # Cross a 128-token compression boundary within a six-draft round.
    batch,context,cap=args.batch,args.context,48
    method=METHODS[args.method]
    if context < 1024:
        raise ValueError('keep the existing 1024-token prompt ending')
    rows=json.loads(args.manifest.read_text())[:batch]
    if len(rows)!=batch or any(len(r['prompt_token_ids'])<context for r in rows):
        raise ValueError('the manifest must contain a full batch at the requested context length')
    prompts=[r['prompt_token_ids'][:context-1024]+r['prompt_token_ids'][-1024:] for r in rows]
    factor=yarn_factor(context+cap)
    rope=rope_parameters(args.checkpoint,factor) if factor else None
    config=engine_options(args.method,batch=batch,context=context,generation=cap,
        checkpoint=args.checkpoint,rope=rope,scheduler='benchmarks.decode.BatchScheduler',
        attention_backend=args.attention_backend)
    if not 0 < args.gpu_mem_util <= .9:
        raise ValueError('integration checks require a memory fraction in (0, .9]')
    config['gpu_memory_utilization']=args.gpu_mem_util
    config['attention_config']['flash_attn_version']=args.prefill_fa_version
    config['worker_extension_cls']='highlight_spec.hierarchical_profile.HierarchicalProfileWorker'
    if args.trace:
        config['profiler_config']=dict(profiler='torch',torch_profiler_dir=str((args.out/'trace').resolve()),
            torch_profiler_with_stack=False,torch_profiler_record_shapes=False,
            torch_profiler_with_memory=False,delay_iterations=2,max_iterations=2)
    root=Path(__file__).resolve().parents[1]
    sources={str(p.relative_to(root)):sha(p) for base in ['highlight_spec','benchmarks']
             for p in (root/base).glob('*.py')}
    plan=dict(method=args.method,batch=batch,context=context,output_cap=cap,
        fixed_drafts=6 if method.speculative else 0,
        eos_respected=True,thinking=False,serving_throughput=False,
        prompt_source=str(args.manifest.resolve()),manifest_sha256=sha(args.manifest),
        prompt_slice=f'first {context-1024} plus last 1024 tokens; functional test, no task-quality claim',
        prompt_ids=[r.get('id') for r in rows],prompt_token_ids=prompts,
        prefill_fa_version=args.prefill_fa_version, attention_backend=args.attention_backend,
        source_sha256=sources,
        runtime={k:importlib.metadata.version(k) for k in ['torch','vllm','triton','spotattention']},
        gpu=torch.cuda.get_device_name(),slurm_job_id=os.getenv('SLURM_JOB_ID'),engine=config)
    save(args.out/'plan.json',plan)
    llm=LLM(**config)
    native=SimpleNamespace(checkpoint=args.checkpoint,mode=method.mode,spec_tokens=6,
        eos_aware=True,gen=cap,out=str(args.out),batch=batch)
    graph_check=None
    def check():
        nonlocal graph_check
        graph_check=llm.collective_rpc('hierarchical_graph_check',kwargs={'num_reqs':batch})[0]
        save(args.out/'graph-check.json',graph_check)
    generate_batch(llm,native,prompts,16,check if method.speculative else None)
    if method.speculative and graph_check is None:
        raise RuntimeError('warm-up ended before the graph audit')
    llm.collective_rpc('hierarchical_cache_info')
    before=llm.collective_rpc('hierarchical_benchmark_begin')[0]
    if method.speculative:
        llm.collective_rpc('hierarchical_profile_start',kwargs={'batch':batch,'capacity':64})
    if args.trace:llm.start_profile()
    outputs,timing,_,_=generate_batch(llm,native,prompts,cap)
    if args.trace:llm.stop_profile()
    profile=llm.collective_rpc('hierarchical_profile_finish')[0] if method.speculative else None
    execution=llm.collective_rpc('hierarchical_benchmark_info')[0]
    execution['draft_graph_replays']-=before['draft_graph_replays']
    cache=llm.collective_rpc('hierarchical_cache_info')[0]
    result=dict(status='recorded',serving_throughput=False,execution=execution,cache=cache,
        graph_check=graph_check,decode_steps=timing['decode_steps'],stage_profile=profile,
        outputs=[dict(token_ids=list(o.outputs[0].token_ids),text=o.outputs[0].text,
                      finish_reason=o.outputs[0].finish_reason) for o in outputs])
    save(args.out/'result.json',result)
    audit_execution(execution,timing['decode_steps'],batch,method.speculative,cap=cap,
                    require_draft_trace=method.speculative)
    events=execution['target_execution'][:len(timing['decode_steps'])]
    full=sum(e['kind']=='graph' and e['tokens']==batch*(7 if method.speculative else 1) for e in events)
    if profile is not None and profile['rounds']!=full:
        raise RuntimeError('profile and full-batch target trace differ')
    if any(sha(root/name)!=value for name,value in sources.items()):
        raise RuntimeError('serving source changed during the integration check')
    result['status']='passed'
    save(args.out/'result.json',result)
    print(json.dumps(dict(status='passed',method=args.method,rounds=len(events),
        full_graph_rounds=full,graph_check=graph_check,
        stage_mean_ms=profile['mean_ms'] if profile else None)),flush=True)


if __name__=='__main__':main()
