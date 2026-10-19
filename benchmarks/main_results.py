"""Prepare, audit and report the hierarchical-only Main Results campaign.

prepare is CPU-only. queue is an explicit, separate launch command. Each
capacity job searches in fresh child engines; each result job uses one engine.
"""
import argparse
import collections
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import socket
import sys

from highlight_spec.hierarchical_config import METHODS

CONTEXTS = (32768, 65536, 130560)
RULER_LENGTHS = (65536, 129024)
REPETITIONS = 3
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def load(path):
    return json.loads(Path(path).read_text())


def flags(method, checkpoint):
    m = METHODS[method]
    return ['--checkpoint', checkpoint, '--mode', m.mode, '--hierarchical-method', method,
            '--draft-select', 'round' if m.speculative else 'step',
            '--verify-fraction', str(m.verify_fraction)]


def make_jobs(root, out, inputs):
    """Every primary measurement maps to exactly one reviewed table cell/repetition."""
    py = str(root/'.venv/bin/python')
    jobs = []
    def add(name, stage, **kw):
        job = dict(name=name, kind='main-'+stage, stage=stage, **kw)
        job['command'] = [py, '-m', 'benchmarks.hierarchical_run', '--run', str(out), '--job', name]
        jobs.append(job)
        return name
    kernels = add('validation/kernels', 'kernels')
    smoke = []
    for model in ('4b','8b'):
        for method in METHODS:
            smoke.append(add(f'validation/{model}/{method}', 'smoke', model=model,
                             method=method, ctx=130560, depends_on=[kernels]))
    capacities = {}
    for model in ('4b','8b'):
        for ctx in CONTEXTS:
            capacities[model,ctx] = [add(f'capacity/{model}/{ctx}/{method}', 'capacity',
                model=model, ctx=ctx, method=method, depends_on=smoke) for method in METHODS]
    all_capacities = [j for values in capacities.values() for j in values]
    timings = []
    for repetition in range(1,REPETITIONS+1):
        for model in ('4b','8b'):
            for ctx in CONTEXTS:
                # Rotate order without changing the paired batch or inputs.
                methods = list(METHODS)
                methods = methods[repetition-1:] + methods[:repetition-1]
                for method in methods:
                    timings.append(add(f'throughput/{model}/{ctx}/{method}/r{repetition}', 'throughput',
                        model=model, ctx=ctx, method=method, repetition=repetition,
                        required_gpu=2 if repetition in (1,3) else 3,
                        depends_on=all_capacities))
    for model in ('4b','8b'):
        for length in RULER_LENGTHS:
            for method in METHODS:
                add(f'ruler/{model}/{length}/{method}', 'ruler', model=model, ctx=length,
                    method=method, expected_rows=192, depends_on=timings)
        for method in METHODS:
            add(f'longbench/{model}/cot/{method}', 'longbench', model=model, method=method,
                expected_rows=60, depends_on=timings)
    return jobs


def counts(jobs):
    by_stage = dict(collections.Counter(j['stage'] for j in jobs))
    return dict(by_stage=by_stage, result_runs=sum(by_stage.get(k,0) for k in ('throughput','ruler','longbench')),
        scheduled_jobs=len(jobs), capacity_engine_processes='adaptive; recorded separately',
        longbench_answer_only=0, measured_batches_per_throughput_process=1)


def campaign_gpus(plan):
    """Validate explicit physical GPU authorization and paired repetition placement."""
    allowed=plan['allowed_gpus']
    mapping=plan.get('repetition_gpus',{'1':2,'2':3,'3':2})
    if (not allowed or len(set(allowed))!=len(allowed)
            or any(type(g) is not int or g not in range(4) for g in allowed)
            or set(mapping)!={'1','2','3'}
            or any(type(g) is not int or g not in allowed for g in mapping.values())):
        raise ValueError('invalid GPU authorization or repetition assignment')
    return allowed,mapping


def make_throughput_jobs(runtime, out, models, contexts, backend, repetition_gpus=None):
    """A selected throughput grid with capacity dependencies only within its scope."""
    full=make_jobs(runtime,out,{})
    jobs=[j for j in full if j['stage'] in ('capacity','throughput')
          and j['model'] in models and j['ctx'] in contexts]
    capacities=[j['name'] for j in jobs if j['stage']=='capacity']
    for job in jobs:
        job['depends_on']=[] if job['stage']=='capacity' else list(capacities)
        job['attention_backend']=backend
        job['command'].insert(1,'-P')
        if job['stage']=='throughput' and repetition_gpus is not None:
            job['required_gpu']=repetition_gpus[str(job['repetition'])]
    return jobs


def prepare_throughput(args):
    """Reuse validated native kernels and frozen prompts without scheduling quality jobs."""
    from benchmarks.queue import source_digest
    prior=load(args.prior_run/'plan.json')
    runtime=Path(prior['runtime_root'])
    out=args.out.resolve()
    models,contexts=tuple(args.models),tuple(args.contexts)
    allowed=list(args.gpus)
    assigned=args.repetition_gpus or [allowed[i%len(allowed)] for i in range(3)]
    repetition_gpus={str(i+1):gpu for i,gpu in enumerate(assigned)}
    campaign_gpus(dict(allowed_gpus=allowed,repetition_gpus=repetition_gpus))
    if len(set(models))!=len(models) or len(set(contexts))!=len(contexts):
        raise ValueError('duplicate campaign scope')
    if out.exists():
        raise ValueError('prepare into a new directory')
    gate_path=getattr(args,'validation_gate',None) or runtime/'outputs/evidence/hierarchical-native-loader.json'
    gate_path=gate_path.resolve()
    gate=load(gate_path)
    if gate['status']!='validated_opt_in' or gate['native_gpu_tests']<13:
        raise ValueError('native hierarchical validation is incomplete')
    manifest_path=runtime/gate['source_manifest']
    tested=load(manifest_path)
    core=['hierarchical_native.py','hierarchical_native_loader.py','hierarchical_native_pipeline.py',
          'hierarchical_paged.py','hierarchical_paged_kernels.py','hierarchical_backend.py',
          'hierarchical_config.py']
    for name in core:
        key='highlight_spec/'+name
        if sha(ROOT/key)!=tested[key]:
            raise ValueError('native implementation changed after validation: '+name)
    checkpoints={m:prior['inputs']['checkpoints'][m] for m in models}
    inputs=dict(checkpoints=checkpoints,data=prior['inputs']['data'],
        smoke_prompts={m:prior['inputs']['smoke_prompts'][m] for m in models})
    hashes={str(gate_path):sha(gate_path),str(manifest_path):sha(manifest_path)}
    for name,digest in prior['input_sha256'].items():
        if name==inputs['data'] or any(Path(name).parent==Path(c) for c in checkpoints.values()):
            hashes[name]=digest
    prompt_source=args.prior_run.resolve()/'prompts'
    for model in models:
        for ctx in contexts:
            path=prompt_source/model/f'ctx-{ctx}/prompts.json'
            rows=load(path)
            if not rows or len({r['id'] for r in rows})!=len(rows) or any(len(r['prompt_token_ids'])!=ctx for r in rows):
                raise ValueError('invalid frozen prompt pool')
            hashes[str(path)]=sha(path)
    source=out/'source';source.mkdir(parents=True)
    for name in ('benchmarks','highlight_spec','tests','third_party'):
        shutil.copytree(ROOT/name,source/name,ignore=shutil.ignore_patterns('__pycache__'))
    import spotattention
    shutil.copytree(Path(spotattention.__file__).parent,source/'spotattention',
                   ignore=shutil.ignore_patterns('__pycache__'))
    jobs=make_throughput_jobs(runtime,out,models,contexts,'fa4',repetition_gpus)
    plan=dict(schema='hierarchical-main-v1',host=socket.gethostname(),runtime_root=str(runtime),
        source_root=str(source),source_sha256=source_digest(source),inputs=inputs,
        input_sha256=hashes,allowed_gpus=allowed,repetition_gpus=repetition_gpus,idle_seconds=60,
        runtime={n:importlib.metadata.version(n) for n in ('torch','vllm','triton','spotattention')},
        report_module='benchmarks.main_results',jobs=jobs,counts=counts(jobs),
        scope=dict(models=list(models),contexts=list(contexts)),prompt_source=str(prompt_source),
        validation_gate=str(gate_path),state='prepared_not_launched',
        protocol=dict(batch='largest common calibrated capacity across all eight methods',
            repetitions=3,batches_per_process=1,prefill_excluded=True,eos_respected=True,
            output_cap=512,spec_tokens=6,async_scheduling=False,prefix_caching=False,
            graph='FULL_DECODE_ONLY; one graph per draft step',weights='bf16',indexer='bf16',
            cache='hierarchical 4+4-bit',attention_backend='fa4'))
    save(out/'plan.json',plan)
    print(json.dumps(dict(run=str(out),**plan['counts']),indent=2))


def capacity_candidate(history):
    """Largest feasible integer batch, bounded by a batch-1 allocation probe.

    A failed engine initialization is not capacity evidence unless its failure
    is explicitly identified as memory exhaustion by the child.
    """
    if not history:
        return 1
    first = history[0]
    if first['batch'] != 1 or not first['fits']:
        raise ValueError('batch one does not fit')
    lower, upper = 1, first['maximum_batch']
    for row in history[1:]:
        if row['fits']:
            lower = max(lower,row['batch'])
        else:
            upper = min(upper,row['batch']-1)
    if lower == upper:
        return None
    if len(history)==1 or history[-1]['fits']:
        return upper
    hint = history[-1].get('maximum_batch',0)
    return hint if lower < hint <= upper else (lower+upper+1)//2


def common_batch(run, model, ctx):
    rows = [load(run/f'capacity/{model}/{ctx}/{method}/capacity.json') for method in METHODS]
    if any(r.get('status')!='complete' for r in rows):
        raise ValueError('all method capacities must be measured first')
    return min(r['maximum_batch'] for r in rows)


def discover_inputs(prior):
    """Reuse the saved benchmark plan; never infer dataset paths from filenames."""
    inputs = dict(checkpoints={},ruler={},questions={})
    for job in load(prior)['jobs']:
        if job['kind'] not in ('ruler','longbench'):
            continue
        command=job['command'];model=job['model']
        inputs['checkpoints'][model]=command[command.index('--checkpoint')+1]
        if job['kind']=='ruler':
            inputs['ruler'][f"{model}/{job['ctx']}"]=command[command.index('--rows')+1]
        elif '--cot' in command:
            inputs['questions'][model]=command[command.index('--questions')+1]
    if (set(inputs['checkpoints'])!={'4b','8b'} or set(inputs['questions'])!={'4b','8b'}
            or len(inputs['ruler'])!=4):
        raise ValueError('prior plan lacks the agreed model and quality inputs')
    return inputs


def prepare(args):
    from benchmarks.queue import source_digest
    inputs=discover_inputs(args.prior_quality_plan)
    inputs['data']=str(args.data.resolve())
    inputs['smoke_prompts']={model:str(args.prompt_root.resolve()/model/'130560/prompts.json') for model in ('4b','8b')}
    out=args.out.resolve()
    if out.exists():
        raise ValueError('prepare into a new directory')
    files=[inputs['data'], *inputs['ruler'].values(), *inputs['questions'].values(), *inputs['smoke_prompts'].values()]
    for checkpoint in inputs['checkpoints'].values():
        files += [str(p) for p in Path(checkpoint).glob('*.json')]
        files += [str(p) for p in Path(checkpoint).glob('*.safetensors')]
    # Hash model weights too: changing a checkpoint invalidates a frozen plan.
    hashes={str(Path(p).resolve()):sha(p) for p in files}
    for model,path in inputs['questions'].items():
        q=load(path)
        if len(q['questions'])!=60 or q['protocol']['thinking'] is not False:
            raise ValueError('LongBench cohort differs')
    for key,path in inputs['ruler'].items():
        r=load(path)
        if len(r['rows'])!=192 or len({x['task'] for x in r['rows']})!=6:
            raise ValueError('RULER cohort differs')
    # A runnable frozen source tree; edits to the live checkout cannot mutate it.
    source=out/'source'
    source.mkdir(parents=True)
    for name in ('benchmarks','highlight_spec'):
        shutil.copytree(ROOT/name,source/name,ignore=shutil.ignore_patterns('__pycache__'))
    import spotattention
    shutil.copytree(Path(spotattention.__file__).parent,source/'spotattention',
                   ignore=shutil.ignore_patterns('__pycache__'))
    runtime_root=(args.runtime_root or ROOT).resolve()
    jobs=make_jobs(runtime_root,out,inputs)
    plan=dict(schema='hierarchical-main-v1',host=socket.gethostname(),runtime_root=str(runtime_root),
        source_root=str(source),source_sha256=source_digest(source),inputs=inputs,
        input_sha256=hashes,allowed_gpus=[2,3],idle_seconds=60,
        runtime={n:importlib.metadata.version(n) for n in ('torch','vllm','triton','spotattention')},
        report_module='benchmarks.main_results',jobs=jobs,counts=counts(jobs),
        protocol=dict(batch='largest common calibrated capacity across all eight methods',
            repetitions=3,batches_per_process=1,prefill_excluded=True,eos_respected=True,
            output_cap=512,spec_tokens=6,async_scheduling=False,prefix_caching=False,
            graph='FULL_DECODE_ONLY; one graph per draft step',longbench='CoT only; 1024 reasoning + 128 answer',
            quality_repetitions=1,weights='bf16',indexer='bf16',cache='hierarchical 4+4-bit'),
        state='prepared_not_launched')
    save(out/'plan.json',plan)
    print(json.dumps(dict(run=str(out),**plan['counts']),indent=2))


def verify_plan(run):
    from benchmarks.queue import source_digest
    plan=load(run/'plan.json')
    _,repetitions=campaign_gpus(plan)
    if plan['schema']!='hierarchical-main-v1' or socket.gethostname()!=plan['host']:
        raise ValueError('host, GPU authorization or schema changed')
    if any(j['stage']=='throughput' and j.get('required_gpu')!=repetitions[str(j['repetition'])]
           for j in plan['jobs']):
        raise ValueError('throughput GPU pairing changed')
    if source_digest(Path(plan['source_root']))!=plan['source_sha256']:
        raise ValueError('frozen source changed')
    for name,version in plan['runtime'].items():
        if importlib.metadata.version(name)!=version:
            raise ValueError('runtime changed: '+name)
    for name,digest in plan['input_sha256'].items():
        if sha(name)!=digest:
            raise ValueError('input changed: '+name)
    return plan


def audit_execution(execution, steps, batch, speculative, cap=512, *, require_draft_trace=False):
    events=execution['target_execution'][:len(steps)]
    if len(events)!=len(steps):
        raise ValueError('target execution trace is incomplete')
    for index,event in enumerate(events):
        if event['requests']!=batch:
            raise ValueError('target did not execute the full batch')
        if event['kind']=='graph':
            expected=sum(min(7,cap-n) for n in steps[index]['before']) if speculative else batch
            if event['tokens']!=expected:
                raise ValueError('target graph query shape differs from the timed batch')
        if event['kind']!='graph':
            # With max_model_len=prompt+cap, native vLLM clips each query to
            # min(7, cap-emitted). Rejection can leave that row unfinished,
            # requiring more than one shortened verification. Match the exact
            # native shape; do not allow arbitrary eager fallback near the cap.
            expected=[min(7,cap-n) for n in steps[index]['before']]
            if (not speculative or event['kind']!='eager'
                    or len(expected)!=batch or min(expected)<1
                    or sum(expected)==batch*7 or event['tokens']!=sum(expected)
                    or event['max_query_len']!=max(expected)):
                raise ValueError('steady decode did not use full CUDA graphs')
    if speculative and execution['draft_graph_replays']<6*len(steps):
        raise ValueError('six draft graphs did not replay per round')
    if speculative:
        trace=execution.get('draft_execution')
        if trace is None:
            if require_draft_trace:
                raise ValueError('per-round draft graph trace is required')
            return  # Legacy evidence has aggregate draft counts only.
        matched={}
        for row in trace:
            index=row['target_event_index']
            if not 0<=index<len(steps):
                continue  # Prefill proposals and work after the timed window.
            if index in matched:
                raise ValueError('duplicate timed draft execution record')
            matched[index]=row
        if len(matched)!=len(steps):
            raise ValueError('timed draft execution trace is incomplete')
        for row in matched.values():
            if (row['requests']!=batch or row['tokens']!=batch
                    or row['steps']!=6 or row['graph_replays']!=6):
                raise ValueError('timed round did not use six full-batch draft graphs')


def audit_answer_format(text):
    """Distinguish an empty closing delimiter from a generated reasoning span.

    Qwen can repeat the prompt's closing delimiter before starting its answer.
    Permit exactly one leading close, followed by nonempty marker-free text.
    Inspect a local view only; never alter saved text, tokens or timing.
    """
    answer=text.strip()
    leading_close=answer.startswith('</think>')
    if leading_close:
        answer=answer[len('</think>'):].strip()
    if not answer or '<think>' in answer or '</think>' in answer:
        raise ValueError('invalid thinking-off generation')
    return dict(leading_closing_think_marker=leading_close)


def audit_throughput(job, result, *, speculative=None, check_cache_transactions=True):
    from benchmarks.multilexsum import output_audit
    from benchmarks.eos_throughput import useful_speculation
    if result['method']!=job['method'] or result['ctx']!=job['ctx'] or result['batches']!=1:
        raise ValueError('result configuration differs')
    if job.get('attention_backend') and result['cache'].get('attention_backend')!=job['attention_backend']:
        raise ValueError('measured attention backend differs from the frozen plan')
    capacity=result['capacity'];batch=result['batch'];steps=result['timing']['decode_steps']
    if not capacity['fits'] or capacity['batch']!=batch or not steps:
        raise ValueError('capacity or full-batch window missing')
    manifest=load(result['manifest_path'])
    if sha(result['manifest_path'])!=result['manifest_sha256']:
        raise ValueError('prompt manifest changed')
    from benchmarks.multilexsum import digest
    if (result['prompt_ids']!=[r['id'] for r in manifest[:batch]]
            or result['prompt_sha256']!=[digest(r['prompt_token_ids']) for r in manifest[:batch]]
            or any(len(r['prompt_token_ids'])!=job['ctx'] for r in manifest[:batch])):
        raise ValueError('timed prompts differ from the frozen batch')
    outputs=result['outputs']
    if len(outputs)!=batch or len(set(result['prompt_ids']))!=batch:
        raise ValueError('batch is incomplete or duplicates prompts')
    previous=[1]*batch;elapsed=0
    if speculative is None:
        speculative=METHODS[job['method']].speculative
    limit=7 if speculative else 1
    for index,step in enumerate(steps):
        if (step['before']!=previous or len(step['after'])!=batch or len(step['finished'])!=batch
                or not math.isfinite(step['elapsed_seconds']) or step['elapsed_seconds']<=elapsed
                or any(step['finished'])!=(index==len(steps)-1)):
            raise ValueError('invalid full-batch timing trace')
        for i,(a,b,finished) in enumerate(zip(step['before'],step['after'],step['finished'],strict=True)):
            if not 1<=b-a<=limit or b>len(outputs[i]['token_ids']) or finished!=(b==len(outputs[i]['token_ids'])):
                raise ValueError('trace and useful outputs disagree')
        previous=step['after'];elapsed=step['elapsed_seconds']
    tokens=sum(previous)-batch
    timing=result['timing']
    if (timing['decode_window']!='all_prefills_complete_to_first_request_finish'
            or timing['decode_tokens']!=tokens or timing['decode_seconds']!=elapsed
            or not math.isclose(timing['decode_tok_s'],tokens/elapsed,rel_tol=1e-12)):
        raise ValueError('throughput is not derived from useful emitted tokens')
    formats=[]
    for output in outputs:
        formats.append(audit_answer_format(output['text']))
        output_audit(output['token_ids'],result['eos_ids'],512,output['finish_reason'])
    graphs=result['execution']
    audit_execution(graphs,steps,batch,speculative,
                    require_draft_trace=result.get('execution_audit_version',1)>=2)
    if speculative:
        if graphs['draft_graph_replays']<6*len(steps) or result['speculation']!=useful_speculation(steps,6,512):
            raise ValueError('draft execution or acceptance accounting differs')
    if check_cache_transactions and any(any(values) for values in result['cache']['errors'].values()):
        raise ValueError('cache transaction failed')
    return dict(decode_tok_s=tokens/elapsed,decode_seconds=elapsed,decode_tokens=tokens,
                acceptance=result['speculation'],peak_memory=graphs['peak_reserved_bytes'],
                output_format=dict(leading_closing_think_markers=sum(
                    f['leading_closing_think_marker'] for f in formats)))


def validate(job,dest):
    if job['stage']=='kernels':
        row=load(dest/'validation.json')
        if row.get('status')!='passed' or row.get('sparse_verifier_graph_matches_reference') is not True:
            raise ValueError('packed sparse-verifier validation missing')
    elif job['stage']=='capacity':
        row=load(dest/'capacity.json')
        if row['status']!='complete' or capacity_candidate(row['probes']) is not None:
            raise ValueError('capacity search is incomplete')
    elif job['stage']=='smoke':
        row=load(dest/'smoke.json')
        if row['status']!='passed' or row['method']!=job['method'] or not row['execution']['target_graph_replays']:
            raise ValueError('integration gate failed')
        if METHODS[job['method']].speculative and (not row['graph_check'] or not row['graph_check']['kv_bitwise_equal']):
            raise ValueError('same-state draft graph audit failed')
    elif job['stage']=='throughput':
        result=load(dest/'throughput.json')
        audit=audit_throughput(job,result)
        if result['batch']!=common_batch(dest.parents[4],job['model'],job['ctx']):
            raise ValueError('throughput did not use the common capacity batch')
        save(dest/'audit.json',audit)
    else:
        config=load(dest/'config.json');finished=load(dest/'finished.json')
        rows=[json.loads(line) for line in (dest/'results.jsonl').read_text().splitlines()]
        if (len(rows)!=job['expected_rows'] or len({r['id'] for r in rows})!=len(rows)
                or config['hierarchical_method']!=job['method'] or config['thinking'] is not False
                or config['ignore_eos'] is not False):
            raise ValueError('quality output or configuration is incomplete')
        expected=config['questions' if job['stage']=='longbench' else 'rows']
        if [r['id'] for r in rows]!=expected:
            raise ValueError('quality cohort or order changed')
        from benchmarks.multilexsum import output_audit
        for row in rows:
            output_audit(row['output_token_ids'],config['eos_audit_ids'],128 if job['stage']=='longbench' else 1024,row['finish_reason'])
            if job['stage']=='longbench':
                if not config['cot']: raise ValueError('answer-only is excluded')
                output_audit(row['reasoning_token_ids'],config['eos_audit_ids'],1024,row['reasoning_finish_reason'])
        if finished['questions' if job['stage']=='longbench' else 'rows']!=len(rows):
            raise ValueError('quality summary is incomplete')


def report(run):
    """Audited machine-readable results. Historical tables are never overwritten."""
    plan=load(run/'plan.json');status=load(run/'status.json')
    measurements=[]
    for job in status['jobs']:
        if job['state']!='complete': continue
        dest=run/job['name']
        if job['stage']=='throughput':
            validate(job,dest)
            measurements.append(dict(model=job['model'],ctx=job['ctx'],method=job['method'],
                                     repetition=job['repetition'],**load(dest/'audit.json')))
        elif job['stage'] in ('ruler','longbench'):
            validate(job,dest)
            measurements.append(dict(model=job['model'],kind=job['stage'],ctx=job.get('ctx'),
                method=job['method'],**load(dest/'finished.json')))
    save(run/'report.json',dict(state=status['state'],counts=plan['counts'],measurements=measurements,
        speedup_reference='dense8, same model/context/batch/GPU/repetition',
        note='Three process results required before a throughput cell is complete.'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action')
    prep=sub.add_parser('prepare')
    for name in ('prior-quality-plan','data','prompt-root','out'):
        prep.add_argument('--'+name,type=Path,required=True)
    prep.add_argument('--runtime-root',type=Path)
    scoped=sub.add_parser('prepare-throughput')
    scoped.add_argument('--prior-run',type=Path,required=True)
    scoped.add_argument('--out',type=Path,required=True)
    scoped.add_argument('--models',nargs='+',choices=('4b','8b'),required=True)
    scoped.add_argument('--contexts',nargs='+',type=int,choices=CONTEXTS,required=True)
    scoped.add_argument('--gpus',nargs='+',type=int,choices=range(4),default=[2,3])
    scoped.add_argument('--repetition-gpus',nargs=3,type=int,choices=range(4))
    scoped.add_argument('--validation-gate',type=Path)
    for action in ('check','queue'):
        child=sub.add_parser(action);child.add_argument('--run',type=Path,required=True)
    p.add_argument('--run',type=Path,help='write the campaign report (queue hook)')
    args=p.parse_args()
    if args.action=='prepare': prepare(args)
    elif args.action=='prepare-throughput': prepare_throughput(args)
    elif args.action in ('check','queue'):
        plan=verify_plan(args.run)
        if args.action=='check': print(json.dumps(dict(status='prepared_not_launched',**plan['counts']),indent=2))
        else:
            source=Path(plan['source_root'])
            os.environ['PYTHONPATH']=str(source)
            os.execv(sys.executable,[sys.executable,'-P','-m','benchmarks.queue','--run',str(args.run.resolve())])
    elif args.run: report(args.run)
    else: p.error('choose prepare, check or queue')


if __name__=='__main__': main()
