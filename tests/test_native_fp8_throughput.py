import copy
import socket
from pathlib import Path

import pytest

from benchmarks.main_results import save, sha
from benchmarks.multilexsum import digest
from benchmarks.native_fp8_throughput import METHOD, CONTEXTS, make_jobs, run_job, validate, verify_plan


@pytest.mark.parametrize('expanded',[False,True])
def test_scoped_parent_produces_three_matched_fp8_runs(tmp_path,monkeypatch,expanded):
    from benchmarks import native_fp8_throughput as native
    mapping={'1':2,'2':3,'3':0 if expanded else 2}
    jobs=make_jobs(Path('/runtime'),tmp_path,{'4b/32768':42},{'4b/32768':'prompts'},
                   models=('4b',),contexts=(32768,),repetition_gpus=mapping)
    assert len(jobs)==3 and {j['required_gpu'] for j in jobs}==set(mapping.values())
    plan=dict(schema='native-fp8-throughput-v2',allowed_gpus=[0,1,2,3] if expanded else [2,3],
              repetition_gpus=mapping,host=socket.gethostname(),
              scope=dict(models=['4b'],contexts=[32768]),jobs=jobs,
              source_root=str(tmp_path),source_sha256='source',runtime={},input_sha256={})
    monkeypatch.setattr(native,'source_digest',lambda root:'source')
    save(tmp_path/'plan.json',plan)
    assert verify_plan(tmp_path)['jobs']==jobs
    plan['jobs'][1]['ctx']=65536
    save(tmp_path/'plan.json',plan)
    with pytest.raises(ValueError,match='scope'):
        verify_plan(tmp_path)


def test_only_eighteen_matched_throughput_processes():
    batches={f"{m}/{c}":i+3 for i,(m,c) in enumerate((m,c) for m in ("4b","8b") for c in CONTEXTS)}
    manifests={key:"/prompts/"+key for key in batches}
    jobs=make_jobs(Path('/runtime'),Path('/run'),batches,manifests)
    assert len(jobs)==18 and len({j['name'] for j in jobs})==18
    assert all(j['stage']=='throughput' and j['method']==METHOD for j in jobs)
    assert {j['required_gpu'] for j in jobs}=={2,3}
    assert all(j['required_gpu']==(2 if j['repetition'] in (1,3) else 3) for j in jobs)
    for model in ('4b','8b'):
        for ctx in CONTEXTS:
            group=[j for j in jobs if (j['model'],j['ctx'])==(model,ctx)]
            assert {j['repetition'] for j in group}=={1,2,3}
            assert {j['batch'] for j in group}=={batches[f'{model}/{ctx}']}


@pytest.mark.parametrize('historical', [True, False])
def test_launch_rejects_old_or_unauthorized_gpu_assignments(tmp_path, historical):
    plan=dict(schema='native-fp8-throughput-v1' if historical else 'native-fp8-throughput-v2',
              allowed_gpus=[0,1] if historical else [2,3], host=socket.gethostname(),
              jobs=[dict(required_gpu=0,repetition=1)]*18)
    save(tmp_path/'plan.json',plan)
    with pytest.raises(ValueError,match='authorization'):
        verify_plan(tmp_path)


@pytest.mark.parametrize('assigned,visible', [(0, '0'), (2, '0'), (3, '2')])
def test_direct_job_cannot_bypass_gpu_assignment(tmp_path, monkeypatch, assigned, visible):
    save(tmp_path/'plan.json', dict(schema='native-fp8-throughput-v2',
         allowed_gpus=[2,3], jobs=[dict(name='test', required_gpu=assigned)]))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
    with pytest.raises(ValueError, match='authorized GPU'):
        run_job(tmp_path, 'test')


@pytest.fixture
def native_result(tmp_path):
    # Archival GPU-0 measurements remain readable; only new launches are restricted.
    manifest=tmp_path/'prompts.json'
    ids=[5]*32768
    save(manifest,[dict(id='a',prompt_token_ids=ids)])
    job=dict(method=METHOD,model='4b',ctx=32768,batch=1,required_gpu=0,repetition=1,manifest=str(manifest))
    result=dict(method=METHOD,model='4b',ctx=32768,batch=1,batches=1,gpu=0,repetition=1,
        manifest_path=str(manifest),manifest_sha256=sha(manifest),prompt_ids=['a'],prompt_sha256=[digest(ids)],
        eos_ids=[99],outputs=[dict(token_ids=[8,99],text='summary',finish_reason='stop')],
        capacity=dict(batch=1,fits=True),speculation=None,
        selection=dict(retained_fraction=.07),
        engine=dict(kv_cache_dtype='fp8_e4m3',calculate_kv_scales=True,flash_attn_version=4,
                    async_scheduling=False,prefix_caching=False),
        cache=dict(backbone_cache='native_fp8_e4m3',indexer_cache='bf16',kv_cache_groups=1,
                   layers=[dict(element_bytes=1,scales=dict(q=.1,k=.2,v=.3))]),
        execution=dict(target_execution=[dict(kind='graph',requests=1,tokens=1)],
                       target_graph_replays=1,draft_graph_replays=0,peak_reserved_bytes=20),
        timing=dict(decode_steps=[dict(before=[1],after=[2],finished=[True],elapsed_seconds=.5)],
                    decode_window='all_prefills_complete_to_first_request_finish',
                    decode_tokens=1,decode_seconds=.5,decode_tok_s=2.))
    return job,result,tmp_path


def test_native_audit_reuses_eos_timing_checks_without_pretending_cache_is_hierarchical(native_result):
    job,result,dest=native_result
    save(dest/'throughput.json',result)
    assert validate(job,dest)['decode_tok_s']==2.
    result['outputs'][0]['token_ids']=[8,99,10]
    save(dest/'throughput.json',result)
    with pytest.raises(ValueError): validate(job,dest)


@pytest.mark.parametrize('change', ['eager','batch','scales','precision','inflated_rate'])
def test_native_audit_rejects_invalid_measurements(native_result,change):
    job,result,dest=native_result
    result=copy.deepcopy(result)
    if change=='eager': result['execution']['target_execution'][0]['kind']='eager'
    elif change=='batch': result['batch']=2
    elif change=='scales': result['cache']['layers'][0]['scales']['k']=float('nan')
    elif change=='precision': result['engine']['kv_cache_dtype']='auto'
    else: result['timing']['decode_tok_s']=200
    save(dest/'throughput.json',result)
    with pytest.raises(ValueError): validate(job,dest)
