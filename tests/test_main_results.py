"""CPU checks of campaign scope, scheduling, capacity search and engine contracts."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.main_results import METHODS, make_jobs, counts, capacity_candidate
from highlight_spec.hierarchical_config import engine_options
from benchmarks.queue import eligible


def test_expanded_gpu_authorization_preserves_three_paired_repetitions(tmp_path):
    from benchmarks.main_results import campaign_gpus, make_throughput_jobs
    plan=dict(allowed_gpus=[0,1,2,3],repetition_gpus={'1':2,'2':3,'3':0})
    allowed,mapping=campaign_gpus(plan)
    jobs=make_throughput_jobs(tmp_path,tmp_path/'run',('4b',),(32768,),'fa4',mapping)
    state=[dict(j,state='pending') for j in jobs]
    assert all(eligible(state,g)['stage']=='capacity' for g in allowed)
    for job in state:
        if job['stage']=='capacity':job['state']='complete'
    assert eligible(state,1) is None
    for rep,gpu in mapping.items():
        assert eligible(state,gpu)['repetition']==int(rep)
    assert counts(jobs)['by_stage']==dict(capacity=8,throughput=24)
    assert all(j['required_gpu']==mapping[str(j['repetition'])] for j in jobs if j['stage']=='throughput')
    with pytest.raises(ValueError,match='authorization'):
        campaign_gpus(dict(allowed_gpus=[2,3],repetition_gpus=mapping))


def test_scoped_native_campaign_has_only_requested_capacity_and_timing_jobs(tmp_path):
    from benchmarks.main_results import make_throughput_jobs
    jobs=make_throughput_jobs(tmp_path,tmp_path/'run',('4b',),(32768,),'fa4')
    assert counts(jobs)['by_stage']==dict(capacity=8,throughput=24)
    capacity={j['name'] for j in jobs if j['stage']=='capacity'}
    assert len({j['name'] for j in jobs})==32
    for job in jobs:
        assert job['model']=='4b' and job['ctx']==32768 and job['attention_backend']=='fa4'
        assert set(job['depends_on'])==(capacity if job['stage']=='throughput' else set())
    assert {j['method'] for j in jobs}==set(METHODS)


def test_native_backend_is_used_by_engine_and_required_by_result_audit(tmp_path):
    from benchmarks.hierarchical_run import engine_args
    from benchmarks.main_results import audit_throughput
    plan=dict(protocol=dict(attention_backend='fa4'),inputs=dict(checkpoints={'4b':'checkpoint'}))
    job,result=fixture_result(tmp_path)
    job.update(model='4b',attention_backend='fa4')
    assert engine_args(plan,job,42).hierarchical_attention_backend=='fa4'
    result['cache']['attention_backend']='triton'
    with pytest.raises(ValueError,match='backend'):
        audit_throughput(job,result)
    result['cache']['attention_backend']='fa4'
    assert audit_throughput(job,result)['decode_tok_s']==2.


def test_scoped_prompt_reuse_does_not_require_unrequested_contexts(tmp_path,monkeypatch):
    from benchmarks import hierarchical_run as runner
    from benchmarks.main_results import save
    prior=tmp_path/'prior'
    rows=[dict(id=str(i),prompt_token_ids=[i]*8) for i in range(3)]
    save(prior/'4b/ctx-32768/prompts.json',rows)
    requested=[]
    def capacity(run,model,ctx):
        requested.append(ctx)
        return 2
    monkeypatch.setattr(runner,'common_batch',capacity)
    data=str((tmp_path/'data').resolve())
    plan=dict(scope=dict(contexts=[32768]),prompt_source=str(prior),
              inputs=dict(data=data),input_sha256={data:'digest'})
    result,path,digest=runner.prepared_prompts(tmp_path/'new',plan,'4b',32768,2)
    assert requested==[32768] and result==rows[:2] and Path(path).exists() and digest


def test_only_reviewed_results_and_no_answer_only(tmp_path):
    jobs=make_jobs(tmp_path,tmp_path/'run',{})
    c=counts(jobs)
    assert c['result_runs']==192 and c['scheduled_jobs']==257
    assert c['by_stage']==dict(kernels=1,smoke=16,capacity=48,throughput=144,ruler=32,longbench=16)
    assert len({j['name'] for j in jobs})==len(jobs)
    assert all('answer-only' not in j['name'] for j in jobs)
    names={j['name'] for j in jobs}
    assert all(set(j.get('depends_on',[]))<=names for j in jobs)
    for j in jobs:
        if j['stage']=='throughput':
            assert len(j['depends_on'])==48
            assert j['required_gpu']==(2 if j['repetition'] in (1,3) else 3)
        if j['stage'] in ('ruler','longbench'): assert len(j['depends_on'])==144


def test_gate_failures_and_gpu_affinity_block_dispatch():
    jobs=[dict(name='gate',state='failed'),dict(name='run',state='pending',depends_on=['gate'],required_gpu=2)]
    assert eligible(jobs,2) is None
    jobs[0]['state']='complete'
    assert eligible(jobs,3) is None
    assert eligible(jobs,2)['name']=='run'


def test_capacity_search_rechecks_graph_memory_at_candidate_batch():
    history=[]
    seen=[]
    while (n:=capacity_candidate(history)) is not None:
        seen.append(n)
        # Per-request graph/scratch allocation lowers the pool at larger batches.
        maximum=51 if n==1 else 48
        history.append(dict(batch=n,maximum_batch=maximum,fits=n<=maximum))
    assert max(r['batch'] for r in history if r['fits'])==48
    assert 49 in seen and len(seen)==len(set(seen))


def test_capacity_initial_failure_cannot_be_hidden():
    with pytest.raises(ValueError,match='batch one'):
        capacity_candidate([dict(batch=1,fits=False,maximum_batch=0)])


@pytest.mark.parametrize('method',METHODS)
def test_fixed_six_native_graph_and_memory_contract(method):
    m=METHODS[method]
    cfg=engine_options(method,batch=17,context=130560,generation=512,checkpoint='ckpt',rope={'factor':4},scheduler='barrier')
    assert cfg['max_model_len']==131072
    assert cfg['dtype']=='bfloat16' and cfg['kv_cache_dtype']=='auto'
    assert cfg['async_scheduling'] is False and cfg['enable_chunked_prefill'] is False
    assert cfg['enable_prefix_caching'] is False and cfg['enforce_eager'] is False
    assert cfg['compilation_config']['cudagraph_mode']=='FULL_DECODE_ONLY'
    assert cfg['hf_overrides']['use_dsa']==m.sparse_draft
    assert cfg['hf_overrides']['molle_verify_fraction']==m.verify_fraction
    assert cfg['hf_overrides']['molle_hierarchical_draft_bits']==m.draft_bits
    assert cfg['speculative_config']['num_speculative_tokens']==6 if m.speculative else cfg['speculative_config'] is None
    assert cfg['compilation_config']['cudagraph_capture_sizes']==([1,17,119] if m.speculative else [1,17])


def test_quality_engine_does_not_use_full_batch_scheduler():
    cfg=engine_options('highlight4-sv25',batch=1,context=129024,generation=1024,checkpoint='ckpt',rope=None,scheduler=None)
    assert 'scheduler_cls' not in cfg


def test_context_limit_is_not_extended_to_fit_speculation():
    with pytest.raises(ValueError):
        engine_options('highlight4-dense8',batch=1,context=130560,generation=519,checkpoint='x',rope=None,scheduler=None)


def fixture_result(tmp_path):
    from benchmarks.multilexsum import digest
    from benchmarks.main_results import sha
    manifest=tmp_path/'prompts.json'
    rows=[dict(id='case',prompt_token_ids=[5]*32768)]
    manifest.write_text(json.dumps(rows))
    result=dict(method='dense8',ctx=32768,batch=1,batches=1,
        capacity=dict(batch=1,fits=True),manifest_path=str(manifest),manifest_sha256=sha(manifest),
        prompt_ids=['case'],prompt_sha256=[digest(rows[0]['prompt_token_ids'])],
        outputs=[dict(token_ids=[12,99],text='summary',finish_reason='stop')],eos_ids=[99],
        execution=dict(target_execution=[dict(kind='graph',requests=1,tokens=1)],
                       target_graph_replays=1,draft_graph_replays=0,peak_reserved_bytes=1234),
        cache=dict(errors={'layer':[0]}),speculation=None,
        timing=dict(decode_steps=[dict(before=[1],after=[2],finished=[True],elapsed_seconds=.5)],
                    decode_window='all_prefills_complete_to_first_request_finish',
                    decode_tokens=1,decode_seconds=.5,decode_tok_s=2.))
    return dict(method='dense8',ctx=32768),result


def test_audit_recomputes_useful_tokens_and_rejects_inflated_rate(tmp_path):
    from benchmarks.main_results import audit_throughput
    job,result=fixture_result(tmp_path)
    assert audit_throughput(job,result)['decode_tok_s']==2.
    result['timing']['decode_tok_s']=4.
    with pytest.raises(ValueError,match='throughput'):
        audit_throughput(job,result)


def test_audit_rejects_changed_input_and_post_eos_tokens(tmp_path):
    from benchmarks.main_results import audit_throughput
    job,result=fixture_result(tmp_path)
    result['prompt_sha256']=['wrong']
    with pytest.raises(ValueError,match='prompts'):
        audit_throughput(job,result)
    job,result=fixture_result(tmp_path)
    result['outputs'][0]['token_ids']=[99,12]
    with pytest.raises(ValueError,match='after EOS'):
        audit_throughput(job,result)


def test_audit_rejects_cache_errors_and_missing_graphs(tmp_path):
    from benchmarks.main_results import audit_throughput
    job,result=fixture_result(tmp_path)
    result['cache']['errors']['layer']=[1]
    with pytest.raises(ValueError,match='cache transaction'):
        audit_throughput(job,result)
    job,result=fixture_result(tmp_path)
    result['execution']['target_execution'][0]['kind']='eager'
    with pytest.raises(ValueError,match='graphs'):
        audit_throughput(job,result)


@pytest.mark.parametrize('text', ['summary', '</think>\n\nsummary', ' \n</think> summary'])
def test_output_format_records_leading_close_without_changing_measurement(tmp_path,text):
    from benchmarks.main_results import audit_throughput
    job,result=fixture_result(tmp_path)
    result['outputs'][0]['text']=text
    before=json.dumps(result,sort_keys=True)
    audit=audit_throughput(job,result)
    assert audit['output_format']['leading_closing_think_markers']==int('</think>' in text)
    assert audit['decode_tokens']==1 and audit['decode_tok_s']==2.
    assert json.dumps(result,sort_keys=True)==before


@pytest.mark.parametrize('text', ['', '  ', '</think>', '</think> \n',
    '<think>reasoning</think>summary', 'reasoning</think>summary',
    '</think></think>summary', '</think><think>reasoning</think>summary'])
def test_output_format_still_rejects_empty_answers_and_reasoning(text):
    from benchmarks.main_results import audit_answer_format
    with pytest.raises(ValueError,match='thinking-off'):
        audit_answer_format(text)


def test_native_shortened_final_verification_is_timed_but_steady_fallback_rejected():
    from benchmarks.main_results import audit_execution
    steps=[dict(before=[501,400]),dict(before=[508,407])]
    execution=dict(target_execution=[dict(kind='graph',requests=2,tokens=14),
        dict(kind='eager',requests=2,tokens=11,max_query_len=7)],draft_graph_replays=12)
    audit_execution(execution,steps,2,True)
    execution['target_execution'].reverse()
    with pytest.raises(ValueError,match='steady decode'):
        audit_execution(execution,steps,2,True)


def test_rejection_can_require_multiple_native_cap_shortened_rounds():
    from benchmarks.main_results import audit_execution
    steps=[dict(before=[506,400]),dict(before=[507,407]),dict(before=[509,414])]
    execution=dict(target_execution=[dict(kind='eager',requests=2,tokens=n,max_query_len=7)
                                    for n in (13,12,10)],draft_graph_replays=18)
    audit_execution(execution,steps,2,True)
    # Merely being near the cap does not permit the wrong query shape.
    execution['target_execution'][0]['tokens']=12
    with pytest.raises(ValueError,match='steady decode'):
        audit_execution(execution,steps,2,True)
    execution['target_execution'][0]['tokens']=13
    execution['target_execution'][0]['max_query_len']=6
    with pytest.raises(ValueError,match='steady decode'):
        audit_execution(execution,steps,2,True)


def test_target_graph_must_match_the_timed_query_shape():
    from benchmarks.main_results import audit_execution
    execution=dict(target_execution=[dict(kind='graph',requests=2,tokens=2)],
                   draft_graph_replays=6)
    with pytest.raises(ValueError,match='query shape'):
        audit_execution(execution,[dict(before=[1,1])],2,True)


@pytest.mark.parametrize('change',['eager_step','missing','duplicate','wrong_batch','wrong_tokens','wrong_steps'])
def test_post_window_graphs_cannot_hide_a_bad_timed_draft_round(change):
    from benchmarks.main_results import audit_execution
    trace=[dict(target_event_index=i,requests=2,tokens=2,steps=6,graph_replays=6)
           for i in (-1,0,1,2)]
    execution=dict(target_execution=[dict(kind='graph',requests=2,tokens=14) for _ in range(3)],
                   draft_graph_replays=1000,draft_execution=trace)
    steps=[dict(before=[1,1]),dict(before=[8,8])]
    audit_execution(execution,steps,2,True,require_draft_trace=True)
    if change=='eager_step':trace[1]['graph_replays']=5
    elif change=='missing':trace.pop(1)
    elif change=='duplicate':trace.append(dict(trace[1]))
    elif change=='wrong_batch':trace[1]['requests']=1
    elif change=='wrong_tokens':trace[1]['tokens']=3
    else:trace[1]['steps']=5
    with pytest.raises(ValueError,match='draft'):
        audit_execution(execution,steps,2,True,require_draft_trace=True)


def test_new_runs_require_trace_without_relabeling_legacy_evidence():
    from benchmarks.main_results import audit_execution
    execution=dict(target_execution=[dict(kind='graph',requests=2,tokens=14)],
                   draft_graph_replays=6)
    steps=[dict(before=[1,1])]
    audit_execution(execution,steps,2,True)
    with pytest.raises(ValueError,match='trace is required'):
        audit_execution(execution,steps,2,True,require_draft_trace=True)
