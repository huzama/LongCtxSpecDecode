"""Scope and failure barriers for the user-authorized 4B-first quality queue."""
from pathlib import Path

import pytest

from benchmarks import main_quality as quality
from benchmarks.main_results import save, load, sha


def test_quality_scope_preserves_cohorts_and_excludes_answer_only():
    jobs = quality.make_jobs(Path('/runtime'), Path('/run'), '4b')
    assert len(jobs) == len({j['name'] for j in jobs}) == 27
    assert {(j['stage'], j['ctx']) for j in jobs} == {('ruler',65536),('ruler',129024),('longbench',None)}
    for method in quality.QUALITY_METHODS:
        group = [j for j in jobs if j['method'] == method]
        assert len(group) == 3 and sum(j['expected_rows'] for j in group) == 444
    assert all(j['model'] == '4b' and 'required_gpu' not in j for j in jobs)
    assert all('answer-only' not in j['name'] for j in jobs)


@pytest.mark.parametrize('failure', [None, '4b-quality', '8b-throughput'])
def test_pipeline_never_advances_past_failed_stage(tmp_path, monkeypatch, failure):
    paths = {name: tmp_path/name for name in ('4b-quality','8b-throughput','8b-quality')}
    for path in paths.values():
        save(path/'plan.json',dict(source_root='/frozen',runtime_root='/runtime'))
    save(tmp_path/'pipeline.json',dict(throughput_run=str(paths['8b-throughput']),
        throughput_plan_sha256=sha(paths['8b-throughput']/'plan.json'),
        quality_runs={m:str(paths[m+'-quality']) for m in ('4b','8b')}))
    seen = []
    class Process:
        pid = 123
        def __init__(self, command, **kwargs):
            self.run = Path(command[-1]);seen.append(self.run.name)
        def wait(self):
            failed = self.run.name == failure
            save(self.run/'status.json',dict(state='failed' if failed else 'complete',report_exit_code=0))
            return int(failed)
    monkeypatch.setattr(quality.subprocess,'Popen',Process)
    if failure:
        with pytest.raises(RuntimeError,match='pipeline stopped'):
            quality.pipeline(tmp_path)
        assert load(tmp_path/'pipeline-status.json')['state'] == 'failed'
    else:
        quality.pipeline(tmp_path)
        assert load(tmp_path/'pipeline-status.json')['state'] == 'complete'
    assert seen == list(paths)[:list(paths).index(failure)+1] if failure else seen == list(paths)


@pytest.fixture
def recovery_parent(tmp_path, monkeypatch):
    parent, source, runtime = (tmp_path/name for name in ('parent', 'source', 'runtime'))
    (source/'benchmarks').mkdir(parents=True)
    (source/'benchmarks/main_quality.py').write_text('# prior launcher\n')
    (source/'highlight_spec').mkdir()
    (source/'highlight_spec/model.py').write_text('# unchanged serving code\n')
    jobs = quality.make_jobs(runtime, parent, '8b')
    save(parent/'plan.json', dict(schema='main-quality-v1', host='old-container', model='8b',
        source_root=str(source), source_sha256=quality.source_digest(source), runtime_root=str(runtime),
        runtime={}, inputs={}, input_sha256={}, protocol={'speculative_tokens':6}, jobs=jobs,
        allowed_gpus=[2,3], report_module='benchmarks.main_quality'))
    save(parent/'status.json', dict(jobs=[dict(j,state='complete' if i==0 else 'running' if i==1 else 'pending')
                                        for i,j in enumerate(jobs)]))
    dest=parent/jobs[0]['name']
    names=('config.json','results.jsonl','finished.json','cache.json')
    for name in names:save(dest/name, {})
    partial=parent/jobs[1]['name']/'results.jsonl'
    save(partial, {'partial':True})
    monkeypatch.setattr(quality, '_verify_payload', lambda plan: None)
    monkeypatch.setattr(quality, 'validate', lambda job, path: dict(raw_sha256={n:sha(path/n) for n in names}))
    return parent, tmp_path/'recovery', partial


def test_recovery_preserves_evidence_and_only_queues_unfinished(recovery_parent):
    parent, out, partial=recovery_parent
    originals={p: p.read_bytes() for p in parent.rglob('*') if p.is_file()}
    result=quality.prepare_recovery(parent,out,[0,1,2,3])
    assert (result['preserved'],result['remaining'])==(1,26)
    plan=quality.verify_plan(out)
    assert plan['allowed_gpus']==[0,1,2,3]
    assert not set(plan['recovery']['completed_audits']) & {j['name'] for j in plan['jobs']}
    assert partial.exists() and all(p.read_bytes()==value for p,value in originals.items())
    assert all(j['command'][-3]==str(out) for j in plan['jobs'])


def test_nested_recovery_restricts_gpus_without_repeating_completed_runs(recovery_parent):
    parent, out, _ = recovery_parent
    quality.prepare_recovery(parent, out, [0, 1, 2, 3])
    plan = load(out/'plan.json')
    jobs = plan['jobs']
    save(out/'status.json', dict(jobs=[dict(j, state='complete' if i==0 else 'failed')
                                     for i,j in enumerate(jobs)]))
    for name in ('config.json','results.jsonl','finished.json','cache.json'):
        save(out/jobs[0]['name']/name, {})
    frozen = {p:p.read_bytes() for p in out.rglob('*') if p.is_file()}
    child = out.parent/'restricted-recovery'
    result = quality.prepare_recovery(out, child, [2, 3])
    checked = quality.verify_plan(child)
    assert result['remaining'] == 25 and checked['allowed_gpus'] == [2, 3]
    assert {j['name'] for j in checked['jobs']} == {j['name'] for j in jobs[1:]}
    assert all(p.read_bytes()==value for p,value in frozen.items())
    original = load(parent/'plan.json')['jobs'][0]['name']
    assert original not in {j['name'] for j in checked['jobs']}


@pytest.mark.parametrize('change', ['protocol','source','preserved_result','parent_status'])
def test_recovery_rejects_changed_protocol_source_or_evidence(recovery_parent, change):
    parent,out,_=recovery_parent
    quality.prepare_recovery(parent,out,[0,1,2,3])
    plan=load(out/'plan.json')
    if change=='protocol':
        plan['protocol']={'speculative_tokens':5};save(out/'plan.json',plan)
    elif change=='source':
        (out/'source/highlight_spec/model.py').write_text('# changed\n')
    elif change=='preserved_result':
        name=next(iter(plan['recovery']['completed_audits']))
        save(parent/name/'finished.json',{'changed':True})
    else:
        state=load(parent/'status.json');state['jobs'][1]['state']='complete';save(parent/'status.json',state)
    with pytest.raises(ValueError):quality.verify_plan(out)
