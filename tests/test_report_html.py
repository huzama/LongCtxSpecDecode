"""HTML publication keeps audited units, pending cells and edited prose intact."""
import copy
import json

import pytest

from benchmarks import report_html as report


@pytest.fixture
def snapshot():
    jobs = [dict(kind='eos-throughput', model=m, ctx=c, expected_rows=32 if (m,c)==('4b',32768) else 1)
            for m in ('4b','8b') for c in report.CONTEXTS for _ in range(3)]
    cell = dict(batch=6, median=100., repetitions=[99.,100.,102.])
    summary = dict(state='running', complete_runs=6, cells={
        '8b/32768/ffn-dense/bf16': cell,
        '8b/32768/int4-highlight/bf16': dict(cell, median=250., repetitions=[249.,250.,255.])})
    plan = dict(jobs=jobs, single_batch_remaining=True, protocol={})
    state = dict(state='running', jobs=jobs, source_sha256='source', plan_sha256='plan')
    return summary, plan, state


def test_shared_reference_and_pending_values(snapshot):
    result = report.render(*snapshot, 0)
    assert '2.500×' in result['ffn']
    assert 'data-absolute="250.00"' in result['ffn']
    assert '100.00' in result['throughput']  # main table reuses the FFN dense reference
    assert 'aria-label="Awaiting three audited process repetitions"></td>' in result['throughput']
    assert 'Retained measurement · 32 batches/process' in result['throughput']
    assert 'SV25 was measured only with MXFP4' in result['throughput']


@pytest.mark.parametrize('change', ['partial', 'nonfinite', 'median', 'batch'])
def test_bad_aggregates_cannot_be_published(snapshot, change):
    summary, plan, state = copy.deepcopy(snapshot)
    cell = summary['cells']['8b/32768/int4-highlight/bf16']
    if change == 'partial': cell['repetitions'].pop()
    elif change == 'nonfinite': cell['repetitions'][1] = float('nan')
    elif change == 'median': cell['median'] = 300.
    else: cell['batch'] = 7
    with pytest.raises(ValueError): report.render(summary, plan, state, 0)


def test_update_preserves_manual_text_and_archived_data(tmp_path, snapshot):
    run = tmp_path/'outputs/runs/campaign'; run.mkdir(parents=True)
    for name,value in zip(('summary','plan','status'),snapshot):
        (run/f'{name}.json').write_text(json.dumps(value))
    notes = tmp_path/'experiments.html'
    manual = '<p>My edited research conclusion, including 32K.</p>'
    text = manual + ''.join(f'<!-- report:{name}:start -->old<!-- report:{name}:end -->' for name in ('status','throughput','ffn')) + '<p>Historical table stays here.</p>'
    notes.write_text(text)
    report.update_report(notes,run)
    assert notes.read_text().startswith(manual) and notes.read_text().endswith('<p>Historical table stays here.</p>')
    assert (tmp_path/'outputs/evidence/eos-throughput.json').exists()
    updated = notes.read_text()
    report.update_report(notes,run)
    assert notes.read_text() == updated
    notes.write_text(updated.replace('<!-- report:ffn:start -->',''))
    before = notes.read_text()
    with pytest.raises(ValueError): report.update_report(notes,run)
    assert notes.read_text() == before


def test_concurrent_prose_edit_is_preserved_for_retry(tmp_path, snapshot, monkeypatch):
    run = tmp_path/'outputs/runs/campaign'; run.mkdir(parents=True)
    for name,value in zip(('summary','plan','status'),snapshot):
        (run/f'{name}.json').write_text(json.dumps(value))
    notes = tmp_path/'experiments.html'
    text = ''.join(f'<!-- report:{name}:start -->old<!-- report:{name}:end -->' for name in ('status','throughput','ffn'))
    notes.write_text(text)
    original_write = report.atomic_write
    def edit_during_evidence_write(path, content):
        original_write(path, content)
        if path.name == 'eos-throughput.json':
            notes.write_text('<p>User correction.</p>' + text)
    monkeypatch.setattr(report, 'atomic_write', edit_during_evidence_write)
    with pytest.raises(report.ReportChangedError):
        report.update_report(notes, run)
    assert notes.read_text() == '<p>User correction.</p>' + text
    monkeypatch.setattr(report, 'atomic_write', original_write)
    report.update_report(notes, run)
    assert notes.read_text().startswith('<p>User correction.</p>')
