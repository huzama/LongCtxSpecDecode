"""Report publication cannot change unrelated evidence or overwrite edited prose."""
import json
from pathlib import Path

import pytest

from study_report import audits, publish_rows, replace_rows


def test_publication_preserves_other_studies_and_escapes_text(tmp_path):
    page = tmp_path/'report.html'
    before = '<p>Main study stays.</p><table><tr data-study-row="case"><td>old</td><td class="num">1</td></tr></table><p>FFN stays.</p>'
    page.write_text(before)
    publish_rows(page, {'case':['A < B', '1.234×']})
    result = page.read_text()
    assert result.startswith('<p>Main study stays.</p>') and result.endswith('<p>FFN stays.</p>')
    assert 'A &lt; B' in result and '1.234×' in result
    publish_rows(page, {'case':['A < B', '1.234×']})
    assert page.read_text() == result


@pytest.mark.parametrize('text,rows', [
    ('<p>Missing row</p>', {'case':['1']}),
    ('<tr data-study-row="case"><td>1</td></tr>'*2, {'case':['2']}),
    ('<tr data-study-row="case"><td>1</td></tr>', {'case':['2','3']}),
])
def test_missing_ambiguous_or_changed_table_fails_before_writing(tmp_path, text, rows):
    page=tmp_path/'report.html';page.write_text(text)
    with pytest.raises(ValueError): publish_rows(page, rows)
    assert page.read_text() == text


def test_concurrent_edit_is_kept_on_retry(tmp_path, monkeypatch):
    page=tmp_path/'report.html';page.write_text('<tr data-study-row="case"><td>old</td></tr>')
    original = Path.read_text
    reads=0
    def edit_when_checked(path, *args, **kwargs):
        nonlocal reads
        if path==page:
            reads+=1
            if reads==2: page.write_text('<p>User edit.</p>'+original(page))
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',edit_when_checked)
    publish_rows(page, {'case':['new']})
    assert page.read_text().startswith('<p>User edit.</p>') and '>new</td>' in page.read_text()
    assert not list(tmp_path.glob('.report.html.*'))


def test_nonfinite_audit_is_not_published(tmp_path):
    cell=tmp_path/'cell';cell.mkdir()
    (cell/'audit.json').write_text(json.dumps({'measurement':float('nan')}))
    with pytest.raises(ValueError, match='Non-finite'): list(audits(tmp_path))
