"""Publish saved Jacobi audits into identified rows of the unified HTML report."""
from html import escape
import json
import math
import os
from pathlib import Path
import re
import tempfile


def replace_rows(text, rows):
    for key, values in rows.items():
        pattern = re.compile(r'(<tr\b[^>]*\bdata-study-row="' + re.escape(key) + r'"[^>]*>)(.*?)(</tr>)', re.S)
        matches = list(pattern.finditer(text))
        if len(matches) != 1:
            raise ValueError(f"Missing or ambiguous report row: {key}")
        match = matches[0]
        cells = re.findall(r'<td\b([^>]*)>.*?</td>', match[2], re.S)
        if len(cells) != len(values):
            raise ValueError(f"Column count changed: {key}")
        rendered = []
        for attrs, value in zip(cells, values):
            classes = ['num'] if re.search(r'class="[^"]*\bnum\b', attrs) else []
            if not value:
                classes.append('empty-cell')
            elif value.endswith('×'):
                classes.append('speedup')
                if float(value[:-1]) < 1:
                    classes.append('slower')
            rendered.append(f'<td class="{" ".join(classes)}">{escape(value)}</td>')
        text = text[:match.start()] + match[1] + '\n' + '\n'.join(rendered) + '\n' + match[3] + text[match.end():]
    return text


def publish_rows(path, rows):
    """Preserve other studies and concurrent prose edits; never rewrite the page."""
    path = Path(path)
    if not rows:
        return
    for _ in range(3):
        before = path.read_text()
        after = replace_rows(before, rows)
        if after == before:
            return
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=f'.{path.name}.', delete=False) as temp:
            temp.write(after)
            temporary = Path(temp.name)
        try:
            if path.read_text() != before:
                continue
            os.chmod(temporary, path.stat().st_mode & 0o777)
            temporary.replace(path)
            return
        finally:
            temporary.unlink(missing_ok=True)
    raise RuntimeError('Report changed during publication; retry without replacing edited text')


def audits(run):
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('Non-finite audited measurement')
        if isinstance(value, dict):
            for child in value.values(): finite(child)
        elif isinstance(value, list):
            for child in value: finite(child)
    for file in sorted(Path(run).glob('*/audit.json')):
        report = json.loads(file.read_text())
        finite(report)
        yield report


def publish_jacobi(path, run):
    rows = {}
    for report in audits(run):
        model, length = report['model'], report['haystack']
        for method, result in report['methods'].items():
            label = {'ar':'AR', 'jacobi':'Jacobi', 'lookahead':'Lookahead'}[method]
            timed = result['timing_valid']
            rows[f'jacobi-{model.lower()}-{length}-{method}'] = [model, f'{length:,}', label,
                f"{result['matching']}/{result['rows']}", f"{result['score']*100:.2f}",
                f"{result['prefill_seconds']:.3f}" if timed else '',
                f"{result['decode_seconds']:.3f}" if timed else '',
                f"{result['speedup']:.3f}×" if timed else '',
                f"{result['tokens_per_call']:.3f}", str(result['query_tokens']),
                f"{result['peak_allocated_bytes']/2**30:.3f}"]
        diagnostic = report.get('first_update')
        if diagnostic:
            count = diagnostic['blocks']
            rows[f'first-update-{model.lower()}-{length}'] = [model, f'{length:,}', str(count),
                f"{diagnostic['prefix_sum']/count:.3f}" if count else 'n/a',
                f"{100*diagnostic['full_matches']/count:.2f}" if count else 'n/a']
    publish_rows(path, rows)


def publish_parallel(path, run):
    rows = {}
    for report in audits(run):
        if 'matching' not in report:
            continue
        model, length, timed = report['model'], report['haystack'], report['timing_valid']
        rows[f'parallel-{model.lower()}-{length}'] = [model, f'{length:,}',
            f"{report['matching']}/{report['rows']}", f"{report['score']*100:.2f}",
            f"{report['decode_seconds']:.3f}" if timed else '',
            f"{report['speedup']:.3f}×" if timed else '',
            f"{report['accepted_per_round']:.3f}", f"{report['emitted_per_round']:.3f}",
            str(report['forward_calls']), f"{report['peak_allocated_bytes']/2**30:.3f}"]
    publish_rows(path, rows)
