"""Publish audited throughput into bounded regions of the primary HTML report.

This module is CPU-only and uses the Python standard library. It consumes
summary.json produced by the raw-output auditor; narrative outside the
three named regions is never regenerated.
"""
import argparse
from datetime import datetime, timezone
from html import escape
import json
import math
from pathlib import Path
import tempfile

CONTEXTS = (32768, 65536, 130560)
METHODS = ("direct", "dv", "sv50", "sv40", "sv25")
FFN = ("bf16-full", "int4-full", "bf16-highlight", "int4-highlight")
SOURCE = "../outputs/worktrees/highlight-spec/outputs/evidence/eos-throughput.json"


class ReportChangedError(RuntimeError):
    """A simultaneous prose edit must be preserved and retried next sync."""


def region(text, name, content):
    start, end = f"<!-- report:{name}:start -->", f"<!-- report:{name}:end -->"
    if text.count(start) != 1 or text.count(end) != 1 or text.index(start) >= text.index(end):
        raise ValueError(f"missing or ambiguous HTML region: {name}")
    return text.split(start)[0] + start + "\n" + content + "\n" + end + text.split(end)[1]


def audited_cells(summary):
    cells = summary["cells"]
    for key, cell in cells.items():
        rates = cell["repetitions"]
        if (len(rates) != 3 or any(not math.isfinite(r) or r <= 0 for r in rates)
                or cell["median"] != sorted(rates)[1] or not isinstance(cell["batch"], int) or cell["batch"] < 1):
            raise ValueError(f"invalid three-process aggregate: {key}")
    return cells


def value_cell(cell, dense=None, reference=False):
    if cell is None or (not reference and dense is None):
        return '<td class="num pending" aria-label="Awaiting three audited process repetitions"></td>'
    if dense and cell["batch"] != dense["batch"]:
        raise ValueError("speedup compared different batch sizes")
    rate = cell["median"]
    spread = f"Three process rates: {', '.join(f'{v:,.2f}' for v in cell['repetitions'])} tokens/s"
    if reference:
        return f'<td class="num baseline" title="{spread}">{rate:,.2f}</td>'
    speedup = rate / dense["median"]
    slower = " slower" if speedup < 1 else ""
    return (f'<td class="num speedup{slower}" data-speedup="{speedup:.3f}×" '
            f'data-absolute="{rate:,.2f}" title="{spread}">{speedup:.3f}×</td>')


def workload(plan, state, model, ctx):
    jobs = [j for j in state.get("jobs", plan["jobs"]) if j["kind"] == "eos-throughput"
            and j["model"] == model and j["ctx"] == ctx]
    counts = {j["expected_rows"] for j in jobs}
    if len(counts) != 1:
        raise ValueError(f"mixed batch repetition protocols in {model}/{ctx}")
    count = counts.pop()
    return (f'<small class="retained">Retained measurement · {count} batches/process</small>' if count > 1
            else '<small>One full batch/process · 3 processes</small>')


def table_start(caption, headers):
    return ('<div class="table-wrap" data-filter-table tabindex="0" role="region" '
            f'aria-label="{escape(caption, quote=True)}"><table><caption class="sr-only">{escape(caption)}</caption>'
            '<thead><tr>' + ''.join(f'<th scope="col" class="{"num" if i else ""}">{h}</th>'
                                   for i, h in enumerate(headers)) + '</tr></thead>')


def render(summary, plan, state, timestamp):
    cells = audited_cells(summary)
    total = sum(j["kind"] == "eos-throughput" for j in plan["jobs"])
    complete = summary["complete_runs"]
    if not 0 <= complete <= total:
        raise ValueError("invalid audited run count")
    state_name = state.get("state", summary["state"])
    labels = {"running": "Measurements in progress", "waiting_for_idle_gpu": "Waiting for idle GPUs",
              "complete": "Measurements complete", "failed": "Queue stopped: inspect evidence",
              "draining_after_failure": "Queue stopped: active runs finishing",
              "report_failed": "Report audit needs attention", "prepared_not_launched": "Prepared for measurement"}
    label = escape(labels.get(state_name, state_name.replace('_', ' ').capitalize()))
    date = datetime.fromtimestamp(timestamp, timezone.utc).strftime('%d %b %Y · %H:%M UTC')
    dot = " failed" if "fail" in state_name else ""
    status = (f'<div class="campaign-status"><div><div class="status-heading"><span class="status-dot{dot}"></span>{label}'
              '<span class="tag">B200 · GPUs 2 / 3</span></div>'
              f'<p class="status-detail">Audited snapshot {date} · FFN comparison runs first.</p></div>'
              f'<div class="status-progress"><div class="progress-label"><span>Timing processes audited</span><strong>{complete} / {total}</strong></div>'
              f'<progress value="{complete}" max="{total}" aria-label="Timing processes audited">{complete} of {total}</progress></div></div>')
    blocks = []
    for model in ("4b", "8b"):
        blocks.append(f'<div class="model-block" data-model-block="{model}"><div class="model-heading"><h3>Qwen3-{model.upper()}</h3>'
                      '<span class="model-label">Matched batch comparison</span></div>')
        headers = ["Indexer", "Dense<small>tokens/s</small>", *[f'{label}<small data-throughput-unit>× dense</small>'
                   for label in ("Direct", "Dense verifier", "SV50", "SV40", "SV25")]]
        blocks.append(table_start(f'Qwen3-{model.upper()} EOS-respecting throughput', headers))
        for ctx in CONTEXTS:
            dense_method = "ffn-dense" if model == "8b" and plan.get("single_batch_remaining") else "dense"
            dense = cells.get(f"{model}/{ctx}/{dense_method}/bf16")
            batch = str(dense["batch"]) if dense else "pending"
            blocks.append(f'<tbody data-context-group><tr class="context-row"><th colspan="7" scope="rowgroup">{ctx:,} input tokens'
                          f'<small>Batch {batch}</small>{workload(plan,state,model,ctx)}</th></tr>')
            for fmt in ("bf16", "mxfp4"):
                blocks.append(f'<tr data-result-row data-model="{model}" data-format="{fmt}"><td><span class="format-badge {fmt}">{fmt.upper()}</span></td>')
                blocks.append(value_cell(dense, reference=True))
                for method in METHODS:
                    blocks.append('<td class="num na" title="SV25 was measured only with MXFP4">n/a</td>' if method == "sv25" and fmt == "bf16"
                                  else value_cell(cells.get(f"{model}/{ctx}/{method}/{fmt}"), dense))
                blocks.append('</tr>')
            blocks.append('</tbody>')
        blocks.append('</table></div></div>')
    blocks.append(f'<p class="metric-legend">Dense is the shared reference. Speedups divide unrounded three-process medians. '
                  f'Dots indicate pending results; n/a indicates an untested configuration. <a href="{SOURCE}">Audited values and provenance ↗</a></p>')
    ffn = [table_start('Qwen3-8B FFN quantization with dense BF16 verification',
           ["Input tokens", "Batch", "Dense<small>tokens/s</small>", *[f'{name}<small data-throughput-unit>× dense</small>'
            for name in ("BF16 / full", "INT4 / full", "BF16 / Highlight", "INT4 / Highlight")]])]
    ffn.append('<tbody>')
    for ctx in CONTEXTS:
        dense = cells.get(f'8b/{ctx}/ffn-dense/bf16')
        ffn.append(f'<tr data-result-row data-model="8b" data-format="shared"><td>{ctx:,}</td>'
                   + (f'<td class="num">{dense["batch"]}</td>' if dense else '<td class="num pending" aria-label="Awaiting audited batch"></td>')
                   + value_cell(dense, reference=True))
        ffn += [value_cell(cells.get(f'8b/{ctx}/{method}/bf16'), dense) for method in FFN]
        ffn.append('</tr>')
    ffn.append('</tbody></table></div><p class="metric-legend">One full batch per process; three fresh processes per cell. '
               'All four drafts use BF16 backbone KV and dense BF16 verification. INT4 changes only draft FFN weights. '
               'The main 8B table shares these dense controls.</p>')
    return {"status": status, "throughput": '\n'.join(blocks), "ffn": '\n'.join(ffn)}


def update_report(notes, run):
    notes, run = Path(notes), Path(run)
    summary = json.loads((run / 'summary.json').read_text())
    plan = json.loads((run / 'plan.json').read_text())
    state = json.loads((run / 'status.json').read_text())
    fragments = render(summary, plan, state, (run / 'summary.json').stat().st_mtime)
    before = notes.read_text()
    result = before
    for name, content in fragments.items():
        result = region(result, name, content)
    # Evidence remains a separate, machine-readable artifact; HTML is the report.
    evidence = run.parents[1] / 'evidence' / 'eos-throughput.json'
    evidence.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(summary, plan_sha256=state['plan_sha256'], source_sha256=state['source_sha256'],
                   campaign=str(run), protocol=plan['protocol'])
    atomic_write(evidence, json.dumps(payload, indent=2) + '\n')
    if notes.read_text() != before:
        raise ReportChangedError("report edited during update; retry without overwriting prose")
    if result != before:
        atomic_write(notes, result)


def atomic_write(path, content):
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=f'.{path.name}.', delete=False) as temp:
        temp.write(content)
        name = Path(temp.name)
    name.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    update_report(args.report, args.run)


if __name__ == '__main__':
    main()
