# Experimental report

[Open the report](experiments.html): one HTML document with five tabs.
It works locally in a browser, without a server or build step.

| Tab                                                | Contents                                                                                                        |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| [Setup & Method](experiments.html#setup-study)     | Shared model, cache updates, baselines, settings and measurement protocol                                       |
| [Main Results](experiments.html#main-study)        | Hierarchical KV performance and task quality; unmeasured cells marked N/A                                       |
| [All Results](experiments.html#all-study)          | Full throughput, task quality, hierarchical KV, indexer precision, selector comparisons and historical evidence |
| [Weight Quantization](experiments.html#ffn-study)  | INT4 FFN results, full-projection diagnostic and supporting evidence                                            |
| [Parallel Drafting](experiments.html#jacobi-study) | Jacobi, carried predictions, prompt retrieval and separate hardware comparisons                                 |

Edit prose directly in `experiments.html`. All Results retains the earlier studies
and implementation evidence. Main Results contains the current hierarchical-cache
method and its controls. Publish only audited results from the matching backend;
keep superseded timings in raw evidence. Cells with `data-result-source` refresh
from their linked tables on page load; keep static values current as well.
Use N/A for unmeasured Main Results cells and preserve original placeholders in
All Results. Shared protocols use links that open the appropriate tab. `report.css` controls
presentation and `report.js` provides tab navigation, persistent model/context
filters, indexer filtering, throughput units, mobile contents and printing.
Print/PDF supports the current tab or the whole report, with all configurations
and expanded evidence. Keep the HTML, CSS and JS together when sharing.

The serving worktree's `benchmarks.report_html` publishes audited throughput
inside the `report:status` and `report:throughput` regions of All Results and
`report:ffn` in Weight Quantization. The stopped campaign snapshot is collapsed under historical
results; it is not a live job monitor. New timing cells require three audited
process repetitions.

`benchmarks/study_report.py` publishes saved Jacobi audits into `data-study-row`
rows in the same file. Preserve these row identifiers, all `report:*` markers
and all `selector-repeatability:*` markers. Publishers preserve surrounding prose.

Raw evidence stays under `outputs/`. Original Markdown is frozen under
[`archive/`](archive/experiments.md); the [literature survey](literature.yaml)
remains separate. Historical measurements retain their original protocols.
