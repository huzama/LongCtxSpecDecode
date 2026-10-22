# Experimental report

[Open the report](experiments.html). One local HTML document, five tabs, no server or build step.

| Tab                                                | Contents                                                                                              |
| -------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| [Method & Setup](experiments.html#setup-study)     | Method, shared cache, compared configurations, workloads, timing protocol and one provenance appendix |
| [Main Results](experiments.html#main-study)        | Hierarchical-cache throughput, acceptance, memory and task quality from the matching implementation   |
| [Supporting Results](experiments.html#all-study)   | BF16-backbone results, indexer precision, selector timing and attention diagnostics                   |
| [Weight Quantization](experiments.html#ffn-study)  | INT4 FFN throughput, full-projection diagnostic and bounded pre-EOS agreement evidence                |
| [Parallel Drafting](experiments.html#jacobi-study) | Jacobi, carry-over and prompt copying, with separate B200 and A6000 baselines                         |

Keep each tab compact: a short explanation and primary tables stay visible;
cache mechanics, detailed protocols and secondary diagnostics use disclosures.
Keep the report scientific and timeless. Do not add queue state, GPU assignments,
launch counts, progress notices, rerun plans, development chronology or test counts.
Operational records belong under `outputs/`. Retain method details that affect
interpretation and the exact settings needed to reproduce a comparison.

Publish only audited measurements from the stated implementation and protocol.
Main throughput cells require three independent process repetitions. Use N/A for
unavailable valid measurements and — for inapplicable metrics. Preserve raw
outputs and frozen sources; keep superseded timings and post-EOS diagnostics in
the evidence archive, outside the report's performance tables. Never mix them
with the main comparison. Missing cells do not support performance claims.

Edit prose in `experiments.html`; presentation and reader controls live in
`report.css` and `report.js`. Preserve table row identifiers, `report:*` and
`selector-repeatability:*` markers. The empty `report:status` region is a legacy
compatibility anchor and must remain empty. Result publishers must update cells
and provenance without inserting operational status or replacing surrounding prose.

Model/context filters, the Supporting Results indexer filter and the throughput
unit switch apply to labeled tables. Dense references stay in tokens/s. Print/PDF
supports one tab or all five, with filters cleared and details expanded. Keep the
HTML, CSS, JS and `evidence.json` together when sharing.

[Evidence snapshot](evidence.json) retains source hashes and audited records,
including archived studies. Original Markdown is frozen under
[`archive/`](archive/experiments.md). The [literature survey](literature.yaml) is
separate. The serving branch's [experiment checkpoint](https://github.com/huzama/LongCtxSpecDecode/blob/highlight-spec/notes/README.md#next-run-checkpoint)
contains operational instructions; they do not belong in the report.
