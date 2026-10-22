# Research report

Branch `thesis` maintains the [HTML report](https://github.com/huzama/LongCtxSpecDecode/blob/thesis/notes/experiments.html)
and its [editing and publication instructions](https://github.com/huzama/LongCtxSpecDecode/blob/thesis/notes/README.md).
In the shared research checkout, [open the local report](../../../../notes/experiments.html)
in a browser.
This serving worktree holds the implementation, campaign audits and raw results.
The [literature survey](https://github.com/huzama/LongCtxSpecDecode/blob/thesis/notes/literature.yaml)
also lives on `thesis`; no report or survey copy is maintained here.

The five tabs are Method & Setup, Main Results, Supporting Results, Weight Quantization
and Parallel Drafting. Protocols and results each have one authoritative location.

## Next-run checkpoint

- The completed throughput grid has three process repetitions per cell. Raw
  runs and frozen source snapshots under `outputs/` must remain unchanged.
- Attention split dispatch is calibrated across nine workloads and six shapes,
  with three fresh processes per shape. Separate draft/verifier scratch supports
  every captured request batch; source/runtime guards reject stale calibration.
  See `outputs/evidence/attention-tuning.json`. Validation: 255 CPU tests,
  45 GPU tests, 54 calibrated kernel cases, all eight hierarchical methods,
  native FP8, and an 8B/64K batch-19 integration check. These are not new throughput
  results. Earlier measurements retain their original implementation labels.
- The authorized rerun uses B200-2 GPUs 0–3 when idle. See
  `outputs/evidence/main-campaign.json` for frozen plan paths and source hashes.
  Scope: seven methods, 126 throughput processes, 42 quality processes and
  42 adaptive capacity checks. Direct Highlight AR with hierarchical KV and
  native FP8 is excluded by explicit user instruction. Completed retained
  measurements are reused with their original source provenance; model and
  attention computation are unchanged. Finish pending 4B quality before 8B quality.
  The 8B contexts can run concurrently, with matched repetition assignments
  across methods within each context. Quality uses FA4 and RULER 64K/128K.
  LongBench v2 uses the same 60 questions with full-document MoLLE answer-tag
  prompts, an 8,192-token cap, greedy decoding, thinking off and EOS stopping.
  Score the letter in the last complete answer tag; a valid capped answer counts.
- Timing keeps the prefill barrier, EOS, one common-capacity batch per process
  and three repetitions. Each context must pass capacity and calibrated-split
  coverage checks before timing. Do not edit frozen plans or source snapshots.
- The queue requires 60 idle seconds and a cooperative GPU lock. Failed jobs
  stop further dispatch; interrupted jobs require inspection before restart.
  Cleanup does not resume any queue.
