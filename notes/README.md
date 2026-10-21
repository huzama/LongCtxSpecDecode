# Research report

Branch `thesis` maintains the [HTML report](https://github.com/huzama/LongCtxSpecDecode/blob/thesis/notes/experiments.html)
and its [editing and publication instructions](https://github.com/huzama/LongCtxSpecDecode/blob/thesis/notes/README.md).
In the shared research checkout, [open the local report](../../../../notes/experiments.html)
in a browser.
This serving worktree holds the implementation, campaign audits and raw results.
The [literature survey](https://github.com/huzama/LongCtxSpecDecode/blob/thesis/notes/literature.yaml)
also lives on `thesis`; no report or survey copy is maintained here.

The five tabs are Setup & Method, Main Results, All Results, Weight Quantization
and Parallel Drafting. Protocols and results each have one authoritative location.

## Next-run checkpoint

- The completed throughput grid has three process repetitions per cell. Raw
  runs and frozen source snapshots under `outputs/` must remain unchanged.
- Outstanding: isolate the attention split change at 8B/64K, batch 19 (one
  split), and 8B/128K, batch 9 (two splits). Compare one and two splits at each
  shape, including output checks, before finalizing cross-context speedup claims.
  That comparison has not run; no split-policy change has been made.
- Quality for the current loader remains unmeasured. Earlier quality results
  in the report retain their original implementation label.
- Prepare a new plan only when requested. Preserve EOS-aware timing, the
  prefill barrier, one common-capacity batch per process and three repetitions.
  GPU authorization and repetition placement belong in that frozen plan.
- The queue requires 60 idle seconds and a cooperative GPU lock. Failed jobs
  stop further dispatch; interrupted jobs require inspection before restart.
  Cleanup does not resume any queue.
- Cleanup verification: 248 CPU tests passed; 45 GPU or opt-in cases skipped.
  All seven serving implementation files still match the validated source
  manifest. B200-1 SSH returned `Permission denied (publickey)` during the
  cleanup; restore access and inspect its queue state before dispatch.
