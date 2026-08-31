# Handoff

What exists and what was measured. Goals and work: [TODO.md](TODO.md). Survey: [literature.yaml](literature.yaml). Install, slurm, benchmark rules: root `CLAUDE.md`.

## Baseline: Vegas

Self-speculative decoding: the target model drafts over a sparse KV subset, verifies over the full cache, losslessly. Its scorer, the rule that picks the subset, is the verify pass's attention scores of two query rows per request, one round stale, at token granularity, with one global ratio (`sparse_attn_ratio`). The fork is vendored at its release (remote `upstream`); their proposer and the overrider plug-in point live in `vllm/v1/spec_decode/sparse_attn/`. Everything of ours lives in `vllm/v1/spec_decode/sparse_attn/longspec/`; a method is one overrider there, registered in the fork's dispatcher. Design: [method.md](method.md).

## Built: vegas on any GPU

Stock vegas is Hopper-only twice: its scores come from a patched FlashAttention-3 kernel (the FA2 op ignores the `scores` argument), and its draft addresses one-token pages (FA2 requires page sizes divisible by 16). A strategy layer under `attn_overrider/` detects what the loaded binary offers and supplies the rest; selection math is untouched, so acceptance is identical on every path.

| Feature | Patched FA3 | Portable | Code |
|---|---|---|---|
| Scores of two query rows per request | written by the kernel | one fused Triton pass over paged K writes the reduced metric directly; no score buffer; one extra K read per verify | `longspec/portable/score_collection.py`, `longspec/kernels/c2q_scores.py` |
| Draft over the selected tokens | page-size-1 table | selection gathered into page-aligned scratch once per propose, one-token append per step | `longspec/portable/draft_kv.py`, `longspec/kernels/draft_gather.py` |
| Choice | `auto` by detection; a forced unavailable path fails at init | | `longspec/portable/kernel_support.py`; config `sparse_attn_score_source`, `sparse_attn_draft_kv` |

Tests: `tests/v1/spec_decode/sparse_attn/longspec/`.

## Built: longspec drafter

The method of [method.md](method.md), as `LongSpecAttnOverrider` in `longspec/`: per-layer, per-request budgets from an attention-mass target, sink and recency reserved, one fused CUDA selection per round over all layers, static attention and whole-layer skip masks, per-layer budget statistics. Two names for one overrider: `coverage` is the selection alone, `longspec` adds the skip masks. Config: `sparse_attn_algorithm="coverage"` or `"longspec"` plus `sparse_attn_theta`, `sparse_attn_sink`, `sparse_attn_recent`, `sparse_attn_skip_attn_layers`, `sparse_attn_skip_layers`. Grid runner: `benchmarks/longspec/grid.py`.

Grid, one pass: Qwen3-4B, one A6000, pg19, greedy, 256 tokens, serial cells, g = 6. Vegas at 7%; coverage (selection alone, no skip masks) with sink 4, recent 64, min 0, cap 15%. 64K and 128K use YaRN (factor 2 and 4 on the 40960 window), 32K none. Decode tok/s; ratio against dense at the same cell; KV read is the mean over layers of the selected fraction of the scored prefix.

| ctx | batch | dense | vegas 7% | coverage θ 0.90 | coverage θ 0.95 | coverage θ 0.98 |
|---|---|---|---|---|---|---|
| 32K | 1 | 49.0 | 45.6 (0.93x), tau 6.38 | 36.6 (0.75x), tau 5.80, 5.5% | 46.6 (0.95x), tau 6.61, 11% | 46.6 (0.95x), tau 6.71, 15% |
| 32K | 4 | 92.5 | 103.8 (1.12x), tau 6.14 | **111.9 (1.21x)**, tau 5.96, 4.7% | 111.1 (1.20x), tau 6.10, 10% | 112.9 (1.22x), tau 6.51, 14% |
| 64K | 1 | 35.8 | 22.2 (0.62x), tau 5.52 | 26.5 (0.74x), tau 5.23, 5.2% | 27.7 (0.77x), tau 5.64, 11% | 28.9 (0.81x), tau 6.05, 15% |
| 64K | 2 | 46.3 | 50.9 (1.10x), tau 5.38 | 48.4 (1.05x), tau 5.11, 5.0% | 51.2 (1.11x), tau 5.44, 11% | 50.2 (1.08x), tau 5.88, 14% |
| 128K | 1 | 23.9 | 19.4 (0.81x), tau 5.43 | 19.6 (0.82x), tau 5.31, 3.9% | 21.8 (0.91x), tau 5.84, 8.6% | 22.1 (0.92x), tau 6.12, 13% |

- Acceptance: θ 0.95 beats vegas's tau at every cell with about 1.5x its bytes; θ 0.90 trails it by 0.1-0.6 with 55-80% of its bytes; θ 0.98 sits at the cap almost everywhere, so it is a flat 15% budget in effect.
- Speed: coverage wins where bytes matter and acceptance is close (32K batch 4: 1.21x vs 1.12x at 4.7% of the KV; 128K batch 1: equal to vegas at 55% of its bytes). At batch 1 the bytes are free and only acceptance counts, so higher θ is simply faster.
- Per-layer profile at θ 0.90: layers 1-3 sit at the cap, layers 5, 7, 11, 23, 25, 30 read under 2%. The profile is stable across context and batch.
- Vegas acceptance at 64K and 128K is far below the baseline pass (tau 5.4-5.5 vs 6.3-6.95) while the 32K cells match; the YaRN configuration of that pass is not recorded. Generations are coherent book continuations at every length. Dense speeds reproduce (within 2% at 32K and 64K, 128K 13% faster).
- Node shared with another user's GPU job from the 32K batch-4 vegas cell onward (load average under 2). Dense cells were unaffected; spec cells at batch 1 may read low by up to ~10%. A quiet-node rerun of the 32K batch-1 row is pending.

Found on the way, fixed in our port: rows padded for a CUDA graph were scored with a block table of -1 (a read outside the cache); the draft's gathered scratch was allocated after vLLM's memory profiling. Found, not fixed: the draft step runs with no CUDA graphs at all (the drafter holds the model loaded before the graph wrapper), which is part of the measured c ≈ 0.8.

## Built: packed verify attention

FA2 packs the query heads of one kv head into the block row dimension only at `seqlen_q == 1`, so the 7-query verify read the KV once per query head: 4x the bytes at 38% occupancy (the round profile's unexplained term). `longspec/verify_attention.py` restores the packing: non-causal prefix call with the group reshaped into rows (`[B*T, Hq, D]` to `[B*T*G, Hk, D]`, split at the last page boundary at or below `seqused_k - T`), tiny causal tail over the last pages, `merge_attn_states` combine; the merged LSE feeds the score reduction. Static shapes, no host reads, FULL-graph safe. Gated per call (uniform multi-query decode shape, FA2, no window/softcap/alibi/sinks); kill switch `sparse_attn_packed_verify`; padded and empty-prefix rows masked so nothing NaNs. Kernel-level test matches the single causal call; parity green.

A/B, coverage θ 0.98 cap 15%, quiet srv09, 512 tokens where marked:

| cell | unpacked | packed |
|---|---|---|
| 32K b4, 256 tok | 111.5, tau 6.44 | **129.6 (+16%)**, tau 6.50 |
| 32K b1, 512 tok | 46.3, tau 6.67 | 47.6, tau 6.89 |
| 64K b1, 512 tok | 30.2, tau 6.01 | 29.9, tau 5.71 |
| 64K b1, 512 tok, fresh prompt (srv07, contended: acceptance only) | tau 6.64 | tau 6.70 |
| 64K b2, 512 tok (srv03) | 55.5, tau 6.38 | 54.9, tau 6.36 |

- Acceptance is trajectory noise, not a systematic loss: the 64K tau gap flips sign on a fresh prompt. Output parity vs dense holds (the emitted tokens are the verify's greedy argmax by construction).
- Batch 1 speed is flat because the prefix launches `B x Hk` blocks and vLLM pins FA2 FULL-graph `num_splits` to 1 (the FA2 wrapper refuses an explicit count above 1). Letting FA2's heuristic split (`num_splits=0`) fills the SMs but moved θ=1 acceptance on Qwen3-0.6B from 0.98+ to 0.946, below the parity gate; parked pending a precision look at the two-call bf16 merge. Small batches stay flat: 64K b2 launches 16 blocks and measures even. The one measured win is 32K b4's 32 blocks; the gain should grow with `B x Hk`.
- A first b2 run on a fresh node looked like a packed collapse (4.8 tok/s). Cause, from a faulthandler stack dump: vLLM's `gather_draft_hidden_states` JIT-compiles its CUDA module on first use, and first use is the first verified round with non-uniform draft counts, so a ~200 s ninja build landed inside the timed window (`~/.cache/torch_extensions` is per node; batch 1 takes the reshape path and never triggers it, which is why only b2 cells showed it, and only the first per node). `SparseAttnProposer.load_model` now builds the module eagerly.

## Profiled: where a round goes

`benchmarks/longspec/round_phases.py`: one cell under vLLM's torch profiler, 5-11 steady full-batch decode rounds, kernels attributed to vLLM's scopes by correlation id. Coverage θ 0.98, cap 15%. GPU ms per round; dense step from the same kind of trace. Kernel times are per-GPU and clean; the node carried four other one-GPU jobs, so CPU-side numbers read high.

| cell | dense step | round GPU busy (steps) | verify (steps) | verify: attention / K scoring / weights | draft per step (steps) | draft: weights / attention / gather | sample + post |
|---|---|---|---|---|---|---|---|
| 32K b1 | 20.3 | 141 (6.9) | 47.3 (2.3) | 31.1 / 4.5 / 11.1 | 15.3 (0.75) | 12.4 / 1.8 / 0.4 | 2.3 |
| 32K b4 | 34.0 | 198 (5.8) | 90.0 (2.6) | 65.0 / 12.3 / 12.1 | 17.7 (0.52) | 12.3 / 3.5 / 1.1 | 2.4 |
| 64K b1 | 27.2 | 185 (6.8) | 83.0 (3.0) | 62.3 / 9.1 / 11.1 | 16.6 (0.61) | 12.4 / 2.8 / 0.8 | 2.3 |

- The verify attention costs 4.4x the dense step's attention over the same KV (31.1 vs 7.5 ms at 32K b1, 62.3 vs 14.4 at 64K b1; 3.3x at 32K b4). Launch geometry from the trace: dense decode `flash_fwd_splitkv` grid (1, 17, 8), FA2 packing the 4 q heads of each kv head into one block (its `seqlen_q == 1` swap) and splitting the KV 17 ways, 136 blocks; the 7-query verify grid (1, 1, 32), one block per q head, no split, 32 blocks on 84 SMs. The draft, one query, packs again: (1, 20, 8), 0.034 ms per layer. So the verify reads the KV four times at 38% occupancy. That is the gap the byte model could not explain; the fix is an attention path with GQA packing and split-KV for small `seqlen_q` (FA2 has the packing only at 1; vLLM's Triton unified attention and FlashInfer pack by construction).
- Weights: every forward reads 8 GB in 11.1-12.4 ms (gemm plus lm_head), 93% of the A6000's bandwidth. The draft step sits on that floor: 12.4 of 15.3 ms at 32K b1. Its attention over the 15% view is 1.8-3.5 ms per step, packed and split; the gather 0.4-1.1 ms.
- Scoring on FA2 is one K read per verify (`_c2q_metric_kernel`): 4.5 / 12.3 / 9.1 ms, 0.2-0.4 dense steps. Selection kernel 0.5-1.0 ms per round. Sampler, rejection sampler and postprocess 2.3 ms.
- CPU side: under the profiler the draft's CPU scope spans the whole round (192 ms against 92 ms of draft GPU work at 32K b1) while the verify replays a graph in 0.7 ms of CPU. Without the profiler the grid's round times sit 2-15% above GPU busy (32K b1 144 vs 141 ms, 64K b1 209 vs 185, 32K b4 231 vs 198, the latter two on a shared node). Launch overhead is second order; the profiler inflates it.
- In dense-step units at 32K b1: verify 2.3 where bytes say 1.2, draft 4.5 where bytes say 4.1, sample 0.1. The excess is the verify attention.

## Fixed: the draft never ran under CUDA graphs

Every draft step since the fork's beginning launched its ~600 kernels one by one from Python. The drafter's dispatcher was initialized with the runner's resolved mode (FULL_AND_PIECEWISE), so uniform-decode draft steps dispatched FULL; but the drafter calls the inner model, which carries only piecewise wrappers, and a mode mismatch makes every wrapper pass through eagerly. Trace evidence: zero `cudaGraphLaunch` in the draft scope of every profile, including the original baselines, against ~590-660 `cudaLaunchKernel` per step. Invisible until W4: the bf16 draft step is GPU-bound at 15.3 ms and hides the ~10 ms of launch CPU; Marlin cut the GPU to ~6 ms and the launch floor became the critical path.

The contract now: the drafter's dispatcher keys are PIECEWISE only and trimmed to the sizes a draft step can reach (one token per request, padded batch bound); a dedicated pass in `capture_model` captures every key inside the capture window with the standard warmup discipline; dummy runs outside the window never capture or replay; `propose` replays. A key miss degrades to eager, never a crash. Effect at 32K b1: bf16 round wall unchanged (GPU-bound), W4 round wall 154 to 98 ms.

Gotcha found on the way: `sparse_attn_draft_weights` must be part of `SpeculativeConfig.compute_hash`. Without it the bf16 and W4 draft copies shared a torch.compile cache directory and loaded each other's inductor artifacts (an arity error at best).

## Measured: W4A16 drafting on coverage (stage B)

`sparse_attn_draft_weights` loads a quantized copy of the target through vLLM's own loader, grafts the target's attention modules into it (KV binding, layer names and overrider call order unchanged), shares embeddings and lm_head, and hands it to the drafter. Verify keeps the target weights; lossless as before, parity green with a bf16 copy. Cost: ~2 GB of Marlin decoder linears. Coverage θ 0.98 cap 15%, RedHat W4A16, same-node pairs, draft graphs fixed:

| cell | bf16 draft | W4 draft | vs dense |
|---|---|---|---|
| 32K b1, 512 tok | 47.5, tau 6.89 | **60.2 (+27%), tau 5.87** | 49.0 dense: **1.23x** |
| 32K b4, 256 tok | 117.4, tau 6.37 | **156.0 (+33%), tau 5.99** | 93.8 dense: **1.66x** |
| 64K b1, 512 tok | 33.3, tau 6.26 | 37.2 (+12%), tau 5.23 | 36.3 dense: 1.02x |

- Acceptance composes about multiplicatively: quant alone costs ~0.07 alpha (stage A 0.93), sparse selection ~0.10, together 0.80-0.84 at 32K. At 64K the composition is harsher (alpha 0.704); tuning θ or the cap for the W4 draft at long contexts is the open lever.
- The W4 round at 32K b1 is 98 ms wall against ~94 ms GPU; the draft is GPU-bound again. Batch 1 finally beats dense.
- More cells, same stack: 64K b2 56.4 to 65.6 (+16%). 64K b1 tune: θ 0.995 cap 15% is best (39.4, alpha 0.754, +18% over bf16); bf16 at θ 0.995 cap 25% reaches alpha 0.935, so at 64K the loss is quantization, not the sparse view. 128K b1: dense 23.5, coverage 21.0, coverage W4 21.8; coverage loses to dense there. Possibly the batch-1 verify occupancy wall, possibly memory starvation from the extra weight copy; unresolved, test before concluding.
- The method is coverage with the W4 draft, one thing. Against vegas as published (its own 7% fixed ratio, same stack): 60.2 vs 46.0 at 32K b1, 156.0 vs 121.7 at b4. Third column, ablation only: W4 grafted onto vegas gives 67.7 at b1 and 146.7 at b4; it separates budget from selection (vegas drafts half the bytes at b1) and motivates the matched-bytes run, coverage tuned to a 7% mean budget.
- 128K b1 losing to dense is not memory starvation: zero preemptions, KV pool 236-250K tokens against the 131K one request needs (peak usage 21%), and the W4 copy only cost the pool 6%. The batch-1 verify occupancy wall stands as the suspect.
- The Blackwell node (sm120) fails before any model code: some engine-init kernel has no sm120 image (torch and vllm `_C` both ship sm_120; a queued probe with blocking launches will name it), and the wheel's FA2 is sm_80 SASS with PTX only, so the method's attention would run through slow PTX JIT there. Parked; benchmarks stay on A6000.

## Measured: matched bytes against vegas

Both sides draft on W4; coverage's θ tuned per context so its mean per-layer budget lands at vegas's fixed 7% (calibration at gen 64 overshoots real decode by ~20%, correct for it). Alpha at matched bytes is the pure selection signal; throughput also carries the packed verify, which only our path has.

| cell | coverage ~7% + W4 (mean) | vegas 7% + W4 |
|---|---|---|
| 32K b1, θ 0.923 | 71.4, tau 6.58, alpha 0.931 (7.3%) | 72.0, tau 6.47, alpha 0.911 |
| 32K b4, θ 0.923 | 154.7, tau 5.54, alpha 0.759 (6.7%) | 143.3, tau 6.06, alpha 0.845 |
| 64K b1, θ 0.926 | 39.1, tau 5.21, alpha 0.704 (7.1%) | 38.4, tau 5.10, alpha 0.683 |
| 64K b2, θ 0.926 | 73.6, tau 5.76, alpha 0.795 (6.7%) | 70.5, tau 5.80, alpha 0.802 |

- Throughput: coverage equal or ahead everywhere. Selection alone: ahead at b1 both contexts, behind at 32K b4 (reproduced twice, same prompts: real, batch-dependent), tied at 64K b2. The paper claim layers accordingly: the system decisively beats vegas as published; the selection row is honest, not triumphant, pending the b4 allocation diagnosis.
- θ is a plateau: pushing 32K b1 from a 5.7% to a 7.3% budget moved alpha 0.925 to 0.931 and cost a little speed. θ near 0.92 beats the pre-W4 default 0.98 (round 90 vs 98 ms at 32K b1); re-baseline the headline grid there.

## Measured: W4A16 control (stage A)

The draft's bottleneck isn't KV anymore; we already killed that with the top-p selection. The draft step is weight-bound (12.4 of 15.3 ms), so weight quant is the lever that composes with coverage, while the verify's full-KV read stays bf16 and lossless.

`benchmarks/longspec/w4_agreement.py` teacher-forces bf16 greedy trajectories (pg19 32K, 256 tokens, 4 prompts) through a quantized copy and reads per-token argmax agreement; that is exactly the acceptance condition of a greedy W4 drafter under a bf16 verify. `tau_sim` walks the agreement in blocks of 6. Speed is the dense grid at 32K with `--model`. Both gates pass.

| model | agreement | late half | tau_sim | b1 | b2 | b4 |
|---|---|---|---|---|---|---|
| bf16 self (sanity) | 0.994 | 0.994 | 6.81 | 48.8 | 72.7 | 94.2 |
| Qwen AWQ | 0.927 | 0.943 | 5.53 | 82.0 (1.68x) | 100.8 | 113.9 (1.21x) |
| RedHat W4A16 (GPTQ) | 0.928 | 0.945 | 5.53 | 81.1 (1.66x) | 102.3 | 115.4 (1.23x) |
| JunHowie GPTQ-Int4 | 0.920 | 0.941 | 5.41 | 81.1 | 102.4 | 115.0 |

- Quantization alone costs about 0.9 of tau (5.5 against the 6.8 bf16 ceiling); agreement is higher in the late half, no long-context decay. The three checkpoints are within 0.008 of each other; the method choice does not matter at 4B.
- Marlin at b1 is 1.66x on the full dense step, consistent with bytes: the step reads 4.8 GB of KV either way, only the 8 GB of weights shrink. The draft step is weight-dominated (12.4 of 15.3 ms), so its gain is larger: about 9 ms of the round per draft step.
- No dequant crossover through b4; W4 stays ahead. At b8 bf16 cannot hold 8 x 33K of KV next to 8 GB of weights and preempts round-robin (9-11 tok/s at 98% pool); the W4 copies free 5.6 GB and fit. Capacity, not kernel speed, but a real effect.
- Composition estimate at 32K b1 on coverage: round 47 ms verify + 6 x ~6.3 ms draft + 2.3 sample = 87 ms at tau ~5.2 (if sparse-KV and quant disagreement compose independently) = ~60 tok/s against 45-47 today. Stage B measures the actual composition.

## Measured

Qwen3-4B, one A6000 (sm86, FA2), pg19, greedy, 256 tokens, serial cells on a quiet node; decode tok/s with prefill separated, ratio against dense at the same cell. The vegas column reproduced within 2% across two passes.

| ctx | batch | dense | vegas | alpha / tau (of 7) |
|---|---|---|---|---|
| 32K | 1 | 49.0 | 43.5 (0.89x) | 0.845 / 6.07 |
| 32K | 4 | 93.8 | **112.6 (1.20x)** | 0.880 / 6.28 |
| 64K | 1 | 36.3 | 33.6 (0.92x) | 0.882 / 6.29 |
| 64K | 2 | 45.9 | **57.0 (1.24x)** | 0.915 / 6.49 |
| 128K | 1 | 21.2 | **24.8 (1.17x)** | 0.991 / 6.95 |

- Vegas acceptance reproduces its paper in every regime, including their own benchmark shape (tau 6.14 vs their ~6.1). Their 1.25-2.81x is 2x H100 at up to 128 concurrent.
- Drafter cost: c ≈ 0.8 of a dense step for a 7% KV read (64K, batch 1); the bandwidth floor is ~0.1. Step overhead, not the mask, is the lever.
- Verification rides along: 7 verify queries cost 1.07-1.49 of one (32K, batch 1).
