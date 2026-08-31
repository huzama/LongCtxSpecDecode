# Make the drafter faster, and better than Vegas

Two goals, nothing else gets built.

1. **Make the drafter faster.** Speedup is t/(g*c + 1): t accepted tokens per round, g draft tokens, c the drafter step cost relative to a dense step. t is set by the model and the budget; c is the only free lever.
2. **Beat Vegas** (arXiv 2602.07223), the baseline for this loop, inside its own harness.

Pointers: [handoff.md](handoff.md) for what exists and what is measured, [literature.yaml](literature.yaml) for the survey, branch `survey-and-prototypes` for everything before the fork.

## Fixed

| | Choice | Why |
|---|---|---|
| Speculation | Self-spec, no draft model | A draft model's KV grows with context |
| Drafter | Same weights, sparse KV view | KV reads dominate the long-context decode step |
| Verify | Full KV, lossless | Exactness is the selling point; acceptance is the only quality metric |
| Harness | Vegas vLLM fork, vendored here | Proposer, scheduler wiring, rejection sampler, CUDA-graph dispatch inherited; a method is one attention overrider |
| Workload | Long context, short generation: pg19 prompts 32K-128K, up to 256 generated tokens | Their AIME benchmark is the opposite shape |
| Model | Qwen3-4B | In their paper's lineup; fits one 48 GB card at 128K |
| Parked | Early exit, whole-layer skip, neuron sparsity, separate drafter | literature.yaml, ideas_rejected |
| Parked | FP8 for the gathered draft scratch | The scratch is a copy of the one bf16 cache, not a second cache; with W4 weights the draft attention is ~1.8 of a ~6 ms step, so the saving is ~5% of the round with acceptance risk attached |
| Chosen, not started | Budget-driven dynamic attention skip; dynamic draft length | The two remaining method components; scheduled after the engine items |

## Done

- [x] Harness: the fork vendored, installed over the prebuilt wheel, no compile.
- [x] Vegas on any GPU: score recompute and draft gather strategies, tests, config knobs.
- [x] Baseline grid: dense and vegas at 32K-128K, batch 1 to saturation, on a quiet node; vegas acceptance reproduces its paper. Numbers in handoff.md.
- [x] Method: longspec drafting designed and implemented as `longspec`, unit tests and parity test green. Design and status in method.md.
- [x] Grid T2-T4: coverage (selection alone) at θ 0.90/0.95/0.98 vs dense and vegas on the five cells. Numbers in handoff.md.
- [x] Parity gate: `test_parity` (θ 1, uncapped) reproduces dense output token for token; `grid.py --parity` checks any cell.
- [x] Packed verify attention: GQA group packed into rows for the verify's prefix, causal tail, LSE merge. +16% at 32K b4, acceptance neutral, lossless; b1 and 64K b2 flat, blocked on splits (to-do 1). Numbers in handoff.md.
- [x] Benchmark hazard fixed: the draft-probs gather module JIT-compiled at first use, mid-decode on any fresh node at batch 2+; it now builds at model load. Details in handoff.md.
- [x] W4A16 draft weights, stage A control: agreement 0.92-0.93, Marlin 1.66x at b1, no dequant crossover through b4, AWQ = GPTQ at 4B. Numbers in handoff.md.
- [x] W4A16 draft weights, stage B: `sparse_attn_draft_weights` drafts on a quantized copy sharing the target's attention, embeddings and lm_head. 60.2 tok/s at 32K b1 (1.23x dense), 156.0 at 32K b4 (1.66x dense). Numbers in handoff.md.
- [x] The draft never ran under CUDA graphs: FULL dispatch against an inner model with only piecewise wrappers ran every draft step eagerly since the fork's beginning. Fixed with piecewise-only keys, a dedicated capture pass and phase-aware dummy runs; W4 round 154 to 98 ms at 32K b1. Details in handoff.md.
- [x] Profile of the round at 32K b1, 32K b4, 64K b1: the verify attention reads the KV four times at 38% occupancy (FA2 packs GQA only at `seqlen_q == 1`); the draft step sits on the 8 GB weight floor; launch overhead is second order. Numbers in handoff.md.
- [x] Matched bytes vs vegas, both on W4, coverage tuned to a ~7% mean budget: coverage throughput equal or ahead in all four cells; alpha ahead at b1 both contexts, behind at 32K b4, tied at 64K b2. θ near 0.92 is a flat, better operating point than 0.98 under W4. Numbers in handoff.md.

## To do

- [ ] 1. The b4 acceptance deficit: at matched bytes coverage loses ~0.09 alpha to vegas at 32K b4 (0.759 vs 0.845, reproduced twice, same prompts), while winning at b1 both contexts. Suspect per-request budget skew when requests share the selection pass. Dump per-request budgets and acceptance in one instrumented 32K b4 cell; if the allocation skews, a per-request floor may clean the selection claim at batch.
- [ ] 2. Small-batch verify occupancy: the packed prefix launches `B x Hk` blocks (8 at b1, 16 at b2, both flat) and FA2's wrapper refuses an explicit split count. Heuristic splits fill the SMs but pushed θ=1 acceptance to 0.946 on Qwen3-0.6B; check the two-call bf16 merge precision against a single-call fp32 path before adopting. Until then batch 1 keeps the 31 ms verify attention, and 128K b1 may be losing to dense on it.
- [ ] 3. 128K b1 coverage below dense (21.0 vs 23.5): memory starvation ruled out from the run logs (zero preemptions, pool at 21% peak); the verify-occupancy wall (to-do 2) is the standing suspect. Confirm by profiling a 128K round after any occupancy fix.
- [ ] 4. Quant loss at long context: 64K W4 best is θ 0.995 cap 15% (39.4, alpha 0.754) while bf16 reaches 0.935 there; run the agreement control at 64K to isolate the quant decay, then decide W4 vs W8 for long contexts.

## Openings against Vegas

| Vegas limitation | Fact |
|---|---|
| Signal needs a patched FA3 kernel, even at prefill | on other GPUs it costs one K read per verify |
| Score buffers scale with context | batch x heads x 2 x max_len, kernel path |
| Token-granular draft, page size 1 | FA3 only; gathered into 16-token pages elsewhere |
| Signal one round stale | selection comes from the previous verify |
| Budget fixed, global, chosen by offline sweep | `sparse_attn_ratio`; tables rebuild every propose, so runtime changes are graph-safe |

## Risks

| Risk | Handling |
|---|---|
| Fork frozen at its base vLLM | pin for the whole project |
| Layer tracking by kernel-call count breaks on non-uniform models | stay on Qwen3 |
| Parity subtle across spec and non-spec paths | item 3 before any speed claim |
| Shared nodes distort CPU-bound cells by up to 6x | serial cells on an idle node; power-of-two batches; prefix caching off |
