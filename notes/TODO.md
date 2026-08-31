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

## Done

- [x] Harness: the fork vendored, installed over the prebuilt wheel, no compile.
- [x] Vegas on any GPU: score recompute and draft gather strategies, tests, config knobs.
- [x] Baseline grid: dense and vegas at 32K-128K, batch 1 to saturation, on a quiet node; vegas acceptance reproduces its paper. Numbers in handoff.md.
- [x] Method: longspec drafting designed and implemented as `longspec`, unit tests and parity test green. Design and status in method.md.
- [x] Grid T2-T4: coverage (selection alone) at θ 0.90/0.95/0.98 vs dense and vegas on the five cells. Numbers in handoff.md.
- [x] Parity gate: `test_parity` (θ 1, uncapped) reproduces dense output token for token; `grid.py --parity` checks any cell.
- [x] Packed verify attention: GQA group packed into rows for the verify's prefix, causal tail, LSE merge. +16% at 32K b4, acceptance neutral, lossless; batch 1 blocked on splits (to-do 1). Numbers in handoff.md.
- [x] Profile of the round at 32K b1, 32K b4, 64K b1: the verify attention reads the KV four times at 38% occupancy (FA2 packs GQA only at `seqlen_q == 1`); the draft step sits on the 8 GB weight floor; launch overhead is second order. Numbers in handoff.md.

## To do

- [ ] 1. Batch-1 verify occupancy: the packed prefix launches `B x Hk` blocks and FA2's wrapper refuses an explicit split count. Heuristic splits fill the SMs but pushed θ=1 acceptance to 0.946 on Qwen3-0.6B; check the two-call bf16 merge precision against a single-call fp32 path before adopting. Until then batch 1 keeps the 31 ms verify attention.
- [ ] 2. Draft with W4A16 weights of the same model, bf16 verify: the draft step is 12.4 ms of weight reads out of 15.3; Marlin on sm86 is free at batch 1-8. Worth ~3 dense steps per round at 32K b1 if acceptance holds (literature: 0.9-0.98 per token, unmeasured with sparse KV).
- [ ] 3. Head-to-head at matched bytes: coverage with θ between 0.90 and 0.95, cap 7% vs 15%, on a quiet node. Then T5: longspec, the skip masks.

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
