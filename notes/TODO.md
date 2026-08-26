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
- [x] Method: coverage-budgeted drafting designed and implemented as `longspec`, unit tests and parity test green. Design and status in method.md.
- [x] Grid T2-T4: coverage at θ 0.90/0.95/0.98 vs dense and vegas on the five cells. Numbers in handoff.md.

## To do

- [ ] 1. Coverage at matched bytes: θ between 0.90 and 0.95, cap 7% vs 15%, on a quiet node; then T5, the skip ablation.
- [ ] 2. Profile the draft step inside `propose()`: attention, weights, sampler, metadata rebuild; 64K batch 1 and 32K batch 4. Tells where c goes.
- [ ] 3. Parity gate: full budget reproduces dense output. Before any speed claim.
- [ ] 4. Head-to-head vs vegas at matched ratio on the same serial grid: acceptance and tok/s.

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
