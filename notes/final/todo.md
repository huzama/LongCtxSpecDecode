# TODO

Working contract for the current phase. The method is fixed; one question remains.

**The idea.** Self-speculative decoding for long context. The drafter is the target model itself reading a sparse KV view; verification runs full KV and is lossless. No separate draft model, no training, no preparation step.

**The one question: how fast can the drafter get.** Speedup is t/(g*c + 1) with t accepted tokens per round, g draft tokens, c the drafter step cost relative to a dense decode step. t is a property of the model and the budget; c is the only free lever. Break-even at batch 1 needs c below ~0.6 and we measure 0.58-0.64 today (batch 1, 32K, tuned selector path, unpaged, no CUDA graph). Every task below either lowers c or proves that lowering c keeps t.

## Locked decisions

| Decision | Choice | Why |
|---|---|---|
| Speculation | Self-spec, no draft model | A draft model's own KV grows with context; dead in production at long context |
| Drafter | Same weights, sparse KV view | KV reads dominate the long-context decode step |
| Scoring | Training-free min/max block bound | Beats the trained selector past its training length (+0.061 4B, +0.081 8B mean mass efficiency, 32K, batch 1); trained selector stays optional below 16K |
| Selection cadence | Once per draft burst | Amortizes scoring; per-round re-selection gains +21-33% captured mass (32K) and is the follow-up knob |
| Budget policy | Per layer, zero allowed | At 1% density the oracle preserves 0.719 total mass shallow vs 0.855 deep (32K); uniform budgets waste the deep layers' slack |
| Depth axis | Attention-sublayer skip, not whole-layer skip | Both remove the layer's KV read; attention is 3-4x more redundant than MLP, and whole-layer skip caps near 1.5x with 0% oracle skip on Qwen3-8B |
| Verify | Full KV, lossless | Output exactness is the selling point; acceptance is the only quality metric |
| Harness | vLLM, pending first baseline run | HF's q=1 to q=2 verify penalty is a kernel artifact (r(9) 1.10-3.55 HF vs 1.07-1.49 selector kernel, 32K, batch 1). One backend for baseline, drafter, and verify, paged and graph-captured |

## Parked

| Direction | Reason |
|---|---|
| Early exit | Untrained exit is the logit lens: 21.6-38.9% head agreement, ~0.90x end to end |
| Whole-layer skip | Family caps near 1.5x; 0% oracle skip ratio on Qwen3-8B |
| Neuron/width sparsity | Vanishes under batching; head sparsity is the only batch-invariant kind |
| Quantized separate drafter | Reintroduces a second KV; same production objection as any draft model |

## Ordered work

| # | Task | Output | Status |
|---|---|---|---|
| 1 | vLLM harness: dense decode baseline, batch 1-32, 32K-128K | step-time table | next |
| 2 | Sparse drafter step in vLLM, paged and graph-captured, cost vs density | c(density) curve | after 1 |
| 3 | Selection cost at the floor: fp16 block-16 metadata reads 6.25% of KV per scoring event; INT4 and block-32 variants | scoring overhead table | open |
| 4 | Acceptance vs density on the same harness | t(density) curve | open |
| 5 | Per-layer budgets with zero as a value; attention-sublayer skip on top (+21-28% projected, not yet run) | c and t vs allocation | open |
| 6 | End to end vs vanilla vLLM and the strongest self-spec baseline | headline table | after 2 and 4 |
| 7 | Online budget control from measured acceptance | adaptive vs fixed | later |
| 8 | Recycled verification logits vs metadata bound at matched budget | positioning vs Vegas | later |

## Baselines

Vanilla vLLM dense. Vegas. Layer-skip family (Draft and Verify, SWIFT, CLaSp). Sparse-only decode without speculation.

## Pointers

| What | Where |
|---|---|
| Method design | [work/exit-sparse-self-spec-training-free.md](../work/exit-sparse-self-spec-training-free.md) |
| Base design, round structure and cost model | [work/exit-sparse-self-spec.md](../work/exit-sparse-self-spec.md) |
| Selection quality results | [work/results/training-free-selection.md](../work/results/training-free-selection.md) |
| Verify scaling results | [work/results/verify-scaling.md](../work/results/verify-scaling.md) |
| Survey | [work/survey/README.md](../work/survey/README.md) |
