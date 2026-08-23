# Verification Scaling

Two questions about the cost of verification, which decides whether speculation can pay at all in the regime this method targets. Raw numbers under `outputs/` (slugs `prelim-verify-scaling-*`, `prelim-draft-cost`); every run archives its config and launch command. Qwen3-4B on one A6000.

**TL;DR, one answer per question**

| Question | Answer |
|---|---|
| Q1. Does a verify pass over many query tokens cost more than a single-token decode step? | **Only on a backend that switches kernels at q > 1.** With one kernel handling any query length, a 9-token pass costs 1.07 to 1.49x a 1-token step in seven of eight cells. On the HuggingFace path the same pass costs 1.10 to 3.55x |
| Q2. Is the draft then cheap enough to break even? | **Not at batch 1.** Break-even needs a draft under about 0.6 of a decode step; the tuned sparse decode measures 0.58 to 0.64. It is a coin flip there, and the margin has to come from batch |

## Why this matters

Every long-context speculative method rests on the claim that verification is memory-bound, so the extra query tokens ride along on a KV read that happens anyway. Two published results contradict it: a production RL framework measures roughly 50% rollout throughput loss with speculation enabled and tells users to leave it off, and a rollout system reports no or negative gain at batch 256. If verification is genuinely compute-bound at scale, no draft design rescues the round and the whole direction fails.

The claim is one measurable ratio, r(q) = t(q) / t(1), the cost of a full-KV forward over q query tokens against the cost of one. It needs no drafter, so it can be measured before anything is built.

## Q1. Does verification scale with query count?

**What we did.** Build a KV cache of a given length and batch, then time forward passes of q in {1, 2, 4, 8, 9, 16} against it, restoring the cache after every call so each repeat sees the same length. Median of seven, after warmup, CUDA-synchronised. Two backends: stock HuggingFace, and a Triton dense kernel that handles arbitrary query length through one path.

**Answer: the penalty is a property of the kernel path, not of the hardware.**

Cost of a 9-token verify pass relative to a 1-token decode step:

| Batch | Context | HuggingFace | Triton dense |
|---|---|---|---|
| 1 | 4K | 1.10 | 1.07 |
| 4 | 4K | 1.94 | 1.08 |
| 16 | 4K | 2.77 | 1.23 |
| 1 | 8K | 1.43 | 1.17 |
| 4 | 8K | 2.42 | 1.38 |
| 16 | 8K | 2.94 | 1.19 |
| 1 | 32K | 3.40 | 2.62 |
| 4 | 32K | 3.55 | 1.49 |

- **On the HuggingFace path the jump is entirely from q=1 to q=2**, after which more query tokens are nearly free (r(16) / r(2) is about 1.05). That is the signature of a path switch: one kernel serves single-token decode, a different and less efficient one serves everything else. It is not compute saturation. At batch 1 and 32K the pass sits ten times off the memory roofline while its arithmetic accounts for about 1.4 ms.
- **On a kernel with no such switch the ratio collapses to 1.07 to 1.49** in seven of eight cells, which is the memory-bound behaviour the theory predicts.
- **This is the most likely explanation for the published negative results.** A framework whose verify path costs three times its decode path needs an accepted length above three just to break even. That is an engineering property of the engine, not a limit of the method, and it is fixable.
- One cell resists: batch 1 at 32K stays at 2.62 even on the good kernel. Batch 1 is where the weight term dominates and there is least to amortise.

## Q2. Is the draft cheap enough?

**What we did.** Turn the measured verify ratio into the break-even draft cost. With round cost gamma·t_draft + t_verify and tau accepted tokens, speedup exceeds 1 when the draft costs less than (tau − r) / gamma of a decode step. Using tau = 6.13 and gamma = 8.

**Answer: the margin is thin at batch 1, and the tuned sparse decode sits right on the line.**

| Setting | Break-even draft cost | Measured sparse decode |
|---|---|---|
| batch 1, 8K | under 0.62 | 0.64 |
| batch 1, 32K | under 0.44 | 0.58 |
| batch 4, 32K | under 0.58 | not measured at this batch |

- At batch 1 a sparse draft is not cheap enough, because sparsity cuts only the KV term while weights still dominate. This agrees with the byte split: KV is 7 to 38% of decode bytes at batch 1, so cutting it hard still leaves most of the step.
- The margin has to come from batch, where KV grows to 91 to 99% of bytes and the sparse draft cost falls toward its coverage fraction. That is the untested cell, and it is now the single most valuable measurement left.
- Our own harness cannot supply it yet. Its sparse path runs unpaged, without CUDA graphs, with top-p's data-dependent budget, and measures 82 to 165 ms against 39 to 134 ms dense, which is slower than dense and clearly untuned. Those numbers are recorded but no conclusion rests on them.
- The tuned figures above come from the selector repo's own benchmark, which is paged, graph-captured, and fixed-budget. That benchmark takes no batch argument, so it cannot be pointed at the missing cell either. Closing the gap means bringing the paged, graph-captured sparse path into our harness, which is a build task rather than a run.

## What this changes

- The crossover risk moves from physics to engineering. Verification is memory-bound where it matters, provided the engine has one kernel for any query length.
- Any speedup claim we make must state which verify kernel it used, since the same model on the same GPU differs by up to 3.5x on this axis alone.
- The next measurement is tuned sparse decode at batch 4 to 32, which turns the break-even table into a verdict. It requires building the paged, graph-captured sparse path into our harness; neither our current path nor the selector repo's batch-1 benchmark can produce it.
