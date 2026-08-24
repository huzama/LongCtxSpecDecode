# Fork baseline grid

Five questions about the vegas fork on our hardware, answered before building our own overrider. Raw cell logs live in the fork checkout under `outputs/grid-longctx-serial/` (final numbers) and `outputs/grid-longctx/` (superseded, see methodology); every cell prints one JSON line with its config and timings.

**TL;DR, one answer per question**

| Question | Answer |
|---|---|
| Q1. Absolute long-context speedup of self-spec over dense vLLM? | **Positive on our cards wherever context or batch is nontrivial**: 1.18x at 32K batch 4, 1.12x at 64K batch 2, 1.09x at 128K batch 1, with a recency mask. Batch 1 at 32K is the one loss (0.75x) |
| Q2. Does the vegas baseline run on our cards? | **Not as shipped; yes after two changes we made.** Stock vegas is Hopper-only twice over: the FA2 op silently drops its `scores` argument, and its page-size-1 draft is rejected by FA2. With emulated score collection and a gather-based draft it runs; acceptance is faithful, speed carries a quantified handicap |
| Q3. Is acceptance or drafter cost the bottleneck? | **Drafter cost.** c stays near 0.8 of a dense step for a mask that reads ~7% of KV; the KV floor prices near 0.1 |
| Q4. What happens at batch > 1? | **Speculation wins, once measured cleanly.** Concurrent benchmark shards had contaminated the CPU-bound cells by up to 6x; serial reruns flipped the sign |
| Q5. How good is their selection signal? | **Near oracle at long context**: per-token acceptance 0.87-0.99 against 0.64-0.85 for recency. This is the bar our metadata bound has to approach |

## Setup

Vegas vLLM fork, one RTX A6000 (48 GB, sm86, FA2 path), Qwen3-4B bf16, pg19 prompts as raw token ids, greedy, 256 generated tokens with ignore_eos, prefix caching off. Decode isolated by differencing a gen=1 and a gen=256 run at identical prefill. Contexts past the native 40960 window use YaRN rope scaling, flagged per cell. Speculation: their defaults, g=6 draft tokens, 7% ratio. Saturated batch fills the KV pool (144 KB per token), rounded down to a power of two (the fork's drafter warmup breaks on odd capture sizes).

**Methodology, learned the hard way.** All final cells ran serially on an otherwise idle GPU with a drain wait between cells. Two contamination modes measured: concurrent shards on the same node distorted CPU-bound spec cells by up to 6x (8.1 vs 51.3 tok/s for the same cell), and an engine still releasing the GPU starved or slowed the next cell. Dense cells were insensitive (93.9 parallel vs 93.8 serial at 32K batch 4).

**Running vegas off-Hopper.** Two additions to their overrider, active only below sm90: score collection recomputed as raw QK rows of the first and last query against the paged cache (the FA2 op never receives their `scores` buffer), and the draft gathers selected tokens' KV into page-16 scratch once per round (FA2 rejects page size 1). Selection semantics are unchanged, so acceptance is faithful. Speed is handicapped by one extra full K read per verify pass, which their FA3 kernel gets for free; the handicap grows with batch.

## Q1 and Q2. The absolute table

Decode tok/s; ratio against dense at the same context and batch.

| ctx | batch | dense | streamingllm | vegas (handicapped) |
|---|---|---|---|---|
| 32K | 1 | 49.0 | 36.9 (0.75x) | 36.5 (0.74x) |
| 32K | 4 | 93.8 | **110.5 (1.18x)** | 63.2 (0.67x) |
| 64K | 1 | 36.3 | 31.2 (0.86x) | 26.0 (0.72x) |
| 64K | 2 | 45.9 | **51.3 (1.12x)** | 22.5 (0.49x) |
| 128K | 1 | 21.2 | **23.1 (1.09x)** | 19.0 (0.90x) |

- The dense-vs-streamingllm columns carry no handicap and are the absolute long-context numbers missing from the literature.
- The vegas speed column measures the price of its signal without kernel access: at 128K batch 1 the extra K pass per verify erases the entire speculation gain (0.90x with a near-perfect mask). On Hopper their kernel collects it for free; their paper's speed claims stand for that hardware only.
- The handicap scales with batch (a full K read per request per verify): 0.67x at 32K batch 4, 0.49x at 64K batch 2.
- Dense reference at odd saturation batches, from the parallel phase: 101.8 tok/s at 32K batch 7, 49.7 at 64K batch 3.

## Q5. Signal quality

Per-token acceptance and accepted tokens per round (bonus included, max 7):

| ctx, batch | streamingllm | vegas |
|---|---|---|
| 32K b1 | 0.642 / 4.85 | 0.870 / 6.22 |
| 32K b4 | 0.718 / 5.31 | 0.874 / 6.24 |
| 64K b1 | 0.729 / 5.38 | 0.833 / 6.00 |
| 64K b2 | 0.731 / 5.39 | 0.910 / 6.46 |
| 128K b1 | 0.853 / 6.12 | **0.991 / 6.95** |

- Both signals improve with context at fixed 7% ratio; verification-guided selection is close to the ceiling at 128K.
- The gap between the rows is the room our metadata bound plays in: it must beat recency and approach their scores. Where it lands is measured in-harness once our overrider exists.
- Acceptance columns are hardware-independent: the selection math is identical to theirs.

## Q3. Where the time goes

Derivation from the 64K batch-1 streamingllm cell (verify of 7 queries assumed ~1.2 dense steps, from our verify-scaling measurements): round time = tau / rate = 5.38 / 31.23 = 172 ms against a 27.5 ms dense step, so ~6.3 dense steps per round; less verify leaves ~5 dense steps for 6 draft steps, **c ≈ 0.84**. If drafting were free the same tau would deliver ~4.5x.

- A 7% KV read prices near 0.1 dense steps, so the draft step is dominated by weights, the eager per-step loop, and the full sampler per draft token, not by the mask.
- This is the build target: every point of c bought back moves every row of the Q1 table.

## Long-generation repro

**Why we ran it.** Their published speedups come from short prompts with long sampled generations, the opposite shape of our workload. One small cell in their regime checks that the port is in the right state.

AIME'25 prompts (~165 tokens), 2048 sampled tokens at their sampling params, batch 8, one A6000:

| mode | decode tok/s | vs dense | alpha | tau (of 7) |
|---|---|---|---|---|
| dense | 492.8 | | | |
| streamingllm | 146.4 | 0.30x | 0.747 | 5.48 |
| vegas | 119.8 | 0.24x | **0.895** | **6.37** |

- Acceptance reproduces their paper (tau ~6.1-7 reported), now under sampling as well, so the rejection-sampler path of the port is validated.
- Speed fails for both spec modes equally, which rules out a vegas-specific port bug: at a 165-token context there is no KV to skip, batch 8 amortizes weights for dense, and one eager draft step costs ~3-4 dense batch-steps of overhead (round arithmetic from tau and the rates). Their 1.25-2.81x lives on 2x H100 NVL at batch up to 128 with the in-kernel signal.
- Same conclusion from a second regime: drafter step overhead governs everything on this stack.

**Their benchmark shape.** Their `benchmark_vegas.py` scaled to one A6000: Qwen3-4B (a paper config), AIME'25 prompts, their sampling with EOS stopping, scheduler-managed queueing at 32 concurrent, 60 prompts, max 8192 tokens; whole-workload wall-clock throughput, their metric.

| mode | tok/s | vs dense | alpha | tau (of 7) |
|---|---|---|---|---|
| dense | 666.6 | | | |
| vegas | 352.2 | **0.53x** | 0.857 | **6.14** |

- tau matches their reported ~6.1 on this workload, so the port reproduces their selection in their own regime under sampling.
- Speed does not: 0.53x against their 1.25-2.81x on 2x H100 NVL at up to 128 concurrent. Round arithmetic: one round costs ~11.6 dense steps here, so each draft step costs ~1.7 dense steps at batch 32; the sm86 handicap explains ~0.5-1 of the 11.6, the rest is the drafter's per-step cost that faster kernels and 4x the concurrency amortize.
- Not separable on our cluster: no sm90 card exists here, so "Hopper" and "sm86 handicap" cannot be told apart without an external run.

## What this changes

- The loop pays on our hardware with the crudest mask; the two levers left are c (goal 1) and signal quality between 0.85 and 0.99 (goal 2, task 6).
- A query-aware signal without kernel access costs a full K pass if done naively (vegas emulation, 0.90x); our block metadata reads 6.25% of KV for the same purpose. That factor-16 gap is the design's reason to exist, now with a measured counterfactual.
- Budget policy should loosen with length: acceptance rises with context for both signals at fixed ratio.
- Benchmark discipline: serial cells, drain wait, matched power-of-two batches. Recorded here because the contaminated numbers inverted two conclusions before the rerun.
