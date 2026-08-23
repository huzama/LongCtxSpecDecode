# Training-Free Selection

Three questions about replacing the trained block-sparse selector with a scorer that needs no training and no preparation, which is what [exit-sparse-self-spec-training-free.md](../exit-sparse-self-spec-training-free.md) depends on. Raw numbers under `outputs/` (slug `prelim-scorer-*`); every run archives its config and launch command. Batch is 1 throughout.

**TL;DR, one answer per question**

| Question | Answer |
|---|---|
| Q1. What does dropping the trained selector cost? | **It depends on context length, and the sign flips.** Inside the selector's training length the trained scorer is ahead by 0.064 mean efficiency; at twice that length the training-free bound is ahead by 0.061 (4B) and 0.081 (8B) |
| Q2. Is that an artifact of how we measured? | **No.** Block alignment falsified in 18 of 18 cells, and switching the oracle's head aggregation moves the gap from +0.084 to +0.093 with 17 of 18 cells keeping their sign |
| Q3. Which training-free scorer? | **The min/max bound, not the mean-plus-spread score.** The smooth score collapses in deep layers as budgets tighten (0.395 at 1% density) while the bound holds (0.921) |

## Scope

Qwen3-4B and Qwen3-8B with their stage-1 selector checkpoints, both trained at 16K sequence length. Contexts 8K and 32K, pg19 and fineweb prompts, greedy decode, records every 16th step across all 36 layers.

At each recorded step and layer, four rankings over the same residual block set are built from identical inputs: the oracle (true attention mass per block), the trained selector's block scores, a min/max key bound in the style of Quest, and a mean-key-plus-spread score. Sink and recency reserves are excluded for every scorer equally, so the numbers isolate scoring quality from the reservation policy. Comparison is at matched k, so no scorer is credited for spending more.

Reported metric is **mass efficiency**: true attention mass captured by a scorer's top-k, divided by the most any selector could capture at that k. 1.0 means the ranking is as good as the oracle's own.

## Q1. What does dropping the trained selector cost?

**Why we asked.** The training-free direction rests on this number, and the survey found no published comparison of trained against training-free selection at matched density, for any scorer pair.

**What we did.** The setup above, sweeping budgets from 50% of residual blocks down to 1%, which reaches the aggressive regime nobody has measured.

**Answer: the trained selector wins inside its training length and loses outside it.**

Mean efficiency gap over all budget and depth cells, positive meaning the training-free bound is ahead:

| Setting | Gap |
|---|---|
| 4B at 8K (inside the 16K training length) | **−0.064** |
| 4B at 32K (twice the training length) | **+0.061** |
| 8B at 32K | **+0.081** |

The mechanism shows in the shallow band, where our earlier calibration campaign already found length decay:

| Budget | Trained, 8K → 32K | Bound, 8K → 32K |
|---|---|---|
| 50% | 0.962 → 0.894 | 0.952 → 0.972 |
| 10% | 0.901 → 0.750 | 0.851 → 0.903 |
| 1% | 0.825 → 0.648 | 0.673 → 0.721 |

- **The trained scorer degrades with length; the bound does not.** A learned low-rank index has to extrapolate; a min/max bound over the actual keys is exact arithmetic with nothing to extrapolate, and it gets slightly *better* at longer context.
- **This is not a verdict against trained selection.** It is a verdict about operating outside the training length. Inside it, training wins on every budget in the shallow band.
- **For the method**: at the lengths this work targets, training-free scoring is the safer default, and the trained selector belongs in the design as an optional accelerator rather than a requirement. That is what the design doc assumes, now measured rather than argued.
- **For the selector itself**: the weakness is localized to length extrapolation, worst in the shallow band, which is a training-recipe question rather than an architectural one.

## Q2. Is the result an artifact of how we measured?

**Why we asked.** Two ways the comparison could be unfair to the trained scorer: its block scores could be misaligned to block indices in our harness, and our oracle averages over query heads while a trained selector may pool them differently.

**What we did.** Two falsification tests on the 4B at 32K. Roll the trained score vector by ±1 block, which must make efficiency *worse* if alignment is correct. Then recompute the oracle with max over query heads instead of mean.

**Answer: neither explains the gap.**

| Test | Result |
|---|---|
| Block alignment | The aligned ranking beats both shifts in **18 of 18** cells. Shifting is catastrophic where the trained scorer is strongest (deep band at 1%: 0.868 aligned against 0.020 shifted), so the scores carry sharp positional signal and are correctly indexed |
| Head aggregation | Mean gap +0.084 under mean-aggregation, +0.093 under max-aggregation, with **17 of 18** cells keeping the same sign |

The alignment test doubles as evidence that the selector is being invoked correctly: a misused scorer would not degrade so sharply under a one-block shift.

## Q3. Which training-free scorer?

**Why we asked.** The bound is the classic choice, but a mean-plus-spread page score is cheaper and reported to be better behaved, so it was worth measuring rather than assuming.

**What we did.** Both scorers in the same runs, compared per depth band and budget.

**Answer: the min/max bound, decisively, and the failure of the alternative is instructive.**

Deep band, 8B at 32K:

| Budget | Bound | Mean-plus-spread | Trained |
|---|---|---|---|
| 10% | 0.947 | 0.744 | 0.863 |
| 5% | 0.935 | 0.651 | 0.861 |
| 2% | 0.925 | 0.519 | 0.869 |
| 1% | 0.921 | **0.395** | 0.883 |

Deep-layer attention is spiky: a few keys carry the mass. A smooth mean-based score averages those peaks away, while a max bound is built to catch them. The same score is competitive in shallow layers, where attention is diffuse. So scorer choice is depth-dependent in the same way budget is, which strengthens the per-layer allocation idea rather than complicating it.

## What this changes

- The training-free default is justified at 32K and above, which is the regime the method targets.
- Keep the trained selector in the design as an optional accelerator, and prefer it below its training length.
- Use the min/max bound as the default scorer. The mean-plus-spread variant is not a safe drop-in at aggressive budgets.
- Open next: whether the efficiency gaps here translate into acceptance differences of the same order, which is a direct measurement on the same harness and the natural follow-up.
