# Preliminary Results

Four questions from the design doc ([exit-sparse-self-spec.md](../exit-sparse-self-spec.md)), each with a measurement behind it. Raw numbers live under `outputs/` (slug `prelim-*`); every run archives its config and launch command. Batch is 1 unless said otherwise.

**TL;DR, one answer per question**

| Question | Answer |
|---|---|
| Q1. Can we trust the selector's top-p inside the decode loop? | **Yes, except at the stack's edges.** Captured mass 0.82-0.89 per depth band vs nominal p=0.95 on both 4B and 8B; the last layer inverts (0.02 / 0.52 captured) |
| Q2. Does aggressive draft sparsity fail exactly on retrieval tokens? | **Only below a floor that depends on model and workload.** 8B at 32K: safe to p = 0.5, crossover at p ≈ 0.45. 4B: floor at p ≈ 0.7. Denser-needle configs push the floor up to p > 0.5. Local text is robust everywhere (α ≥ 0.82) |
| Q3. What does early exit cost at long context, zero training? | **Usable depth shifts right with context until nothing remains.** α ≥ 0.74 from 37.5% depth at 8K; at 32K only ≥62.5% depth works (0.57-0.75); at 64K every depth fails (≤0.34). Exit training at long context is the enabling step |
| Q4. What baseline speed must we beat? | **33.8 ms/token: sparse-AR at 32K (8B, fixed-K, A6000, batch 1), flat in context.** Dense: 50.5 at 32K, 63 at 64K. Sparse-AR's lead over dense grows 1.07x (8K) to 1.50x (32K) |

## Scope

Selector-side measurements are Qwen3 4B and 8B; exit-side are Meta's LayerSkip 1B and 8B; all batch 1. Coverage: calibration at 3 context lengths x 4 coverage targets x 2 backbones plus two drift horizons; acceptance by token type at 3 lengths x 2 scales x 9 draft budgets plus a matched-density control and both draft mechanisms; exit acceptance over 7 depths x 4 lengths plus a single-layer scan; throughput on the dense and the selector kernel paths.

Selector runs cap at 32K: Qwen3's context limit is 40960, so 64K would measure an out-of-range model. Reaching past 32K needs the YaRN rope-scaling recipe and a re-check that the selector's calibration survives it.

## Q1. Can we trust the selector's top-p inside the decode loop?

**Why we asked.** The whole design reads budgets off the selector's calibrated distribution (top-p). Its published calibration was measured on full-model runs; in our loop, every query is computed through sparse attention. If calibration drifts there, top-p budgets become arbitrary and the divergence-bound plan dies (kill-test condition 3).

**What we did.** Decoded 32K-token prompts (pg19 books + fineweb web text, 2 samples each, 64 greedy steps, batch 1) with dual top-p active at p = 0.95, on both stage-1 checkpoints. At every decode step and layer we compared the selector's block ranking against the true attention block mass computed from the same query and cache.

**Answer: calibration holds in-loop, on both backbones.**

| Depth band | 4B captured | 4B blocks (of ~2048) | 8B captured | 8B blocks |
|---|---|---|---|---|
| Shallow third | 0.863 | ~934 (46%) | 0.824 | ~957 (47%) |
| Middle third | 0.890 | ~809 (40%) | 0.892 | ~776 (38%) |
| Deep third | 0.823 | ~421 (21%) | 0.863 | ~502 (25%) |

- **The gap to nominal is 6-13 points, not a collapse.** Sparse-computed states still give the selector enough signal. Kill-test condition 3 looks pass-shaped.
- **The depth asymmetry is live inside the loop.** Shallow layers spend ~2x the blocks deep layers spend for the same coverage, on both backbones. The draft runs exactly the expensive layers; the design doc's caveat is now measured fact.
- **The stack's edges misbehave.** The last layer captures 0.022 (4B) / 0.522 (8B) of its residual mass with negative rank correlation; at 8B the two earliest layers are next-weakest (0.66, 0.74). The last layer parks most of its mass in the sink + recency reserves (residual 0.11 vs 0.33-0.51 elsewhere), so damage is bounded, but the practical rule is: run the first and last layers dense or reserves-only. They are cheap layers to exempt.
- Rank correlation varies widely (near 0 to 0.75) even where captured mass is high: in diffuse layers, coverage comes from breadth, not precise ranking.

**The knob across p.** Captured true residual mass (shallow / middle / deep bands) as nominal p moves:

| Nominal p | 4B captured | 4B shallow blocks | 8B captured |
|---|---|---|---|
| 0.99 | 0.96 / 0.97 / 0.89 | ~1442 | — |
| 0.95 | 0.86 / 0.89 / 0.82 | ~934 | 0.82 / 0.89 / 0.86 |
| 0.90 | — | — | 0.73 / 0.82 / 0.80 |
| 0.80 | 0.64 / 0.69 / 0.69 | ~435 | 0.58 / 0.69 / 0.71 |

- The response is smooth and monotone: p is a real control knob. But the gap to nominal grows as p drops (2-10 points at 0.99, up to 9-22 at 0.8), shallow layers undershooting most.
- Consequence for the divergence-bound plan: state bounds in *realized* retained mass and carry a measured p-to-realized map per model, rather than trusting nominal p. The map is four cheap runs per checkpoint.
- Budgets scale steeply with p (4B shallow band: 435 → 934 → 1442 blocks for p = 0.8 → 0.95 → 0.99), so the speed-quality trade lives almost entirely inside p ∈ [0.8, 0.99].

**The knob across context length** (captured at p = 0.95, shallow / middle / deep):

| Prompt | 4B | 8B |
|---|---|---|
| 8K | 0.88 / 0.89 / 0.85 | 0.89 / 0.89 / 0.86 |
| 16K | 0.88 / 0.90 / 0.85 | 0.87 / 0.89 / 0.86 |
| 32K | 0.86 / 0.89 / 0.82 | 0.82 / 0.89 / 0.86 |

Middle and deep bands are flat in length; only the shallow band decays as context extends past the selector's 16K training length, and gently (2-6 points by 32K). Length generalization of in-loop calibration is a shallow-layer problem, which is again where the draft lives.

**No drift over generation.** Captured mass over 1024 greedy decode steps (8B, p = 0.95, 32K prompt, one sample) is flat: 0.848 / 0.862 / 0.861 / 0.857 per 256-step quarter; a separate 256-step run shows the same at finer grain, shallow band included. Selection is recomputed every step, so nothing accumulates; the staleness problem that frozen-index methods carry does not exist in this design. The refresh-controller question dissolves for the basic loop.

## Q2. Does aggressive draft sparsity fail exactly on retrieval tokens?

**Why we asked.** Rejections that pile up on retrieval-dependent tokens would gut the method on exactly the tasks that make long context matter (the TriForce eviction lesson, one level up).

**What we did.** Needle task at 32K on Qwen3-8B (stage-1 selector, 6 needles, 4 samples, batch 1): generate with the gentle setting (p = 0.95) as target, re-score with aggressive draft settings, split acceptance by token label (retrieval-hit vs local). Teacher-forced sparse forwards use the prefill path (selection shared per 16-query tile), so these α are the safe lower-bound side.

**Answer: not until p drops below ~0.5. At p = 0.3 the predicted failure appears, retrieval-first.**

α_greedy (α_overlap), Qwen3-8B, target p = 0.95, 8 samples, needles recalled in 48/48 runs:

| Draft p | Retrieval tokens | Local tokens |
|---|---|---|
| 0.95 | 1.000 (1.000) | 1.000 (1.000) |
| 0.90 | 1.000 (0.998) | 0.977 (0.981) |
| 0.80 | 0.998 (0.997) | 0.951 (0.966) |
| 0.70 | 1.000 (0.997) | 0.934 (0.958) |
| 0.50 | 0.972 (0.951) | 0.922 (0.937) |
| 0.30 | **0.486 (0.452)** | 0.848 (0.859) |

- **Safe floor located.** Down to p = 0.5, retrieval tokens accept as well as or better than local ones (query-aware selection keeps the needle blocks; the TriForce retrieval-vs-eviction lesson working for us, one level up), and overall α stays at or above 0.92.
- **The failure mode is real, just further out, and its onset is sharp but its decay is smooth.** A bracket run (8 samples, p = 0.45/0.40/0.35) pins the 8B crossover at p ≈ 0.45, where retrieval meets local (0.913 vs 0.921); below it retrieval decays steadily (0.77 at 0.40, 0.64 at 0.35, 0.49 at 0.30) while local text barely moves (0.92 to 0.85). Below the floor, needle blocks are what the selector drops first, exactly the correlated-rejection risk the design doc names. Draft budgets live in p ∈ [0.5, 0.9] with headroom known to within 0.05 of coverage.
- α_sparse enters the compounding product from a high floor: ≥0.92 anywhere in the safe range at 32K.
- **The floor moves with scale.** On Qwen3-4B (8 samples, needles recalled 24/24) the inversion already appears at p = 0.5: retrieval 0.694 vs local 0.869, while p = 0.7 is still safe (0.965 / 0.910). Smaller model, higher floor: 8B tolerates p = 0.5, 4B needs p ≥ 0.7.
- **The floor also moves with context, in the favorable direction: longer context tolerates lower p.** At fixed p = 0.5 and six needles, retrieval α worsens as context shortens: 0.97 at 32K, 0.60 at 16K, 0.44 at 8K (8 samples each, all needles recalled; local stays ~0.87 throughout). A matched-density control (3 needles at 16K, same needles-per-token as 6 at 32K) recovers only part of the gap (0.76 at p = 0.5), so density explains some of the shift and genuine length dependence explains the rest. Practical rules: the safe p is a property of (model, context, workload), exactly the α(budget, length, task) surface the design's measurement agenda calls for, and aggressive coverage is safest precisely in the long-context regime this method targets.
- Caveats: one task family, 8-digit needle values are easy to copy once attended.

**Exit-draft version: same answer, where it is measurable at all.** At 8B scale (layerskip-llama3-8B, native 8K, 3 needles, 8/8 recalled, 50% exit): retrieval tokens accept *better* than local ones, 0.850 vs 0.692 greedy. Copying an attended needle is an easy prediction the half-depth model already gets right; free prose is the hard part. So within each mechanism's working range, rejections never concentrate on retrieval tokens; the failures that exist are range failures (context beyond the exit boundary in Q3; p below the floor here), and only sparse-below-floor is retrieval-first. The 1B could not host this split at all (zero needle recall at 16K and 32K); its one usable scrap: exit α on needle-task text is lower than on plain prose (0.63 vs 0.79 at 16K, 50% exit), so exit acceptance is task-dependent on top of depth- and length-dependent.

## Q3. What does early exit cost at long context, zero training?

**Why we asked.** No published early-exit result exists past 16K. Before spending on LoRA training we want the free estimate: Meta's LayerSkip checkpoints, exit depths from 12.5% to 87.5% of layers, 8-64K contexts.

**What we did.** Position-wise acceptance (greedy match + distribution overlap) of exit-k drafts against the full model's own continuation, on pg19 text, 4 samples per cell, batch 1. Models: layerskip-llama3.2-1B (128K-native) and layerskip-llama3-8B (8K-native, so only its 8K cells are in-range).

**Answer: exit survives to 16K, and 32K forces a choice: exit shallower or train.**

α_greedy, layerskip-llama3.2-1B (16 layers), 4-8 samples per cell:

| Exit depth | 8K | 16K | 32K | 64K |
|---|---|---|---|---|
| 12.5% | 0.27 | 0.26 | 0.18 | — |
| 25% | 0.37 | 0.31 | 0.22 | — |
| 37.5% | 0.74 | 0.59 | 0.24 | — |
| 50% | 0.78 | 0.79 | 0.23 | 0.06 |
| 62.5% | 0.82 | 0.77 | 0.57 | 0.05 |
| 75% | 0.87 | 0.89 | 0.66 | 0.18 |
| 87.5% | 0.94 | 0.94 | 0.75 | 0.34 |

layerskip-llama3-8B at its native 8K rises smoothly 0.66 → 0.93 across 12.5-87.5% depth (higher floor than the 1B at every depth). Its 16K/32K cells are RoPE-out-of-range; ignore them.

- **This is acceptance data past 16K for early exit, which the survey found nowhere in the literature.**
- **Longer context shifts the usable exit depth rightward, as a step, not a slope, and the step is one layer wide.** At 8-16K, α clears 0.7 from ~40% depth. At 32K everything below 62.5% sits on a flat ~0.2 plateau, and a 16-sample per-layer scan pins the jump to a single layer: 0.276 at layer 9 of 16, 0.530 at layer 10, then a gradual climb (0.543 at 11, 0.641 at 12). Whatever integrates long-range context in this model passes through layer 10; every exit before it is blind at 32K no matter how close it sits.
- **At 64K, zero-shot exit is dead at every depth**: even 87.5% reaches only 0.34 greedy (distribution overlap 0.57, so the distributions still resemble each other while argmax agreement is gone). The step boundary marches rightward with context (~37.5% at 8-16K, ~62.5% at 32K, past 87.5% at 64K) until nothing is left.
- **Design consequence: the exit boundary is context-dependent, and training is the whole game at 64K.** The kill-test condition (usable α at 50% exit at 64K) cannot pass zero-shot; the long-context LoRA adapter is not an optimization but the enabling step, and its precise job is to pull the step boundary back down at long range.
- Scale caveat: the clean curve comes from a 1B; no long-context LayerSkip checkpoint exists at 8B scale, so the 8B curve needs our own adapter first. The 8B's higher floor at 8K suggests scale helps, as it did for the sparse floor in Q2.

## Q4. What baseline speed must we beat?

**Why we asked.** Our contract makes the sparse model the deployed target, so the honest denominator is sparse AR decode, not dense. Nobody gets to quote a speedup here without this table.

**What we did.** Dense HF decode timing across prompt lengths, batch 1, greedy, steady-state steps only (CUDA events, warmup excluded); selector-side numbers via the selector repo's own latency benchmark in its fixed-K mode.

**Answer (dense half): the KV term takes an 8B from ~28 to ~63 ms/token by 64K on these 48 GB cards.**

ms/token, batch 1:

| Prompt | Llama-3.1-8B | Qwen3-8B |
|---|---|---|
| 4K | 27.9 | 32.7 |
| 8K | 30.4 | 33.6 |
| 16K | 35.1 | 38.9 |
| 32K | 44.4 | 49.3 |
| 64K | 63.2 | out of range |

- The shape matches the cost model: weight-bound floor at short context, KV bytes adding ~0.55 ms per extra 1K tokens of context. Every speedup we ever quote divides into these numbers.

**Sparse-AR half** (selector repo's own benchmark, fixed-K mode k8192, CUDA graph, RTX A6000, batch 1, per-checkpoint isolated runs):

| Qwen3-8B | 8K | 32K |
|---|---|---|
| Vanilla decode | 34.5 ms/tok | 50.5 ms/tok |
| Selector sparse decode | 32.3 ms/tok | 33.8 ms/tok |
| Sparse advantage | 1.07x | **1.50x** |

- **The sparse path is flat with context** (32.3 to 33.8 ms/tok across 4x the KV), so its advantage grows with length exactly as the KV-bytes arithmetic says. The number every speculation result must beat at 32K is 33.8 ms/tok, not 50.5.
- Cross-validation: our dense harness and their vanilla mode agree within 2.5% at 32K (49.3 vs 50.5 ms/tok).
- The 4B pair (vanilla 33.8, sparse 21.8-23.3 ms/tok) shows a larger gap, but its vanilla decode matches the 8B's almost exactly, so at this model size the HF vanilla path is overhead-bound at batch 1 and flatters the comparison; the 8B pair is the honest one.
- Top-p latency is unmeasured: the benchmark's CUDA-graph path rejects data-dependent K. Fixed-K stands in as the sparse-AR baseline; the top-p kernel cost question stays open.

## Next

1. Train the long-context exit adapter. Its target is pulling the one-layer step (layer 10 of 16 at 32K on the 1B) back down; without it the exit axis contributes nothing at 64K.
2. Run the speculation loop end to end at the measured operating point (verify p = 0.95, draft p = 0.5-0.7 at 8B/32K) against the 33.8 ms/token sparse-AR baseline.
3. The YaRN check to unlock 64K on the selector side.
