# DrafterGoesBurrrr

Lossless self-speculative decoding for long contexts. One model plays both roles. The drafter is the target made cheap three ways at once: it attends to a small KV view selected per layer and per request from the model's own attention mass, it runs on 4-bit weights (W4A16), and it replays CUDA graphs. The verifier is the unmodified target: full KV, bf16, one forward per round. Output equals dense greedy decoding token for token.

| Cell | vs dense | vs vegas |
|---|---|---|
| 32K, batch 1 | 1.46x | 1.55x |
| 32K, batch 4 | 1.65x | 1.27x |
| 64K, batch 1 | 1.08x | 1.28x |
| 64K, batch 2 | 1.60x | 1.30x |
| 128K, batch 1 | 0.80x | 0.97x |

This document holds the method, every result measured so far, and what comes next. Engine notes: [handoff.md](handoff.md). Work list: [TODO.md](TODO.md).

## Setting

| | |
|---|---|
| Model | Qwen3-4B, 36 layers, 32 query heads, 8 KV heads, bf16 weights 8 GB. YaRN factor 2 at 64K, 4 at 128K. |
| Workload | pg19 book prompts of 32K, 64K, 128K tokens; greedy; 256 or 512 generated tokens; batch 1 to 4. |
| Hardware | One A6000 48 GB (sm86, FlashAttention-2 path). Serial cells on a quiet node; A/B pairs on the same node. |
| Software | The vegas vLLM fork. Proposer, scheduler wiring, rejection sampler and graph dispatch are inherited; a method is one attention overrider. |
| Baselines | Dense decoding. Vegas, rerun on this stack with its paper's configuration: fixed top-k ratio 0.07, floor 256 tokens, bf16 draft. |
| Metrics | Decode tok/s = batch x (gen - 1) / (t_gen - t_1). tau = 1 + accepted / rounds, at most 7. alpha = accepted / drafted. Budget = mean over layers of the selected fraction of the scored prefix. |

## Method

### One round

L layers, B requests, g draft tokens per round.

1. **Scores.** During the verify pass, every layer's attention hook recomputes the two query rows' dot products against the paged K, rematerializes the softmax weights from the kernel's log-sum-exp, and averages over rows and query heads. The result is one attention mass per KV token per layer, written into a per-layer buffer. One extra K read per layer on FA2; no patched kernel. Prefill scores its last row, so drafting starts at the first decode step. Rows padded for a graph are never read.
2. **Selection.** At the last layer of the same pass, one launch over all layers and requests. Per layer and request: the sink (first tokens) and the recent window (last tokens) are always kept; among the rest, the smallest set of top tokens whose mass, with the reserved mass, reaches θ of the total; clamped to the floor and the cap. The indices go to a table. Static shapes, no host sync: the verify pass replays as a CUDA graph.
3. **Table build.** First draft step, once for all layers: logical index to physical slot through the block table, the tokens after the scored prefix appended, K/V of the selected slots gathered into each layer's page-aligned scratch.
4. **Draft steps.** g steps, one token per request, on the quantized weight copy under piecewise CUDA graphs. Per layer: the new token's K/V is written to the cache and the scratch, then attention over exactly the selected tokens, the reserved ranges, and every token after the scored prefix. Greedy sampling with the target's sampler.
5. **Verify.** Target forward over the g draft tokens plus one bonus token, full KV, bf16 weights, packed attention. K/V at the draft positions are overwritten with full-attention values. The rejection sampler accepts the longest matching prefix and emits the verify's own argmax at the first mismatch. This pass is step 1 of the next round.

Speedup per round is tau / (g c + 1) with c the draft step cost relative to a dense step. The design lowers c and protects tau.

### Selection

| Piece | Rule |
|---|---|
| Budget | Per layer, per request: the smallest number of top tokens holding fraction θ of the attention mass (top-p over tokens). Layers differ; requests differ. |
| Reserved | Sink and recent tokens are always kept and excluded from the candidates. The recent window is clipped to the prefix. |
| Bounds | A floor (0 allowed) and a cap (a fraction of the prefix). The cap sizes the scratch; θ does the work. θ = 1 keeps every token. |
| Kernel | One fused CUDA kernel, one 1024-thread block per row, grid L x B. Pass 1: total and reserved mass, count and mass histograms of the candidates on the high byte. Pass 2: same on the low byte inside the chosen bucket, tie count from the threshold value. Passes 3 and 4 only if the clamp moved the count: count-based radix select. Pass 5: compaction, strictly better keys front to back, ties back to front, then the reserved indices. No workspace, no allocation. |
| Determinism | Shared-memory float atomics move the count by one element at near-exact crossings; tests allow that band and require exact agreement elsewhere. |
| Freshness | Recomputed every round from the latest verify. Tables rebuild every propose, so budget changes are graph safe. |

Fixed-k top-k (vegas) is the special case: same signal, uniform k. Coverage differs only in where the bytes go: concentrated layers and requests get less, spread ones get more.

### Drafting

| Piece | Rule |
|---|---|
| KV view | Selected tokens gathered from the paged cache into per-layer page-aligned scratch once per round; later steps append only the newest token. |
| KV writes | Draft steps write K/V into the one real paged cache at the round's reserved slots; the verify overwrites the same slots. There is exactly one KV cache. |
| Weights | A separately loaded W4A16 copy of the target on the Marlin path, about 2 GB. The copy holds decoder projections and norms only; the target's attention modules, embeddings and lm_head are grafted in, so KV binding, layer identity and losslessness are untouched. The checkpoint id is part of the compile hash. |
| Graphs | The draft has its own dispatcher: piecewise-only keys, trimmed to the sizes a draft step can reach, captured in a dedicated pass. Dummy runs outside the capture window run eagerly; propose replays. A key miss degrades to eager. |
| Logits | The target's own lm_head, shared, bf16. |

### Verifying

The unmodified target: full KV, bf16 weights, one forward over g + 1 tokens. Nothing of the method touches it. Its attention runs packed for GQA on FA2, an engine fix any multi-token verify needs; see Time per round.

### Configuration

Explicit typed fields on `SpeculativeConfig`. `sparse_attn_algorithm="coverage"` is the method; `"longspec"` adds the static skip masks, an ablation instrument.

| Knob | Field | Value |
|---|---|---|
| θ | `sparse_attn_theta` | 0.923 at 32K, 0.926 at 64K and 128K (a 7% mean budget) |
| Sink, recent | `sparse_attn_sink`, `sparse_attn_recent` | 4, 64 |
| Floor | `sparse_attn_min_tokens` | 0 |
| Cap | `sparse_attn_ratio`, fraction of the prefix; 1 means uncapped | 0.15 |
| g | `num_speculative_tokens` | 6 |
| Draft weights | `sparse_attn_draft_weights` | `RedHatAI/Qwen3-4B-quantized.w4a16`; AWQ and GPTQ measure the same |
| Packed verify | `sparse_attn_packed_verify` | on |
| Skip masks | `sparse_attn_skip_attn_layers`, `sparse_attn_skip_layers` | empty |

### Cost and memory

| Item | Cost per round at 32K, batch 1 |
|---|---|
| Scores | one K read per layer per verify, 4.5 ms |
| Selection | one launch, L x B blocks, 0.5 to 1.0 ms |
| Gather | the selected tokens once, 0.4 ms |
| Draft | g x (Marlin weights 3.4 ms + attention over the selected view 1.7 ms + gather and lm_head) |
| Verify | one dense step of g + 1 queries: attention 31.1, scores 4.5, weights 11.1 ms |

Memory: target weights 8 GB; 4-bit copy 2 GB; KV 0.147 MB per token; score buffer of one bf16 per layer, request and token, 75 MB at 36 layers, 8 requests, 128K; draft table width: cap plus sink, recent and 2g + 1; the scratch is a bounded copy of the selected slots. The KV pool at 0.9 utilization is about 32 GB with the 4-bit copy; one 128K request peaks at 21% of it.

## Results

### Against dense and vegas

Ours: the configuration above. Vegas: the baseline configuration, same stack. tau and alpha in parentheses; the budget is coverage's mean selected fraction.

| ctx | batch | dense | vegas | ours | vs dense | vs vegas |
|---|---|---|---|---|---|---|
| 32K | 1 | 49.0 | 46.0 (6.22, 0.873) | **71.4** (6.58, 0.931, 7.3%) | 1.46x | 1.55x |
| 32K | 4 | 93.8 | 121.7 (6.11, 0.855) | **154.7** (5.54, 0.759, 6.7%) | 1.65x | 1.27x |
| 64K | 1 | 36.3 | 30.6 (5.60, 0.767) | **39.1** (5.21, 0.704, 7.1%) | 1.08x | 1.28x |
| 64K | 2 | 45.9 | 56.7 (5.84, 0.807) | **73.6** (5.76, 0.795, 6.7%) | 1.60x | 1.30x |
| 128K | 1 | 23.5 | 19.5 (5.31, 0.724) | 18.9 (4.52, 0.590, 5.9%) | 0.80x | 0.97x |

- Dense reproduces within 2% across passes at 32K and 64K, within 13% at 128K.
- Vegas acceptance reproduces its paper at 32K (tau 6.1 to 6.3 in every pass); at 64K and 128K it sits 0.5 to 1.6 below the first baseline pass, a YaRN or trajectory effect never pinned down.
- The gain moves with batch: bytes matter more, and the packed verify wins more.

### Selection at equal budget

Both sides draft on the 4-bit copy; θ tuned per context so coverage's mean budget lands at vegas's fixed 7%. Alpha here is the pure selection signal. Throughput also carries the packed verify, which only our path has.

| ctx | batch | ours (budget) | vegas, 4-bit draft |
|---|---|---|---|
| 32K | 1 | 71.4, 6.58, 0.931 (7.3%) | 72.0, 6.47, 0.911 |
| 32K | 2 | 80.6, 5.12, 0.688 (8.1%) | 95.3, 5.18, 0.696 |
| 32K | 3 | 134.2, 6.05, 0.841 (7.0%) | 108.3, 5.80, 0.804 |
| 32K | 4 | 154.7, 5.54, 0.759 (6.7%) | 143.3, 6.06, 0.845 |
| 64K | 1 | 39.1, 5.21, 0.704 (7.1%) | 38.4, 5.10, 0.683 |
| 64K | 2 | 73.6, 5.76, 0.795 (6.7%) | 70.5, 5.80, 0.802 |

- Selection alone: ahead at batch 1 in both contexts and at 32K batch 3, tied at 32K batch 2 and 64K batch 2, behind at 32K batch 4 (0.759 vs 0.845, reproduced twice on the same prompts). The deficit is prompt-dependent, not monotonic in batch.
- Throughput: ahead or equal in five of six cells. The 32K batch 2 cell is a single run and reads low on both tau and alpha; treat as unconfirmed.
- Calibration curve, mean budget by θ: 32K 0.80 / 0.85 / 0.90 / 0.93 gives 1.42 / 3.01 / 6.28 / 9.25%; 64K gives 1.25 / 2.21 / 4.57 / 7.33%. Calibration at 64 generated tokens overshoots real decode by about 20%.
- θ is a plateau. At 32K batch 1 a 5.7% budget gives 72.6 tok/s, tau 6.55, alpha 0.925; 7.3% gives 71.4, 6.58, 0.931. θ 0.92 beats the pre-quantization default 0.98: round 90 ms against 98 ms.

Selection alone before quantization, packing and draft graphs (bf16 draft, cap 15%; tau and budget in parentheses):

| ctx | batch | dense | vegas | θ 0.90 | θ 0.95 | θ 0.98 |
|---|---|---|---|---|---|---|
| 32K | 1 | 49.0 | 45.6 (6.38) | 36.6 (5.80, 5.5%) | 46.6 (6.61, 11%) | 46.6 (6.71, 15%) |
| 32K | 4 | 92.5 | 103.8 (6.14) | 111.9 (5.96, 4.7%) | 111.1 (6.10, 10%) | 112.9 (6.51, 14%) |
| 64K | 1 | 35.8 | 22.2 (5.52) | 26.5 (5.23, 5.2%) | 27.7 (5.64, 11%) | 28.9 (6.05, 15%) |
| 64K | 2 | 46.3 | 50.9 (5.38) | 48.4 (5.11, 5.0%) | 51.2 (5.44, 11%) | 50.2 (5.88, 14%) |
| 128K | 1 | 23.9 | 19.4 (5.43) | 19.6 (5.31, 3.9%) | 21.8 (5.84, 8.6%) | 22.1 (6.12, 13%) |

- θ 0.95 beats vegas's tau at every cell with about 1.5x its bytes; θ 0.90 trails by 0.1 to 0.6 with 55 to 80% of its bytes. The 32K batch 1 spec cells of this pass ran on a shared node and may read low by up to 10%.
- Per-layer profile at θ 0.90: layers 1 to 3 sit at the cap; layers 5, 7, 11, 23, 25 and 30 read under 2%. Stable across context and batch.

### Quantized drafting

Agreement: bf16 greedy trajectories teacher-forced through a quantized copy, per-token argmax agreement. That is exactly the acceptance condition of a greedy 4-bit drafter under a bf16 verify. tau_sim walks the agreement in blocks of 6. Dense tok/s: the quantized model decoding alone at 32K.

| Model | 32K agreement (tau_sim) | 64K agreement (tau_sim) | Dense b1 / b2 / b4 |
|---|---|---|---|
| bf16 self | 0.994 (6.81) | 0.984 (6.66) | 48.8 / 72.7 / 94.2 |
| Qwen AWQ | 0.927 (5.53) | 0.846 (4.28) | 82.0 / 100.8 / 113.9 |
| RedHat GPTQ W4A16 | 0.928 (5.53) | 0.848 (4.46) | 81.1 / 102.3 / 115.4 |
| JunHowie GPTQ-Int4 | 0.920 (5.41) | 0.865 (4.51) | 81.1 / 102.4 / 115.0 |

- Quantization alone costs 0.9 tau at 32K and 2.3 at 64K. The loss grows with context; within a generation the late half agrees better than the early half.
- Marlin: 1.66x on the dense step at batch 1 (the 8 GB of weights shrink, the KV read does not); no crossover through batch 4; at batch 8 bf16 preempts on KV capacity (9 to 11 tok/s at a 98% pool) while the 4-bit copy fits.
- Draft weights per step 12.4 to 3.4 ms. Sparse view and quantization compose about multiplicatively: alpha 0.93 x 0.90 gives the 0.80 to 0.84 seen at 32K.

Integrated, θ 0.98 cap 15%, same-node pairs:

| Cell | bf16 draft | 4-bit draft |
|---|---|---|
| 32K, batch 1 | 47.5 (6.89) | 60.2 (5.87) |
| 32K, batch 4 | 117.4 (6.37) | 156.0 (5.99) |
| 64K, batch 1 | 33.3 (6.26) | 37.2 (5.23) |
| 64K, batch 2 | 56.4 | 65.6 |

- 64K batch 1 tune with the 4-bit draft: θ 0.995 cap 15% gives 39.4, alpha 0.754; θ 0.98 cap 25% gives 38.1, 0.760; θ 0.995 cap 25% gives 35.9, 0.732. A bf16 draft at θ 0.995 cap 25% reaches alpha 0.935, so at 64K the loss is quantization, not the view.
- Ablation only: the 4-bit draft grafted onto vegas gives 67.7 at 32K batch 1 and 146.7 at batch 4 on its own node pair; the equal-budget table above is the paired comparison.

### Time per round

GPU ms per round from vLLM's profiler, θ 0.98, steady full-batch rounds. Dense step from the same kind of trace.

| Cell | Dense step | Round | Verify: attention / scores / weights | Draft per step: weights / attention / gather | Sample |
|---|---|---|---|---|---|
| 32K b1, bf16 draft | 20.3 | 141 | 47.3: 31.1 / 4.5 / 11.1 | 15.3: 12.4 / 1.8 / 0.4 | 2.3 |
| 32K b4, bf16 draft | 34.0 | 198 | 90.0: 65.0 / 12.3 / 12.1 | 17.7: 12.3 / 3.5 / 1.1 | 2.4 |
| 64K b1, bf16 draft | 27.2 | 185 | 83.0: 62.3 / 9.1 / 11.1 | 16.6: 12.4 / 2.8 / 0.8 | 2.3 |
| 32K b1, 4-bit draft | 20.3 | 94 (98 wall) | 47.3 | 7.4: 3.4 / 1.7 / 0.4 | 2.3 |

- Unpacked, the verify attention costs 4.4x the dense step's attention over the same KV (31.1 vs 7.5 ms at 32K b1): one block per query head, 32 blocks on 84 SMs, no split. The dense step packs the GQA group and splits the KV 17 ways, 136 blocks.
- Every bf16 forward reads 8 GB of weights in 11.1 to 12.4 ms, 93% of the card's bandwidth. The bf16 draft step sits on that floor.
- Wall time sits 2 to 15% above GPU busy. Launch overhead is second order once the draft replays graphs.

Packed verify attention. FA2 packs the query heads of one KV head into block rows only at query length 1, so the multi-token verify read the KV once per query head. The verify attention is decomposed instead: a non-causal prefix call with the GQA group reshaped into rows, a causal tail over the last pages, and a log-sum-exp merge that also feeds the score reduction. Static shapes, no host reads, graph safe, gated per call with a kill switch. The prefix launches B x Hk blocks, so the win scales with batch. A/B, θ 0.98, bf16 draft:

| Cell | Unpacked | Packed | Blocks |
|---|---|---|---|
| 32K, batch 4 | 111.5 (6.44) | **129.6** (6.50) | 32 |
| 32K, batch 1 | 46.3 (6.67) | 47.6 (6.89) | 8 |
| 64K, batch 1 | 30.2 (6.01) | 29.9 (5.71) | 8 |
| 64K, batch 2 | 55.5 (6.38) | 54.9 (6.36) | 16 |

Draft graphs: every draft step launched about 600 kernels from Python before the fix. The bf16 round was GPU-bound and hid it; with the 4-bit draft the round wall went from 154 to 98 ms at 32K batch 1.

### Losslessness

Output equals dense greedy decoding token for token: selection alone at θ = 1 uncapped on Qwen3-0.6B, and selection with the 4-bit draft on Qwen3-4B at 8K, batch 2. Any grid cell can be checked the same way with `--parity`.

### Limits

| Where | Fact |
|---|---|
| 128K, batch 1 | Below dense: 18.9 at θ 0.926, 21.8 at θ 0.98, dense 23.5. Not memory starvation: zero preemptions, pool at 21%. Suspects: the batch-1 verify occupancy wall and the quantization decay. |
| 32K, batch 4 | Selection loses 0.09 alpha to vegas at equal bytes, reproduced twice; batch 2 and 3 do not show it. Per-request budget skew under a shared selection pass is the suspect. |
| Small batches | The packed prefix launches B x Hk blocks; 8 to 16 blocks measure flat. A heuristic split fills the SMs but moved θ = 1 acceptance to 0.946, below the parity gate; parked pending the merge precision. |
| Blackwell | Engine init hits a kernel with no sm120 image before any model code; the wheel's FA2 is sm80 SASS with PTX. Benchmarks stay on A6000. |
| Measurement | Greedy trajectories diverge across engine configurations; tau moves by 0.3 on the same prompt. Only same-node paired cells are compared. |

## Lessons

| Lesson | Evidence |
|---|---|
| Each lever exposes the next | The sparse view makes the draft weight-bound; 4-bit weights make it launch-bound; graphs make it GPU-bound again. |
| The draft's bottleneck is not the KV | 12.4 of 15.3 ms per bf16 step are weights; the selected view costs 1.8. |
| Bytes do not explain the verify | Unpacked it reads the KV four times at 38% occupancy; packing fixes it where blocks suffice. |
| θ is a plateau | 0.92 and 0.98 accept alike; 0.92 halves the bytes and shortens the round. |
| Quantization loss grows with context | Agreement 0.93 at 32K, 0.85 at 64K; the 64K deficit is quantization, not selection. |
| Acceptance is trajectory noise across configs | Pair cells on one node; compare ratios within pairs. |

## Next

### Budget-driven attention skip

The selection already prices every layer. A layer whose budget lands near the floor is a layer the model barely reads at that position; its draft attention is not worth a launch.

| Piece | Rule |
|---|---|
| Decision | Per round, per layer: budget below a skip floor (about 2% of context) skips that layer's draft attention for the round. |
| Execution | Attention output zeroed in place; residual and MLP unchanged; K/V of drafted tokens still written, so the verify and later rounds see a complete cache. |
| Safety | Decided where the tables are built, frozen per round; the layer counter advances through skipped layers. Tables rebuild every propose, so the skip set may change every round without touching graphs. |
| Replaces | The static mask, which needs an offline sweep and is batch-uniform. |

### Dynamic draft length

Fixed g drafts through positions the drafter has already given up on. The expected value of step i is the probability that all earlier draft tokens are accepted.

| Piece | Rule |
|---|---|
| Signal | The drafter's top-1 probability at each step, already computed for the rejection sampler, accumulated as a running product per request. |
| Decision | Stop the request's draft when the product falls below a threshold; g becomes per request, bounded by g_max. |
| Batch | A stopped request pads its remaining steps; the batch keeps a uniform shape for graph replay. The rejection sampler already handles non-uniform draft counts. |
| Effect | Cuts g c exactly where tau is small: long contexts and quantized drafting. Where acceptance is high the threshold never triggers. |

## Invariants

- One KV cache. Every extra structure is a bounded copy or an index over it.
- Lossless by construction: emitted tokens are the verify's argmax; the draft moves speed only.
- Graph safe: selection, gather, slot conversion and the draft forward use static shapes and no host reads; anything data-dependent is frozen per round at table-build time.
- Layer identity is call order: every draft step issues exactly one attention call per layer, or advances the counter explicitly when a layer is skipped.
- The drafter shares by reference everything that defines the model: attention modules, KV cache, embeddings, lm_head, sampler. It owns only cheap replacements: quantized projections, its scratch, its graphs.
