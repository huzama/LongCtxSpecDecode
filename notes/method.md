# Coverage-budgeted self-speculative drafting

**TL;DR.** The target model drafts g tokens over a per-layer, per-request KV subset chosen once per round from the last verify pass's real attention weights: sink and recency reserved, top-p over the rest. Budget zero is allowed. Static per-layer skip masks are an ablation knob. Verify is full KV, lossless.

Status: implemented as `CoverageAttnOverrider` in `longspec/`, unit tests green, parity test in place. Goals and status: [TODO.md](TODO.md). Measured baseline: [handoff.md](handoff.md).

## Symbols

| Symbol | Meaning |
|---|---|
| L, B, g | layers (36 on Qwen3-4B), requests in the batch, draft tokens per round |
| P | scored prefix length of a request: the tokens whose weights the last verify pass produced |
| m_l[t] | score of token t < P at layer l: mean over the 2 query rows and all query heads of the softmax weight |
| tail | tokens after P: the just-verified tokens plus this round's draft tokens. Always attended |
| S, R | sink [0, S) and recency [P-R, P). Always attended |
| θ | coverage target, one value for all layers |
| k_l | top-p count of layer l, per request |
| used_l | tokens the draft attends at layer l: S + R + k_l + tail |

## One round, step by step

### 1. Scores

During the previous verify pass, per layer, inside the attention hook: recompute the two query rows' dot products against the paged K of that layer, rematerialize softmax weights from the kernel's log-sum-exp, mean over rows and heads. Output m_l[0..P) into row l of a per-layer buffer `[L, B, max_len]`. One extra K read per layer. Prefill: last row only. Single-query verify: that row only. Rows padded for a CUDA graph (no query) get P = 0 and are never read.

### 2. Selection

At the last layer of the same pass: one launch of the fused kernel over all L × B rows. Per row:

| Step | Rule |
|---|---|
| reserved mass | Σ m_l over [0, S) ∪ [P-R, P) |
| candidates | [S, P-R), ranked by a radix select over their bf16 keys |
| total | Σ_{t<P} m_l[t] |
| k* | smallest count whose mass, with the reserved mass, reaches θ · total; ties at the threshold resolved from the threshold value |
| k_l | clamp(k*, k_min, k_max); a moved count re-selects by rank |
| selected_l | top-k_l candidates, then sink and recency indices, used_l = S + R + k_l |

Edge cases: recency is clipped to [S, P); empty candidates select all of [0, P); θ = 1 selects every candidate. Static shapes, no host sync: the verify pass replays as a CUDA graph.

### 3. Table build

First draft step of the round, once for all layers: logical index to physical slot through the request's block table, the tail's slots appended, K/V of the selected slots gathered into each layer's page-aligned scratch.

### 4. Draft steps

g steps, one token per request, full model forward. Per layer:

| Layer state | What runs |
|---|---|
| sparse, k_l > 0 | the new token's K/V written to cache and scratch, used_l += 1, attend over exactly used_l tokens |
| local-only, k_l = 0 | same, over sink + recency + tail |
| attn-skip, static mask | attention output is zero, the residual passes through; q/k/v/o projections still run |
| layer-skip, static mask | whole layer bypassed; eager mode only; the overrider is told so its layer counter stays aligned |

After the forward: sample the next draft token with the target sampler, keep the hidden state for the draft logits.

### 5. Verify

Target forward over the g draft tokens plus one bonus token, full attention, full KV. K/V at the draft positions are overwritten with full-attention values. The rejection sampler accepts the longest matching prefix; the sequence rolls back past the first rejection. This pass is step 1 of the next round.

### 6. First round

Step 1 runs on the prefill pass itself, so drafting starts at the first decode step.

## Cost per round

| Item | Cost |
|---|---|
| scores | one K read per layer (FA2 path) |
| selection | one launch, L × B blocks, up to five passes over each row's prefix |
| gather | Σ_l used_l tokens, once |
| draft | g × (weights of unskipped sublayers + Σ_l used_l × KV bytes per token) |
| verify | one dense step of g+1 queries |

## Parameters

Explicit typed fields on `SpeculativeConfig`, the vLLM convention.

| Knob | Config field | Start |
|---|---|---|
| algorithm | `sparse_attn_algorithm = "coverage"` | |
| θ | `sparse_attn_coverage` | 0.9 |
| S | `sparse_attn_sink` | 4 |
| R | `sparse_attn_recent` | 64 |
| k_min | `sparse_attn_min_tokens` | 0 |
| k_max | `sparse_attn_ratio` × P, existing field, now a cap; 1 means uncapped | 0.07 |
| g | `num_speculative_tokens` | 6 |
| attn-skip mask | `sparse_attn_skip_attn_layers` | none |
| layer-skip mask | `sparse_attn_skip_layers` | none |

## Invariants

- Lossless: output equals dense decoding under the same sampling. θ = 1 with ratio 1 reproduces dense output token for token, and the drafter agrees with the target except for accumulation-order effects (alpha 0.99+).
- Selection is frozen within a round. Budgets are per layer and per request. Skip masks are batch-uniform.
- Static shapes and no `.item()` anywhere on the per-round path.
- Draft table width = k_max + S + R + 2g + 1; k_max, not θ, bounds memory.
- Metric buffer `[L, B_max, max_len]` bf16: 75 MB at 36 layers, 8 requests, 128K.

## Selection kernel

`longspec/kernels/coverage_select.py`, CUDA via `load_inline`, one 1024-thread block per row. Same key mapping and compaction as the fork's radix top-k, different objective:

| Pass | Work |
|---|---|
| 1 | total and reserved mass; count and mass histograms of the candidates on the high byte |
| 2 | same on the low byte inside the chosen bucket; the number of tied elements needed from the threshold value |
| 3, 4 | only if the clamp moved k: count-based radix select for the new k |
| 5 | compaction: strictly better keys front to back, ties back to front, then the reserved indices |

Shared-memory float atomics make the count vary by one element at near-exact crossings; tests allow that band and require exact agreement elsewhere. No multi-block path: parallelism comes from the L × B grid.

## Package

Everything of ours is one package. The fork is entered at three seams.

```
vllm/v1/spec_decode/sparse_attn/
  proposer.py                fork, +1 line: bind_model hook
  attn_overrider/
    __init__.py              fork, +1 branch: dispatch "coverage"; bind_model no-op
    vegas.py                 fork, imports our portable layer and slot table
    streamingllm.py, utils/  fork, untouched
  longspec/                  ours
    __init__.py              exports CoverageAttnOverrider
    config.py                typed view of the knobs, cross-field validation
    overrider.py             verify hook, draft hook, budgets [L, B]
    layer_skip.py            static whole-layer masks, eager only
    stats.py                 per-layer budget accumulators
    kernels/                 c2q_scores, draft_gather, slot_table, coverage_select
    portable/                score_collection, draft_kv, kernel_support

tests/v1/spec_decode/sparse_attn/longspec/   one test file per module, test_parity end to end
benchmarks/longspec/grid.py                  pg19 grid runner
```

Rules: imports from the fork are the base overrider class, `varlen_reduce`, the forward context, and the flash-attn function, nothing else. One public symbol per module. Type hints everywhere. Docstrings state constraints only. Every module has a test.

## Grid runner

`benchmarks/longspec/grid.py`: one cell per process, serial cells with a drain wait, pg19 prompts as token ids cached under `outputs/prompts/`, YaRN beyond 40960, decode isolated as gen=1 vs gen=N, acceptance from vLLM's spec-decode counters, per-layer budgets from the overrider, one JSON line per cell, `--parity` compares spec output with dense token for token.

## Tests

| # | Question | Measure |
|---|---|---|
| T1 | parity | `test_parity.py`: θ = 1, ratio 1 on Qwen3-0.6B reproduces dense greedy output, alpha ≥ 0.98 |
| T2 | how sparse per layer | `budget.mean_ratio_per_layer` from the grid vs θ ∈ {0.8, 0.9, 0.95, 0.99} |
| T3 | acceptance | alpha, tau vs θ at 32K/64K/128K batch 1, 32K batch 4 |
| T4 | speed | decode tok/s vs baseline at matched bytes and at matched θ |
| T5 | skip ablation | attn-skip and layer-skip per band of 4 layers at 32K batch 1: alpha, tau, step time |

Benchmark rules from the baseline grid apply: serial cells on an idle node, drain wait, power-of-two batches, prefix caching off.

## Later

Online skip rule from a residual-contribution signal, per-head union, all verify rows as signal, EMA across rounds, page-level selection, a multi-block path for the selection kernel if profiling asks for it.
