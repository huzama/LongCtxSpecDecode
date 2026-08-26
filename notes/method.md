# Coverage-budgeted self-speculative drafting

**TL;DR.** The target model drafts g tokens over a per-layer, per-request KV subset chosen once per round from the last verify pass's real attention weights: sink and recency reserved, top-p over the rest. Budget zero is allowed. Static per-layer skip masks are an ablation knob. Verify is full KV, lossless.

Scope: the first testable cut. Everything under "Later" is out until the tests below have numbers. Goals and status: [TODO.md](TODO.md). Measured baseline: [handoff.md](handoff.md).

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

At the end of the previous verify pass, per layer, inside the attention hook: recompute the two query rows' dot products against the paged K of that layer, rematerialize softmax weights from the kernel's log-sum-exp, mean over rows and heads. Output m_l[0..P). One extra K read per layer. Prefill: last row only. Single-query verify: that row only.

### 2. Selection

Same hook, right after the scores. Per layer l, per request:

| Step | Rule |
|---|---|
| reserved mass | Σ m_l over [0, S) ∪ [P-R, P) |
| candidates | [S, P-R), sorted by m_l descending, cumulative mass c[k] |
| total | Σ_{t<P} m_l[t] |
| k* | min k with reserved + c[k] ≥ θ · total |
| k_l | clamp(k*, k_min, k_max) |
| selected_l | sink ∪ recency ∪ top-k_l candidates |
| output | logical indices into row l of the draft table, used_l = S + R + k_l |

Edge cases: recency is clipped to [S, P); empty candidates select all of [0, P). Selection runs on static shapes with no host sync: the verify pass replays as a CUDA graph.

### 3. Table build

First draft step of the round, once per layer: logical index to physical slot through the request's block table, the tail's slots appended, K/V of the selected slots gathered into the layer's page-aligned scratch. Layers under a skip mask gather nothing.

### 4. Draft steps

g steps, one token per request, full model forward. Per layer:

| Layer state | What runs |
|---|---|
| sparse, k_l > 0 | the new token's K/V written to cache and scratch, used_l += 1, attend over exactly used_l tokens |
| local-only, k_l = 0 | same, over sink + recency + tail |
| attn-skip, static mask | attention output is zero, the residual passes through; q/k/v/o projections still run |
| layer-skip, static mask | whole layer bypassed; eager mode only |

After the forward: sample the next draft token with the target sampler, keep the hidden state for the draft logits.

### 5. Verify

Target forward over the g draft tokens plus one bonus token, full attention, full KV. K/V at the draft positions are overwritten with full-attention values. The rejection sampler accepts the longest matching prefix; the sequence rolls back past the first rejection. This pass is step 1 of the next round.

### 6. First round

Step 1 runs on the prefill pass itself, so drafting starts at the first decode step.

## Cost per round

| Item | Cost |
|---|---|
| scores | one K read per layer (FA2 path) |
| selection | L sorts of length max_len (first cut); one fused radix pass later |
| gather | Σ_l used_l tokens, once |
| draft | g × (weights of unskipped sublayers + Σ_l used_l × KV bytes per token) |
| verify | one dense step of g+1 queries |

## Parameters

Explicit typed fields on `SpeculativeConfig`, the vLLM convention: one field per knob with type, validator, and docstring.

| Knob | Config field | Start |
|---|---|---|
| algorithm | `sparse_attn_algorithm = "coverage"` | |
| θ | `sparse_attn_coverage` | 0.9 |
| S | `sparse_attn_sink` | 4 |
| R | `sparse_attn_recent` | 64 |
| k_min | `sparse_attn_min_tokens` | 0 |
| k_max | `sparse_attn_ratio` × P, existing field, now a cap | 0.07 |
| g | `num_speculative_tokens` | 6 |
| attn-skip mask | `sparse_attn_skip_attn_layers` | none |
| layer-skip mask | `sparse_attn_skip_layers` | none |

## Invariants

- Lossless: output equals dense decoding under the same sampling. θ = 1 must reproduce dense output token for token.
- Selection is frozen within a round. Budgets are per layer and per request. Skip masks are batch-uniform.
- Static shapes and no `.item()` anywhere in selection.
- Draft table width = k_max + S + R + 2g + 1.
- k_max bounds memory: scratch and table are sized from it, never from θ.

## Selection kernel

Selection is two operations on one metric row: find k by mass, then pick indices by rank.

| Cut | Find k | Pick indices | Kernels |
|---|---|---|---|
| first | `torch.sort` descending, `cumsum`, `searchsorted` on reserved + c[k] ≥ θ · total | existing radix top-k (`varlen_topk`) with per-layer k | 0 new; graph-safe, static shapes |
| later | same 8-bit radix passes as top-k, with a mass histogram next to the count histogram; the bucket where reserved + cumulative mass crosses θ · total gives the threshold | same compaction pass, unchanged | 1 fused kernel in `longspec/kernels/`; reads the row per pass instead of sorting; reserved ranges summed in the first pass and excluded from candidates |

## Package

Everything of ours is one package. The fork is entered at three seams.

```
vllm/v1/spec_decode/sparse_attn/
  proposer.py                fork, +1 line: bind_model hook
  attn_overrider/
    __init__.py              fork, +1 branch: dispatch "coverage"
    vegas.py                 fork, imports our portable layer and slot table
    streamingllm.py, utils/  fork, untouched
  longspec/                  ours
    __init__.py              exports CoverageAttnOverrider
    config.py                reads the knobs off SpeculativeConfig, validates the masks
    overrider.py             verify hook, draft hook, budgets [L, B]
    budget.py                coverage_budget(): reserved mass, top-p count
    layer_skip.py            static masks, eager only
    stats.py                 per-layer k_l running sums
    kernels/                 c2q_scores, draft_gather, slot_table, later top-p
    portable/                score_collection, draft_kv, kernel_support

tests/v1/spec_decode/sparse_attn/longspec/   one test file per module
benchmarks/longspec/                         grid runner
```

Rules: imports from the fork are the base overrider class, `varlen_topk`, `varlen_reduce`, and the flash-attn function, nothing else. One public symbol per module. Type hints everywhere. Docstrings state constraints only. Every module has a test.

## Implementation map

`longspec/` is `vllm/v1/spec_decode/sparse_attn/longspec/`.

| File | Change |
|---|---|
| `vllm/config/speculative.py` | `coverage` in `sparse_attn_algorithm`; fields `sparse_attn_coverage`, `sparse_attn_sink`, `sparse_attn_recent`, `sparse_attn_skip_attn_layers`, `sparse_attn_skip_layers`; `sparse_attn_min_tokens` accepts 0 |
| `attn_overrider/__init__.py` | dispatch `coverage`; table width includes S + R; `bind_model()` no-op on the base class |
| `proposer.py` | `load_model` calls `attn_overrider.bind_model(model)` |
| `attn_overrider/vegas.py` | imports the slot-table kernel from `longspec/kernels/`; math unchanged |
| `longspec/config.py`, new | typed view of our knobs, mask validation against L |
| `longspec/overrider.py`, new | `CoverageAttnOverrider`: verify hook (scores, budget, top-k, table row), draft hook (per-layer used, attn-skip zeros), budgets `[L, B]` |
| `longspec/budget.py`, new | `coverage_budget()`: reserved mass, cumulative mass, k per layer. Torch ops first; the fused kernel replaces its body later |
| `longspec/layer_skip.py`, new | wraps decoder layers for the static masks; eager only |
| `longspec/stats.py`, new | running k_l sums per layer, read by the benchmark |
| `longspec/kernels/slot_table.py`, moved out of `vegas.py` | index-to-slot kernel with a per-layer budget stride; vegas passes stride 0 |
| `benchmarks/longspec/grid.py`, new | pg19 grid runner: dense, vegas, coverage; alpha, tau, decode tok/s, Σ_l used_l per round |
| `tests/.../longspec/test_budget.py`, `test_overrider.py`, new | budget against a reference: reserved, clamps, zero, short prompt; table row layout |

## Tests

| # | Question | Measure |
|---|---|---|
| T1 | parity | θ = 1 output == dense greedy, 32K |
| T2 | how sparse per layer | k_l distribution and Σ_l used_l vs θ ∈ {0.8, 0.9, 0.95, 0.99} |
| T3 | acceptance | alpha, tau vs θ at 32K/64K/128K batch 1, 32K batch 4 |
| T4 | speed | decode tok/s vs baseline at matched bytes and at matched θ |
| T5 | skip ablation | attn-skip and layer-skip per band of 4 layers at 32K batch 1: alpha, tau, step time |

Benchmark rules from the baseline grid apply: serial cells on an idle node, drain wait, power-of-two batches, prefix caching off.

## Later

Online skip rule from a residual-contribution signal, per-head union, all verify rows as signal, EMA across rounds, page-level selection, the fused radix top-p kernel.
