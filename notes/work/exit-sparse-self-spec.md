# Exit-Sparse Self-Speculation

Our method: self-speculative decoding where drafting and verification both run sparse through one external selector, with early exit as an option on the draft side. Extends ideas #0 and #1 in [survey/ideas-kept.md](survey/ideas-kept.md); evidence lives in [survey/landscape.md](survey/landscape.md).

**Notation**: α = per-token acceptance rate, τ = accepted tokens per round, γ = draft tokens per round, k = exit depth as a fraction of the L layers, p = attention-mass coverage of a sparse view, S = context length, W = model weight bytes, AR = autoregressive decoding. A *gentle* view keeps enough budget to match dense quality; an *aggressive* view keeps far less.

**TL;DR**

- One model, one selector, no second drafter. Drafting runs the shallow layers (exit at k) over an aggressive view of the KV cache. Verification runs full depth over a gentle view. Both views come from our block-sparse selector (method under review). Neither pass ever computes full attention.
- The output matches the gentle-sparse model exactly. SSV ([2605.19893](https://arxiv.org/abs/2605.19893)) set this precedent for natively sparse models. The gentle-sparse model is the deployed model; its quality claim comes from the selector method's published results.
- Draft KV is temporary. Verification recomputes accepted tokens at full depth and commits only what it computed itself. Error cannot accumulate, and aggressive drafting costs acceptance only, never correctness.
- Expect 1.5-2x per token over sparse AR at batch 1, 32-128K (model estimate, numbers below). The value is elsewhere: the first system where neither side needs full attention, an exact quality claim, and the measured (k, p, γ) surface.
- Kill test: α over exit depth × draft budget at 32-64K on Llama-3.1-8B. Three conditions decide, listed under first experiments.

## One round

```
committed KV  (every entry written by a full-depth gentle pass)
     │
     │  draft × γ  (sequential)
     │    layers 1..k, LoRA exit head
     │    reads: aggressive top-p view of committed KV + temp KV of this burst
     │    writes: temp KV only
     │
     └─ verify  (one parallel pass over the γ+1 tokens)
          layers 1..L, gentle top-p view
          accept or reject against the gentle-sparse distribution
          recompute and commit fresh KV, drop temp, roll back rejects
```

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Quality claim | Output matches the gentle-sparse target exactly | SSV precedent; the quality burden stays on the selector method's own results |
| KV handling | Draft KV is temporary; verification recomputes and commits its own | Draft sparsity never leaks into the verified model's state, and error cannot accumulate. Costs ~15% on the weight term, paid knowingly (numbers below) |
| Exit training | LoRA on the exit branch only, base frozen | Kangaroo-style ([2404.18911](https://arxiv.org/abs/2404.18911)); the target never moves, so this works on any frozen checkpoint |
| Exit rule | Fixed layer k; per-token adaptive exit comes later | Keeps the kill test one-dimensional per axis |
| Selection | Top-p over selector block scores, with a hard budget cap | p controls retained attention mass, the quantity a divergence bound is written in; the cap bounds worst-case cost |

## Neighbors, and why nobody runs both sides sparse

| Method | What runs sparse | Where it stops |
|---|---|---|
| Vegas ([2602.07223](https://arxiv.org/abs/2602.07223)) | Drafting: 7% of KV, token-level top-k picked from its own verification attention. τ~6.1 of 7 at 96-120K (Qwen3-8B, batch 4-20) | Verification must stay full: the selection signal is read off the full verify pass. Speedup vs dense at long context unpublished; +18-29% over other speculative baselines |
| Dustin ([2606.24957](https://arxiv.org/abs/2606.24957)) | Verification: 512 tokens at 32K, picked by drafter lookahead plus target attention history. 9.17x end-to-end (Qwen2.5-72B) | Needs a separate full-attention 0.5B drafter as scout. Not lossless, no drift control, nothing past 32K |
| SpecPV ([2512.02337](https://arxiv.org/abs/2512.02337)) | Verification, partially, with periodic full refresh | Error accumulates between refreshes: GovReport ROUGE-L falls 77.3 to 56.1 at a 2K budget |
| LayerSkip, Kangaroo, Mirror-SD | Draft depth: early exit | All at 16K or less, all full attention. No early-exit result past 16K exists |
| SSV ([2605.19893](https://arxiv.org/abs/2605.19893)) | The target itself, trained sparse, with speculation on top | Needs a sparse-trained model. Ours is a frozen checkpoint plus a retrofit selector; we adopt their quality claim |

The pattern: every published signal is parasitic on a full-attention pass. Vegas reads its selection off full verification; sparsify verification and the signal is gone. Dustin's blind verifier needs a full-attention drafter to scout for it; drop the second model and the scout is gone. Our selector scores importance without reading any attention output, so it keeps working when no pass is full.

Skipping the separate drafter is a production decision, not a taste: no second model to deploy, no per-target training, no drafter prefill, no drafter KV. And acceptance holds at long context because the drafter is the target: self-drafters that share the target's state hold acceptance to 128K, while separately trained draft heads collapse by 8-32K (survey).

## What is new

1. The first self-speculative system where draft and verify are both sparse, with the selection signal external to both.
2. Budgets profiled over depth: aggressive shallow drafting, gentle full-depth verification, exit boundary k as a control. Full depth (k=L) is a special case.
3. Top-p mass coverage instead of top-k counts: the budget adapts per query, and retained mass is what a divergence bound is written in.
4. Commit semantics that make sparse self-speculation drift-free by construction and free the draft budget from any effect on output quality.
5. The measured α(k, p, S, task) surface, with the batch and context map of where each half pays.

## Where the speedup comes from, and its size

Per round: the draft makes γ sequential passes over k of the layers, verification makes one parallel pass over all layers. Weight traffic is γ·k·W + W for τ accepted tokens. Against sparse AR at W per token:

| γ | k | τ | Speedup on weights |
|---|---|---|---|
| 5 | 0.50 | 4.5 | 1.29x |
| 5 | 0.25 | 4.0 | 1.78x |
| 3 | 0.25 | 2.8 | 1.60x |

Reusing the draft's shallow compute at verification would save the k·W rerun (γ·k·W + (1-k)·W, ~15% better at k=0.5). But then the shallow layers of the verified model would have run at aggressive sparsity, and we would no longer match the model we claim to match. We pay the rerun.

Roofline sketch, Llama-3.1-8B BF16, batch 1, 64K, H100-class bandwidth, memory-bound terms only (model estimate, not a measurement): dense AR ~7.3ms per token, gentle-sparse AR at 25% ~5.4ms, our round (k=50%, 5% draft view, γ=5, τ≈4.5) ~3.4ms. That is 2.1x against dense but **1.6x against the correct baseline**, the sparse model itself, since that is the model our output matches. Every reported speedup must separate the selector's gain from the speculation gain.

Where the numbers move:

- **Longer context**: verification's KV traffic grows linearly in S while the draft view stays near constant, so the multiple rises with context. Same effect MagicDec measured.
- **Larger batch**: weights amortize across the batch, so early exit stops paying and sparse verification starts to (KV bytes dominate). The two halves cover opposite regimes; we show the map instead of hiding it.
- **MLA targets**: decode there is weight-bound (arithmetic table in [survey/landscape.md](survey/landscape.md)), so early exit is the half that transfers. That is our insurance against the architecture shift.
- **Ceilings**: τ is logarithmically bounded ([2512.11718](https://arxiv.org/abs/2512.11718)), and γ stays small because draft weight cost grows linearly in γ while τ saturates.

## Risks

What kills this, and how we cover each.

| Risk | Why it kills | Hedge |
|---|---|---|
| α collapses when exit meets sparsity | The two acceptance losses may compound worse than multiplicatively, the failure recorded for rejected idea #12 ([survey/ideas-rejected.md](survey/ideas-rejected.md)) | Kill-test condition 2 measures exactly this |
| Exit fails on retrieval tokens | Retrieval integration sits in mid and deep layers; rejections would pile up on the tokens that make the task long-context | Calibration splits α by token type, needle vs local |
| Early exit past 16K is unmeasured | No published early-exit result beyond 16K, and depth redundancy is shrinking in newer checkpoints ([2603.23701](https://arxiv.org/abs/2603.23701)) | LayerSkip checkpoints give an estimate for free before we spend on LoRA |
| Selector scores do not track attention mass | Top-p budgets become arbitrary and no divergence bound can be written | Calibration condition: predicted retained mass must correlate with true mass |
| Block granularity misses needle tokens | Vegas beat block-approximate selection by 15-29% end-to-end; selection misses cost α 0.05 vs 0.99 on needle tasks (TriForce, 120K) | Gentle budget on the verify side; the aggressive side risks acceptance only, never correctness |
| The multiple looks small next to dense-baseline numbers | 1.6x vs sparse AR at batch 1, 64K (estimate) | Report selector gain and speculation gain separately; the exact quality claim, the (k, p, γ) surface, and the regime map are the paper |

## First experiments

### Calibration, about a week, no training

| Measurement | What it decides | Cost |
|---|---|---|
| LayerSkip public checkpoints, α_exit vs k at 8-32K | The exit half, before any LoRA spend | Hours |
| Selector score mass vs true attention mass, correlation across tasks at 32K | Whether top-p and the divergence bound are possible | ~1 GPU-day |
| α split by token type, retrieval-hit vs local, at 32K | The pile-up-on-retrieval-tokens risk | Same runs |
| Sparse-AR and dense-AR throughput at 32/64K | The baseline under every speedup we will ever quote | Hours |

### Kill test, two weeks

Llama-3.1-8B at 32-64K. Gentle verify budget comes from the selector's quality-vs-budget curve, at the point matching dense.

| Axis | Values |
|---|---|
| Exit k | 25% and 50% of layers |
| Draft coverage p | gentle, mid, aggressive; exact values from calibration |
| Tasks | Needle retrieval, summarization, long chain-of-thought, reported separately |

All three must hold:

1. α at k=50% stays above the cost-model break-even at 64K.
2. α_comb ≥ α_exit · α_sparse: acceptance with both mechanisms on is at least the product of each alone.
3. Predicted retained mass correlates with true attention mass.

If one fails, that axis moves to [survey/ideas-rejected.md](survey/ideas-rejected.md) with the data, and the rest continues: k=L drops the exit and keeps both sides sparse; p=1 on the verify side keeps full verification with selector-driven drafting.

## Rules for the build

- Parity bypass: p=1 on both sides and k=L must match the vanilla model path bit-exactly. Named validation command, recorded in CLAUDE.md.
- Every model patch (mask injection, temp-KV buffers, exit branch) sits behind a reversible context manager: snapshot on enter, restore on exit.
- Every run lives under `outputs/<slug>-<timestamp>/` with its launch command archived.

## Order of work

1. Calibration. Decides whether the exit axis and top-p survive.
2. Kill test. Decides whether the composition survives.
3. Kernels and wall-clock: block-sparse draft path, gather-based gentle verify, the batch regime map.
4. Divergence bound: retained-mass vs realized-KL curves on our own system. Survey idea #2 applied to ourselves; Dustin's attention-recovery appendix showed the correlation, we want the bound.
5. Write the paper: quality claim, system, surface, regime map.

## Open questions

- Chain vs tree drafting. A draft node here costs k·W of weights, so tree width is priced very differently than in methods whose draft step is a small head.
- γ policy, fixed vs entropy-based, once the α surface exists.
- Reusing the verify pass's own attention: the gentle pass produces attention over retained blocks; re-ranking inside the retained set might help, or might double-count the selector.
- Top-p at batch > 1: per-request budgets break kernel batching (risks table in [survey/ideas-kept.md](survey/ideas-kept.md)). The hard cap is the likely answer.

## References

Survey entry point: [survey/README.md](survey/README.md). Ideas and kill reasons: [survey/ideas-kept.md](survey/ideas-kept.md), [survey/ideas-rejected.md](survey/ideas-rejected.md). Full annotated bibliography: [survey/bibliography.md](survey/bibliography.md). Inline arXiv links above cover this file's own claims; everything else is in the bibliography.
