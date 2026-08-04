# Ideas: Rejected

Kill reasons on record so no dead idea is re-litigated. IDs are shared with [ideas-kept.md](ideas-kept.md); evidence lives in [landscape.md](landscape.md).

**TL;DR**
- The starting hypotheses did not survive the literature. The corrected forms below are what the kept bets stand on.
- #12–#14 are dead on arithmetic: layer-skip as a core framing caps at ~1.5x regardless of context length.
- Append-only. Reopen an entry only by attacking its recorded reason with new evidence.

## Corrected starting hypotheses

| Hypothesis | Verdict | The correction |
|---|---|---|
| H1: long-ctx spec decode is fundamentally KV-limited | ⚠️ **NUANCED** (sign error) | KV-boundedness is the *opportunity*; the real limits are the full-KV verify pass + the log(P) acceptance ceiling. On MLA it is the weight term. |
| H2: an extra drafter makes it worse (its KV grows too) | ❌ **REFUTED** | Constant-KV, O(1)-state, and zero-KV drafter classes all exist. The real killer is acceptance collapse from short training. |
| H3: only self-spec / early-exit / sparse drafting is viable | ❌ **REFUTED** as "only" | Corrected: draft against the target's *own* state, a class that includes retrieval, RAG, and O(1)-state drafters H3 excludes. Early exit is the triad's weakest member. |
| Hidden: acceptance holds/fails uniformly with context | ⚠️ **BIFURCATES** | By draft-state construction: shared/quantized/retrieved holds to 128K; evicted/short-trained collapses by 8–32K. Nothing measured at 256K–1M. |

### H1: "fundamentally limited by KV memory/bandwidth"
- **For**: decode is KV-bound for MHA/GQA and worsens with S (VeriCache: 5ms@5K → 25ms@500K, ~60GB KV); MagicDec's model; GQA measurably weakens gains by lightening KV (MHA 1.63x vs GQA 1.47x at 32K/b128). KV footprint really is the control variable.
- **Against**: (a) KV-boundedness is the *opportunity*: speedup grows with batch/context beyond S_inflection (MagicDec 2.51x; QuantSpec 2.49x@128K; SpecPV 6.29x@60K). (b) Architecture-contingent, not fundamental: MLA is >100x MHA intensity, compute-bound-ish ([2507.15465](https://arxiv.org/abs/2507.15465)); GLA kernels 2x FlashMLA at q>1 ([2505.21487](https://arxiv.org/abs/2505.21487)); DeepSeek MTP runs 1.8x at 128K production. (c) At high acceptance the binding ceiling is informational: log(P) ([2512.11718](https://arxiv.org/abs/2512.11718)). (d) At production load the limiter is scheduling/verify compute (Meta 1.4–2.0x; EAGLE 3.1 2.03x→1.66x).
- **Corrected form**: *Full-KV verification bytes are the binding cost at long context for MHA/GQA; KV-boundedness makes cheap drafting profitable, and the residual limit is the verify pass itself plus the log(P) acceptance ceiling. On MLA it is the weight term instead.*

### H2: "an independent drafter makes it worse"
- **Steelman**: full-attention drafters trained short do fail (OWL 0.81x; SpecExtend 16K collapse; BudgetDraft ~0% at 8K); drafter KV is genuinely material: Qwen3-1.7B *and* 0.6B both carry 112KB/tok (GQA floors at 8 kv-heads) = 44% of a Qwen3-32B target's KV, cutting 8xH100 max batch at 128K by 31%; TransKV shows draft pages eat scheduler budget.
- **Refutation**: whole drafter classes have no growing KV: constant-KV (MagicDec 2.51x, LongSpec 3.26x with cross-attention into *target* KV), O(1)-state (OWL LSTM 2.35x, Mamba, ReDrafter), zero-KV retrieval (SuffixDecoding 2.8x over EAGLE-3; TR <2MB). Even a *larger* independent drafter wins at 120K (RAPID 2.10–2.69x, quality-improving). **The real failure driver is acceptance/distribution mismatch from short training, not drafter KV growth.**

### H3: "only self-spec, early exit, and sparse-attention drafting are viable"
- **Steelman**: everything measured at 96K–128K *is* drafting against the target's own sparse/compressed state (TriForce 2.31x@122K; QuantSpec 2.49x@128K; Vegas 96–120K; SparseSpec 2.13x@b256; VeriCache 4x). No alternative family has a 128K acceptance curve.
- **Refutation**: (a) retrieval/suffix drafting (SuffixDecoding 5.3x; HOWL τ6.14, 3.08x at ≤64K); (b) RAG drafting (RAPID at 120K); (c) block-diffusion (DFlash >6x, recoverable to 32K); (d) O(1)-state SSM/RNN drafters; (e) compressed-KV drafting is *compression not sparsity*, and QuantSpec shows quantized beats sparse. Meanwhile **early exit, inside the triad, has zero evidence past 16K** and layer-skip is arithmetically capped at ~1.5x.
- **Surviving corrected claim**: at ≥96K, published wins come from drafting against the target's own (sparse, compressed, or retrieved) state rather than short-trained independent transformer drafters, a class that *includes* retrieval, RAG, and O(1)-state drafters H3 excludes.

### Hidden assumption: "acceptance holds or fails uniformly with context"
- Holds/rises: QuantSpec 91.64→94.31% (16K→128K); TriForce 0.9649–0.9878@120K; MagicDec 0.84@4K→0.79@100K; VeriCache 25–40 tok/round; SparseSpec-L 52–84.6% to 64K (recallable).
- Collapses: BudgetDraft 55.4%→0.3%→~0.1% (4K→8K→16K), post-training only ~18%@16K; eviction caches 0.05–0.07 on needle; EAGLE-3 → τ1.28; DFlash → τ2.09 @32K; TriForce falls to ~0.29/token at T=1.0 and staleness grows with generation.
- **Nothing has an acceptance curve at 256K–1M. Anywhere.**

## Rejected directions

| ID | Direction | Kill reason | Reopen / salvage |
|---|---|---|---|
| #12 | 2D (layer × KV-budget) knapsack drafter [G2] | Arithmetic: composition adds ~2% at B≥8/128K and acceptance multiplies *down* (landscape, arithmetic #7) | Only as batch-1, 512K–1M self-spec (+30–40% predicted); reopen test below |
| #13 | Layer-skip/early-exit acceptance curves 64K–1M + LayerSkip training at 128K [G1] | Ceiling ~1.5x (landscape, arithmetic #5): a measurement, not a project | As a measurement section inside another paper (e.g. #12's reopen test). Cite KnapSpec, AdaSkip ([2501.02336](https://arxiv.org/abs/2501.02336)), [2603.23701](https://arxiv.org/abs/2603.23701) (redundancy shrinking in modern checkpoints) |
| #14 | Layer-skip self-spec under continuous batching [G5] | Serving-engineering project on a 1.5x-capped direction; SparseSpec deliberately routed around it | Only if #12's niche pans out |

**#12 reopen test**: the composition is fully novel and fits our DP/selector skills; ingredients are KnapSpec + Vegas. Measure α of (50% skip × 4K sparse) jointly at 64K. If α_comb < α_skip·α_sparse (super-multiplicative damage), the rejection stands; if not, pursue only in the batch-1, 512K–1M niche.
