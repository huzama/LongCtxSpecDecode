# Long-Context Speculative Decoding: Research Dossier
*Literature snapshot: 2026-08-03 · evidence-first · all numbers as published · arXiv IDs linked inline.*

**TL;DR**
- Sparse self-drafting (draft against the target's own sparse/compressed/retrieved KV; verify full-KV, lossless) is the **converged recipe** — it already works to 128K.
- The open frontier is the **verification side** (85–95% of round time) and the **unmeasured regimes** (256K–1M, KV-quantized targets, MLA/hybrid architectures).
- Top bets: **#1 both-sides-sparse**, **#2 divergence certification**, **#3–#4 the measurement papers**.
- Avoid layer-skip-centric framings — arithmetic-capped at ~1.5x.

| § | Section | Answers | Layer |
|---|---|---|---|
| 1 | 🎯 Hypothesis scorecard | Was the starting intuition right? | decision |
| 2 | 🚀 Ranked gaps | What should we work on? | decision |
| 3 | ☠️ Kill risks | What breaks the project? | decision |
| 4 | ✅ What is settled | What does the field agree on? | evidence |
| 5 | ⚖️ What is contested | Where do papers disagree? | evidence |
| 6 | 🧮 The arithmetic | What does the roofline math allow? | evidence |
| 7 | 📚 Bibliography | Which paper said what? (generated — reference, not reading list) | reference |

---

## 1. 🎯 HYPOTHESIS SCORECARD

| Hypothesis | Verdict | The correction |
|---|---|---|
| H1 — long-ctx spec decode is fundamentally KV-limited | ⚠️ **NUANCED** (sign error) | KV-boundedness is the *opportunity*; the real limits are the full-KV verify pass + the log(P) acceptance ceiling. On MLA it's the weight term. |
| H2 — an extra drafter makes it worse (its KV grows too) | ❌ **REFUTED** | Constant-KV, O(1)-state, and zero-KV drafter classes all exist. The real killer is acceptance collapse from short training. |
| H3 — only self-spec / early-exit / sparse drafting is viable | ❌ **REFUTED** as "only" | Corrected: draft against the target's *own* state — a class that includes retrieval, RAG, and O(1)-state drafters H3 excludes. Early exit is the triad's weakest member. |
| Hidden — acceptance holds/fails uniformly with context | ⚠️ **BIFURCATES** | By draft-state construction: shared/quantized/retrieved holds to 128K; evicted/short-trained collapses by 8–32K. Nothing measured at 256K–1M. |

### H1 detail — "fundamentally limited by KV memory/bandwidth"
- **For**: decode is KV-bound for MHA/GQA and worsens with S (VeriCache: 5ms@5K → 25ms@500K, ~60GB KV); MagicDec's model; GQA measurably weakens gains by lightening KV (MHA 1.63x vs GQA 1.47x at 32K/b128) — KV footprint really is the control variable.
- **Against**: (a) KV-boundedness is the *opportunity*: speedup grows with batch/context beyond S_inflection (MagicDec 2.51x; QuantSpec 2.49x@128K; SpecPV 6.29x@60K). (b) Architecture-contingent, not fundamental: MLA is >100x MHA intensity, compute-bound-ish ([2507.15465](https://arxiv.org/abs/2507.15465)); GLA kernels 2x FlashMLA at q>1 ([2505.21487](https://arxiv.org/abs/2505.21487)); DeepSeek MTP runs 1.8x at 128K production. (c) At high acceptance the binding ceiling is informational — log(P) ([2512.11718](https://arxiv.org/abs/2512.11718)). (d) At production load the limiter is scheduling/verify compute (Meta 1.4–2.0x; EAGLE 3.1 2.03x→1.66x).
- **Corrected form**: *Full-KV verification bytes are the binding cost at long context for MHA/GQA; KV-boundedness makes cheap drafting profitable, and the residual limit is the verify pass itself plus the log(P) acceptance ceiling. On MLA it's the weight term instead.*

### H2 detail — "an independent drafter makes it worse"
- **Steelman**: full-attention drafters trained short do fail (OWL 0.81x; SpecExtend 16K collapse; BudgetDraft ~0% at 8K); drafter KV is genuinely material — Qwen3-1.7B *and* 0.6B both carry 112KB/tok (GQA floors at 8 kv-heads) = 44% of a Qwen3-32B target's KV, cutting 8xH100 max batch at 128K by 31%; TransKV shows draft pages eat scheduler budget.
- **Refutation**: whole drafter classes have no growing KV — constant-KV (MagicDec 2.51x, LongSpec 3.26x with cross-attention into *target* KV), O(1)-state (OWL LSTM 2.35x, Mamba, ReDrafter), zero-KV retrieval (SuffixDecoding 2.8x over EAGLE-3; TR <2MB). Even a *larger* independent drafter wins at 120K (RAPID 2.10–2.69x, quality-improving). **The real failure driver is acceptance/distribution mismatch from short training, not drafter KV growth.**

### H3 detail — "only self-spec, early exit, and sparse-attention drafting are viable"
- **Steelman**: everything measured at 96K–128K *is* drafting against the target's own sparse/compressed state (TriForce 2.31x@122K; QuantSpec 2.49x@128K; Vegas 96–120K; SparseSpec 2.13x@b256; VeriCache 4x). No alternative family has a 128K acceptance curve.
- **Refutation**: (a) retrieval/suffix drafting (SuffixDecoding 5.3x; HOWL 6.14 acceptance/3.08x at ≤64K); (b) RAG drafting (RAPID at 120K); (c) block-diffusion (DFlash >6x, recoverable to 32K); (d) O(1)-state SSM/RNN drafters; (e) compressed-KV drafting is *compression not sparsity*, and QuantSpec shows quantized beats sparse. Meanwhile **early exit — inside the triad — has zero evidence past 16K** and layer-skip is arithmetically capped at ~1.5x.
- **Surviving corrected claim**: at ≥96K, published wins come from drafting against the target's own (sparse, compressed, or retrieved) state rather than short-trained independent transformer drafters — a class that *includes* retrieval, RAG, and O(1)-state drafters H3 excludes.

### Hidden-assumption detail — acceptance vs context
- Holds/rises: QuantSpec 91.64→94.31% (16K→128K); TriForce 0.9649–0.9878@120K; MagicDec 0.84@4K→0.79@100K; VeriCache 25–40 tok/round; SparseSpec-L 52–84.6% to 64K (recallable).
- Collapses: BudgetDraft 55.4%→0.3%→~0.1% (4K→8K→16K), post-training only ~18%@16K; eviction caches 0.05–0.07 on needle; EAGLE-3 → 1.28; DFlash → 2.09@32K; TriForce falls to ~0.29/token at T=1.0 and staleness grows with generation.
- **Nothing has an acceptance curve at 256K–1M. Anywhere.**

---

## 2. 🚀 RANKED GAPS

Ranked by impact × tractability × fit for a sparse-attention expert with a block-sparse selector method under review. Audit status in brackets; details for each gap follow the table — **read #1–#4 first**.

| # | Gap | Audit | 2-week kill test |
|---|---|---|---|
| **1** | Both-sides-sparse: sparse draft + sparse verify via external selector oracle | partially-solved | selector-restricted verify @12.5% budget — acceptance holds within ~5% → project lives |
| **2** | Divergence certification: KV budget → output-distribution bound | open | attention-mass ↔ realized-KL correlation @32K — one strong plot validates the program |
| **3** | Acceptance-vs-(budget, length) surface, 128K–1M | partially-solved | fit α(B, L) over {32K, 128K, 256K} × budgets; sublinear vs linear vs task-forked |
| **4** | EAGLE-3/MTP acceptance vs KV-quantized targets (the production config) | open | 70B + EAGLE-3, KV {BF16, FP8, INT8} × {4K, 32K, 64K} × 4 tasks |
| 5 | Staleness/refresh policy for verification-recycled importance | open | freeze indices {8–1024} steps; plot acceptance-vs-staleness decay |
| 6 | Adaptive full-verification triggering for partial verification | open | attention-mass-missed vs realized drift: GovReport (cliff) vs QA (safe) |
| 7 | Length-robust *trained* sparse-KV drafting past 32K | open | retrain BudgetDraft 68M with Anchor-Offset @32K; does 16K acceptance move off ~18%? |
| 8 | Online (drafter budget × γ) controller across (batch, context, load) | partially-solved | (B, γ) grid per cell; 2-feature predictor recovers ≥90% oracle throughput |
| 9 | Self-speculation for sequential-hybrid / SSM targets | partially-solved | LayerSkip-style finetune on 0.5–1B hybrid; does α recover from 0.038? |
| 10 | No-regret draft-path adaptation over 30K–100K generations | partially-solved | EXP3 budget selection on 13K-token AIME traces vs fixed |
| 11 | SSM drafter acceptance past 8K | open | Mamba-130M + Llama-3.1-8B on LongSpecBench @128K vs OWL-LSTM |
| 12 | 2D (layer × KV-budget) knapsack drafter | deprioritized by §6.7 | α_comb vs α_skip·α_sparse @64K — super-multiplicative damage kills it |
| 13 | Layer-skip acceptance curves 64K–1M | low ceiling (~1.5x) | measurement section inside another paper, not a project |
| 14 | Layer-skip self-spec under continuous batching | lowest priority | only if #12's niche pans out |

### #1 — Both-sides-sparse: compose sparse drafting with sparse verification, breaking oracle circularity [G11]
- **Why #1**: verification is 85–95% of round time (§6.6) — the only place large headroom remains; every verification-recycled drafter (Vegas/PillarAttn/SparseSpec-L) structurally *requires* full-KV verification for its signal. **Your block-sparse selector is a third-party importance oracle that doesn't need full verification attention — the exact missing piece.**
- **Novel delta**: a self-spec system where both draft and verify run sparse, with an external (selector-based) or Dustin-style fused signal replacing full-verification attention, plus recall of dropped-but-relevant tokens without a third scoring pass; measured interaction of the two speedups (each side's 2–9x is only measured in isolation).
- **Cite/beat**: Vegas ([2602.07223](https://arxiv.org/abs/2602.07223)) + Dustin ([2606.24957](https://arxiv.org/abs/2606.24957)) — the two sides, never composed; SSV ([2605.19893](https://arxiv.org/abs/2605.19893), natively-sparse targets — different contract); SpecPV (crude periodic refresh).
- **2-week experiment**: Llama-3.1-8B at 32–64K: run Vegas-style drafting with verification KV restricted to your selector's blocks at budgets {25%, 12.5%, 6%}; measure acceptance and downstream quality vs full-verify Vegas and vs Dustin-style fusion. If acceptance holds within ~5% at 12.5% verify budget, the project lives; if it collapses, that's the (publishable) circularity confirmation.

### #2 — Divergence certification for sparse verification: KV budget → output-distribution bound [G10]
- **Novel delta**: model target-under-sparse-KV as a perturbed distribution; per-token KL/TV certificates as a function of KV budget / dropped attention mass, composed with the level-set acceptance regions of [2606.30265](https://arxiv.org/abs/2606.30265); explains [2607.26627](https://arxiv.org/abs/2607.26627)'s failure taxonomy. Turns Dustin's unfalsifiable "negligible loss" into a knob. Directly enables #1 and #3.
- **Cite/beat**: [2606.30265](https://arxiv.org/abs/2606.30265) (certificates, no KV axis); VeriCache's linear-KL-growth measurement ([2605.17613](https://arxiv.org/abs/2605.17613)) as the empirical template; SpecPV's ROUGE cliff as the motivating failure.
- **2-week experiment**: measure per-token KL(full ‖ sparse-verify) vs retained attention mass across budgets/tasks at 32K; test whether captured-attention-mass is a usable online divergence estimator (correlation with realized KL). A single strong correlation plot validates the whole program.

### #3 — Acceptance-vs-(budget, length) surface and the B(L) law, 128K–1M [G7 + G14]
- **Novel delta**: swept surfaces for *retrieval/verification-recycled* sparse drafters (not window-local MTP) at 128K–1M; reconcile Vegas ("budget must grow") vs Windowed-MTP [2607.21535](https://arxiv.org/abs/2607.21535) ("constant window suffices at 1M"); unify quantized (QuantSpec) vs sparse (BudgetDraft) compression axes into MagicDec's (batch, seqlen) model → choose (bits, budget) from (B, S, HBM). Must anchor on Windowed-MTP and QuantSpec-128K.
- **Cite/beat**: TriForce budget curve @120K; Vegas 7%@96–120K; BudgetDraft table to 16K; QuantSpec 4K–128K single setting; MagicDec model (no compression axis).
- **2-week experiment**: one model (Llama-3.1-8B-1M or Qwen3), contexts {32K, 128K, 256K}, budgets {1K, 2K, 4K, 8K, %-based}, quantized vs top-k selection: fit α(B, L) and check whether iso-acceptance budget grows sublinearly, linearly, or is task-forked. 256K on 2 GPUs is feasible with offloaded verify (VeriCache pattern).

### #4 — Measured acceptance of EAGLE-3/MTP against KV-quantized targets (the production config) [G13]
- **Novel delta**: the config everyone runs (FP8-KV target + EAGLE-3) has *zero* published acceptance data — only an uncited dev.to claim (0.3–1.5 accepted-token drop) and vLLM issue [#37618](https://github.com/vllm-project/vllm/issues/37618). Sweep target KV precision (FP8/INT8/INT4, per-axis) × context × task; quantify double-lossy compounding (BF16-trained drafter features vs FP8-KV verification) and name the contract problem ("lossless relative to the wrong distribution").
- **Cite/beat**: SpecKV-γ ([2605.02888](https://arxiv.org/abs/2605.02888), weight-quant only — explicitly says the KV interaction is unstudied); [vLLM AMD EAGLE-3 blog](https://vllm.ai/blog/2026-07-13-eagle-3-amd-instinct) (AL 2.77, no KV-quant data); VeriCache (compressed KV as drafter only).
- **2-week experiment**: vLLM, Llama-3.3-70B + EAGLE-3 head, KV in {BF16, FP8, INT8} × contexts {4K, 32K, 64K} × 4 tasks; report AL and speedup deltas. Fully executable in days; near-guaranteed publishable finding either way.

### #5 — Staleness/refresh policy for verification-recycled importance over 10K+ generated tokens [G9]
- **Novel delta**: measure acceptance decay of stale sparse indices over long generations (nobody has: SparseSpec-L caps at 128 tokens; PillarAttn fixes k=8 with no ablation); build an acceptance-aware refresh controller (refresh when predicted acceptance loss > gather cost). Your selector's block-level statistics are natural drift features.
- **Cite/beat**: TriForce (first staleness observation, 2024); SparseSpec/PillarAttn (fixed k=8); SparseSpec-L (entropy controller for γ, not refresh); Dustin's lookahead fusion as an estimator ingredient.
- **2-week experiment**: PillarAttn-style setup on AIME long-CoT; freeze indices for {8, 64, 256, 1024} steps; plot acceptance vs staleness. The decay curve alone is a paper section; the controller follows.

### #6 — Adaptive full-verification triggering / error-accumulation control for partial verification [G3]
- **Novel delta**: replace SpecPV's fixed buffer-overflow refresh with an online divergence trigger from verification-side statistics (captured attention mass); bound divergence as f(budget, refresh interval). Pairs with #2; together they make lossy verification respectable.
- **Cite/beat**: SpecPV ([2512.02337](https://arxiv.org/abs/2512.02337), the quantified failure + heuristic); [2607.26627](https://arxiv.org/abs/2607.26627) (failure taxonomy, no control); Dustin (fixed heuristics).
- **2-week experiment**: instrument SpecPV; correlate attention-mass-missed with realized output drift on GovReport (the known cliff) vs QA (known-safe); if the signal separates the two task families, the trigger works.

### #7 — Length-robust *trained* sparse-KV drafting past 32K [G8]
- **Novel delta**: multi-budget sparse-view training (BudgetDraft) × long-position schemes (LongSpec Anchor-Offset) × drift-fix normalization (Attention Drift post-norm), evaluated at 64K+. Nobody has combined them; BudgetDraft's 16K collapse is a position-range artifact (drafter's native 2048 limit) begging for this fix.
- **Cite/beat**: BudgetDraft ([2606.00144](https://arxiv.org/abs/2606.00144), budget-robust not length-robust); LongSpec ([2502.17421](https://arxiv.org/abs/2502.17421), architecture without budget training); [2605.09992](https://arxiv.org/abs/2605.09992) (norm fixes).
- **2-week experiment**: retrain BudgetDraft's llama-68m with Anchor-Offset positions on 32K data; test whether 16K acceptance moves off ~18%. Cheap (68M params), decisive.

### #8 — Online (drafter KV budget × γ) controller across (batch, context, load) [G12]
- **Novel delta**: the γ axis is covered (Nightjar, FASER, SpecKV-γ); the *budget* axis is untouched at runtime — Vegas fixes sparsity per run, BudgetDraft never deploys its robustness. Needs an α(task, L, B) predictor (which #3 produces) + joint policy under HBM pressure. Position as adding the budget dimension to Nightjar/FASER/SpecKV.
- **2-week experiment**: with a budget-robust drafter, grid-search (B, γ) offline per (batch, context) cell; show the optimum moves and a 2-feature predictor recovers ≥90% of oracle throughput.

### #9 — Self-speculation for sequential-hybrid / SSM targets [G4]
- **Novel delta**: (a) LayerSkip-style training on a hybrid checkpoint (never done); (b) shared-recurrent-state self-spec semantics (ReplaySSM handles separate-path rollback only); (c) checkpoint/replay overhead at 100K. High strategic value (2026 frontier is hybrid) but moderate fit and heavier lift.
- **Cite/beat**: [2605.01106](https://arxiv.org/abs/2605.01106) (α=0.038 diagnosis); ReplaySSM RFCs ([#47572](https://github.com/vllm-project/vllm/issues/47572); concurrent infra, cite not claim); SpecLA ([2607.16673](https://arxiv.org/abs/2607.16673), 1.70x ceiling).
- **2-week experiment**: apply layer-dropout + early-exit loss finetune to a small sequential hybrid (0.5–1B); check whether self-spec α recovers from 0.038 toward the ~12x early-exit recovery already reported.

### #10 — No-regret draft-path adaptation over 30K–100K-token generations [G6]
- **Novel delta**: bandit machinery exists (BanditSpec [2505.15141](https://arxiv.org/abs/2505.15141), Not-a-Bandit [2510.20064](https://arxiv.org/abs/2510.20064), OnlineSpec [2603.12617](https://arxiv.org/abs/2603.12617)) but never for combinatorial path selection (skip set / KV budget) under self-generation drift on long CoT. Must position against those explicitly.
- **2-week experiment**: run EXP3-style budget selection vs fixed budget on 13K-token AIME traces inside PillarAttn; report dynamic-regret and wall-clock delta.

### #11 — SSM/linear-attention drafter acceptance past 8K [bonus gap]
- **Novel delta**: acceptance-vs-state-capacity curve for a Mamba/GDN drafter at 32K–128K vs a transformer target — the regime O(1) state was built for, never measured (Mamba Drafters stop at 8K).
- **2-week experiment**: Mamba-130M drafter + Llama-3.1-8B target on LongSpecBench extended to 128K; compare vs OWL-LSTM.

### #12 — 2D (layer × KV-budget) knapsack drafter [G2]
- The composition is fully novel and your DP/selector skills fit — but §6.7 shows it's ~+2% at B≥8/128K and acceptance-coupled downward. **Only worth doing framed as batch-1, 512K–1M self-spec** (+30–40% predicted). Cite KnapSpec + Vegas as the ingredients. Kill test: measure α of (50% skip × 4K sparse) jointly at 64K — if α_comb < α_skip·α_sparse (super-multiplicative damage), kill it.

### #13 — Layer-skip/early-exit acceptance curves at 64K–1M + LayerSkip training at 128K [G1]
- Novel but the arithmetic caps the payoff at ~1.5x. Worth a *measurement section* inside another paper (e.g., #12's kill test), not a project. Cite KnapSpec, AdaSkip ([2501.02336](https://arxiv.org/abs/2501.02336)), [2603.23701](https://arxiv.org/abs/2603.23701) (redundancy shrinking in modern checkpoints).

### #14 — Layer-skip self-spec under continuous batching [G5]
- Real gap (all layer-skip is batch-1; SparseSpec deliberately routed around it), but it's a serving-engineering project attacking a direction with a 1.5x ceiling. Only pursue if #12's niche pans out.

---

## 3. ☠️ WHAT WOULD KILL THIS PROJECT

| # | Risk | Why it kills | Hedge |
|---|---|---|---|
| 1 | Lossless headroom nearly exhausted | Sparse drafting already drives c→0; verification is 85–95% of round time, hard-capped at τ·BW/(W+S·kvB) — 30 tok/s at 1M on H100 — and τ is log-bounded ([2512.11718](https://arxiv.org/abs/2512.11718)). If QuantSpec-style quantized KV, VeriCache offload-overlap, and MLA kernels suffice, "better sparse drafting" papers add single-digit percent. | The frontier *is* the verify side — bets #1/#2 target it directly, not drafting. |
| 2 | Architecture ground shifting | Frontier long-context models are MLA and sequential hybrids: MLA makes sparse-KV drafting worth 1.01–1.43x below batch ~64 (§6.9); sequential hybrids draft at α=0.038 ([2605.01106](https://arxiv.org/abs/2605.01106)). "Does this matter for DeepSeek/Qwen3.5-class models?" — honest answer today: no. | Gap #9 (hybrid self-spec) is the transfer play; measurement gaps #3/#4 stay valid on the deployed GQA fleet. |
| 3 | Field converged fast, crowded | Vegas, PillarAttn, SparseSpec-L, Dustin, SpecPV, BudgetDraft, VeriCache all landed within ~12 months on the same recipe; residual deltas are narrower than they look; ReplaySSM already ate part of G4. | Move fast on the weeks-not-months measurement papers (#3/#4); position explicitly against each named neighbor. |
| 4 | Lossy-verification defection | If Dustin/SpecPV-style lossiness (6–9x) becomes the accepted norm, lossless long-context spec decode becomes a niche contract; the winning program is quality-bounded lossy inference. | Certification (#2) is the entry ticket to that program either way — the stated hedge. |
| 5 | Headline regime experimentally inaccessible | 256K–1M needs multi-node full-KV verification; academic setups cap at 64K (single A40) or 128K. Extrapolations can invert — Windowed-MTP ([2607.21535](https://arxiv.org/abs/2607.21535)) showed a constant window suffices at 1M for MTP heads. | Offloaded verify (VeriCache pattern) reaches 256K on 2 GPUs; claim only where you can measure. |
| 6 | Acceptance is checkpoint-idiosyncratic | OWL-collapse vs (unverified) MiniMax-flat; DFlash fixed by 1.6K fine-tuning samples. An acceptance-drift phenomenon may be a training-data artifact patched by a fine-tune, not a law. | Evaluate on multiple checkpoints; report per-checkpoint, never as a universal claim. |
| 7 | Serving reality | Engines can't compose the features (TRT-LLM: no dynamic trees on MLA/SWA; vLLM PP+SD gaps; batched-SD correctness only fixed in 2025); gains compress 2–3x → 1.4–2.0x at production concurrency; heterogeneous per-request sparse paths break kernel homogeneity. A batch-1 bespoke-kernel method will not ship. | Design batch-compatible from day one — SparseSpec routed around batched layer-skip for exactly this reason. |
| 8 | The cheap baselines are embarrassing | PLD/SAMD/SuffixDecoding: zero training, zero GPU state, τ2.75–4.98 at 4–64K, beat EAGLE-3, already in vLLM. | Benchmark against SAMD+TR (τ4.98) and HOWL (τ6.14) from day one — not just AR decoding. |

**Net position**: the defensible bets for this profile are #1 (both-sides-sparse via an external block-sparse oracle), #2+#6 (certification + control of sparse verification), and #3+#4 (the measurement papers nobody has done, executable in weeks). Avoid layer-skip-centric framings (arithmetic-capped) and pure drafting improvements (bottleneck moved).

---

## 4. ✅ WHAT IS SETTLED

| # | Fact | Evidence |
|---|---|---|
| 1 | Short-trained draft heads fail at long context | EAGLE-3: τ1.28, 0.81x on 4K–64K (OWL [2510.07535](https://arxiv.org/abs/2510.07535)); vanilla SD τ0.5–0.6 @16K ([2505.20776](https://arxiv.org/abs/2505.20776)); naive sparse drafter ~0% α by 8K ([2606.00144](https://arxiv.org/abs/2606.00144)); vLLM/SGLang consensus: EAGLE-family decays past trained window (>32K) |
| 2 | At long context + batch, MHA/GQA decode is KV-bandwidth-bound — and that's an *opportunity*: speedup grows with batch and context | MagicDec 2.51x @batch 32–256 ([2408.11049](https://arxiv.org/abs/2408.11049)); QuantSpec 1.35x@4K→2.49x@128K ([2502.10424](https://arxiv.org/abs/2502.10424)); SpecPV 2.88x@10K→6.29x@60K ([2512.02337](https://arxiv.org/abs/2512.02337)); roofline validated to 3% vs VeriCache's 25ms@500K |
| 3 | The converged recipe: draft against the target's own sparse/compressed/retrieved state, verify full-KV lossless | TriForce, MagicDec, LongSpec, QuantSpec, Vegas, PillarAttn, SparseSpec-L, VeriCache all instantiate it |
| 4 | Selection mechanism matters more than budget: attention-retrieval holds, permanent eviction collapses | TriForce @120K needle: retrieval 0.9878 vs StreamingLLM 0.0519, H2O 0.0739; SnapKV > StreamingLLM at equal budget (MagicDec) |
| 5 | Quantized drafter KV is acceptance-robust; sparse token-dropping is not | QuantSpec acceptance *rises* to 94.31% @128K; quantized beats sparse for draft-target agreement; VeriCache sustains 25–40 tok/round at 4x compaction |
| 6 | Verification attention is a free importance oracle for drafting | Vegas, PillarAttn, SparseSpec-L, SpecExtend recycle it at near-zero cost; beats static budgets on shifting saliency (SparseSpec 1.36x over MagicDec, 1.76x over TriForce) |
| 7 | Zero-KV retrieval drafting beats trained heads on long grounded inputs | OWL third-party: PLD τ2.75, TR τ3.16, SAMD τ3.18, SuffixDecoding τ3.41, SAMD+TR τ4.98 vs EAGLE-3 τ1.28 |
| 8 | Accepted tokens per iteration are log-bounded in verifier capacity | [2512.11718](https://arxiv.org/abs/2512.11718), tight vs EAGLE-3 — giant trees have sharply diminishing returns |
| 9 | Fixed γ is wrong | Optimal draft length shifts with batch ([2310.18813](https://arxiv.org/abs/2310.18813)), load ([2406.14066](https://arxiv.org/abs/2406.14066)), precision ([2605.02888](https://arxiv.org/abs/2605.02888)); optimal policy is a threshold stopping rule ([2405.19715](https://arxiv.org/abs/2405.19715)) |
| 10 | Batched speculation was silently incorrect in major open-source stacks as late as 2025 | [2510.22876](https://arxiv.org/abs/2510.22876); gains compress at production concurrency (Meta 1.4–2.0x; P-EAGLE +5–25% @c=64) |
| 11 | MLA changes the physics | Arithmetic intensity >100x MHA, toward compute-bound ([2507.15465](https://arxiv.org/abs/2507.15465)); MLA KV 69KB/tok vs Qwen3-32B 256KB/tok; DeepSeek shipped weight-amortizing MTP (1.8x), not KV-sparse drafting — correctly, per the arithmetic |
| 12 | No published acceptance/speedup datapoint exists past 128K, in any family | TriForce (2024) + QuantSpec remain the high-water marks; Windowed-MTP ([2607.21535](https://arxiv.org/abs/2607.21535)) is the lone 1M draft-KV datapoint, window-local MTP heads only |

---

## 5. ⚖️ WHAT IS CONTESTED

| # | Question | One side | Other side | Where it stands |
|---|---|---|---|---|
| 1 | Does acceptance decay with context length? | OWL: EAGLE-3 collapses (τ1.28, 0.81x) | MiniMax-M3 blog: AL flat 2.64→2.63 from 1K→32K — **unverifiable, not load-bearing** | It's a property of draft-state construction (shared/quantized/retrieved holds to 128K; evicted/short-trained collapses by 8–32K) + training coverage. No law — measure per checkpoint |
| 2 | Must the sparse budget grow with context? | Vegas: fixed budgets underperform at extreme scale | Windowed-MTP ([2607.21535](https://arxiv.org/abs/2607.21535)): constant sink+window suffices at 1M for MTP heads (~99% draft KV dropped) | Likely drafter-family- and task-dependent; unreconciled → gap #3 |
| 3 | Is sparse verification's quality loss "negligible"? | Dustin / [2512.21911](https://arxiv.org/abs/2512.21911): yes, empirically @32K | SpecPV's own tables: ROUGE-L −27% on GovReport while QA stays within 1–3% | Uncertified either way; no KV-budget→divergence bound exists → gap #2 |
| 4 | Does dynamic layer-skip beat static at long inputs? | Dynamic should adapt | KnapSpec: CLaSp (dynamic) 1.22x *below* SWIFT (static) 1.33x @16K; DEL "fails to accelerate" base models | Dynamic ≠ better |
| 5 | Is verification "free" at scale? | Memory-bound argument: extra verify tokens ride along | Holds only while B(L+1) < ridge (~296 FLOP/B on H100 → breaks at B>59, L=4); 3/5 consumer configs decelerate ([2607.17283](https://arxiv.org/abs/2607.17283)) | Long context re-anchors memory-boundedness; short context + big batch does not |
| 6 | Do diffusion drafters survive long context? | DFlash decays to τ2.09 @32K zero-shot | A 1.6K-sample fine-tune recovers τ3.56 | Data patch or architectural limit? No data past 32K |
| 7 | KV-reuse drafting: principle vs wall-clock | KVShot ([2604.26412](https://arxiv.org/abs/2604.26412)): reading target KV improves long-range acceptance | e2e speedup marginal — shallow drafters can't estimate target queries; sparse gradients | Contested whether block-wise training fixes it |
| 8 | Thin-evidence zones | — | — | SSM drafters past 8K: nothing; cross-tokenizer at length: nothing; MTP acceptance-vs-length in production: never published; 256K–1M: nothing; StreamServe's 11–18x: implausible baselines (4 GPUs, 80 queries) |

---

## 6. 🧮 THE ARITHMETIC

Constants: H100 3.35TB/s, 990 TF BF16, ridge 296 FLOP/B. Model validated to 3% vs VeriCache. Speedup = τ/(L·c+1), c = draft/target step cost.

| # | Question | The math | Verdict |
|---|---|---|---|
| 1 | KV per token | Qwen3-32B 256KB/tok → 8.6/34.4/137/275 GB @32K/128K/512K/1M; Llama-3.1-8B 128KB/tok; DeepSeek MLA 69KB/tok; Qwen3-1.7B *and* 0.6B both 112KB/tok (GQA floors at 8 kv-heads) | Small GQA models do NOT have small KV; a full-KV drafter = +44% KV/request → −31% max batch @128K on 8xH100 |
| 2 | Weights vs KV — which term dominates? | At B=1/128K a GQA 32B is still weight-dominated (KV = 34% of bytes); KV dominates at (B≥8, 128K+) or (B=1, 512K+) | Which term you attack depends on (B, S) |
| 3 | Independent full-KV drafter | c → 0.4375 asymptotically as B·S grows | Speedup ceiling **1.09x** even at perfect acceptance — dead twice (arithmetic + acceptance) |
| 4 | Sparse-KV drafter (4K budget) | c = 0.039 @128K/B1 → 0.016 @128K/B32; 2.6→3.0x at τ=3; **3.85x at τ=4.1, B=32, 128K** | Speedup monotonically approaches τ with S and B — the arithmetic loves exactly this direction |
| 5 | Layer skip | c = 1−k, S-invariant; 50% skip at τ=3, L=4 → **1.00x**; matching a 4K sparse draft @128K/B8 needs 78% skip | Reproduces the whole literature (KnapSpec 1.47x cap); 78% skip is acceptance-fatal |
| 6 | Verify dominance | With cheap drafting, verification = **85–95% of round time**; ceiling τ·BW/(W + S·kvB): Qwen3-32B B=1 → 135/101/49/**30 tok/s @1M** (τ=3; 59 @τ=6; B200 only 2.4x better) | τ is log-bounded → **lossless verify-byte reduction is the frontier, not better drafting** |
| 7 | Layer-skip × sparse-KV composition | Multiplicative in cost, sub-additive in speedup (B=32/1M: 2.87x→2.93x, +2%); acceptance multiplies *down* (0.9×0.85→0.765 → τ −23%) | Can be net-negative; genuine niche = batch-1, 512K–1M self-spec (+30–40%) |
| 8 | Self-spec regime gate | Wins iff B·S·kvB > W → S_inflection = 256K/B for Qwen3-32B (B=8→32K; B=1→~400K); ~26K for MHA-7B | Explains why TriForce used MHA and why MagicDec needs batch ≥32 |
| 9 | MLA targets | KV = 1.4–30% of decode bytes at B=1–32 up to 128K → sparse-KV drafting max gain 1.01–1.43x | **Arithmetic malpractice below batch ~64**; weight-amortizing MTP is the only lever (measured 1.8x); KV-centric research does not transfer |
| 10 | Batch ceiling | Verify exits memory-bound at B(L+1) ≳ 296 (B>59 at L=4); long context postpones the ridge (KV intensity (L+1)·g ≈ 40 ≪ 296) | The arithmetic behind all measured concurrency decay; **speculation and batching are complementary at 128K–1M exactly where they conflict at 4K**; γ must shrink with B |

---

## 7. 📚 ANNOTATED BIBLIOGRAPHY: THE LANDSCAPE BY METHOD FAMILY

<!-- GENERATED by build.py from papers.yaml — edit those, not this section. -->

*Reference material — do not read linearly. Each family opens with its verdict;
that is the takeaway. Pull individual papers only when a gap or experiment needs them.*

**Two regimes with opposite physics.** Short-context decoding is weight-bound
(EAGLE-3/DFlash arms race, 5–7x). Long-context decoding is KV-bound for MHA/GQA,
where short-trained heads go *sub-1x*. Every family verdict is relative to the
32K–1M input / long-generation regime. Acceptance cells use α = rate, τ = accepted
length (tokens/round) — they are not comparable across papers without the qualifiers.

| Family | Verdict | Papers |
|---|---|---|
| [Independent AR drafters](#independent-ar-drafters) | ☠️ Dead at long context — GQA floors drafter KV (speedup ceiling ~1.1x) and acceptance collapses by 8–16K. | 2 |
| [Draft heads (Medusa / EAGLE / MTP)](#draft-heads-medusa-eagle-mtp) | ❌ Die beyond the trained window unless drafter state is O(1) or norm-fixed; trained-short is the failure, not the head architecture. | 10 |
| [Self-speculation: layer skip](#self-speculation-layer-skip) | 🪦 Capped at ~1.0–1.5x — cost ratio is context-invariant, so it cannot touch the KV term that dominates 128K+. Secondary multiplier at best. | 9 |
| [Early exit](#early-exit) | ❓ Zero long-context evidence anywhere; structurally 1M-friendly (shared exit-layer KV) but unmeasured past 16K and no 128K training recipe exists. | 5 |
| [Sparse-KV drafting](#sparse-kv-drafting) | ✅ The converged recipe — survives and strengthens with context and batch. Constraints: selection must be attention-retrieval not eviction; budget likely grows with S; verification becomes the bottleneck. | 10 |
| [KV-compression drafting](#kv-compression-drafting) | ✅ Strongest acceptance-vs-length results in the literature; the 128K high-water marks live here or adjacent. | 5 |
| [Sparse / partial verification](#sparse-partial-verification) | 🔥 The 2026 frontier — attacks the true bottleneck (verify KV bytes = 85–95% of round time) but silently abandons the target distribution. No divergence certificates exist. | 5 |
| [Retrieval / n-gram / suffix drafting](#retrieval-n-gram-suffix-drafting) | ✅ Only family trivially length-robust on the draft side (zero drafter KV); beats trained heads at 4–64K in third-party evals; task-dependent; unmeasured past 64K. | 9 |
| [Block-diffusion drafters](#block-diffusion-drafters) | ⚠️ SOTA short-context drafting; decays hard by 32K zero-shot; bidirectional attention over long prompts is quadratic and unanalyzed; nothing past 32K. | 6 |
| [Tree speculation & theory](#tree-speculation-theory) | 📐 No theory anywhere has a context-length axis; the log(P) acceptance ceiling kills giant trees; optimal tree-vs-context is unsolved. | 14 |
| [Serving systems & runtime control](#serving-systems-runtime-control) | 🏭 The engines serving 128K–1M cannot compose the needed features (no dynamic trees on MLA/SWA in TRT-LLM; vLLM PP+SD gaps); gains compress to 1.4–2.0x at production concurrency. | 13 |
| [SSM / linear / hybrid drafters & targets, cross-tokenizer](#ssm-linear-hybrid-drafters-targets-cross-tokenizer) | 🕳️ The O(1)-state drafters that should dominate at 128K have never been measured there; sequential-hybrid targets are effectively un-draftable via component reuse (α=0.038). | 8 |

### Independent AR drafters

**Verdict: ☠️ Dead at long context — GQA floors drafter KV (speedup ceiling ~1.1x) and acceptance collapses by 8–16K.**

| Paper | Summary | Lossless | Max ctx | Acceptance | Speedup | Long-ctx note |
|---|---|---|---|---|---|---|
| [Speculative Decoding (Leviathan et al.)](https://arxiv.org/abs/2211.17192) '23 | Original draft-then-verify with rejection sampling; introduced α, γ and the speedup formula | ✓ | <2K | — | 2–3.4x vs AR (T5-XXL) | Drafter carries full growing KV; no KV-bound analysis |
| [Speculative Sampling (Chen et al.)](https://arxiv.org/abs/2302.01318) '23 | Concurrent discovery; modified rejection sampling; 4B drafter for Chinchilla-70B | ✓ | <2K | — | 2–2.5x vs AR (Chinchilla-70B) | No long-context analysis |

### Draft heads (Medusa / EAGLE / MTP)

**Verdict: ❌ Die beyond the trained window unless drafter state is O(1) or norm-fixed; trained-short is the failure, not the head architecture.**

| Paper | Summary | Lossless | Max ctx | Acceptance | Speedup | Long-ctx note |
|---|---|---|---|---|---|---|
| [Medusa](https://arxiv.org/abs/2401.10774) '24 | Parallel FFN heads off last hidden state + tree attention; optional lossy typical acceptance | ~ | <4K | — | 2.2–3.6x vs AR | Constant drafter memory, but acceptance degrades on long inputs |
| [EAGLE](https://arxiv.org/abs/2401.15077) '24 | Feature-level AR head with token-shift disambiguation; static draft tree | ✓ | <4K | — | 2.7–3.5x vs AR | Head keeps its own growing feature-KV; acceptance collapses past trained window |
| [EAGLE-2](https://arxiv.org/abs/2406.16858) '24 | Confidence-calibrated dynamic draft trees (drafter confidence ≈ acceptance) | ✓ | <4K | τ4–5.5 (short ctx) | 3.05–4.26x vs AR | Sub-1.0x at batch ~24 in SGLang; long-context decay inherited from EAGLE-1 |
| [EAGLE-3](https://arxiv.org/abs/2503.01840) '25 | Direct token prediction + multi-layer feature fusion; production default in vLLM/SGLang/TRT-LLM | ✓ | 64K (via OWL) | τ1.28 @4–64K (OWL) | ≤6.5x short ctx; 0.81x long ctx | Long context is its known failure mode; +38% throughput at batch 64 short-ctx |
| [EAGLE 3.1](https://vllm.ai/blog/2026-05-26-eagle-3-1) '26 | FC/post-norm fixes for attention drift at deep speculation | ✓ | 32K | ~2x AL vs EAGLE-3 (long ctx) | 2.03x @c=1 → 1.66x @c=16 | No acceptance-vs-context curve past 32K |
| [P-EAGLE](https://arxiv.org/abs/2602.01469) '26 | Single-pass parallel drafting (K=7) | ✓ | — | — | 1.69x over EAGLE-3; +55–69% @c=1 → +5–25% @c=64 | Gains compress with concurrency |
| [OWL](https://arxiv.org/abs/2510.07535) '25 | O(1)-state LSTM drafter off final hidden state + [SPEC] token; hybrid tree/non-tree | ✓ | 100K+ | τ4.00–4.27; HOWL hybrid τ6.14 | 2.35x (HOWL 3.08x) on Llama-3.3-70B | Introduces LongSpecBench; acceptance ~length-independent by construction |
| [Attention Drift](https://arxiv.org/abs/2605.09992) '26 | Diagnoses drafter hidden-state magnitude growth pulling attention off the prompt; post-norm fix | ✓ | — | — | 1.18x on long-context tasks | Norm fix transfers across head-based drafters |
| [DeepSeek-V3 MTP](https://arxiv.org/abs/2412.19437) '24 | Weight-amortizing multi-token-prediction head, shipped in 128K production | ✓ | 128K (production) | 85–90% second-token | 1.8x TPS | No acceptance-vs-length breakdown ever published |
| [Windowed-MTP](https://arxiv.org/abs/2607.21535) '26 | Sink + constant window KV for MTP heads (~99% draft KV dropped) | ✓ | 1M | matched vs full-KV MTP @1M | — | Lone 1M draft-KV datapoint; deflates 'budget must grow' stories for MTP heads |

### Self-speculation: layer skip

**Verdict: 🪦 Capped at ~1.0–1.5x — cost ratio is context-invariant, so it cannot touch the KV term that dominates 128K+. Secondary multiplier at best.**

| Paper | Summary | Lossless | Max ctx | Acceptance | Speedup | Long-ctx note |
|---|---|---|---|---|---|---|
| [Draft & Verify](https://arxiv.org/abs/2309.08168) '24 | Bayesian-optimized static skipped-layer set; zero extra params or memory | ✓ | <4K | ~92% α | ≤1.99x | Drafting still attends full KV — cost ratio context-invariant |
| [SWIFT](https://arxiv.org/abs/2410.06916) '25 | Plug-and-play online discovery of the optimal skipped-layer set per input stream | ✓ | 16K (via KnapSpec) | 98–100% token accept | 1.3–1.6x; 1.33x @16K | — |
| [CLaSp](https://arxiv.org/abs/2505.24196) '25 | In-context dynamic layer skip via per-step DP | ✓ | 16K (via KnapSpec) | — | 1.24–1.73x; 1.22x @16K | Dynamic ≠ better — below static SWIFT at 16K |
| [DEL](https://arxiv.org/abs/2504.05598) '25 | Dynamic exit-layer and draft-length selection | ✓ | — | — | 2.16–2.62x | 'Fails to provide acceleration' on base models (KnapSpec) |
| [CAS-Spec](https://arxiv.org/abs/2510.26843) '25 | Cascaded adaptive self-speculation | ✓ | — | — | 1.1–2.3x | — |
| [DVI](https://arxiv.org/abs/2510.05421) '25 | Online drafter learning during inference | ✓ | — | — | 2.16x | — |
| [KnapSpec](https://arxiv.org/abs/2602.20217) '26 | 0/1 knapsack over attention/MLP sublayers with context-dependent latency model; cosine acceptance proxy (0.837 corr.) | ✓ | 16K | — | 1.47x peak @~16K | First context-dependent skip selection; still confirms the ~1.5x family cap |
| [AdaSkip](https://arxiv.org/abs/2501.02336) '25 | Adaptive sublayer skipping for long-context inference | ✓ | — | — | — | Skip-selection ingredient for gap #13 |
| [Layer-redundancy analysis](https://arxiv.org/abs/2603.23701) '26 | Modern pretraining reduces layer redundancy — shrinking skip/exit headroom | — | — | — | — | Analysis, not a method; threat to the whole family |

### Early exit

**Verdict: ❓ Zero long-context evidence anywhere; structurally 1M-friendly (shared exit-layer KV) but unmeasured past 16K and no 128K training recipe exists.**

| Paper | Summary | Lossless | Max ctx | Acceptance | Speedup | Long-ctx note |
|---|---|---|---|---|---|---|
| [LayerSkip](https://arxiv.org/abs/2404.16710) '24 | Layer-dropout + early-exit-loss training; draft shares exit-layer KV — zero drafter memory | ✓ | <4K | — | 1.82–2.16x | Structurally the most 1M-friendly design; no 128K-recipe checkpoint exists |
| [Kangaroo](https://arxiv.org/abs/2404.18911) '24 | Fixed shallow sub-network + adapter with double early exit | ✓ | <4K | — | 2.04x | — |
| [EESD](https://arxiv.org/abs/2406.03853) '24 | Early-exit drafting with Thompson-sampling draft length | ✓ | — | — | 2.29x (70B GSM8K) | — |
| [PPSD](https://arxiv.org/abs/2509.19368) '25 | Verify-while-draft pipelining across exit points | ✓ | — | — | 2.01–3.81x | — |
| [Mirror-SD](https://arxiv.org/abs/2510.13161) '25 | Bidirectional early-exit speculation across heterogeneous accelerators | ✓ | — | — | 2.8–5.8x; +30% over EAGLE-3 | Strongest early-exit numbers — entirely short-context |

### Sparse-KV drafting

**Verdict: ✅ The converged recipe — survives and strengthens with context and batch. Constraints: selection must be attention-retrieval not eviction; budget likely grows with S; verification becomes the bottleneck.**

| Paper | Summary | Lossless | Max ctx | Acceptance | Speedup | Long-ctx note |
|---|---|---|---|---|---|---|
| [TriForce](https://arxiv.org/abs/2404.11912) '24 | Target self-drafts on retrieval-KV (~4K budget), full-KV lossless verify; hierarchical with tiny drafter below | ✓ | 122K | 0.96–0.99 α @120K | 2.31x @122K (A100); 7.78x offloaded | Canonical 128K result; eviction baselines collapse (α 0.05–0.07 on needle); staleness grows with generation; α ~0.29 @T=1.0 |
| [MagicDec](https://arxiv.org/abs/2408.11049) '25 | StreamingLLM/SnapKV-cache drafter + (batch, seqlen) regime analysis: speedup grows with batch beyond S_inflection | ✓ | 100K | 0.84 α @4K → 0.79 α @100K | up to 2.51x @batch 32–256 | SnapKV > StreamingLLM selection; GQA raises S_inflection (MHA 2.0x vs GQA 1.84x @32K) |
| [LongSpec](https://arxiv.org/abs/2502.17421) '25 | Constant 512-token drafter window + cross-attention into the target's KV (zero duplicate storage); Anchor-Offset positions; hybrid tree attention | ✓ | 32K | — | 3.26x; 2.25–2.34x AIME24/QwQ | Degrades at 25–32K |
| [SpecExtend](https://arxiv.org/abs/2505.20776) '25 | Training-free retrofit: target attention selects the drafter's KV | ✓ | 16K | vanilla-SD baseline τ0.5–0.6 @16K (measured) | 2.65–3.21x @16K (from 1.38–1.61x) | — |
| [Vegas](https://arxiv.org/abs/2602.07223) '26 | Verification attention recycled as the drafting mask — zero selection cost | ✓ | 120K | τ~6.1/7 @7% sparsity | 1.25–2.81x vs vLLM; 1.15–1.29x vs sparse SOTA | Longest self-spec evaluation published (96–120K); warns fixed budgets fail at extreme scale |
| [SparseSpec (PillarAttn)](https://arxiv.org/abs/2512.01278) '26 | Only serving-grade self-spec (batch 256, TP); dynamic pillar-token selection | ✓ | — | 6.16/8 accepted on reasoning CoT (EAGLE-3/n-gram <2) | 2.13x vs vLLM; 1.36x vs MagicDec; 1.76x vs TriForce | Static budgets fail on shifting saliency; k=8 fixed with no staleness ablation |
| [SparseSpec-L](https://arxiv.org/abs/2607.27735) '26 | Training-free recallable eviction — dense KV retained off the hot path | ✓ | 64K | 52–84.6% α | 2.79x @~10% KV ratio | Single-A40 cap; 128-token generations only |
| [BudgetDraft](https://arxiv.org/abs/2606.00144) '26 | Multi-view budget training for budget-robust sparse drafters | ✓ | 16K | 91.6% α @4K → ~18% α @16K | 6.55x / 4.46x / 2.10x @4K/8K/16K | Budget-robust ≠ length-robust; drafter's native 2048 position range is the suspected artifact (gap #7) |
| [SPIRe](https://arxiv.org/abs/2504.06419) '25 | Sparse-KV self-spec variant | ✓ | — | — | +35% over MagicDec-style (modeled) | Modeled, not measured |
| [KVShot](https://arxiv.org/abs/2604.26412) '26 | Drafter reads target KV to rescue long-range acceptance | ✓ | — | improves long-range acceptance | marginal e2e | Shallow drafters can't estimate target queries; sparse gradients; block-wise training contested |

### KV-compression drafting

**Verdict: ✅ Strongest acceptance-vs-length results in the literature; the 128K high-water marks live here or adjacent.**

| Paper | Summary | Lossless | Max ctx | Acceptance | Speedup | Long-ctx note |
|---|---|---|---|---|---|---|
| [QuantSpec](https://arxiv.org/abs/2502.10424) '25 | Self-spec with hierarchical INT4/INT8 shared quantized KV + 4-bit weights | ✓ | 128K | 91.6% α @16K → 94.3% α @128K (rises with length) | 1.35x @4K → 2.49x @128K | Quantized beats sparse drafter KV for draft-target agreement |
| [VeriCache](https://arxiv.org/abs/2605.17613) '26 | Same model drafts on compressed KV; full-KV verify off-GPU, overlapped with PCIe | ✓ | 500K–1M (~327GB KV @1M) | 25–40 accepted tokens/round (small drafters: 2–3) | 4x throughput, bit-identical | Measures KL(compressed‖full) growing ~linearly (~6 nats/250 steps @KVzip 4x) — why unverified lossy KV fails |
| [SpecKV-γ](https://arxiv.org/abs/2605.02888) '26 | Optimal draft length shifts with weight precision (FP16: 2–4; INT8: 6–8; NF4: 4–6) | ✓ | — | ~0.70 α across precisions | +56% over fixed γ=4 | Explicitly flags KV-quant x SD interaction as unstudied (gap #4) |
| [Lynx](https://arxiv.org/abs/2607.01831) '26 | Bit-plane KV-transfer speculation | ✓ | — | — | 1.43x TTFT | — |
| [MLA-drafter conversion](https://arxiv.org/abs/2607.27269) '26 | MLA-converting drafters break acceptance via attention-function error; calibration fixes 37/64 cells | ✓ | — | — | — | — |

### Sparse / partial verification

**Verdict: 🔥 The 2026 frontier — attacks the true bottleneck (verify KV bytes = 85–95% of round time) but silently abandons the target distribution. No divergence certificates exist.**

| Paper | Summary | Lossless | Max ctx | Acceptance | Speedup | Long-ctx note |
|---|---|---|---|---|---|---|
| [Dustin](https://arxiv.org/abs/2606.24957) '26 | Draft-lookahead + target attention history select the verification-KV subset | ✗ | 32K | — | 27.85x self-attention; 9.17x e2e (Qwen2.5-72B @32K) | 'Negligible' quality loss is uncertified |
| [SpecPV](https://arxiv.org/abs/2512.02337) '25 | Partial-KV verification + periodic full refresh (draft module is EAGLE-3) | ✗ | 60K | — | 2.88x @10K → 6.29x @60K | GovReport ROUGE-L 77.3 → 56.1 (−27%) at 2K budget — loss is task-dependent and unbounded |
| [SSV / SpecSA](https://arxiv.org/abs/2605.19893) '26 | Speculation for natively-sparse (NSA) targets | ~ | — | — | 3.49x vs AR-NSA; 6.86x kernel | Lossless w.r.t. the sparse target — a different contract |
| [Sparse verification (attn+FFN+MoE)](https://arxiv.org/abs/2512.21911) '25 | Sparsifies attention, FFN, and MoE in the verify pass; claims stable acceptance | ✗ | 32K | — | — | Quality claims empirical only |
| [Sparse-verification failure taxonomy](https://arxiv.org/abs/2607.26627) '26 | Taxonomy of where sparse verification fails | — | — | — | — | No control mechanism — motivates certification (gap #2) and triggering (gap #6) |

### Retrieval / n-gram / suffix drafting

**Verdict: ✅ Only family trivially length-robust on the draft side (zero drafter KV); beats trained heads at 4–64K in third-party evals; task-dependent; unmeasured past 64K.**

| Paper | Summary | Lossless | Max ctx | Acceptance | Speedup | Long-ctx note |
|---|---|---|---|---|---|---|
| [PLD (Prompt Lookup Decoding)](https://github.com/apoorvumang/prompt-lookup-decoding) '23 | N-gram match into the prompt as free drafts | ✓ | length-agnostic | τ2.75 (OWL third-party eval) | 2–4x on grounded tasks | Improves with more copyable context; trivially 1M-safe; in vLLM |
| [REST](https://arxiv.org/abs/2311.08252) '24 | Retrieval from an external datastore as drafts | ✓ | — | — | 1.62–2.36x | — |
| [SuffixDecoding](https://arxiv.org/abs/2411.04975) '25 | Suffix automaton over prompt + history; ~10.75 bytes/token CPU-RAM state | ✓ | — | τ3.41 (OWL eval) | 5.3x AgenticSQL; 2.8x vs EAGLE-2/3 | Shipped in vLLM |
| [Token Recycling](https://arxiv.org/abs/2408.08696) '24 | Recycles candidate tokens from earlier verify steps; <2MB state | ✓ | — | τ3.16 (OWL eval) | ~2x | — |
| [SAM-Decoding](https://arxiv.org/abs/2411.10666) '24 | O(1) suffix-automaton matching | ✓ | 64K (OWL eval) | τ3.18; SAMD+TR hybrid τ4.98 @4–64K | — | Beats EAGLE-3 (τ1.28) on long grounded inputs — the embarrassing cheap baseline |
| [AdaPLD](https://arxiv.org/abs/2606.05742) '26 | Adaptive prompt-lookup | ✓ | — | — | 3.10x | — |
| [DReSD](https://arxiv.org/abs/2502.15572) '25 | Dense (semantic) retrieval drafting instead of exact n-gram match | ✓ | — | +87% vs sparse retrieval | ≤4.64x | External corpus only |
| [RAPID](https://arxiv.org/abs/2502.20330) '25 | RAG drafter — even larger than the target; upward speculation | ~ | 120K | — | 2.10–2.69x @120K | Quality-improving (InfiniteBench 42.83 → 49.98), so not distribution-preserving; >1x only beyond ~32K |
| [TokenSwift](https://arxiv.org/abs/2502.18890) '25 | Accelerates 100K-token generation (long output, not input) | ✓ | 100K generated | — | >3x (5h → 90min) | — |

### Block-diffusion drafters

**Verdict: ⚠️ SOTA short-context drafting; decays hard by 32K zero-shot; bidirectional attention over long prompts is quadratic and unanalyzed; nothing past 32K.**

| Paper | Summary | Lossless | Max ctx | Acceptance | Speedup | Long-ctx note |
|---|---|---|---|---|---|---|
| [DFlash](https://arxiv.org/abs/2602.06036) '26 | 16-token block-diffusion draft per forward pass, conditioned on target features | ✓ | 32K | τ4.9–5.3 @1K → τ2.09 @32K zero-shot; τ3.56 @32K after 1.6K-sample LongAlign FT | >6x; 2.5x vs EAGLE-3 | Only diffusion context curve; is length-robustness a data patch or architectural limit? |
| [SpecDiff-2](https://arxiv.org/abs/2511.00606) '25 | Diffusion drafting, second generation | ✓ | — | — | 5.5x | — |
| [FailFast](https://arxiv.org/abs/2512.20573) '25 | Length-adaptive diffusion speculation (dLLM cost ~length-independent) | ✓ | — | 70-token accepts | 4.9x | — |
| [Graft](https://arxiv.org/abs/2605.20104) '26 | Hybrid tree construction: draft less, retrieve more | ✓ | — | — | 5.41x; +21.8% vs EAGLE-3 (Qwen3-235B) | — |
| [Spec-AUF](https://arxiv.org/abs/2607.01893) '26 | Diffusion drafting variant | ✓ | — | — | — | — |
| [DeLS-Spec](https://arxiv.org/abs/2607.07409) '26 | Diffusion drafting variant | ✓ | — | — | — | — |

### Tree speculation & theory

**Verdict: 📐 No theory anywhere has a context-length axis; the log(P) acceptance ceiling kills giant trees; optimal tree-vs-context is unsolved.**

| Paper | Summary | Lossless | Max ctx | Acceptance | Speedup | Long-ctx note |
|---|---|---|---|---|---|---|
| [SpecInfer](https://arxiv.org/abs/2305.09781) '23 | Token-tree speculation with tree verification | ✓ | — | — | — | — |
| [Sequoia](https://arxiv.org/abs/2402.12374) '24 | Hardware-aware DP-optimal tree shape | ✓ | — | — | 4.04x | No context axis in the DP |
| [SpecExec](https://arxiv.org/abs/2406.02532) '24 | Massive parallel speculation for offloaded inference | ✓ | — | — | 10–18x offloaded | — |
| [DeFT](https://arxiv.org/abs/2404.00242) '25 | Flash tree attention on paged KV | ✓ | — | — | 73–99% tree KV-IO cut | — |
| [SpecTr](https://arxiv.org/abs/2310.15141) '23 | Multi-draft selection as optimal transport; (1−1/e)-approximation guarantee | ✓ | — | — | 2.13x; 1.37x vs single-draft | — |
| [Markov-chain optimality](https://arxiv.org/abs/2411.00841) '24 | Optimal multi-draft SD via a Markov-chain view | ✓ | — | — | — | — |
| [Block Verification](https://arxiv.org/abs/2403.10444) '24 | Jointly verify draft blocks instead of token-by-token | ✓ | — | — | +5–8% free | — |
| [Multi-path block verification](https://arxiv.org/abs/2602.16961) '26 | Extends block verification to multiple paths | ✓ | — | — | +30% block efficiency | — |
| [Multi-draft OT bound](https://arxiv.org/abs/2502.18779) '25 | Tight acceptance bound for multi-draft speculation | ✓ | — | — | — | — |
| [SpecDec++](https://arxiv.org/abs/2405.19715) '24 | Draft-length choice as an optimal stopping problem | ✓ | — | — | +7–11% | — |
| [BRW speed-of-light bound](https://arxiv.org/abs/2512.11718) '25 | E[accepted/iter] ≤ (μ+μ₂)log(P)/μ² + O(1) — only logarithmic in verifier capacity | ✓ | — | — | — | Tight vs EAGLE-3; kills giant trees; the informational ceiling behind kill-risk #1 |
| [Acceptance level-set certificates](https://arxiv.org/abs/2606.30265) '26 | Certified acceptance regions for speculative decoding | ✓ | — | — | — | No KV axis — composition target for gap #2 |
| [MLA arithmetic intensity](https://arxiv.org/abs/2507.15465) '25 | MLA decode is >100x MHA arithmetic intensity — moves toward compute-bound | — | — | — | — | Why KV-centric spec decode doesn't transfer to MLA targets |
| [GLA kernels](https://arxiv.org/abs/2505.21487) '25 | Grouped latent attention kernels, 2x FlashMLA at q>1 | — | — | — | — | Makes MLA verify-friendly — supports MTP-style speculation on MLA |

### Serving systems & runtime control

**Verdict: 🏭 The engines serving 128K–1M cannot compose the needed features (no dynamic trees on MLA/SWA in TRT-LLM; vLLM PP+SD gaps); gains compress to 1.4–2.0x at production concurrency.**

| Paper | Summary | Lossless | Max ctx | Acceptance | Speedup | Long-ctx note |
|---|---|---|---|---|---|---|
| [SmartSpec / TurboSpec](https://arxiv.org/abs/2406.14066) '24 | Goodput-driven speculation-length control | ✓ | — | — | 3.2x latency | Load-aware, not length-aware |
| [AdaServe](https://arxiv.org/abs/2501.12162) '25 | SLO-customized speculative serving | ✓ | — | — | 4.3x fewer SLO violations | — |
| [FASER](https://arxiv.org/abs/2604.20503) '26 | Adaptive speculation in serving | ✓ | — | — | +53% throughput | — |
| [Saguaro](https://arxiv.org/abs/2603.03251) '26 | Overlap drafting with verification | ✓ | — | — | ~30% | — |
| [Batch SD Done Right](https://arxiv.org/abs/2510.22876) '25 | Open-source batched SD was silently wrong (draft-distribution bonus tokens, corrupted KV alignment) | — | — | — | — | Correctness result — audit any batched baseline against it |
| [Meta production study](https://arxiv.org/abs/2508.08192) '25 | Speculation at production concurrency | ✓ | — | — | 1.4–2.0x at production batch (vs 2–3x @bs=1) | The concurrency-compression datapoint |
| [DSpark](https://arxiv.org/abs/2607.05147) '26 | DeepSeek-V4 speculative serving | ✓ | — | — | +60–85% per-user at matched throughput | — |
| [TransKV](https://www.techrxiv.org/doi/full/10.36227/techrxiv.177101038.80960856/v1) '26 | Transactional KV: draft tokens stay in token-sized buffers until accepted, not block-granular pages | ✓ | — | — | 1.78x branch concurrency @B=16 | Draft pages otherwise eat scheduler token budget |
| **ReSpec** '26 | Speculative decoding for RL rollouts | ✓ | — | — | 4.5x rollouts | — |
| [Consumer-hardware audit](https://arxiv.org/abs/2607.17283) '26 | 3/5 consumer configs decelerate under speculation on serialized backends | — | — | — | — | Speculation is not free off-datacenter |
| [BanditSpec](https://arxiv.org/abs/2505.15141) '25 | Bandit selection of drafting configuration | ✓ | — | — | — | Runtime-control prior art for gap #10 |
| [Not-a-Bandit](https://arxiv.org/abs/2510.20064) '25 | Full-information online draft selection | ✓ | — | — | — | Runtime-control prior art for gap #10 |
| [OnlineSpec](https://arxiv.org/abs/2603.12617) '26 | Online speculation adaptation under drift | ✓ | — | — | — | Runtime-control prior art for gap #10 |

### SSM / linear / hybrid drafters & targets, cross-tokenizer

**Verdict: 🕳️ The O(1)-state drafters that should dominate at 128K have never been measured there; sequential-hybrid targets are effectively un-draftable via component reuse (α=0.038).**

| Paper | Summary | Lossless | Max ctx | Acceptance | Speedup | Long-ctx note |
|---|---|---|---|---|---|---|
| [Mamba Drafters](https://arxiv.org/abs/2506.01206) '25 | Mamba drafter with O(1) state (52GB vs 72GB peak @8K) | ✓ | 8K | τ2.80–2.87 vs EAGLE+YaRN τ2.64 @8K | — | Never measured past 8K — the regime O(1) state was built for (gap #11) |
| [ReDrafter](https://arxiv.org/abs/2403.09919) '24 | RNN drafter with O(1) state and beam drafting | ✓ | — | — | 2.3–2.8x | — |
| [SpecLA](https://arxiv.org/abs/2607.16673) '26 | Speculation for linear-attention targets | ✓ | — | — | ≤1.70x (recurrence serializes verification) | — |
| [Component-Aware SD](https://arxiv.org/abs/2605.01106) '26 | Hybrid-target drafting via component reuse | ✓ | — | parallel hybrids α=0.68; sequential α=0.038 (18x gap) | — | Sequential hybrids effectively un-draftable via reuse; LayerSkip-style exit recovers ~12x (gap #9) |
| [ReplaySSM](https://github.com/vllm-project/vllm/issues/47572) '26 | O(1) rollback for SSM speculation via input ring-buffer | ✓ | — | — | — | Concurrent infrastructure — cite, don't claim (ate part of gap #9) |
| [Heterogeneous-vocab lossless SD](https://arxiv.org/abs/2502.05202) '25 | Lossless speculation across different tokenizers | ✓ | — | — | 2.8x incl. long-context tasks | No per-length breakdown |
| [TokenTiming](https://arxiv.org/abs/2510.15545) '25 | Cross-tokenizer draft alignment | ✓ | — | — | 1.57x | — |
| [OmniDraft](https://arxiv.org/abs/2507.02659) '25 | Universal drafter across targets | ✓ | — | — | 1.5–2x | — |
