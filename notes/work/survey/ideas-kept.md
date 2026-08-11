# Ideas: Kept

Live bets. Evidence lives in [landscape.md](landscape.md); dead directions and the corrected starting hypotheses live in [ideas-rejected.md](ideas-rejected.md). IDs #0–#14 are stable across both files.

**TL;DR**
- Our bets: **#0/#1** (both-sides-sparse via an external block-sparse oracle; #0 adds the early-exit depth axis), **#2+#6** (certification + control of sparse verification), **#3+#4** (the measurement papers nobody has done, executable in weeks).
- Avoid pure drafting improvements (bottleneck moved to verification) and layer-skip framings (rejected, arithmetic-capped).
- Every idea carries a falsifiable 2-week kill test. Run it: the idea graduates to active work or moves to rejected with data.

## 🚀 Menu

Ranked by impact × tractability × fit for us (sparse attention; block-sparse selector method under review). Audit = how much is already published; [G*] in headings points into data/raw-data.json. Read #0–#4 in detail first.

| # | Idea | Audit | 2-week kill test |
|---|---|---|---|
| **0** | Early-exit both-sides-sparse: depth-profiled budgets, shared sparse prefix | open | α(k × budget) grid @32–64K; α usable at k=50% and α_comb ≥ α_exit·α_sparse |
| **1** | Both-sides-sparse: sparse draft + sparse verify via external selector oracle | partially-solved | selector-restricted verify @12.5% budget; acceptance holds within ~5% → project lives |
| **2** | Divergence certification: KV budget → output-distribution bound | open | attention-mass ↔ realized-KL correlation @32K; one strong plot validates the program |
| **3** | Acceptance-vs-(budget, length) surface, 128K–1M | partially-solved | fit α(B, L) over {32K, 128K, 256K} × budgets; sublinear vs linear vs task-forked |
| **4** | EAGLE-3/MTP acceptance vs KV-quantized targets (the production config) | open | 70B + EAGLE-3, KV {BF16, FP8, INT8} × {4K, 32K, 64K} × 4 tasks |
| 5 | Staleness/refresh policy for verification-recycled importance | open | freeze indices {8–1024} steps; plot acceptance-vs-staleness decay |
| 6 | Adaptive full-verification triggering for partial verification | open | attention-mass-missed vs realized drift: GovReport (cliff) vs QA (safe) |
| 7 | Length-robust *trained* sparse-KV drafting past 32K | open | retrain BudgetDraft 68M with Anchor-Offset @32K; does 16K acceptance move off ~18%? |
| 8 | Online (drafter budget × γ) controller across (batch, context, load) | partially-solved | (B, γ) grid per cell; 2-feature predictor recovers ≥90% oracle throughput |
| 9 | Self-speculation for sequential-hybrid / SSM targets | partially-solved | LayerSkip-style finetune on 0.5–1B hybrid; does α recover from 0.038? |
| 10 | No-regret draft-path adaptation over 30K–100K generations | partially-solved | EXP3 budget selection on 13K-token AIME traces vs fixed |
| 11 | SSM drafter acceptance past 8K | open | Mamba-130M + Llama-3.1-8B on LongSpecBench @128K vs OWL-LSTM |

### #0 Early-exit both-sides-sparse: depth-profiled budgets over a shared sparse prefix
- **Why #0**: sparse-KV self-drafting is weight-bound at batch 1 below S_inflection (≈400K for a GQA 32B; landscape, arithmetic #8): sparsity cannot touch the weight term, early exit divides it by the exit fraction. It is also the one drafting lever that transfers to MLA targets (arithmetic #9), our hedge against risk #2. Generalizes #1: full depth is the k=L special case.
- **Design**: one selector, a depth-profiled budget B(layer), exit boundary k. Draft = sparse prefix + exit head on a frozen target (Kangaroo-style adapter). Verify = continue the deep layers, reusing the prefix compute and KV. Draft and verify must share the sparse pattern inside the prefix; "different sparsity" lives across depth, not between the two passes. Contract: lossless w.r.t. the sparse-prefix target (SSV precedent, [2605.19893](https://arxiv.org/abs/2605.19893)) or certified via #2.
- **Not #12**: layer skip reruns skipped layers at verification and its cost ratio is context-invariant (rejected). Early exit shares the prefix, so draft work counts toward verification and only rejected tokens pay.
- **Baselines**: LayerSkip ([2404.16710](https://arxiv.org/abs/2404.16710)), Kangaroo ([2404.18911](https://arxiv.org/abs/2404.18911)), Mirror-SD ([2510.13161](https://arxiv.org/abs/2510.13161)): all early exit, all ≤16K, all full attention; Vegas + Dustin via #1; [2603.23701](https://arxiv.org/abs/2603.23701) (layer-redundancy shrinkage, the headwind to beat); Component-Aware SD ([2605.01106](https://arxiv.org/abs/2605.01106), exit training recovers hybrids, idea #9's evidence).
- **Risk**: early exit has zero published evidence past 16K anywhere (family verdict), and it is a batch-1 story first (risk #7): frame as local long context or show the weight saving below the batch ridge.
- **2-week kill test**: Llama-3.1-8B + Kangaroo-style adapter at 32–64K; grid k ∈ {25%, 50%} × draft budget ∈ {full, 25%, 6%}. Gates: α at k=50% stays usable at 64K, and α_comb ≥ α_exit·α_sparse (the super-multiplicativity check inherited from #12's reopen test). Day-one calibration before any training: Meta's LayerSkip checkpoints at native context.

### #1 Both-sides-sparse: compose sparse drafting with sparse verification, breaking oracle circularity [G11]
- **Why #1**: verification is 85–95% of round time (landscape, arithmetic #6), the only place large headroom remains; every verification-recycled drafter (Vegas/PillarAttn/SparseSpec-L) structurally *requires* full-KV verification for its signal. **Our block-sparse selector is a third-party importance oracle that does not need full verification attention: the exact missing piece.**
- **Novel delta**: a self-spec system where both draft and verify run sparse, with an external (selector-based) or Dustin-style fused signal replacing full-verification attention, plus recall of dropped-but-relevant tokens without a third scoring pass; measured interaction of the two speedups (each side's 2–9x is only measured in isolation).
- **Baselines**: Vegas ([2602.07223](https://arxiv.org/abs/2602.07223)) + Dustin ([2606.24957](https://arxiv.org/abs/2606.24957)): the two sides, never composed; SSV ([2605.19893](https://arxiv.org/abs/2605.19893), natively-sparse targets, a different contract); SpecPV (crude periodic refresh).
- **2-week experiment**: Llama-3.1-8B at 32–64K: run Vegas-style drafting with verification KV restricted to our selector's blocks at budgets {25%, 12.5%, 6%}; measure acceptance and downstream quality vs full-verify Vegas and vs Dustin-style fusion. If acceptance holds within ~5% at 12.5% verify budget, the project lives; if it collapses, that is the (publishable) circularity confirmation.

### #2 Divergence certification for sparse verification: KV budget → output-distribution bound [G10]
- **Novel delta**: model target-under-sparse-KV as a perturbed distribution; per-token KL/TV certificates as a function of KV budget / dropped attention mass, composed with the level-set acceptance regions of [2606.30265](https://arxiv.org/abs/2606.30265); explains [2607.26627](https://arxiv.org/abs/2607.26627)'s failure taxonomy. Turns Dustin's unfalsifiable "negligible loss" into a knob. Feeds #1 and #3 directly.
- **Baselines**: [2606.30265](https://arxiv.org/abs/2606.30265) (certificates, no KV axis); VeriCache's linear-KL-growth measurement ([2605.17613](https://arxiv.org/abs/2605.17613)) as the empirical template; SpecPV's ROUGE cliff as the motivating failure.
- **2-week experiment**: measure per-token KL(full ‖ sparse-verify) vs retained attention mass across budgets/tasks at 32K; test whether captured-attention-mass is a usable online divergence estimator (correlation with realized KL). A single strong correlation plot validates the whole program.

### #3 Acceptance-vs-(budget, length) surface and the B(L) law, 128K–1M [G7 + G14]
- **Novel delta**: swept surfaces for *retrieval/verification-recycled* sparse drafters (not window-local MTP) at 128K–1M; reconcile Vegas ("budget must grow") vs Windowed-MTP [2607.21535](https://arxiv.org/abs/2607.21535) ("constant window suffices at 1M"); unify quantized (QuantSpec) vs sparse (BudgetDraft) compression axes into MagicDec's (batch, seqlen) model → choose (bits, budget) from (B, S, HBM). Must anchor on Windowed-MTP and QuantSpec-128K.
- **Baselines**: TriForce budget curve @120K; Vegas 7%@96–120K; BudgetDraft table to 16K; QuantSpec 4K–128K single setting; MagicDec model (no compression axis).
- **2-week experiment**: one model (Llama-3.1-8B-1M or Qwen3), contexts {32K, 128K, 256K}, budgets {1K, 2K, 4K, 8K, %-based}, quantized vs top-k selection: fit α(B, L) and check whether iso-acceptance budget grows sublinearly, linearly, or is task-forked. 256K on 2 GPUs is feasible with offloaded verify (VeriCache pattern).

### #4 Measured acceptance of EAGLE-3/MTP against KV-quantized targets (the production config) [G13]
- **Novel delta**: the config everyone runs (FP8-KV target + EAGLE-3) has *zero* published acceptance data: only an uncited dev.to claim (0.3–1.5 accepted-token drop) and vLLM issue [#37618](https://github.com/vllm-project/vllm/issues/37618). Sweep target KV precision (FP8/INT8/INT4, per-axis) × context × task; quantify double-lossy compounding (BF16-trained drafter features vs FP8-KV verification) and name the contract problem ("lossless relative to the wrong distribution").
- **Baselines**: SpecKV-γ ([2605.02888](https://arxiv.org/abs/2605.02888), weight-quant only; explicitly says the KV interaction is unstudied); [vLLM AMD EAGLE-3 blog](https://vllm.ai/blog/2026-07-13-eagle-3-amd-instinct) (AL 2.77, no KV-quant data); VeriCache (compressed KV as drafter only).
- **2-week experiment**: vLLM, Llama-3.3-70B + EAGLE-3 head, KV in {BF16, FP8, INT8} × contexts {4K, 32K, 64K} × 4 tasks; report AL and speedup deltas. Fully executable in days; near-guaranteed publishable finding either way.

### #5 Staleness/refresh policy for verification-recycled importance over 10K+ generated tokens [G9]
- **Novel delta**: measure acceptance decay of stale sparse indices over long generations (nobody has: SparseSpec-L caps at 128 tokens; PillarAttn fixes k=8 with no ablation); build an acceptance-aware refresh controller (refresh when predicted acceptance loss > gather cost). Our selector's block-level statistics are natural drift features.
- **Baselines**: TriForce (first staleness observation, 2024); SparseSpec/PillarAttn (fixed k=8); SparseSpec-L (entropy controller for γ, not refresh); Dustin's lookahead fusion as an estimator ingredient.
- **2-week experiment**: PillarAttn-style setup on AIME long-CoT; freeze indices for {8, 64, 256, 1024} steps; plot acceptance vs staleness. The decay curve alone is a paper section; the controller follows.

### #6 Adaptive full-verification triggering / error-accumulation control for partial verification [G3]
- **Novel delta**: replace SpecPV's fixed buffer-overflow refresh with an online divergence trigger from verification-side statistics (captured attention mass); bound divergence as f(budget, refresh interval). Pairs with #2; together they make lossy verification respectable.
- **Baselines**: SpecPV ([2512.02337](https://arxiv.org/abs/2512.02337), the quantified failure + heuristic); [2607.26627](https://arxiv.org/abs/2607.26627) (failure taxonomy, no control); Dustin (fixed heuristics).
- **2-week experiment**: instrument SpecPV; correlate attention-mass-missed with realized output drift on GovReport (the known cliff) vs QA (known-safe); if the signal separates the two task families, the trigger works.

### #7 Length-robust *trained* sparse-KV drafting past 32K [G8]
- **Novel delta**: multi-budget sparse-view training (BudgetDraft) × long-position schemes (LongSpec Anchor-Offset) × drift-fix normalization (Attention Drift post-norm), evaluated at 64K+. Nobody has combined them; BudgetDraft's 16K collapse is a position-range artifact (drafter's native 2048 limit) begging for this fix.
- **Baselines**: BudgetDraft ([2606.00144](https://arxiv.org/abs/2606.00144), budget-robust not length-robust); LongSpec ([2502.17421](https://arxiv.org/abs/2502.17421), architecture without budget training); [2605.09992](https://arxiv.org/abs/2605.09992) (norm fixes).
- **2-week experiment**: retrain BudgetDraft's llama-68m with Anchor-Offset positions on 32K data; test whether 16K acceptance moves off ~18%. Cheap (68M params), decisive.

### #8 Online (drafter KV budget × γ) controller across (batch, context, load) [G12]
- **Novel delta**: the γ axis is covered (Nightjar [2512.22420](https://arxiv.org/abs/2512.22420), FASER, SpecKV-γ); the *budget* axis is untouched at runtime: Vegas fixes sparsity per run, BudgetDraft never deploys its robustness. Needs an α(task, L, B) predictor (which #3 produces) + joint policy under HBM pressure. Position as adding the budget dimension to Nightjar/FASER/SpecKV.
- **2-week experiment**: with a budget-robust drafter, grid-search (B, γ) offline per (batch, context) cell; show the optimum moves and a 2-feature predictor recovers ≥90% of oracle throughput.

### #9 Self-speculation for sequential-hybrid / SSM targets [G4]
- **Novel delta**: (a) LayerSkip-style training on a hybrid checkpoint (never done); (b) shared-recurrent-state self-spec semantics (ReplaySSM handles separate-path rollback only); (c) checkpoint/replay overhead at 100K. High strategic value (2026 frontier is hybrid) but moderate fit and heavier lift.
- **Baselines**: [2605.01106](https://arxiv.org/abs/2605.01106) (α=0.038 diagnosis); ReplaySSM RFCs ([#47572](https://github.com/vllm-project/vllm/issues/47572); concurrent infra, cite not claim); SpecLA ([2607.16673](https://arxiv.org/abs/2607.16673), 1.70x ceiling).
- **2-week experiment**: apply layer-dropout + early-exit loss finetune to a small sequential hybrid (0.5–1B); check whether self-spec α recovers from 0.038 toward the ~12x early-exit recovery already reported.

### #10 No-regret draft-path adaptation over 30K–100K-token generations [G6]
- **Novel delta**: bandit methods exist (BanditSpec [2505.15141](https://arxiv.org/abs/2505.15141), Not-a-Bandit [2510.20064](https://arxiv.org/abs/2510.20064), OnlineSpec [2603.12617](https://arxiv.org/abs/2603.12617)) but none handle combinatorial path selection (skip set / KV budget) under self-generation drift on long CoT. Must position against those explicitly.
- **2-week experiment**: run EXP3-style budget selection vs fixed budget on 13K-token AIME traces inside PillarAttn; report dynamic-regret and wall-clock delta.

### #11 SSM/linear-attention drafter acceptance past 8K [bonus]
- **Novel delta**: acceptance-vs-state-capacity curve for a Mamba/GDN drafter at 32K–128K vs a transformer target: the regime O(1) state was built for, never measured (Mamba Drafters stop at 8K).
- **2-week experiment**: Mamba-130M drafter + Llama-3.1-8B target on LongSpecBench extended to 128K; compare vs OWL-LSTM.

## ☠️ Risks

What kills these bets, and how we cover each.

| # | Risk | Why it kills | Hedge |
|---|---|---|---|
| 1 | Lossless headroom nearly exhausted | Sparse drafting already drives c→0; verification is 85–95% of round time, hard-capped at τ·BW/(W+S·kvB), 30 tok/s at 1M on H100, and τ is log-bounded ([2512.11718](https://arxiv.org/abs/2512.11718)). If QuantSpec-style quantized KV, VeriCache offload-overlap, and MLA kernels suffice, "better sparse drafting" papers add single-digit percent. | The frontier *is* the verify side: ideas #1/#2 target it directly, not drafting. |
| 2 | Architecture ground shifting | Frontier long-context models are MLA and sequential hybrids: MLA makes sparse-KV drafting worth 1.01–1.43x below batch ~64 (landscape, arithmetic #9); sequential hybrids draft at α=0.038 ([2605.01106](https://arxiv.org/abs/2605.01106)). "Does this matter for DeepSeek/Qwen3.5-class models?" Today: no. | Idea #9 (hybrid self-spec) is the transfer play; measurement ideas #3/#4 stay valid on the deployed GQA fleet. |
| 3 | Field converged fast, crowded | Vegas, PillarAttn, SparseSpec-L, Dustin, SpecPV, BudgetDraft, VeriCache all landed within ~12 months on the same recipe; residual deltas are narrower than they look; ReplaySSM already ate part of idea #9 (G4). | Move fast on the weeks-not-months measurement papers (#3/#4); position explicitly against each named neighbor. |
| 4 | Lossy-verification defection | If Dustin/SpecPV-style lossiness (6–9x) becomes the accepted norm, lossless long-context spec decode becomes a niche contract; the winning program is quality-bounded lossy inference. | Certification (#2) is the entry ticket to that program either way. |
| 5 | Headline regime experimentally inaccessible | 256K–1M needs multi-node full-KV verification; academic setups cap at 64K (single A40) or 128K. Extrapolations can invert: Windowed-MTP ([2607.21535](https://arxiv.org/abs/2607.21535)) showed a constant window suffices at 1M for MTP heads. | Offloaded verify (VeriCache pattern) reaches 256K on 2 GPUs; claim only what we can measure. |
| 6 | Acceptance is checkpoint-idiosyncratic | OWL-collapse vs (unverified) MiniMax-flat; DFlash fixed by 1.6K fine-tuning samples. An acceptance-drift phenomenon may be a training-data artifact patched by a fine-tune, not a law. | Evaluate on multiple checkpoints; report per-checkpoint, never as a universal claim. |
| 7 | Serving reality | Engines cannot compose the features (TRT-LLM: no dynamic trees on MLA/SWA; vLLM PP+SD gaps; batched-SD correctness only fixed in 2025); gains compress 2–3x → 1.4–2.0x at production concurrency; heterogeneous per-request sparse paths break kernel homogeneity. A batch-1 bespoke-kernel method will not ship. | Design batch-compatible from day one: SparseSpec routed around batched layer-skip for exactly this reason. |
| 8 | The cheap baselines are embarrassing | PLD/SAMD/SuffixDecoding: zero training, zero GPU state, τ2.75–4.98 at 4–64K, beat EAGLE-3, already in vLLM. | Benchmark against SAMD+TR (τ4.98) and HOWL (τ6.14) from day one, not just AR decoding. |
