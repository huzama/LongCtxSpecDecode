# Landscape

The field's state of knowledge. Settled is the floor we stand on, contested is where the open questions are, arithmetic says which questions can pay off. The bets built on it live in [ideas-kept.md](ideas-kept.md).

**TL;DR**
- The converged recipe is settled and works to 128K: draft against the target's own sparse/compressed/retrieved state, verify full-KV, lossless.
- Verification is the bottleneck: 85–95% of round time once drafting is cheap, and accepted tokens per round are log-bounded in verifier capacity.
- No published acceptance or speedup datapoint exists past 128K, in any family.
- MLA and sequential hybrids change the physics: KV-centric methods stop transferring.
- Training-free query-aware KV selection matches trained selection at moderate density. The measured difference is selector cost, and quality separates only at aggressive budgets that nobody has compared.

## ✅ Settled

Facts the field agrees on, strongest citations attached. Build on them, never re-prove them, never bet against them.

| # | Fact | Evidence |
|---|---|---|
| 1 | Short-trained draft heads fail at long context | EAGLE-3: τ1.28, 0.81x on 4K–64K (OWL [2510.07535](https://arxiv.org/abs/2510.07535)); vanilla SD τ0.5–0.6 @16K ([2505.20776](https://arxiv.org/abs/2505.20776)); naive sparse drafter ~0% α by 8K ([2606.00144](https://arxiv.org/abs/2606.00144)); vLLM/SGLang consensus: EAGLE-family decays past trained window (>32K) |
| 2 | At long context + batch, MHA/GQA decode is KV-bandwidth-bound, and that is an *opportunity*: speedup grows with batch and context | MagicDec 2.51x @batch 32–256 ([2408.11049](https://arxiv.org/abs/2408.11049)); QuantSpec 1.35x@4K→2.49x@128K ([2502.10424](https://arxiv.org/abs/2502.10424)); SpecPV 2.88x@10K→6.29x@60K ([2512.02337](https://arxiv.org/abs/2512.02337)); roofline validated to 3% vs VeriCache's 25ms@500K |
| 3 | The converged recipe: draft against the target's own sparse/compressed/retrieved state, verify full-KV lossless | TriForce, MagicDec, LongSpec, QuantSpec, Vegas, PillarAttn, SparseSpec-L, VeriCache all instantiate it |
| 4 | Selection mechanism matters more than budget: attention-retrieval holds, permanent eviction collapses | TriForce @120K needle: retrieval 0.9878 vs StreamingLLM 0.0519, H2O 0.0739; SnapKV > StreamingLLM at equal budget (MagicDec) |
| 5 | Quantized drafter KV is acceptance-robust; sparse token-dropping is not | QuantSpec acceptance *rises* to 94.31% @128K; quantized beats sparse for draft-target agreement; VeriCache sustains 25–40 tok/round at 4x compaction |
| 6 | Verification attention is a free importance oracle for drafting | Vegas, PillarAttn, SparseSpec-L, SpecExtend recycle it at near-zero cost; beats static budgets on shifting saliency (SparseSpec 1.36x over MagicDec, 1.76x over TriForce) |
| 7 | Zero-KV retrieval drafting beats trained heads on long grounded inputs | OWL third-party: PLD τ2.75, TR τ3.16, SAMD τ3.18, SuffixDecoding τ3.41, SAMD+TR τ4.98 vs EAGLE-3 τ1.28 |
| 8 | Accepted tokens per iteration are log-bounded in verifier capacity | [2512.11718](https://arxiv.org/abs/2512.11718), tight vs EAGLE-3; giant trees have sharply diminishing returns |
| 9 | Fixed γ is wrong | Optimal draft length shifts with batch ([2310.18813](https://arxiv.org/abs/2310.18813)), load ([2406.14066](https://arxiv.org/abs/2406.14066)), precision ([2605.02888](https://arxiv.org/abs/2605.02888)); optimal policy is a threshold stopping rule ([2405.19715](https://arxiv.org/abs/2405.19715)) |
| 10 | Batched speculation was silently incorrect in major open-source stacks as late as 2025 | [2510.22876](https://arxiv.org/abs/2510.22876); gains compress at production concurrency (Meta 1.4–2.0x; P-EAGLE +5–25% @c=64) |
| 11 | MLA changes the physics | Arithmetic intensity >100x MHA, toward compute-bound ([2507.15465](https://arxiv.org/abs/2507.15465)); MLA KV 69KB/tok vs Qwen3-32B 256KB/tok; DeepSeek shipped weight-amortizing MTP (1.8x), not KV-sparse drafting; correctly so, per the arithmetic |
| 12 | No published acceptance/speedup datapoint exists past 128K, in any family | TriForce (2024) + QuantSpec remain the high-water marks; Windowed-MTP ([2607.21535](https://arxiv.org/abs/2607.21535)) is the lone 1M draft-KV datapoint, window-local MTP heads only |
| 13 | Training-free query-aware selection matches trained selection at moderate density | Matched budget at K=0.5L, our selector vs Twilight vs Quest: RULER .988/.975/.988, BABILong .330/.303/.330, inside bootstrap error. NSA at 2560 activated tokens: Quest .392 vs oracle top-k .423, so heuristic-versus-oracle scoring is worth .031. A trained indexer beats mean-pooled blocks by 1.51 LongBench points. The separation appears only at aggressive budgets: AIME-24 @2K, Quest 18.15 vs dense 74.48 |
| 14 | Untrained early exit through the model's own head is broken, and it is the logit lens | Tuned Lens ([2303.08112](https://arxiv.org/abs/2303.08112)): no interpretable predictions before layer 21 of 32, bias 4–5 bits, cause is hidden-state covariance drift. SimLens ([2507.17618](https://arxiv.org/abs/2507.17618)): mean accuracy 0.251. PPD ([2307.05908](https://arxiv.org/abs/2307.05908)): 21.6–38.9% top-1 agreement at layer 20/40, which the speedup formula turns into ~0.90x. Layer *skipping* escapes this by keeping the final layers, which is why it alone needs no training |
| 15 | Under batching, neuron sparsity vanishes and attention-head sparsity survives | Polar Sparsity ([2505.14884](https://arxiv.org/abs/2505.14884)), the only sweep from batch 1 to 512: the union of active MLP neurons across a batch goes dense, while head sparsity is batch-invariant |
| 16 | Skipping an attention sublayer removes its KV as well as its weights | LLM-Drop ([2406.15786](https://arxiv.org/abs/2406.15786)): attention is 3–4x more redundant than MLP, and dropping 20 of 40 attention sublayers is also a ~50% KV cut. The only depth mechanism that touches the term dominating at batch and length |
| 17 | Sparse attention inflates generation length | Quest at a 2K budget on AIME-24 generates 30.0K tokens against 14.8K dense (+103%). Any sparse-attention throughput claim must report tokens generated; lossless speculation is immune by construction |

## ⚖️ Contested

Questions where published papers directly disagree, both sides shown with a verdict. These are open targets: resolving one is a paper contribution (row 2 became idea #3, row 3 became idea #2).

| # | Question | One side | Other side | Where it stands |
|---|---|---|---|---|
| 1 | Does acceptance decay with context length? | OWL: EAGLE-3 collapses (τ1.28, 0.81x) | MiniMax-M3 blog: AL flat 2.64→2.63 from 1K→32K; **unverifiable, not load-bearing** | It is a property of draft-state construction (shared/quantized/retrieved holds to 128K; evicted/short-trained collapses by 8–32K) + training coverage. No law; measure per checkpoint |
| 2 | Must the sparse budget grow with context? | Vegas: fixed budgets underperform at extreme scale | Windowed-MTP ([2607.21535](https://arxiv.org/abs/2607.21535)): constant sink+window suffices at 1M for MTP heads (~99% draft KV dropped) | Likely drafter-family- and task-dependent; unreconciled → idea #3 |
| 3 | Is sparse verification's quality loss "negligible"? | Dustin / [2512.21911](https://arxiv.org/abs/2512.21911): yes, empirically @32K | SpecPV's own tables: ROUGE-L −27% on GovReport while QA stays within 1–3% | Uncertified either way; no KV-budget→divergence bound exists → idea #2 |
| 4 | Does dynamic layer-skip beat static at long inputs? | Dynamic should adapt | KnapSpec: CLaSp (dynamic) 1.22x *below* SWIFT (static) 1.33x @16K; DEL "fails to accelerate" base models | Dynamic ≠ better |
| 5 | Is verification "free" at scale? | Memory-bound argument: extra verify tokens ride along | Holds only while B(L+1) < ridge (~296 FLOP/B on H100 → breaks at B>59, L=4); 3/5 consumer configs decelerate ([2607.17283](https://arxiv.org/abs/2607.17283)) | Long context re-anchors memory-boundedness; short context + big batch does not |
| 6 | Do diffusion drafters survive long context? | DFlash decays to τ2.09 @32K zero-shot | A 1.6K-sample fine-tune recovers τ3.56 | Data patch or architectural limit? No data past 32K |
| 7 | KV-reuse drafting: principle vs wall-clock | KVShot ([2604.26412](https://arxiv.org/abs/2604.26412)): reading target KV improves long-range acceptance | e2e speedup marginal: shallow drafters cannot estimate target queries; sparse gradients | Contested whether block-wise training fixes it |
| 8 | Thin-evidence zones | — | — | SSM drafters past 8K: nothing; cross-tokenizer at length: nothing; MTP acceptance-vs-length in production: never published; 256K–1M: nothing; StreamServe's 11–18x: implausible baselines (4 GPUs, 80 queries) |
| 9 | Does speculation pay at rollout-scale batch? | MagicDec: yes above S_inflection, 2.51x at batch 32–256; EfficientRollout ([2606.18967](https://arxiv.org/abs/2606.18967)): the batch drains as sequences finish, restoring memory-boundedness | verl measures ~50% rollout throughput *loss* with MTP and tells users to leave it off; SpecActor ([2511.16193](https://arxiv.org/abs/2511.16193)) reports no or negative speedup at batch 256 | Unresolved. Two-roof arithmetic puts verification memory-bound at ≥8K context up to batch 512, contradicting both negatives, so the disagreement is empirical. Measure before betting → idea #15's first experiment |
| 10 | Does speculation give RL training what it needs? | Rejection sampling preserves the target distribution exactly, so rollouts are unbiased | It preserves the *engine's* distribution, not the trainer's; and enabling it currently zeroes generation logprobs, so importance sampling cannot be applied at all ([NeMo-RL #1785](https://github.com/NVIDIA-NeMo/RL/pull/1785)) | Both true. Speculation removes the bias sparsity would add but not the engine-versus-trainer mismatch. A lossless verifier computes exact target logprobs anyway, so exposing them dissolves the conflict → idea #15 |

## 🧮 Arithmetic

What the hardware allows, independent of what anyone published. Every idea must survive this table: it rejects #0 and #12–#14, points #1 at verification, and points #15 at attention sublayers.

Constants: H100 3.35TB/s, 990 TF BF16, ridge 296 FLOP/B. Model validated to 3% vs VeriCache. Speedup = τ/(L·c+1), c = draft/target step cost.

| # | Question | The math | Verdict |
|---|---|---|---|
| 1 | KV per token | Qwen3-32B 256KB/tok → 8.6/34.4/137/275 GB @32K/128K/512K/1M; Llama-3.1-8B 128KB/tok; DeepSeek MLA 69KB/tok; Qwen3-1.7B *and* 0.6B both 112KB/tok (GQA floors at 8 kv-heads) | Small GQA models do NOT have small KV; a full-KV drafter = +44% KV/request → −31% max batch @128K on 8xH100 |
| 2 | Weights vs KV: which term dominates? | At B=1/128K a GQA 32B is still weight-dominated (KV = 34% of bytes); KV dominates at (B≥8, 128K+) or (B=1, 512K+) | Which term to attack depends on (B, S) |
| 3 | Independent full-KV drafter | c → 0.4375 asymptotically as B·S grows | Speedup ceiling **1.09x** even at perfect acceptance: dead twice (arithmetic + acceptance) |
| 4 | Sparse-KV drafter (4K budget) | c = 0.039 @128K/B1 → 0.016 @128K/B32; 2.6→3.0x at τ=3; **3.85x at τ=4.1, B=32, 128K** | Speedup monotonically approaches τ with S and B; the arithmetic loves exactly this direction |
| 5 | Layer skip | c = 1−k, S-invariant; 50% skip at τ=3, L=4 → **1.00x**; matching a 4K sparse draft @128K/B8 needs 78% skip | Reproduces the whole literature (KnapSpec 1.47x cap); 78% skip is acceptance-fatal |
| 6 | Verify dominance | With cheap drafting, verification = **85–95% of round time**; ceiling τ·BW/(W + S·kvB): Qwen3-32B B=1 → 135/101/49/**30 tok/s @1M** (τ=3; 59 @τ=6; B200 only 2.4x better) | τ is log-bounded → **lossless verify-byte reduction is the frontier, not better drafting** |
| 7 | Layer-skip × sparse-KV composition | Multiplicative in cost, sub-additive in speedup (B=32/1M: 2.87x→2.93x, +2%); acceptance multiplies *down* (0.9×0.85→0.765 → τ −23%) | Can be net-negative; genuine niche = batch-1, 512K–1M self-spec (+30–40%) |
| 8 | Self-spec regime gate | Wins iff B·S·kvB > W → S_inflection = 256K/B for Qwen3-32B (B=8→32K; B=1→~400K); ~26K for MHA-7B | Explains why TriForce used MHA and why MagicDec needs batch ≥32 |
| 9 | MLA targets | KV = 1.4–30% of decode bytes at B=1–32 up to 128K → sparse-KV drafting max gain 1.01–1.43x | **Arithmetic malpractice below batch ~64**; weight-amortizing MTP is the only lever (measured 1.8x); KV-centric research does not transfer |
| 10 | Batch ceiling | Verify exits memory-bound at B(L+1) ≳ 296 (B>59 at L=4); long context postpones the ridge (KV intensity (L+1)·g ≈ 40 ≪ 296) | The arithmetic behind all measured concurrency decay; **speculation and batching are complementary at 128K–1M exactly where they conflict at 4K**; γ must shrink with B |
| 11 | Attention-sublayer skip vs uniform layer skip | Row 7 assumes depth reduction leaves KV intact, which is true only for uniform skip. Skipping a fraction s of *attention* sublayers scales the draft's KV read by (1−s): draft cost = (W − s·W_attn) + (1−s)·p·B·S·kvB. At s=50%, p=10%, γ=8, τ=6.1 (Qwen3-4B): 2.02→2.44x at B=32/8K, 3.24→**4.13x** at B=128/32K | **+21–28%, not row 7's +2%.** The depth axis survives only in the attention-sublayer form, and only composed with per-layer KV budgets → idea #15 |
| 12 | Byte split across (batch, context) | Qwen3-4B KV share of decode bytes: 7/13/38% at B=1 (4K/8K/32K); 71/83/95% at B=32; 91/95/99% at B=128 | Above batch ~8 the KV term is the only one worth attacking. Weight-side levers (quantized draft, width pruning, uniform layer skip) are rounding errors there |
| 13 | Rollout arithmetic | If generation is 70–80% of an RL step, a 2.9–4.1x rollout speedup is 1.9–2.5x end to end | The motivation's ceiling. It needs the rollout share to hold, and it needs row 9's contested question resolved first |
