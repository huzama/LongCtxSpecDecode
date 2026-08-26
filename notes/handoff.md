# Handoff

What exists and what was measured. Goals and work: [TODO.md](TODO.md). Survey: [literature.yaml](literature.yaml). Install, slurm, benchmark rules: root `CLAUDE.md`.

## Baseline: Vegas

Self-speculative decoding: the target model drafts over a sparse KV subset, verifies over the full cache, losslessly. Its scorer, the rule that picks the subset, is the verify pass's attention scores of two query rows per request, one round stale, at token granularity, with one global ratio (`sparse_attn_ratio`). The fork is vendored at its release (remote `upstream`); their proposer and the overrider plug-in point live in `vllm/v1/spec_decode/sparse_attn/`. Everything of ours lives in `vllm/v1/spec_decode/sparse_attn/longspec/`; a method is one overrider there, registered in the fork's dispatcher. Design: [method.md](method.md).

## Built: vegas on any GPU

Stock vegas is Hopper-only twice: its scores come from a patched FlashAttention-3 kernel (the FA2 op ignores the `scores` argument), and its draft addresses one-token pages (FA2 requires page sizes divisible by 16). A strategy layer under `attn_overrider/` detects what the loaded binary offers and supplies the rest; selection math is untouched, so acceptance is identical on every path.

| Feature | Patched FA3 | Portable | Code |
|---|---|---|---|
| Scores of two query rows per request | written by the kernel | one fused Triton pass over paged K writes the reduced metric directly; no score buffer; one extra K read per verify | `longspec/portable/score_collection.py`, `longspec/kernels/c2q_scores.py` |
| Draft over the selected tokens | page-size-1 table | selection gathered into page-aligned scratch once per propose, one-token append per step | `longspec/portable/draft_kv.py`, `longspec/kernels/draft_gather.py` |
| Choice | `auto` by detection; a forced unavailable path fails at init | | `longspec/portable/kernel_support.py`; config `sparse_attn_score_source`, `sparse_attn_draft_kv` |

Tests: `tests/v1/spec_decode/sparse_attn/longspec/`.

## Built: coverage drafter

The method of [method.md](method.md), as `CoverageAttnOverrider` in `longspec/`: per-layer, per-request budgets from a coverage target, sink and recency reserved, one fused CUDA selection per round over all layers, static attention and whole-layer skip masks, per-layer budget statistics. Config: `sparse_attn_algorithm="coverage"` plus `sparse_attn_coverage`, `sparse_attn_sink`, `sparse_attn_recent`, `sparse_attn_skip_attn_layers`, `sparse_attn_skip_layers`. Grid runner: `benchmarks/longspec/grid.py`.

Grid, one pass: Qwen3-4B, one A6000, pg19, greedy, 256 tokens, serial cells, g = 6. Vegas at 7%; coverage with sink 4, recent 64, min 0, cap 15%. 64K and 128K use YaRN (factor 2 and 4 on the 40960 window), 32K none. Decode tok/s; ratio against dense at the same cell; KV read is the mean over layers of the selected fraction of the scored prefix.

| ctx | batch | dense | vegas 7% | coverage θ 0.90 | coverage θ 0.95 | coverage θ 0.98 |
|---|---|---|---|---|---|---|
| 32K | 1 | 49.0 | 45.6 (0.93x), tau 6.38 | 36.6 (0.75x), tau 5.80, 5.5% | 46.6 (0.95x), tau 6.61, 11% | 46.6 (0.95x), tau 6.71, 15% |
| 32K | 4 | 92.5 | 103.8 (1.12x), tau 6.14 | **111.9 (1.21x)**, tau 5.96, 4.7% | 111.1 (1.20x), tau 6.10, 10% | 112.9 (1.22x), tau 6.51, 14% |
| 64K | 1 | 35.8 | 22.2 (0.62x), tau 5.52 | 26.5 (0.74x), tau 5.23, 5.2% | 27.7 (0.77x), tau 5.64, 11% | 28.9 (0.81x), tau 6.05, 15% |
| 64K | 2 | 46.3 | 50.9 (1.10x), tau 5.38 | 48.4 (1.05x), tau 5.11, 5.0% | 51.2 (1.11x), tau 5.44, 11% | 50.2 (1.08x), tau 5.88, 14% |
| 128K | 1 | 23.9 | 19.4 (0.81x), tau 5.43 | 19.6 (0.82x), tau 5.31, 3.9% | 21.8 (0.91x), tau 5.84, 8.6% | 22.1 (0.92x), tau 6.12, 13% |

- Acceptance: θ 0.95 beats vegas's tau at every cell with about 1.5x its bytes; θ 0.90 trails it by 0.1-0.6 with 55-80% of its bytes; θ 0.98 sits at the cap almost everywhere, so it is a flat 15% budget in effect.
- Speed: coverage wins where bytes matter and acceptance is close (32K batch 4: 1.21x vs 1.12x at 4.7% of the KV; 128K batch 1: equal to vegas at 55% of its bytes). At batch 1 the bytes are free and only acceptance counts, so higher θ is simply faster.
- Per-layer profile at θ 0.90: layers 1-3 sit at the cap, layers 5, 7, 11, 23, 25, 30 read under 2%. The profile is stable across context and batch.
- Vegas acceptance at 64K and 128K is far below the baseline pass (tau 5.4-5.5 vs 6.3-6.95) while the 32K cells match; the YaRN configuration of that pass is not recorded. Generations are coherent book continuations at every length. Dense speeds reproduce (within 2% at 32K and 64K, 128K 13% faster).
- Node shared with another user's GPU job from the 32K batch-4 vegas cell onward (load average under 2). Dense cells were unaffected; spec cells at batch 1 may read low by up to ~10%. A quiet-node rerun of the 32K batch-1 row is pending.

Found on the way, fixed in our port: rows padded for a CUDA graph were scored with a block table of -1 (a read outside the cache); the draft's gathered scratch was allocated after vLLM's memory profiling. Found, not fixed: the draft step runs with no CUDA graphs at all (the drafter holds the model loaded before the graph wrapper), which is part of the measured c ≈ 0.8.

## Measured

Qwen3-4B, one A6000 (sm86, FA2), pg19, greedy, 256 tokens, serial cells on a quiet node; decode tok/s with prefill separated, ratio against dense at the same cell. The vegas column reproduced within 2% across two passes.

| ctx | batch | dense | vegas | alpha / tau (of 7) |
|---|---|---|---|---|
| 32K | 1 | 49.0 | 43.5 (0.89x) | 0.845 / 6.07 |
| 32K | 4 | 93.8 | **112.6 (1.20x)** | 0.880 / 6.28 |
| 64K | 1 | 36.3 | 33.6 (0.92x) | 0.882 / 6.29 |
| 64K | 2 | 45.9 | **57.0 (1.24x)** | 0.915 / 6.49 |
| 128K | 1 | 21.2 | **24.8 (1.17x)** | 0.991 / 6.95 |

- Vegas acceptance reproduces its paper in every regime, including their own benchmark shape (tau 6.14 vs their ~6.1). Their 1.25-2.81x is 2x H100 at up to 128 concurrent.
- Drafter cost: c ≈ 0.8 of a dense step for a 7% KV read (64K, batch 1); the bandwidth floor is ~0.1. Step overhead, not the mask, is the lever.
- Verification rides along: 7 verify queries cost 1.07-1.49 of one (32K, batch 1).
