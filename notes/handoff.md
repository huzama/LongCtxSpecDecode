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

First cell, 32K batch 1, defaults (θ 0.9, S 4, R 64, cap 0.07, min 0), same setup as the baseline grid, one pass:

| mode | decode tok/s | alpha | tau (of 7) | KV read per draft step |
|---|---|---|---|---|
| dense | 48.8 | | | |
| vegas | 44.2 | 0.877 | 6.22 | 7.2% of prefix |
| coverage | 41.8 | 0.822 | 5.93 | 4.0% mean over layers |

- Layers 0-3 sit at the 7% cap at θ 0.9; layers 5, 11, 25, 30-32 use under 2%. The profile T2 asked for exists; the θ sweep and matched-bytes comparison are the next grid.
- At batch 1 fewer KV bytes buy nothing, as the drafter cost analysis predicted; acceptance decides, and 0.9 coverage is below vegas's flat 7%.

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
