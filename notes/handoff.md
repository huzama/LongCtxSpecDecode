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
