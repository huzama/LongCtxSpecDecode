# Handoff

What exists in the code and what was found building it. Method and numbers: [DrafterGoesBurrrr.md](DrafterGoesBurrrr.md). Work list: [TODO.md](TODO.md). Install, slurm, benchmark rules: root `CLAUDE.md`.

## Layout

The vegas fork is vendored at its release (remote `upstream`). Its proposer and the overrider plug-in point live in `vllm/v1/spec_decode/sparse_attn/`. Everything of ours lives in `vllm/v1/spec_decode/sparse_attn/longspec/`, plus `draft_weights.py` beside the proposer; a method is one overrider, registered in the fork's dispatcher. Tests: `tests/v1/spec_decode/sparse_attn/longspec/`, one file per module.

## Vegas on any GPU

Stock vegas is Hopper-only twice: its scores come from a patched FlashAttention-3 kernel (the FA2 op ignores the `scores` argument), and its draft addresses one-token pages (FA2 requires page sizes divisible by 16). A strategy layer under `attn_overrider/` detects what the loaded binary offers and supplies the rest; selection math is untouched, so acceptance is identical on every path.

| Feature | Patched FA3 | Portable | Code |
|---|---|---|---|
| Scores of two query rows per request | written by the kernel | one fused Triton pass over paged K writes the reduced metric directly; no score buffer; one extra K read per verify | `longspec/portable/score_collection.py`, `longspec/kernels/c2q_scores.py` |
| Draft over the selected tokens | page-size-1 table | selection gathered into page-aligned scratch once per propose, one-token append per step | `longspec/portable/draft_kv.py`, `longspec/kernels/draft_gather.py` |
| Choice | `auto` by detection; a forced unavailable path fails at init | | `longspec/portable/kernel_support.py`; config `sparse_attn_score_source`, `sparse_attn_draft_kv` |

Found and fixed in the port: rows padded for a CUDA graph were scored with a block table of -1 (a read outside the cache); the draft's gathered scratch was allocated after vLLM's memory profiling.

## Longspec drafter

`LongSpecAttnOverrider` in `longspec/`: per-layer, per-request budgets from an attention-mass target, sink and recency reserved, one fused CUDA selection per round over all layers, static attention and whole-layer skip masks, per-layer budget statistics. `sparse_attn_algorithm="coverage"` is the selection alone; `"longspec"` adds the skip masks.

## Packed verify attention

FA2 packs the query heads of one kv head into the block row dimension only at `seqlen_q == 1`, so the multi-query verify read the KV once per query head. `longspec/verify_attention.py` restores the packing: non-causal prefix call with the group reshaped into rows (`[B*T, Hq, D]` to `[B*T*G, Hk, D]`, split at the last page boundary at or below `seqused_k - T`), tiny causal tail over the last pages, `merge_attn_states` combine; the merged LSE feeds the score reduction. Static shapes, no host reads, FULL-graph safe. Gated per call (uniform multi-query decode shape, FA2, no window/softcap/alibi/sinks); kill switch `sparse_attn_packed_verify`; padded and empty-prefix rows masked so nothing NaNs. Kernel-level test matches the single causal call.

Small batches stay flat because the prefix launches `B x Hk` blocks and vLLM pins FA2 FULL-graph `num_splits` to 1 (the FA2 wrapper refuses an explicit count above 1). Letting FA2's heuristic split (`num_splits=0`) fills the SMs but moved θ=1 acceptance on Qwen3-0.6B from 0.98+ to 0.946, below the parity gate; parked pending a precision look at the two-call bf16 merge.

## Draft CUDA graphs

Every draft step since the fork's beginning launched its ~600 kernels one by one from Python. The drafter's dispatcher was initialized with the runner's resolved mode (FULL_AND_PIECEWISE), so uniform-decode draft steps dispatched FULL; but the drafter calls the inner model, which carries only piecewise wrappers, and a mode mismatch makes every wrapper pass through eagerly. Trace evidence: zero `cudaGraphLaunch` in the draft scope of every profile, against ~590-660 `cudaLaunchKernel` per step. Invisible until the 4-bit draft: the bf16 draft step is GPU-bound and hides the ~10 ms of launch CPU.

The contract now: the drafter's dispatcher keys are PIECEWISE only and trimmed to the sizes a draft step can reach (one token per request, padded batch bound); a dedicated pass in `capture_model` captures every key inside the capture window with the standard warmup discipline; dummy runs outside the window never capture or replay; `propose` replays. A key miss degrades to eager, never a crash.

Gotcha: `sparse_attn_draft_weights` must be part of `SpeculativeConfig.compute_hash`. Without it the bf16 and 4-bit draft copies shared a torch.compile cache directory and loaded each other's inductor artifacts.

## Quantized draft copy

`sparse_attn_draft_weights` loads a quantized copy of the target through vLLM's own loader (`draft_weights.py`), grafts the target's attention modules into it (KV binding, layer names and overrider call order unchanged), shares embeddings and lm_head, and hands it to the drafter. Verify keeps the target weights. Layer count and the shapes that define the model are validated against the target at load.

The Blackwell node (sm120) fails before any model code: some engine-init kernel has no sm120 image (torch and vllm `_C` both ship sm_120), and the wheel's FA2 is sm_80 SASS with PTX only, so the method's attention would run through slow PTX JIT there. Parked; benchmarks stay on A6000.

## Tools

| Tool | What it does |
|---|---|
| `benchmarks/longspec/grid.py` | One cell per process, serial cells with a drain wait, pg19 prompts as token ids cached under `outputs/prompts/`, YaRN beyond 40960, decode isolated as gen=1 vs gen=N, acceptance from vLLM's spec-decode counters, per-layer budgets from the overrider, one JSON line per cell. `--parity` compares spec output with dense token for token. `--draft-weights` for the quantized copy. |
| `benchmarks/longspec/table.py` | Markdown table of a grid run. |
| `benchmarks/longspec/w4_agreement.py` | Teacher-forces bf16 greedy trajectories through a quantized copy and reads per-token argmax agreement: the acceptance condition of a greedy quantized drafter under a bf16 verify. `tau_sim` walks the agreement in blocks of g. |
| `benchmarks/longspec/round_phases.py` | One cell under vLLM's torch profiler, kernels attributed to vLLM's scopes by correlation id; GPU ms per round by phase. |

Hazard, fixed: vLLM's `gather_draft_hidden_states` JIT-compiles its CUDA module on first use, and first use is the first verified round with non-uniform draft counts, so a ~200 s ninja build landed inside the timed window on every fresh node at batch 2 and above (`~/.cache/torch_extensions` is per node). `SparseAttnProposer.load_model` now builds the module eagerly.

## Measurement rules

Serial cells on an idle node; power-of-two batches; prefix caching off; same-node pairs for any A/B. Greedy trajectories diverge across engine configurations, so tau moves by up to 0.3 on the same prompt between configurations; compare ratios within pairs only. Budget calibration at 64 generated tokens overshoots real decode by about 20%. Shared nodes distort CPU-bound cells by up to 6x.

## Baseline pass

The first vegas pass on this stack, before any of ours: Qwen3-4B, one A6000, pg19, greedy, 256 tokens. Vegas acceptance reproduces its paper in every regime, including their own benchmark shape (tau 6.14 vs their ~6.1). Later passes at 64K and 128K read 0.5 to 1.6 lower in tau; the YaRN configuration of this pass is not recorded.

| ctx | batch | dense | vegas | alpha / tau |
|---|---|---|---|---|
| 32K | 1 | 49.0 | 43.5 | 0.845 / 6.07 |
| 32K | 4 | 93.8 | 112.6 | 0.880 / 6.28 |
| 64K | 1 | 36.3 | 33.6 | 0.882 / 6.29 |
| 64K | 2 | 45.9 | 57.0 | 0.915 / 6.49 |
| 128K | 1 | 21.2 | 24.8 | 0.991 / 6.95 |
