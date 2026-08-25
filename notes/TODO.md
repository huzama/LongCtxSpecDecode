# Make the drafter faster, and better than Vegas

The working contract. Two goals, nothing else gets built.

1. **Make the drafter faster.** Self-speculative decoding at long context: the drafter is the target model on a sparse KV view, verify is full KV and lossless. Speedup is t/(g*c + 1) with t accepted tokens per round, g draft tokens, c the drafter step cost relative to a dense decode step. t is set by the model and the budget; c is the only free lever.
2. **Show how it is better than Vegas.** Vegas (arXiv 2602.07223) proved the loop. We build inside their vLLM fork so their method and streamingllm run as in-harness baselines, and beat them on the target axes below.

Code lives in the sibling fork checkout `../vegas`, branch `ampere`. Literature and settled findings live in [literature.yaml](literature.yaml). Everything that preceded the fork (survey prose, design docs, HF-era prototypes and their results) is on branch `survey-and-prototypes`.

## Done

### Decisions

| Decision | Choice | Why |
|---|---|---|
| Speculation | Self-spec, no draft model | A draft model's own KV grows with context; dead in production at long context |
| Drafter | Same weights, sparse KV view | KV reads dominate the long-context decode step |
| Scoring | Training-free min/max block bound | Beats the trained selector past its training length (+0.061 4B, +0.081 8B mean mass efficiency, 32K, batch 1, matched budget); a mean-plus-spread score collapses in deep layers at tight budgets (0.395 vs 0.921 efficiency at 1%) |
| Selection cadence | Once per draft burst, rescoring cheap if needed | Per-round re-selection gains +21-33% captured mass (32K) and is a follow-up knob |
| Budget policy | Per layer, zero allowed | Oracle at 1% density preserves 0.719 total mass shallow vs 0.855 deep (32K); uniform budgets waste the deep layers' slack |
| Depth axis | Attention-sublayer skip, not whole-layer skip | Both remove the layer's KV read; attention is 3-4x more redundant than MLP; whole-layer skip caps near 1.5x with 0% oracle skip on Qwen3-8B |
| Verify | Full KV, lossless | Output exactness is the selling point; acceptance is the only quality metric |
| Harness | Vegas vLLM fork, our branch `ampere` | Their proposer, scheduler wiring, rejection sampler, and CUDA-graph dispatch are inherited; our method is one more attention overrider |
| Workload | Long context, short generation: pg19 prompts 32K-128K, up to 256 generated tokens | Their AIME benchmark is the opposite shape (short prompts, 40K-token generations) |
| Model | Qwen3-4B | Also in their paper's lineup; fits one 48 GB card at 128K |
| Parked | Early exit (logit lens, ~0.90x); whole-layer skip (1.5x cap); neuron sparsity (vanishes under batching); separate drafter (second KV) | See literature.yaml, ideas_rejected |

### Harness facts

| Fact | Value |
|---|---|
| Plug-in point | `vllm/v1/spec_decode/sparse_attn/attn_overrider/`: one class per method (`vegas`, `streamingllm`, ours next). The streamingllm overrider is the template: block-granular selection, pruned block table, stock FA kernel |
| Our GPUs | RTX A6000, sm86, 48 GB, FA2 path. No sm90 anywhere on the cluster |
| Stock vegas on sm86 | Cannot run: the FA2 op silently drops its `scores` argument and rejects its page-size-1 draft tables (both measured) |
| Our sm86 port of vegas | Emulated score rows (one extra K read per verify) plus a gather draft into page-16 scratch. Acceptance faithful, speed handicapped |
| Reference math | `attn_overrider/utils/block_bound.py`, gated by `tests/v1/spec_decode/sparse_attn/test_block_bound.py` |
| Benchmarks | `benchmarks/longctx_bench.py` (one grid cell), `run_grid.sh` (serial driver with GPU drain wait), `longgen_bench.py`, `benchmark_vegas_a6000.py` (their benchmark shape) |
| Benchmark discipline | Serial cells only: concurrent shards on one node distorted CPU-bound cells by up to 6x. Power-of-two batches: the fork's drafter warmup breaks on odd capture sizes. Prefill-differenced decode, prefix caching off |

### Measurements

Long-context grid, Qwen3-4B, one A6000, pg19, greedy, 256 tokens, decode tok/s with prefill separated; ratio against dense at the same cell. Vegas speed carries the sm86 handicap; its acceptance is faithful.

| ctx | batch | dense | streamingllm | vegas |
|---|---|---|---|---|
| 32K | 1 | 49.0 | 36.9 (0.75x) | 36.5 (0.74x) |
| 32K | 4 | 93.8 | **110.5 (1.18x)** | 63.2 (0.67x) |
| 64K | 1 | 36.3 | 31.2 (0.86x) | 26.0 (0.72x) |
| 64K | 2 | 45.9 | **51.3 (1.12x)** | 22.5 (0.49x) |
| 128K | 1 | 21.2 | **23.1 (1.09x)** | 19.0 (0.90x) |

Per-token acceptance / accepted per round (max 7): streamingllm 0.642/4.85 (32K b1) rising to 0.853/6.12 (128K b1); vegas 0.870/6.22 rising to **0.991/6.95**. Both rise with context at a fixed 7% ratio.

Their regime (short AIME prompts, long sampled generation): vegas tau 6.14 on their benchmark shape (their paper ~6.1), 6.37 at batch 8; speed 0.53x and 0.24x against dense on one A6000. Their 1.25-2.81x is 2x H100 NVL at up to 128 concurrent.

| Derived fact | Value |
|---|---|
| Drafter cost | c ≈ 0.8 of a dense step for a 7% KV read (64K b1); ~1.7 dense steps per draft step at batch 32 short context. The KV floor is ~0.1: step overhead, not the mask, is the enemy |
| Scoring floor | fp16 block-16 metadata reads 6.25% of KV per scoring event; a naive full-K query-aware pass reads 100% and erases the speculation gain at 128K (measured) |
| Verify scaling | 7 verify queries cost 1.07-1.49 of one (32K, batch 1, tuned kernel), so verification rides along |
| Signal bar | recency 0.64-0.85, verification-guided 0.87-0.99 per-token acceptance; ours must land between and approach the top |

## Next

### Beat-Vegas targets

| Their limitation | Our move |
|---|---|
| Signal needs an instrumented FA3 kernel, even at prefill | stock kernels on both passes; metadata scoring |
| Score buffers scale with context (batch x heads x 2 x max_len) | per-page min/max, updated incrementally |
| Token-granular draft, page size 1, Hopper-only | block-16 tables on the stock paged kernel |
| Signal one round stale | bound scored against the current draft query |
| Budget fixed, global, offline sweep | per-layer budgets with zero; acceptance-driven online control (their tables already rebuild every propose, so both are graph-safe) |
| No absolute long-context number | measured above; ours joins the table |

### Ordered work

| # | Task | Serves | Status |
|---|---|---|---|
| 1 | Design choices: head aggregation (sum of per-head bounds vs per-head top-k union); reserves (sink and recent sizes); metadata dtype (fp16 vs INT4); metadata allocation (overrider-owned vs cache-config registered); rescoring cadence (per propose vs mid-burst); zero-budget layers (reserves-only vs skip the read) | design | next, maintainer decision |
| 2 | Profile the draft step inside `propose()`: attention vs weights vs sampler vs metadata rebuild, 64K b1 and 32K b4 | goal 1 | open |
| 3 | Our overrider: stock verify, per-page min/max metadata from the slots each step writes, top-k block table per layer at the first draft step of each propose, one global ratio first | goal 1 | open |
| 4 | Parity gate: ratio 1.0 reproduces dense vLLM output | correctness | open |
| 5 | Head-to-head at matched 7% ratio vs streamingllm and vegas, same serial grid, acceptance and tok/s | goal 2 | open |
| 6 | Per-layer budgets, zero allowed | goals 1 and 2 | open |
| 7 | Acceptance-driven budget controller | goal 2 | open |
| 8 | Attention-sublayer skip inside the draft pass (+21-28% projected) | goal 1 | later |

### Risks

| Risk | Handling |
|---|---|
| Fork frozen at its base vLLM | pin for the whole project |
| Layer tracking by kernel-call count breaks on non-uniform models | stay on Qwen3 |
| Parity subtle across spec and non-spec paths | task 4 before any speed claim |
| sm86 vs Hopper not separable for vegas speed | their paper numbers stand for Hopper; ours are the Ampere numbers |
