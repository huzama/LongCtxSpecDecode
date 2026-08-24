# Make the drafter faster, and better than Vegas

The two questions of this phase. Nothing else gets built.

1. **Make the drafter faster.** Self-speculative decoding at long context. The drafter is the target model on a sparse KV view; verify is full KV and lossless. Speedup is t/(g*c + 1) with t accepted tokens per round, g draft tokens, c the drafter step cost relative to a dense decode step. t is set by the model and the budget; c is the only free lever.
2. **Show how it is better than Vegas.** Vegas (arXiv 2602.07223) proved the loop: self-spec, sparse draft, full lossless verify, selection once per round. We build inside their vLLM fork so vegas and streamingllm run as in-harness baselines, and we beat them on the target axes below.

## Harness

Vegas ships as a vLLM fork with a plug-in point: `vllm/v1/spec_decode/sparse_attn/attn_overrider/` dispatches one overrider class per method (`vegas`, `streamingllm`). Our method is a third overrider. The streamingllm overrider is the template: block-granular selection, pruned block table built by a small Triton kernel, stock FA kernel, no kernel changes.

| Fact | Value |
|---|---|
| Checkout | sibling of this repo: `../vegas` (github.com/platformxlab/vegas, Apache-2.0) |
| Env | own venv inside that checkout, Python 3.12, source build, CUDA 12.8, `TORCH_CUDA_ARCH_LIST=8.6`; build log at `build.log` there |
| Our GPUs | RTX A6000, sm86, 48 GB. FA3 needs sm90, so vLLM runs its FA2 path here. Whether their verify-score collection exists in the FA2 path decides if the vegas baseline runs on our cards |
| Model | Qwen3-4B, also in the vegas paper's lineup |
| Their kernel dep | companion fork github.com/npz7yyk/vllm-flash-attn exposes a `scores` param on `flash_attn_varlen_func`; only the vegas overrider needs it |
| Workload | long context, short generation: pg19/fineweb prompts at 32K-128K, generation up to 256 tokens. Their AIME'25 benchmark is the opposite shape (short prompts, 40K-token generations) and is out of scope |

## Beat-Vegas targets

| Their limitation | Where it lives | Our move |
|---|---|---|
| Selection signal needs an instrumented kernel | verify pass writes per-token logits or rematerialized weights via the FA fork | stock kernels on both passes; metadata scoring is backend-portable |
| Score buffers scale with context | `_attn_score_buffer` is batch x heads x 2 x max_model_len bf16, ~0.5 GB at batch 32, 32 heads, 128K | per-page min/max key metadata, updated incrementally, read at 6.25% of KV per scoring event (fp16, block 16) |
| Token-granular draft gather | draft k/v reshaped to page size 1; `_index_to_slot_kernel` builds slot-level page tables per layer | block-16 tables = vLLM pages, contiguous reads, stock paged kernel |
| Signal is one round stale | scores come only from the previous verify; no mid-burst refresh | bound is scored against the current draft query; rescoring is cheap |
| Even prefill needs the patched kernel | their overrider collects scores during prefill (its prefill branch) to seed the first draft mask | our metadata update is plain tensor math over new keys; any prefill kernel works |
| Budget fixed, global, offline | one `sparse_attn_ratio` for all layers, requests, and time | per-layer budgets with zero allowed; acceptance-driven online control. Their page tables already carry a layer dimension, and tables rebuild every propose, so both are graph-safe in their own design |
| No absolute long-context number | paper reports 1.25-2.81x on ~182-token inputs; at 96-120K only +18-29% over sparse self-spec baselines | measure vegas, streamingllm, and ours against dense vLLM at 32K-128K and publish the table |

## Ordered work

| # | Task | Serves | Status |
|---|---|---|---|
| 0 | Build the fork in its own env on sm86 | harness | in progress |
| 1 | Smoke on one A6000: `benchmarks/smoke_a6000.py` runs `vegas`, `streamingllm`, `off` with Qwen3-4B on pg19 32K prompts, 256 generated tokens | baselines run on our cards, our workload | next |
| 2 | Absolute baseline table: vegas and streamingllm vs dense vLLM, 32K-128K, batch 1-32 | goal 2 | open |
| 3 | Read `proposer.py` end to end; one-page design note: metadata placement (per-page min/max from keys seen at prefill and each verify), draft block-table build, cold-start path | design | open |
| 4 | Our overrider: stock verify, incremental metadata, top-k block table, fixed global ratio first | goal 1 | open |
| 5 | Parity gate in the fork: ratio 1.0 must reproduce dense vLLM output | correctness rule | open |
| 6 | Head-to-head at matched ratio: acceptance and throughput, short and long context | goal 2 | open |
| 7 | Per-layer budgets, zero allowed | goals 1 and 2 | open |
| 8 | Acceptance-driven budget controller | goal 2 | open |
| 9 | Attention-sublayer skip inside the draft pass | goal 1 | later |

## What transfers from our measurements

| Asset | Becomes |
|---|---|
| Min/max bound beats the trained selector past its training length (+0.061 4B, +0.081 8B mean mass efficiency, 32K, batch 1, matched budget) | the scoring rule of the overrider |
| Depth asymmetry: oracle at 1% density preserves 0.719 total mass shallow vs 0.855 deep (32K) | per-layer budget policy |
| Scoring floor: fp16 block-16 metadata reads 6.25% of KV per scoring event | overhead budget for the metadata path |
| Verify scaling: r(9) = 1.07-1.49 on a tuned kernel (32K, batch 1) | why the loop pays; no new work |
| `validate-scorers` gate and the reference scorer in this repo | reference the fork port must match |
| HF-side experiments | frozen evidence base |

## Parked

| Direction | Reason |
|---|---|
| Early exit | untrained exit is the logit lens: 21.6-38.9% head agreement, ~0.90x end to end |
| Whole-layer skip | family caps near 1.5x; 0% oracle skip ratio on Qwen3-8B |
| Neuron/width sparsity | vanishes under batching |
| Separate draft model | second KV grows with context; dead in production |

## Risks

| Risk | Handling |
|---|---|
| Vegas baseline may not run on sm86 if score collection is FA3-only | ours needs stock kernels either way; streamingllm baseline runs regardless; vegas numbers then come from the paper until a sm90 card is available |
| Fork is frozen at its base vLLM; upstream moves | pin to their base for the whole project |
| Layer tracking by call count breaks on non-uniform models | stay on Qwen3, same as them |
| Full-budget parity may be subtle across spec and non-spec paths | that is what task 5 exists for; no speed claim before it passes |

## Pointers

| What | Where |
|---|---|
| Method design | [work/exit-sparse-self-spec-training-free.md](../work/exit-sparse-self-spec-training-free.md) |
| Base design, round structure and cost model | [work/exit-sparse-self-spec.md](../work/exit-sparse-self-spec.md) |
| Selection quality results | [work/results/training-free-selection.md](../work/results/training-free-selection.md) |
| Verify scaling results | [work/results/verify-scaling.md](../work/results/verify-scaling.md) |
| Survey | [work/survey/README.md](../work/survey/README.md) |
| Fork code layout | `../vegas/README.md` |
