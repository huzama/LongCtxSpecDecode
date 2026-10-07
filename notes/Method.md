# Method

Self-speculative decoding at long context, on the Vegas vLLM fork. The model drafts its own next tokens from part of its KV cache, then checks them against the whole cache. The KV cache stores the attention keys and values for past tokens.

## Our method

| Part | Rule |
|---|---|
| Draft | A copy of the model with 4-bit weights and 16-bit activations (W4A16). Drafts 6 tokens per round. |
| What the draft reads | In each layer, keep the first 4 and last 64 tokens, then add the highest-weight past tokens until their combined attention mass reaches 85%. The additional selection is capped at 15% of the cache. |
| Where the weights come from | The previous verify pass: its first and last new token, averaged over heads. Updated every round, per layer and per request. |
| Verify | The full-precision model reads the whole cache and checks all 6 drafted tokens in one pass. It keeps them up to the first disagreement, then adds its own token. |
| Output | Full-cache target verification. Different batch sizes or numbers of tokens processed together can change floating-point results and the highest-scoring token. |

## Proposed: sparse verify

| Part | Rule |
|---|---|
| What the verify reads | Select cached tokens to reach a higher cumulative attention weight, for example p = 0.99, keeping more tokens than the draft. Always includes the draft's tokens. |
| Confidence check | A drafted token is accepted only if the sparse verify picks the same token and the score difference between its top two choices exceeds a threshold, still to be chosen. At the first token that fails, a full-cache verify takes over. |
| Refresh | A full-cache verify periodically rescans the whole cache to update the selection. The number of rounds between scans is still to be chosen. |
| Output | Not guaranteed equal to dense. Measured as wrong tokens per 1000. |

How we check agreement with dense:

| Step | Decides |
|---|---|
| 1. A full-precision draft uses the proposed sparse-verification KV selection and is checked using the whole cache. No new code. | The smallest attention-mass target p that meets the allowed disagreement rate; that rate is still to be chosen |
| 2. Log the score difference between the top two choices at each disagreement with dense. | The minimum score difference required to accept a token |
| 3. Run the full method and check every output token against dense. | The final wrong tokens per 1000 |

## Evaluation

| Item | Choice |
|---|---|
| Dataset | LongBench v2 only: the official chain-of-thought prompt in the chat template, the document cut in the middle to exactly the context length |
| Models | Qwen3-4B, Qwen3-8B |
| Contexts | 32K, 64K, 128K. YaRN beyond 40K. |
| Batch | 1, with 8 questions per model and context length |
| Generation | Stop at the end-of-sequence token (EOS), capped at 512 tokens. An answer cut off during reasoning can be used to measure speed, but not final-answer accuracy. |
| Baselines | Dense decoding. Vegas: fixed 7% of the cache, full-precision draft. |
| Speed | First-to-last-token time in one generation, against dense and Vegas. Draft, verify and round time from profiling. |
| Acceptance | Accepted drafted tokens / drafted tokens |
| Sparse verify quality | Wrong tokens per 1000 against dense |
| Rules | One A6000. Every comparison on one node. Benchmark configurations run one after another. |

## What we tried and dropped

Some ideas were tested and rejected. Others were dropped without a measured
result. Removing their code does not mean they failed a test.

| Approach | Status | What we measured or know |
|---|---|---|
| Activation sparsity: zero small activations to skip weight reads | Tested, rejected; experimental code removed | Qwen3-4B at 32K on A100: With TEAL-style sparsity at 50% / 70%, next-token agreement was 89.26% / 74.02%, against 99.41% for the dense control. All models were given the same text prefixes. Batch-1 forward steps took 8.70 / 7.69 ms, against 10.09 ms for the unmodified dense model. These tests measured prediction agreement and forward-step time separately. They did not measure the speed of a complete speculative decoding run. |
| Static whole-layer and attention skipping | Implemented, then dropped; code removed | The records checked so far do not show whether these changes improved speed. Removing them does not establish that they failed. |
| Early exit, draft trees, copy drafting, dynamic draft length and budget-driven attention skipping | No longer planned | No test result is documented here. The older method document listed the last two as future work. |
| FlashAttention's automatic KV split for packed verification | Tested; further work deferred | On Qwen3-0.6B with the whole cache selected, recorded acceptance fell from above 0.98 to 0.946, below the required acceptance rate of 0.98. Other ways to split the KV calculation remain untested. |

The activation-sparsity agreement test used four prompts with 256 tokens after
each prompt. Sparsity was applied only to those later tokens. This agreement
measurement is separate from acceptance during speculative decoding. Saved results
are in directories whose names start with `actsparse` under `outputs/runs/`. The split trial and former proposals are recorded in Git history
for `notes/handoff.md` and `notes/DrafterGoesBurrrr.md`. These historical results
are not final LongBench v2 evaluation results.

## Code

| Piece | Where |
|---|---|
| Selection and draft attention | `vllm/v1/spec_decode/sparse_attn/longspec/overrider.py`, `longspec/kernels/mass_select.py` |
| Attention weights from the verify | `longspec/portable/score_collection.py`, `longspec/kernels/c2q_scores.py` |
| Verify attention | `longspec/verify_attention.py` |
| 4-bit draft copy | `vllm/v1/spec_decode/sparse_attn/draft_weights.py` |
| Settings | `sparse_attn_*` in `vllm/config/speculative.py` |
| Runner | `benchmarks/longspec/grid.py` |
| Time per round | `benchmarks/longspec/round_phases.py` |
| Agreement | `benchmarks/longspec/w4_agreement.py` |

Draft buffers are allocated during model loading so their memory is counted before
space is reserved for the KV cache. Tests with fixed synthetic token sequences check
for identical output; LongBench runs report where outputs differ. Dense rescoring
compares next-token predictions on the same text prefixes, rebuilding the dense
model's KV cache. It reports ties for the highest score separately and gives lower
and upper disagreement counts per 1000 tokens, depending on how ties are counted.
