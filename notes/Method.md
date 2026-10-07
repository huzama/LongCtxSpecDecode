# Method

Self-speculative decoding at long context: a 4-bit copy drafts from selected
KV-cache tokens; the BF16 target verifies against the whole cache.

## Our method

| Part | Rule |
|---|---|
| Draft | Same model with 4-bit weights and 16-bit activations (W4A16). Draft 6 tokens per round. |
| Selection | Per layer and request, keep the first 4 and last 64 tokens. Add the highest-attention tokens to reach 85% attention mass, with the additional selection capped at 15% of the cache. |
| Attention scores | From the previous target verification: average its first and last new token's attention over heads. Update every round. |
| Verify | The BF16 target checks the drafts against the whole cache. Accept the matching prefix, then emit one target token. |
| Memory | Retain the target's full KV cache. The extra W4 weights and draft buffers consume GPU memory and can reduce the batch that fits. |

Full-cache verification preserves the target's greedy decoding rule. Different
batch shapes can still change floating-point results and token choices. Check
agreement by rescoring the same text prefixes with dense attention, reporting
ties for the highest score separately.

Experiment setup, batch selection, baselines, and result tables are in
[experiments.md](experiments.md).

### TODO next: sparse verification

Planned extension of our method; not yet implemented. Verify against a larger
cache selection than the draft uses. Accept only when the score gap between the top two choices exceeds a
threshold; otherwise use full-cache verification. Refresh the selection with
periodic full-cache passes. The selection size, score threshold, and refresh
interval remain undecided. Sparse verification can change target outputs and requires
measuring disagreements against dense attention.

## Technical details

Current A6000 implementation (FlashAttention 2):

| What | How |
|---|---|
| Load the draft | Load the matching W4A16 checkpoint. Share the target's embeddings, output head, and attention modules; retain separate draft projections and norms. |
| Multiply quantized weights | Marlin reads packed weights and reconstructs approximate BF16 values in registers using scales and zero points. BF16 multiplies accumulate in FP32 within the same kernel; no full BF16 draft-weight copy is stored. |
| Obtain attention scores | A Triton kernel rereads cached keys, computes the selected verification queries' dot products, and normalizes with FlashAttention's log-sum-exp. Reduce over queries and heads into one BF16 score per cached token. |
| Select cache entries | One CUDA block per layer/request finds the score threshold by radix selection. Reserved tokens count toward the 85% mass target; the 15% additional-token cap can prevent reaching it. Convert selected token indices to physical KV slots. |
| Read KV during drafting | Gather selected target KV and the new tail into per-layer BF16 buffers on the first draft step. Later steps append only the newest KV entry. Keep original token positions for rotary embeddings. |
| Verify and discard rejected drafts | The target recomputes the draft positions with full-cache attention, replacing their temporary draft KV. Accept the matching prefix; shorten the valid sequence length to exclude rejected positions. |
| CUDA graphs | Replay one verifier graph (forward, logits, acceptance, cache selection, next-input preparation), then one graph containing all 6 draft steps and sampling. The first draft gathers the selected KV; later drafts append inside the same graph. No CPU token readback between drafts. |
| Account for memory | Allocate draft buffers before vLLM sizes the target KV pool, so the extra weights and buffers reduce available cache capacity. |

Each draft step generates all 6 proposed tokens with **one CUDA graph launch**.
Each verification step checks those tokens with **one CUDA graph launch**.
A complete round contains one step of each kind.

This applies to greedy Qwen3 runs on one GPU using FlashAttention 2, for batch
sizes captured at startup and requests with all 6 drafts available to verify.
Other configurations use the existing execution path. The CPU still prepares
inputs, schedules requests, and receives outputs. GPU validation is pending.

## What we tried and dropped

| Approach | Outcome |
|---|---|
| Activation sparsity | Rejected. At 50% / 70% sparsity, same-prefix agreement was 89.26% / 74.02%, versus dense 99.41% (Qwen3-4B, 32K, A100). |
| Static layer and attention skipping | Code removed; no measured speed benefit recorded. |
| Early exit, draft trees, copy drafting, dynamic draft length, budget-driven attention skipping | Plans dropped; no recorded results. |
| FlashAttention automatic KV splitting | Deferred. Qwen3-0.6B full-cache verification: acceptance 0.946, below the required 0.98. |

Historical checks, not final LongBench results.
