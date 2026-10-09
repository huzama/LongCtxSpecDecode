# Method

One BF16 Qwen3 model drafts and verifies with thinking off and greedy decoding.
Keep 7% of history for drafting and 50% for verification per layer/request; also
test 40% verification. Refresh scores only for retained verifier keys. Sparse verification
can change the full-cache model's outputs; corrected speed and accuracy results
are pending.

## One speculative round

| Step | Implementation |
|---|---|
| Select history | Reuse the preceding verifier's scores, initialized during full-attention prefill. Rank per layer/request. Reserve the first 4 and last 64 scored positions, all newer unscored history, and all current query positions allowed by causality. These required tokens can exceed the requested fraction. |
| Verify | FA4 reads selected entries from the persistent BF16 KV cache using one-token page tables. All model layers and linear projections still run. Recompute the proposed positions' KV with the verifier. |
| Accept | Accept consecutive draft tokens matching the verifier. Emit its correction at the first mismatch, or a bonus token if all 6 match. Shorten valid cache lengths to discard rejected positions. |
| Refresh scores | Recompute first/last verifier-query QK products only for retained keys; normalize with sparse FA4 log-sum-exp. Average over query heads and those queries, store BF16 scores at original token positions. Unselected positions receive -1, below even zero attention weight. Score only positions visible to the first query. |
| Draft | A batched radix-selection kernel selects fixed 7% independently for every layer/request. Gather selected verifier KV once, then generate 6 draft tokens, appending each new KV entry. Keep original rotary positions. |

The correction/bonus token uses temporary draft KV until the next verification
replaces it. Accepted KV comes from the sparse verifier and can differ from
full-cache decoding.

Prefill, incomplete rounds and unsupported shapes use full attention. Only
complete rounds in CUDA graphs use sparse verification. There is no periodic
full-key scan: omitted history can re-enter after full attention refreshes its
scores, but this does not reconstruct earlier full-cache KV.

## Execution

- B200, FA4, BF16 activations/KV. The full KV cache stays allocated; sparsity
  reduces attention reads.
- One verifier CUDA graph and one graph for all 6 drafts per complete round.
  Each graph contains multiple kernels. CPU staging, scheduling and result
  transfer remain outside; graph counters cover the whole generation.
- Our draft ratio includes reserved tokens. The unchanged Vegas selector uses
  a 7% non-reserved top-k budget plus sink/recent tokens and a 256-token floor.
  Report actual retained fractions; these budgets are not exactly identical.
- Qwen3 native context is 32,768 input+output tokens. Above it, static YaRN uses
  base 32,768 and factor ceil(total budget / 32,768), with total at most 131,072.

Use `--mode coverage --draft-weights target --fixed-budget --ratio .07
--verify-ratio .5 --verify-score-scope selected --flash-attn-version 4` in the
benchmark. The engine defaults to full verification; these flags enable the method.

## What we tried

| Approach | Decision |
|---|---|
| Quantized drafts | Optional `--draft-weights-scope ffn/gate_up/down` quantizes all FFN projections, gate/up only, or down only. Other projections, norms, embeddings, output head, activations and KV stay BF16. Full-verification pilots found no large gain; gate/up was more sensitive in a small fixed-prefix check. See Experiments. |
| Attention-mass selection (top-p) | Replaced by fixed 7%: past attention mass did not predict future draft choices reliably; a larger minimum KV budget improved acceptance. |
| Full-key score refresh | Keep as a comparison: its extra scans reduced the measured speed benefit. |
| Activation sparsity, static skipping | Dropped after output disagreement or no measured speed benefit. |

[Experiments](experiments.md) contains run settings, empty paper tables and earlier
results that need rerunning with corrected settings.
