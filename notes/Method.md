# Method

Approximate self-speculative decoding: one BF16 Qwen3 supplies both draft and
verifier weights. Keep 7% of draft KV and 50% of verifier history per layer and
request; refresh attention scores only over retained verifier keys. The 40%
verifier is an ablation. This trades output fidelity for decode throughput;
it does not preserve the full-cache target's greedy choices.

## One speculative round

| Step | Implementation |
|---|---|
| Select history | Reuse scores from the preceding verifier, initialized by full-attention prefill. Rank separately per layer/request. Reserve the first 4 and recent 64 positions of the scored prefix. Keep unscored history and the complete causal query tail. Mandatory tokens may exceed the nominal fraction. |
| Verify | FA4 reads selected entries from the persistent BF16 KV cache using one-token page tables. All model layers and linear projections still run. Recompute the proposed positions' KV with the verifier. |
| Accept | Greedy comparison accepts the matching draft prefix, then emits a correction or bonus token. Shorten valid cache lengths to discard rejected positions. Acceptance is against this approximate verifier, not the dense target. |
| Refresh scores | Recompute first/last verifier-query QK products only for retained keys; normalize with sparse FA4 log-sum-exp. Average over query heads and those queries, store BF16 scores at original token positions. Unselected positions receive -1, below even zero attention weight. Score only positions visible to the first query. |
| Draft | A batched radix-selection kernel selects fixed 7% independently for every layer/request. Gather selected verifier KV once, then generate 6 draft tokens, appending each new KV entry. Keep original rotary positions. |

The next correction/bonus token has no verifier KV yet. Its draft KV is temporary
until the next verification replaces it. Accepted KV is verifier-generated, but
after sparse verification it can differ from KV produced by full-cache decoding.
A full-attention fallback refreshes scores; it does not reconstruct earlier dense KV.

No periodic full-key scan runs during sparse verification. Omitted history can
re-enter after a full-attention fallback. Prefill, incomplete rounds, and other
unsupported shapes use full attention. Sparse verification is admitted only in
the captured complete-round path, so a short prefill cannot trigger it by shape.
Reserved recency applies to the known scored prefix; newer unscored positions
are additionally retained.

## Execution and controls

- B200, FA4, BF16 weights/activations/KV; no extra draft weights. Full KV remains
  allocated. Sparsity reduces attention reads, not the persistent KV allocation.
- One verifier CUDA graph and one graph for all 6 drafts per complete round.
  Each graph contains multiple kernels. CPU staging, scheduling and result
  transfer remain outside; graph counters cover the whole generation.
- Our draft ratio includes reserved tokens. The unchanged Vegas selector uses
  a 7% non-reserved top-k budget plus sink/recent tokens and a 256-token floor.
  Report actual retained fractions; these budgets are not exactly identical.
- Plain greedy, thinking off, temperature 0, no penalties or logit processors.
  This is a controlled decoding protocol, not Qwen's recommended sampled quality
  configuration. Prefill and full verification remain available as controls.
- Qwen3 native context: 32,768 **input plus output** tokens. Above it, static YaRN
  uses original window 32,768 and factor ceil(total budget / 32,768). Cap total
  length at the documented 131,072-token range. Selection never renumbers positions.

Use `--mode coverage --draft-weights target --fixed-budget --ratio .07
--verify-ratio .5 --verify-score-scope selected --flash-attn-version 4` in the
benchmark. For controls, set verifier ratio 1 and scoring scope `full`; 50% with
`full` scoring isolates the cost of full-key score refresh. Engine defaults retain
full verification; paper runs explicitly select the approximate method.

## What we tried and dropped

| Approach | Decision |
|---|---|
| Quantized drafts | Keep shared BF16 weights. Quantization added draft disagreement in development checks. Optional checkpoints remain supported outside the paper method. |
| Attention-mass selection | Keep fixed 7%. A threshold on averaged past attention did not preserve future draft choices; a larger minimum KV budget recovered the measured deficit. |
| Full-key sparse score refresh | Retain as an ablation. Its extra full-key normalization/scoring scans largely removed the attention-read savings. |
| Activation sparsity, static skipping | Dropped after disagreement or no demonstrated speed benefit. |
| Early exit, trees, copy drafting, dynamic draft length | Not part of this implementation. |

[Experiments](experiments.md) defines the protocol, pending paper tables and
separately labelled development evidence. No corrected paper results exist yet.
