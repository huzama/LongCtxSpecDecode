# Method and experiments

## Method

Highlight-drafted self-speculative decoding on Qwen3-4B/8B. The drafter and
verifier share one BF16 model and persistent KV cache. Highlight's learned
indexer selects attention blocks per layer/request. The full KV cache stays
allocated; sparsity reduces attention reads. Prefill uses full attention.

| Method | Generation/draft attention | Verification |
|---|---|---|
| Dense baseline | Full attention, one token per step | None |
| Direct Highlight | 7%, scores refreshed every step | None |
| Our dense verifier | 7%, scores refreshed once per draft round | Full attention |
| Our SV50 | Same draft | 50% |
| Our SV40 | Same draft | 40% |

One speculative round:

1. Draft six tokens. Score history on the first draft step; reuse the scores
   on later steps while updating selection and KV.
2. Verify up to seven positions together, using full attention or freshly
   scored blocks at the 50%/40% verifier budget.
3. Accept the matching greedy prefix. Emit the verifier's correction at the
   first rejection, or a bonus token if all proposals match.
4. Verification rewrites the proposed KV; discard rejected positions from
   valid history before continuing.

Budgets use the original prompt length, rounded to 16-token blocks. Keep the
last 64 tokens inside the selection budget and preserve causal access to
verifier queries. These are attention budgets, not KV-memory reductions.

Our variants use `--mode highlight --draft-select round --spec-tokens 6`
with `--verify-fraction 0`, `.5` or `.4`. Direct Highlight uses
`--mode direct`; dense uses `--mode dense`. The runtime is a vLLM 0.27.1
V2-runner plugin over MoLLE's SpotAttention, Torch 2.13 and FA4, with separate
scored/reused draft graphs and six reserved lookahead slots.

Sparse verification changes attention and KV history, so it does not guarantee
full-attention outputs. Dense verification follows the greedy speculative
contract; GPU numerical differences limit exact-output claims. See the
[sanity checks](#sanity-checks).

## Main results

One B200, BF16 weights/KV, greedy decoding, thinking off. Blank cells are
unmeasured. Throughput and scored quality use separate workloads.

### Throughput

Tokens/s and speedup over the same runtime's dense baseline. Same prompts
and batch, 512 fixed outputs; pool tokens/time per repetition, then take the
median of three repetitions. Hold each request after its first token until
all prefills finish; time decode until the first request finishes. Exclude
warm-up, prefill, first tokens and the shrinking-batch tail. Async scheduling
and prefix caching are off; chunked prefill is on. Reserve KV for the full
output plus speculative lookahead; reject any KV preemption.

Matched batches nearly fill the sparse runtime's KV capacity; dense has more
headroom (4B/32K, batch 26: about 98% versus 79%, including output reserve).
The 0.9 GPU-memory setting alone does not imply KV saturation. Static YaRN
factors 2/3/4 for the three lengths, native base 32768.

| Model | Prompt | Batch | Dense | Direct Highlight | Our dense verifier | Our SV50 | Our SV40 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4B | 32768 | 26 | 864.3 (1.000x) | 2126.8 (2.461x) | 2350.3 (2.719x) | 2654.0 (3.071x) | 2826.8 (3.271x) |
| 4B | 65536 | 13 | 448.8 (1.000x) | 1110.2 (2.474x) | 1229.6 (2.740x) | 1395.9 (3.111x) | 1489.5 (3.319x) |
| 4B | 130560 | 6 | 263.2 (1.000x) | 564.3 (2.144x) | 660.9 (2.511x) | 715.2 (2.717x) | 758.7 (2.882x) |
| 8B | 32768 | 25 | 806.0 (1.000x) |  | 2008.2 (2.492x) | 2291.5 (2.843x) | 2359.2 (2.927x) |
| 8B | 65536 | 12 | 418.6 (1.000x) |  | 1008.3 (2.409x) | 1121.8 (2.680x) | 1207.8 (2.885x) |
| 8B | 130560 | 6 | 253.7 (1.000x) | 505.0 (1.991x) | 532.7 (2.100x) | 586.7 (2.313x) | 621.0 (2.448x) |

Direct Highlight at 4B/130560 uses three repetitions, 558.7–571.3 tokens/s.
New direct timing and the saved baselines use different physical B200 hosts.

### RULER

Mean retrieval score (%), six needle tasks × 32 rows = 192 per cell. Batch 1,
EOS stopping, cap 1024. Lengths below are haystack tokens; prompts add 37–74.

| Model | Haystack tokens | Dense | Direct Highlight | Our dense verifier | Our SV50 | Our SV40 |
|---|---:|---:|---:|---:|---:|---:|
| 4B | 65536 | 86.98 |  |  | 86.98 | 86.98 |
| 4B | 129024 | 68.75 | 68.75 | 68.75 | 68.75 | 69.27 |
| 8B | 65536 | 86.98 |  |  | 87.50 | 88.02 |
| 8B | 129024 | 75.52 |  |  | 76.56 | 76.56 |

At 4B/129024, direct Highlight and SV50 each lose one row and recover one:
delta 0.00 pp, paired 95% interval [-1.56, 1.56]. Our dense verifier matches
dense row scores on all 192 rows. Full paired changes and intervals are in the audited evidence.
These samples test short retrieval, not general accuracy preservation.

### LongBench v2

Accuracy (%), 60 questions/model, batch 1, original prompts 65893–114176.
Answer-only cap: 16384 tokens. CoT: 1024 reasoning + 128 answer tokens.
Answer-only is a prompting protocol, not direct Highlight.

| Model | Protocol | Dense | Direct Highlight | Our dense verifier | Our SV50 | Our SV40 |
|---|---|---:|---:|---:|---:|---:|
| 4B | Answer-only | 10.00 |  |  | 8.33 | 8.33 |
| 8B | Answer-only | 31.67 |  |  | 31.67 | 31.67 |
| 4B | CoT | 16.67 |  |  | 20.00 | 16.67 |
| 8B | CoT | 18.33 |  |  | 31.67 | 23.33 |

Dense accuracy is weak on this subset; equal scores can hide different errors.
CoT reasoning hits the cap on 7–10/60 questions for 4B and 24–27/60 for 8B.
Quality tasks have no matched timing: their scores and the throughput table
are separate evidence, not measured points on a speed–quality curve.

### Missing evidence

- Direct Highlight: 2 throughput, 3 RULER, 4 LongBench cells remain in the authorized queue.
- Our dense verifier: 0 throughput, 3 RULER, 4 LongBench cells remain in the authorized queue.
- Comparable timing on scored workloads is still missing; the current tables do not establish a speed–quality curve.

### Evidence

[Audited results](../outputs/evidence/methods.json) include per-task scores, paired changes,
intervals and 338 source hashes. [Audit script](../outputs/evidence/methods-audit.py).
[Vegas comparison](../outputs/evidence/chapter.json): our dense verifier is 1.032–1.162× faster,
but the runtimes differ.

## Sanity checks

| Check | Finding |
|---|---|
| Dense verifier replay | Bitwise-equal hidden states and logits at the tested 4B/128K round |
| Forced rejection at draft positions 1, 3, 6 | Tokens, accepted KV and rollback correct; rejected KV did not affect continuation |
| State restoration | All six saved 128-token prefixes preserved |

Dense-verifier outputs agree through EOS in five of six throughput cells.
At 4B/130560, 126/144 repeated appearances agree; all 18 differences involve
one prompt, starting at output token 70. The original numerical mismatch
remains unresolved: the matched compiled replay did not reproduce its tie
flip. Do not claim identical dense sequences or extend this sanity result
to sparse verification. [Details](../outputs/evidence/sanity.json);
[output agreement](../outputs/evidence/eos-agreement.json).

## Historical experiments

- INT4 drafting, early Spot/Vegas selector pilots and old batch sweeps are
  development evidence. Earlier YaRN/thinking results are not current results.
- Acceptance probes suggest a ranking gap; the serving/probe difference is
  unresolved. Earlier validation passed 31 plugin tests plus an EOS regression.
- Detailed legacy notes and artifacts remain in the
  [development archive](../../../archive/development.tar.gz). Obsolete plans and empty legacy tables
  are removed from this document.
