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
| 8B | 32768 | 25 | 806.0 (1.000x) | 1934.9 (2.401x) | 2008.2 (2.492x) | 2291.5 (2.843x) | 2359.2 (2.927x) |
| 8B | 65536 | 12 | 418.6 (1.000x) | 964.4 (2.304x) | 1008.3 (2.409x) | 1121.8 (2.680x) | 1207.8 (2.885x) |
| 8B | 130560 | 6 | 253.7 (1.000x) | 505.0 (1.991x) | 532.7 (2.100x) | 586.7 (2.313x) | 621.0 (2.448x) |

Direct Highlight at 4B/130560 uses three repetitions, 558.7–571.3 tokens/s.
New direct timing and the saved baselines use different physical B200 hosts.

### RULER

Mean retrieval score (%), six needle tasks × 32 rows = 192 per cell. Batch 1,
EOS stopping, cap 1024. Lengths below are haystack tokens; prompts add 37–74.

| Model | Haystack tokens | Dense | Direct Highlight | Our dense verifier | Our SV50 | Our SV40 |
|---|---:|---:|---:|---:|---:|---:|
| 4B | 65536 | 86.98 | 86.98 | 86.98 | 86.98 | 86.98 |
| 4B | 129024 | 68.75 | 68.75 | 68.75 | 68.75 | 69.27 |
| 8B | 65536 | 86.98 | 86.46 | 86.98 | 87.50 | 88.02 |
| 8B | 129024 | 75.52 | 72.40 | 75.52 | 76.56 | 76.56 |

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
| 4B | Answer-only | 10.00 | 8.33 | 10.00 | 8.33 | 8.33 |
| 8B | Answer-only | 31.67 | 31.67 | 31.67 | 31.67 | 31.67 |
| 4B | CoT | 16.67 | 5.00 | 20.00 | 20.00 | 16.67 |
| 8B | CoT | 18.33 | 5.00 | 23.33 | 31.67 | 23.33 |

Dense accuracy is weak on this subset; equal scores can hide different errors.
CoT reasoning hits the cap on 7–10/60 questions for 4B and 24–27/60 for 8B.
Quality tasks have no matched timing: their scores and the throughput table
are separate evidence, not measured points on a speed–quality curve.

### Selector comparisons

Complete: 4B/8B × 32768/65536/130560 tokens, **batch 1**, first six existing
prompts, layers 2/18/35. All checks passed for 108 captures and both budget
protocols (216 evaluations). These measure per-layer selector costs.

Each cell is the median across 18 capture medians. Per capture: five warmups,
20 CUDA-graph samples of 20 calls; initialization uses one call/sample.
No independent process repetitions. Capture, transfers, loading, compilation
and warm-up are excluded. All six cells ran on the same B200 GPU.

Total is score + rank/index + gather, excluding one-time initialization. It is
a derived sum of stage medians, not a separately measured end-to-end latency.
Highlight requires no gather, so its contribution to Total is zero.

Score includes acquisition and reduction. Vegas reports fused-attention time
minus matched dense-attention time, plus reduction; paired samples retain
negative measurement noise. Highlight uses native production kernels; Vegas
is our FP32 FA4 port; TriForce uses the PyTorch retrieval reference.

Initialization builds Highlight indexer keys or TriForce chunk-key means;
scoring/ranking/gathering remain separate. `n/a` means no Highlight KV gather
or no separate Vegas auxiliary-key initialization measurement.

Extra KV counts Highlight indexer keys or the gathered Vegas/TriForce KV,
excluding the shared full cache and scratch. Vegas collection buffers and
TriForce chunk means each need about 8/16/31.88 MiB at the three lengths;
these are identified buffers, not peak memory measurements.

**Matched token budgets**

| Model | Context | Selector | Tokens/head | Score (µs) | Rank/index (µs) | Gather (µs) | Total (µs) | Initialization (ms) | Extra KV (MiB) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4B | 32768 | Highlight | 2304 | 23.21 | 9.40 | n/a | 32.61 | 1.189 | 32.00 |
| 4B | 32768 | Vegas | 2304 | 391.10 | 63.55 | 54.35 | 508.99 | n/a | 9.00 |
| 4B | 32768 | TriForce | 2304 | 2.47 | 40.44 | 158.02 | 200.94 | 0.029 | 36.00 |
| 4B | 65536 | Highlight | 4592 | 28.61 | 10.54 | n/a | 39.15 | 2.358 | 64.00 |
| 4B | 65536 | Vegas | 4592 | 759.97 | 119.48 | 88.78 | 968.22 | n/a | 17.94 |
| 4B | 65536 | TriForce | 4592 | 4.18 | 54.53 | 320.11 | 378.82 | 0.051 | 71.75 |
| 4B | 130560 | Highlight | 9152 | 43.73 | 11.31 | n/a | 55.04 | 4.702 | 127.50 |
| 4B | 130560 | Vegas | 9152 | 1470.56 | 127.92 | 169.57 | 1768.05 | n/a | 35.75 |
| 4B | 130560 | TriForce | 9152 | 4.54 | 57.77 | 656.56 | 718.87 | 0.088 | 143.00 |
| 8B | 32768 | Highlight | 2304 | 24.08 | 9.47 | n/a | 33.55 | 1.522 | 32.00 |
| 8B | 32768 | Vegas | 2304 | 391.75 | 63.59 | 54.24 | 509.58 | n/a | 9.00 |
| 8B | 32768 | TriForce | 2304 | 2.44 | 40.53 | 157.91 | 200.88 | 0.029 | 36.00 |
| 8B | 65536 | Highlight | 4592 | 29.69 | 10.41 | n/a | 40.11 | 3.013 | 64.00 |
| 8B | 65536 | Vegas | 4592 | 739.77 | 113.95 | 88.36 | 942.09 | n/a | 17.94 |
| 8B | 65536 | TriForce | 4592 | 4.17 | 54.12 | 318.90 | 377.20 | 0.051 | 71.75 |
| 8B | 130560 | Highlight | 9152 | 44.65 | 11.31 | n/a | 55.96 | 6.000 | 127.50 |
| 8B | 130560 | Vegas | 9152 | 1468.24 | 129.01 | 169.59 | 1766.83 | n/a | 35.75 |
| 8B | 130560 | TriForce | 9152 | 4.50 | 57.79 | 656.61 | 718.90 | 0.090 | 143.00 |

**Paper token budgets**

| Model | Context | Selector | Tokens/head | Score (µs) | Rank/index (µs) | Gather (µs) | Total (µs) | Initialization (ms) | Extra KV (MiB) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4B | 32768 | Highlight | 2288 | 23.31 | 9.45 | n/a | 32.76 | 1.190 | 32.00 |
| 4B | 32768 | Vegas | 2294 | 391.10 | 63.11 | 54.24 | 508.45 | n/a | 8.96 |
| 4B | 32768 | TriForce | 4096 | 2.47 | 40.74 | 281.70 | 324.91 | 0.029 | 64.00 |
| 4B | 65536 | Highlight | 4576 | 28.56 | 10.43 | n/a | 38.99 | 2.357 | 64.00 |
| 4B | 65536 | Vegas | 4588 | 759.89 | 118.48 | 88.77 | 967.14 | n/a | 17.92 |
| 4B | 65536 | TriForce | 4096 | 4.18 | 54.14 | 284.05 | 342.38 | 0.051 | 64.00 |
| 4B | 130560 | Highlight | 9136 | 43.29 | 11.25 | n/a | 54.54 | 4.703 | 127.50 |
| 4B | 130560 | Vegas | 9140 | 1470.65 | 127.85 | 169.54 | 1768.04 | n/a | 35.70 |
| 4B | 130560 | TriForce | 4096 | 4.54 | 57.03 | 285.70 | 347.28 | 0.088 | 64.00 |
| 8B | 32768 | Highlight | 2288 | 23.50 | 9.43 | n/a | 32.94 | 1.524 | 32.00 |
| 8B | 32768 | Vegas | 2294 | 391.73 | 63.59 | 54.23 | 509.55 | n/a | 8.96 |
| 8B | 32768 | TriForce | 4096 | 2.45 | 40.78 | 280.86 | 324.09 | 0.030 | 64.00 |
| 8B | 65536 | Highlight | 4576 | 29.06 | 10.41 | n/a | 39.47 | 3.013 | 64.00 |
| 8B | 65536 | Vegas | 4588 | 739.70 | 113.98 | 88.29 | 941.97 | n/a | 17.92 |
| 8B | 65536 | TriForce | 4096 | 4.17 | 53.93 | 283.75 | 341.85 | 0.051 | 64.00 |
| 8B | 130560 | Highlight | 9136 | 44.07 | 11.24 | n/a | 55.32 | 5.995 | 127.50 |
| 8B | 130560 | Vegas | 9140 | 1468.03 | 129.08 | 169.47 | 1766.59 | n/a | 35.70 |
| 8B | 130560 | TriForce | 4096 | 4.51 | 57.26 | 285.52 | 347.28 | 0.090 | 64.00 |

Matched budgets round 7% up to a common 16-token boundary. Paper budgets
preserve native rules: Highlight floors 7% to blocks of 16, Vegas takes
ceil(7%), and TriForce keeps 4096 tokens in chunks of 8. Highlight retains
the last 64 tokens inside K and reads compacted page tables without copying KV.

Inputs use the main grid’s checkpoints, prompts and YaRN factors with dense
FA4. Replay seven queries from six greedy continuation tokens; Vegas observes
both boundary queries, while Highlight/TriForce use the last prefill query on
the same prompt KV. Fixed-length capture can include EOS; it is not a quality
or acceptance experiment. Only input hashes and small result records are saved.

Refresh frequency differs: Highlight scores once per draft round, Vegas
refreshes after verification, and TriForce selects once at prefill then updates
its tail. Tail-update latency is not measured here. Per-event costs do not
establish per-token speedups or selector performance at saturated batches.

Selector-quality evidence comes from [Chapter 2’s attention-map and retrieval
analysis](https://github.com/huzama/PhDThesis/blob/76271e73a045ff853f33f0f608305cccd8b072a3/thesis/chapters/02-distilled-selectors.tex#L190)
(`fig:distilled:attention-maps`, `fig:distilled:attention-retrieval`). These
timings do not establish equal selection quality against Vegas/TriForce at 7%.

[Selector results and source hashes](../outputs/evidence/selectors.json).


### MXFP4 Highlight

MXFP4 indexer keys and native Blackwell scoring; model weights and backbone KV
remain BF16. Existing matched throughput batches and frozen accuracy prompts
are reused. Quantization adds capacity headroom; these batches are not retuned
to the MXFP4 saturation limit. Dense and external-selector baselines above are reused.

Queue: 0/111 jobs complete, 0 running, 0 failed. This includes one native-kernel validation job and 110 measurement jobs.

**Decode throughput (tokens/s)**. Same prefill barrier, 512 outputs, asynchronous
scheduling off, decode-only window and three repetitions as BF16. Cells remain
blank until all three repetitions pass validation.

| Model | Context | Batch | Direct Highlight | Dense verifier | SV50 | SV40 |
|---|---:|---:|---:|---:|---:|---:|
| 4B | 32768 | 26 |  |  |  |  |
| 4B | 65536 | 13 |  |  |  |  |
| 4B | 130560 | 6 |  |  |  |  |
| 8B | 32768 | 25 |  |  |  |  |
| 8B | 65536 | 12 |  |  |  |  |
| 8B | 130560 | 6 |  |  |  |  |

**RULER (%)**. Same six needle tasks × 32 rows, batch 1, cap 1024.

| Model | Haystack tokens | Direct Highlight | Dense verifier | SV50 | SV40 |
|---|---:|---:|---:|---:|---:|
| 4B | 65536 |  |  |  |  |
| 4B | 129024 |  |  |  |  |
| 8B | 65536 |  |  |  |  |
| 8B | 129024 |  |  |  |  |

**LongBench v2 (%)**. Same 60 questions/model and answer-only/CoT token caps.

| Model | Protocol | Direct Highlight | Dense verifier | SV50 | SV40 |
|---|---|---:|---:|---:|---:|
| 4B | Answer-only |  |  |  |  |
| 4B | CoT |  |  |  |  |
| 8B | Answer-only |  |  |  |  |
| 8B | CoT |  |  |  |  |

**Highlight selector timing**. Same batch-1 inputs, layers, budgets and CUDA-graph
aggregation as the BF16 selector table. Total sums score and rank medians;
native paged attention needs no KV gather. Extra KV includes the MXFP4 scales.

| Model | Context | Budget | Tokens/head | Score (µs) | Rank (µs) | Total (µs) | Initialization (ms) | Extra KV (MiB) |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 4B | 32768 | Matched |  |  |  |  |  |  |
| 4B | 32768 | Paper |  |  |  |  |  |  |
| 4B | 65536 | Matched |  |  |  |  |  |  |
| 4B | 65536 | Paper |  |  |  |  |  |  |
| 4B | 130560 | Matched |  |  |  |  |  |  |
| 4B | 130560 | Paper |  |  |  |  |  |  |
| 8B | 32768 | Matched |  |  |  |  |  |  |
| 8B | 32768 | Paper |  |  |  |  |  |  |
| 8B | 65536 | Matched |  |  |  |  |  |  |
| 8B | 65536 | Paper |  |  |  |  |  |  |
| 8B | 130560 | Matched |  |  |  |  |  |  |
| 8B | 130560 | Paper |  |  |  |  |  |  |

[MXFP4 results and source hashes](../outputs/evidence/mxfp4.json).

### Findings

Dense verification: 1.038–1.171× direct Highlight throughput.
SV50: 1.162–1.267× direct Highlight throughput.
SV40: 1.219–1.344× direct Highlight throughput.
RULER changes versus dense (pp): dense verifier +0.00 to +0.00; SV50 +0.00 to +1.04; SV40 +0.00 to +1.04.
LongBench changes versus dense (pp): dense verifier +0.00 to +5.00; SV50 -1.67 to +13.33; SV40 -1.67 to +5.00.
These observed ranges apply to the measured workloads; consult paired intervals before claiming a quality advantage.

### Missing evidence

- The planned throughput, RULER, LongBench and batch-1 selector timing grids are complete.
- Comparable timing on scored workloads is still missing; the current tables do not establish a speed–quality curve.

### Evidence

[Audited results](../outputs/evidence/methods.json) include per-task scores, paired changes,
intervals and 392 source hashes. [Audit script](../outputs/evidence/methods-audit.py).
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
