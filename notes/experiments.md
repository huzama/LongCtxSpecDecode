# Experiments

**Paper measurements are pending. Run only when the user instructs.**
The method is BF16/fixed-7% drafting with 50% sparse verification and selected-key
scoring. Use 40% verification as an ablation. Old results below are development
evidence, not final accuracy or speedup claims.

## Fixed protocol

| Setting | Rule |
|---|---|
| Models | Qwen3-4B and Qwen3-8B; freeze the model/tokenizer commit before inference. |
| Hardware | One B200 per comparison, FA4 4.0.0b33, Torch 2.9.1/cu128. Only GPUs 2/3 on `b200-2`; weights in that host's local cache. |
| Methods | Dense, Vegas, ours (50%). Ablations: ours with full verification; 50% with full-key scoring; 40% with selected-key scoring. |
| Precision | BF16 target/drafter/activations/KV; shared weights; no weight quantization. |
| Decoding | Thinking explicitly OFF; temperature 0, top-p 1, top-k disabled, min-p 0, repetition penalty 1, presence/frequency penalties 0. No inherited sampling defaults. |
| Context | Native 32,768 input+output tokens; static YaRN base 32,768, factor ceil(total budget / 32,768) above native. Total at most 131,072. Same RoPE settings across methods within each comparison. |
| Runtime | Same 90% memory allowance; async scheduling and prefix caching off. Capture requested batches; record graph replays and actual retained KV. |
| Data | LongBench v2 revision `2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9`. Freeze IDs, token IDs, prompt hashes, model revision, source commit/diff and dependency versions. |

Greedy decoding is the experiment's controlled setting, not an official Qwen
quality score. [Qwen warns against greedy thinking](https://huggingface.co/Qwen/Qwen3-8B#best-practices);
that combination is excluded. Qwen's EOS IDs remain active in answer/latency
runs. Only fixed-length throughput ignores EOS. Special tokens remain in saved text.

## Throughput and batch selection

Use exact prompt budgets 32,768 / 65,536 / 130,560 tokens and 512 outputs. The
last setting reserves outputs within the 128K total limit; do not call it a 128K
prompt. Factors are 2 / 3 / 4. Timing prompts may truncate document middles;
these prompts are not used for answer accuracy.

1. Find each method's Bmax; reserve all outputs and lookahead, confirm Bmax+1
   fails capacity, and require zero preemption and complete generation.
2. Bshared is the minimum Bmax. Sweep batches 1..Bmax; Bbest is fastest measured,
   breaking exact ties toward smaller batches. Report batch sizes beside throughput.
3. Fix N=max(8, largest Bmax) questions across methods. Run N batches of B
   consecutive questions with wraparound, shifting the start by one. This weights
   every question equally. Use fresh engines and rotate method order.
   Freeze quality questions first and exclude their IDs from timing with
   `grid --exclude <question-manifest-or-ID-list>`.
4. Hold completed prefills until all B are ready. Time from the last prefill's
   first token until the first request finishes. Exclude prefill tokens and the
   partial-batch tail; drain every request. Acceptance uses the same timed window.
5. Confirm Bshared and each Bbest with three repetitions after the sweep.
   Aggregate emitted tokens / timed seconds per repetition; report median/range
   and ratios of medians. Selection/capacity runs are not confirmation repetitions.

For B=1 latency, use the same frozen pool, stop at EOS or 512 tokens, and report
first-to-last-token time with the first token excluded. Keep these results separate
from fixed-length saturation. No claim of end-to-end serving throughput follows.

| Model | Prompt tokens | Bshared | Dense tok/s | Vegas tok/s | Ours tok/s | Ours/dense | Ours/Vegas |
|---|---|---|---|---|---|---|---|
| 4B | 32768 | | | | | | |
| 4B | 65536 | | | | | | |
| 4B | 130560 | | | | | | |
| 8B | 32768 | | | | | | |
| 8B | 65536 | | | | | | |
| 8B | 130560 | | | | | | |

| Model | Prompt tokens | Dense Bbest | Dense tok/s | Vegas Bbest | Vegas tok/s | Our Bbest | Our tok/s |
|---|---|---|---|---|---|---|---|
| 4B | 32768 | | | | | | |
| 4B | 65536 | | | | | | |
| 4B | 130560 | | | | | | |
| 8B | 32768 | | | | | | |
| 8B | 65536 | | | | | | |
| 8B | 130560 | | | | | | |

## Answer accuracy

Freeze complete, untruncated documents before inference using
`python -m benchmarks.longspec.accuracy prepare`. The tracked
[development exclusions](../benchmarks/longspec/development_ids.json) contain
175 previously used questions. Add IDs from timing, capacity and tuning runs
via `--exclude`; never select sparsity using the paper test set. Stop preparation
if too few eligible examples remain, rather than silently reducing sample size.
Report question counts and domain/difficulty composition; this is a subset study.
The old study exhausted the 28 then-unused 32–64K questions. A fresh independent
long-context accuracy set is still a prerequisite; changing the prompt or thinking
mode does not make a used question held out again.

Use batch 1, the same frozen questions, thinking off, greedy decoding and a
16,384-output-token ceiling. Prompt ranges: (16K,32K], (32K,64K], and (64K,112K];
112K+16K fits the 128K total limit. A native-window control can use at most 16K
input plus 16K output without YaRN. Do not mix these variable-length accuracy
runs with exact-length throughput. The generation budget controls RoPE scaling,
so do not change it between compared methods.

`accuracy run` saves every answer, output token, termination reason and parsed
choice. Score the final A–D choice; capped answers and missing final choices
count incorrect. `accuracy compare` checks matching manifests/protocols, reports
paired losses/recoveries, caps, parse failures and a question-level paired
bootstrap interval. Report by context/domain/difficulty. Equal net accuracy is
not choice preservation; sparse acceptance is not a quality metric.

| Model | Prompt range | N | Dense correct | Vegas correct | 50% correct | 40% correct | 50% lost/recovered | 50% capped/unparsed |
|---|---|---|---|---|---|---|---|---|
| 4B | 16–32K | | | | | | | |
| 4B | 32–64K | | | | | | | |
| 4B | 64–112K | | | | | | | |
| 8B | 16–32K | | | | | | | |
| 8B | 32–64K | | | | | | | |
| 8B | 64–112K | | | | | | | |

## Development evidence only

These measurements used the old 40,960 YaRN reference/threshold and thinking ON
with greedy decoding. Both 60-question accuracy groups used factor 2 and a 16K
output cap. They also predate the explicit complete-round sparse-verifier gate.
The following counts were rescored from saved texts and paired by question/hash.
All capped answers lacked a final choice; no question was removed.

| Verifier | 16–32K correct | 32–64K correct | Total | Dense-correct lost | Dense-wrong recovered | Capped |
|---|---|---|---|---|---|---|
| Dense | 16/32 | 10/28 | 26/60 | 0 | 0 | 4 |
| 50% | 17/32 | 10/28 | 27/60 | 9 | 10 | 5 |
| 40% | 15/32 | 11/28 | 26/60 | 6 | 6 | 6 |
| 30% | 12/32 | 10/28 | 22/60 | 11 | 7 | 9 |
| 25% | 17/32 | 8/28 | 25/60 | 10 | 9 | 7 |
| 15% | 14/32 | 8/28 | 22/60 | 12 | 8 | 12 |

Old Qwen3-8B saturation rates, recomputed from emitted tokens and decode times:

| Context | Dense batch / tok/s | Vegas batch / tok/s | Ours 50% batch / tok/s |
|---|---|---|---|
| 32K | 31 / 955.14 | 29 / 1986.80 | 29 / 2747.79 |
| 64K | 15 / 480.76 | 14 / 1014.43 | 14 / 1390.17 |

Ours used one batch; dense/Vegas are earlier three-repetition medians over larger
prompt pools. Their ratios are not controlled final speedup estimates. A 32K
B=1 check slowed down with selected scoring; it remains unexplained. Earlier
quantization/top-p/FA2 tables are retired from the paper plan. Raw artifacts and
the numerical audit remain under `outputs/runs/`; no historical output was rewritten.
