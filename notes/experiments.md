# Experiments

**Paper measurements are pending. Run only when the user instructs.**
[Method](Method.md) describes the implementation. Earlier results below need
rerunning with corrected settings.

## Run settings

| Setting | Rule |
|---|---|
| Models | Qwen3-4B and Qwen3-8B; freeze the model/tokenizer commit before inference. |
| Hardware | One B200 per comparison, FA4 4.0.0b33, Torch 2.9.1/cu128. Only GPUs 2/3 on `b200-2`; weights in that host's local cache. |
| Methods | Dense, Vegas, ours (50% verifier KV, scores refreshed over retained keys). Additional comparisons: full verification; 50% scoring all keys; 40% scoring retained keys. |
| Precision | BF16 target/drafter/activations/KV; shared weights; no weight quantization. |
| Decoding | Thinking explicitly OFF; temperature 0, top-p 1, top-k disabled, min-p 0, repetition penalty 1, presence/frequency penalties 0. No inherited sampling defaults. |
| Context | Native 32,768 input+output tokens; static YaRN base 32,768, factor ceil(total budget / 32,768) above native. Total at most 131,072. Same RoPE settings across methods within each comparison. |
| Runtime | Same 90% memory allowance; async scheduling and prefix caching off. Capture requested batches; record graph replays and actual retained KV. |
| Data | LongBench v2 revision `2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9`. Freeze IDs, token IDs, prompt hashes, model revision, source commit/diff and dependency versions. |

Thinking is off because [Qwen warns against greedy thinking](https://huggingface.co/Qwen/Qwen3-8B#best-practices).
These tests use greedy decoding, not Qwen's recommended sampling settings.
Answer/latency runs stop at EOS; fixed-length throughput ignores EOS. Saved text
includes special tokens.

## Throughput and batch selection

Use 32,768 / 65,536 / 130,560 prompt tokens plus 512 outputs, with YaRN factors
2 / 3 / 4. The last setting is 128K total, including output. Timing prompts may
truncate document middles; accuracy prompts remain complete.

1. Bmax is the largest batch that fits all inputs, outputs and speculative
   lookahead. Confirm Bmax+1 fails capacity. Require no preemption and all outputs.
2. Bshared is the smallest Bmax across methods. Sweep batches 1..Bmax; Bbest is
   the fastest measured batch, choosing the smaller batch on exact ties.
3. Fix N=max(8, largest Bmax) questions across methods. Run N batches of B
   consecutive questions with wraparound, shifting the start by one. This weights
   every question equally. Use fresh engines and rotate method order.
   Freeze accuracy questions first and exclude them from timing with
   `grid --exclude <question-manifest-or-ID-list>`.
4. Hold each request after prefill until all B are ready. Measure from the last
   prefill's first token until the first request finishes. Exclude first tokens
   and the later partial-batch period; finish all requests. Measure acceptance
   over the same interval.
5. Confirm Bshared and each Bbest with three repetitions after the sweep.
   Aggregate emitted tokens / timed seconds per repetition; report median/range
   and ratios of medians. Exclude batch-search runs from these repetitions.

For B=1 latency, use the same questions and stop at EOS or 512 outputs. Report
first-to-last-token time; exclude the first token from token/s. These decode-only
measurements exclude prefill and do not measure end-to-end serving throughput.

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

Freeze complete documents with `python -m benchmarks.longspec.accuracy prepare`.
It excludes [175 previously used questions](../benchmarks/longspec/development_ids.json).
Add IDs from timing, capacity and tuning runs via `--exclude`. Stop if too few
unused questions remain. Choose sparsity before testing; report question counts
by context, domain and difficulty, since this tests only a subset of LongBench v2.

**Fresh long-context accuracy questions are still needed.** Earlier tests used
all 28 then-unused 32–64K questions. Changing prompts or thinking settings does
not make them unused again.

Use batch 1, identical questions and a 16,384-token output limit. Prompt ranges:
(16K,32K], (32K,64K], (64K,112K]; 112K+16K fits the 128K total limit. To test
without YaRN, use at most 16K input plus 16K output. Keep output limits identical
across methods because they determine RoPE scaling.

`accuracy run` saves answers, tokens and stop reasons. Score the final A–D choice;
answers hitting the output limit or missing a choice count incorrect.
`accuracy compare` checks identical questions/settings and reports losses
(dense correct, ours wrong), recoveries (dense wrong, ours correct), capped and
unparsed answers, and a paired bootstrap confidence interval. Equal accuracy
can hide different mistakes; draft acceptance does not measure answer accuracy.

| Model | Prompt range | N | Dense correct | Vegas correct | 50% correct | 40% correct | 50% lost/recovered | 50% capped/unparsed |
|---|---|---|---|---|---|---|---|---|
| 4B | 16–32K | | | | | | | |
| 4B | 32–64K | | | | | | | |
| 4B | 64–112K | | | | | | | |
| 8B | 16–32K | | | | | | | |
| 8B | 32–64K | | | | | | | |
| 8B | 64–112K | | | | | | | |

## FFN-only INT4 test

Quick test: BF16 versus FFN-only INT4 drafting, Qwen3-8B, 32,768 input
tokens, B=1, two previously used prompts and 128 fixed outputs. Both use 7%
draft KV and full verification. Keep attention, activations and KV BF16.
Corrected settings above; two repetitions with GPUs 2/3 swapped. INT4 checkpoint:
`RedHatAI/Qwen3-8B-quantized.w4a16`, commit
`32527053243a382bc56e941c964fc516a5014d39`.

| Drafter | Pooled decode tok/s | Repetition range | Acceptance |
|---|---|---|---|
| BF16 | 187.46 | 184.39–190.64 | 95.55% |
| FFN-only INT4 | 200.42 | 198.65–202.23 | 89.08% |
| Gate/up-only INT4 | 180.55 | 174.68–186.84 | 84.17% |
| Down-only INT4 | 197.64 | 194.04–201.36 | 93.46% |

Full-FFN INT4 gained 6.9%; gate/up-only lost 3.7%; down-only gained 5.4%.
The projection splits were later runs of the same protocol.
Pool emitted tokens / decode seconds and accepted / drafted
tokens; exclude prefill and first tokens. Both repetitions matched the saved
dense outputs exactly (two 128-token sequences); draft/verifier graphs replayed.
This small B=1 pilot does not establish saturation throughput or answer accuracy.

Controlled check: 16 shared verifier states, six fixed-token draft positions each,
identical KV and selection across variants. Eager prediction matches were BF16
95/96, gate/up 92/96, down 95/96, full FFN 94/96. Relative to BF16, these added
3/0/1 mismatches respectively, with no recoveries; down-only predictions were
identical to BF16. This measures fixed-prefix agreement, not graph acceptance.

Separate CUDA-event medians: six drafts took 29.40 ms BF16 and 24.66 ms INT4;
verification took 6.91/6.95 ms. A single-replay kernel trace attributed 14.91/9.09 ms
to FFN projections. Other draft work and verification limit the gain; at these
phase timings, equal acceptance would imply roughly 15% faster rounds.
Profiling was separate from throughput; do not add trace durations to event medians.
Raw commands, source hashes, outputs and summary are in `outputs/runs/`.

## Spot at 7% draft KV

Quick test: does Spot draft as well as Vegas at 7% KV? Qwen3-8B, one B200, B=1,
the two FFN-test questions cut to 32,640 prompt tokens so 128 outputs fit the
native window without YaRN. Greedy, thinking off, 6 drafts, full verification.

| Selector | Runtime | Accepted / drafted | Acceptance | Per prompt | Tokens / round |
|---|---|---|---|---|---|
| Vegas: 7% top-k tokens + recent, min 256 | This engine, FA4 | 212/234 | 90.6% | 92.1%, 89.2% | 6.44 |
| Spot top-K: 7% of prompt, 16-token blocks | Spot's vLLM 0.27.1 runtime | 208/234 | 88.9% | 89.5%, 88.3% | 6.33 |
| Spot top-K, last 64 tokens always kept | Spot's vLLM 0.27.1 runtime | 209/228 | 91.7% | 89.5%, 93.9% | 6.50 |
| Spot top-p 0.85: sink 128, recent 256, at least 512 tokens | Spot's vLLM 0.27.1 runtime | 216/234 | 92.3% | 89.2%, 95.6% | 6.54 |

Spot checkpoint `qwen3-8b-sparsekl-seq16k-k8192`. Its rounds are simulated:
Spot's prefill is dense, so 7 greedy tokens after a dense prefix give the
verifier's token, then 6 sparse drafts; acceptance is the prefix matching the
dense greedy continuation. The prefix cache is reset each round. One of 40
rounds is excluded: its prefill token differed from the reference at a near
tie, so its drafts followed a token the verifier did not choose. Differences:
Spot selects every draft step from the current query, Vegas once per round
from verifier scores; Spot was trained at K=8,192 of 16K (50%). Each selector
is scored against its own runtime's dense output; these agree on prompt 0 and
diverge at token 58 on prompt 1.

Spot's top-K keeps no recent window by default. The kept-recent row raises the
scores of the blocks holding the last 64 tokens before Spot's unchanged
selection kernel, inside the 7% budget; Spot's serving source is not edited.
It changed prompt 1 only. Remaining first-draft misses were plausible
rewordings, e.g. " semi-structured" where dense chose " multigrid".

Spot top-p 0.85 kept 25.1% of blocks during drafting (layers 2.6% to 54.9%),
from Spot's own counters. Its p is mass under the learned block distribution,
not attention mass: verifier-attention top-p 0.85 kept 1.2–1.6% in the earlier
runs. Compare selectors at equal kept fraction, not equal p.

Top-p 0.85 on the same prompts and setting. Ours ranks by verifier attention
(FA4, BF16 drafter, `--no-fixed-budget --ratio 1 --min-tokens 0`: no cap, no
floor); `--theta-scope` sets which mass the 85% applies to.

| Selector | 85% of | Reserved | KV kept | Layers | Accepted / drafted | Acceptance |
|---|---|---|---|---|---|---|
| Verifier attention | all mass | first 4, last 64 | 2.0% | 0.3–13.3% | 203/287 | 70.7% |
| Verifier attention | non-reserved mass | first 4, last 64 | 6.9% | 0.6–26.9% | 204/270 | 75.6% |
| Spot | non-reserved mass | first 128, last 256; floor 512 | 25.1% | 2.6–54.9% | 216/234 | 92.3% |

Counting only non-reserved mass raises ours 3.4x, not to Spot's 25%. Spot also
reserves 384 tokens, keeps 16-token blocks and uses its learned scores; these
were not separated. At about 7% kept, attention-mass top-p accepts 75.6%
against 90–92% for top-K (Vegas, Spot); some layers fall to 0.6%. Our kept
fraction averages the whole generation, Spot's the draft steps only.

## Selector time: Spot vs Vegas

Selection GPU time, Qwen3-8B, 32,640-token context, B=1, one B200, 36 layers.
Vegas: profiler trace of 10 real rounds in this engine (FA4 recomputes the
scores). Spot: its indexer-op microbenchmark (`op_micro.py`, CUDA-graph replay,
synthetic tensors, K=2,284 tokens) plus its projection GEMM timed alone.

| Per layer, µs | Vegas | Spot |
|---|---|---|
| Score | 24.1 (QK over all keys, 2 queries) | 19.8 (projection 4.5, norm/RoPE/cache write 5.3, block scores 9.0) |
| Top-k | 25.0 | 9.9 |
| Total | 49.2 | 29.8 |

| Per round | Vegas | Spot, select per draft step | Spot, select once per round |
|---|---|---|---|
| Selections | 1 | 6 | 1 |
| Selection ms | 1.77 | 6.44 | 1.07 |
| Share of a 38.8 ms Vegas round | 4.6% | 16.6% | 2.8% |

Vegas also gathers the selected KV for drafting: 0.82 ms per round. Spot reads
selected blocks in place through a compacted block table. The acceptance test
above used Spot's per-step selection. Microbenchmark and trace timings are not
the same instrument; treat differences under about 0.2 ms as unresolved.

## Earlier results: must rerun with corrected settings

These tests used the incorrect 40,960 YaRN base/threshold and thinking ON with
greedy decoding. They also ran before sparse verification was restricted to
complete decode rounds. All 60 accuracy questions used factor 2 and a 16K output
limit. Counts were rescored from saved answers, matched by question/prompt hash;
all capped answers lacked a final choice, and no questions were removed.

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

Ours used one batch; dense/Vegas are three-repetition medians over larger question
pools. These mismatched runs cannot establish the final speedup. Scoring only
retained keys slowed a 32K B=1 test; the cause is unresolved. Raw results and the
numerical audit remain in `outputs/runs/`.
