> Frozen at the HTML transition on 15 October 2026. Read the [current report](../experiments.html) for updated results.

# Learned selection and speculative decoding: method and experiments

We evaluate Highlight's learned attention-block selection and its use in
self-speculative decoding. BF16 and MXFP4 indexer variants are
compared at matched attention budgets and batch sizes. Model weights and
backbone KV remain BF16 throughout.

Highlight has the lowest summed selector-stage latency among the measured
implementations. MXFP4 reduces indexer storage. In the historical
forced-continuation workload, it improves Direct Highlight throughput in
every measured setting. With MXFP4, sparse verification remains
faster than Direct Highlight, while dense verification is slower. Accuracy
results vary by task; they do not establish general quality preservation.

## 1. Method and experimental setup

### Generation methods

The drafter and verifier share one model and persistent backbone KV cache.
Prefill uses full attention. Highlight selects attention blocks per layer
and request; the full backbone cache stays allocated. Sparsity reduces
attention reads, not backbone KV storage.

| Method | Generation or draft attention | Verification attention |
|---|---|---|
| Dense | Full attention; one token per step | None |
| Direct Highlight | 7%; scores refreshed every step | None |
| Dense verifier | 7%; scores refreshed once per draft round | Full attention |
| SV50 | Same draft as dense verifier | 50% |
| SV40 | Same draft as dense verifier | 40% |

Dense verifier, SV50 and SV40 are our self-speculative variants. One round:

1. Draft six tokens. Score history on the first step, then reuse the scores
   while updating selection and KV on subsequent steps.
2. Verify up to seven positions together, using full attention or freshly
   scored blocks at the 50%/40% verification budget.
3. Accept the matching greedy prefix. Emit the verifier's correction at the
   first rejection, or a bonus token if all proposals match.
4. Verification rewrites the proposed KV. Remove rejected positions from
   valid history before continuing.

Budgets use each request's original prompt length and 16-token blocks. Blocks
covering the last 64 tokens receive priority inside the selection budget.
Short CoT answer prompts can have too little budget to retain that window;
the diagnosis below identifies this exception. Verification preserves causal
access to its query positions. Sparse verification changes attention and KV
history, so it does not guarantee full-attention outputs. Dense verification
follows the greedy speculative contract, subject to the numerical limitations
in [validation](#6-validation-and-reproducibility).

### Indexer formats and shared settings

| Setting | Protocol |
|---|---|
| Models | Qwen3-4B and Qwen3-8B with trained Highlight indexers |
| Hardware | One NVIDIA B200 per benchmark |
| Runtime | vLLM 0.27.1 V2 runner, MoLLE SpotAttention plugin, Torch 2.13, FlashAttention 4 |
| Generation | Greedy; model thinking mode off; explicit CoT prompting evaluated separately |
| BF16 indexer | BF16 auxiliary indexer keys and native scoring |
| MXFP4 indexer | MXFP4 auxiliary keys with block scales and native Blackwell quantized scoring |
| Backbone | BF16 weights and full KV cache for both indexer formats |
| Throughput lengths | 32,768 / 65,536 / 130,560 prompt tokens; static YaRN factors 2 / 3 / 4 |
| Native context window | 32,768 input plus output tokens |

Both formats use the same checkpoints, prompts, attention budgets and matched
throughput batches. Dense has no Highlight indexer: its saved measurement is
repeated in both format rows for reference. Saved Vegas/TriForce selector
measurements are also reused. RULER lengths denote haystack tokens; selector
and throughput lengths denote prompt tokens.

## 2. Selector latency and memory

### Measurement protocol

<!-- selector-repeatability:protocol:start -->
Batch 1; the first six existing prompts per model/context; layers 2, 18 and 35.
Each cell is the median of 18 capture medians. Each capture uses five warmups
and 20 CUDA-graph samples of 20 calls. Initialization uses one call per sample.
Loading, capture, transfers, compilation and warm-up are excluded. Captures
and timing samples are not independent process repetitions.

**Sum of stages** adds the score, rank/index and gather medians before rounding.
It excludes initialization and is not an independently measured end-to-end
latency. Highlight accesses selected pages without a KV gather; `n/a` gather
therefore contributes zero to the sum.

Three fresh process repetitions per model/context and indexer format are being
collected separately with the same inputs and settings. This report retains
the original single-process measurements. Three-run medians and ranges will
be added after the complete repetition grid passes audit.

<!-- selector-repeatability:protocol:end -->

Score includes acquisition and reduction. Vegas measures fused-attention
time minus matched dense-attention time, plus reduction; paired samples retain
negative measurement noise. Highlight uses native production kernels; Vegas
uses our FP32 FA4 port; TriForce uses the PyTorch retrieval reference.

**Extra KV** is per-layer, batch-1 auxiliary storage: Highlight indexer keys
(including MXFP4 scales), or gathered Vegas/TriForce KV. It excludes the shared
full cache and scratch. Vegas collection buffers and TriForce chunk means
each add about 8 / 16 / 31.88 MiB at the three lengths. These are identified
buffers, not peak-memory measurements. The indexer-format column applies only
to Highlight; all gathered backbone KV remains BF16.

### Matched token budgets

All selectors receive 7% of the context rounded up to a common 16-token
boundary. Highlight retains the last 64 tokens inside that budget. Timings
below are microseconds; storage is MiB. Paper-budget results and one-time
initialization costs are in the [appendix](#7-appendix).

<!-- selector-repeatability:matched:start -->
#### Qwen3-4B

| Context | Selector | Indexer | Tokens/head | Score (µs) | Rank/index (µs) | Gather (µs) | Sum of stages (µs) | Extra KV (MiB) |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 32,768 | Highlight | BF16 | 2,304 | 23.21 | 9.40 | n/a | 32.61 | 32.00 |
| 32,768 | Highlight | MXFP4 | 2,304 | 22.10 | 9.23 | n/a | 31.33 | 8.50 |
| 32,768 | Vegas | n/a | 2,304 | 391.10 | 63.55 | 54.35 | 508.99 | 9.00 |
| 32,768 | TriForce | n/a | 2,304 | 2.47 | 40.44 | 158.02 | 200.94 | 36.00 |
|  |  |  |  |  |  |  |  |  |
| 65,536 | Highlight | BF16 | 4,592 | 28.61 | 10.54 | n/a | 39.15 | 64.00 |
| 65,536 | Highlight | MXFP4 | 4,592 | 23.95 | 10.38 | n/a | 34.34 | 17.00 |
| 65,536 | Vegas | n/a | 4,592 | 759.97 | 119.48 | 88.78 | 968.22 | 17.94 |
| 65,536 | TriForce | n/a | 4,592 | 4.18 | 54.53 | 320.11 | 378.82 | 71.75 |
|  |  |  |  |  |  |  |  |  |
| 130,560 | Highlight | BF16 | 9,152 | 43.73 | 11.31 | n/a | 55.04 | 127.50 |
| 130,560 | Highlight | MXFP4 | 9,152 | 29.04 | 11.24 | n/a | 40.28 | 33.87 |
| 130,560 | Vegas | n/a | 9,152 | 1,470.56 | 127.92 | 169.57 | 1,768.05 | 35.75 |
| 130,560 | TriForce | n/a | 9,152 | 4.54 | 57.77 | 656.56 | 718.87 | 143.00 |

#### Qwen3-8B

| Context | Selector | Indexer | Tokens/head | Score (µs) | Rank/index (µs) | Gather (µs) | Sum of stages (µs) | Extra KV (MiB) |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 32,768 | Highlight | BF16 | 2,304 | 24.08 | 9.47 | n/a | 33.55 | 32.00 |
| 32,768 | Highlight | MXFP4 | 2,304 | 23.01 | 9.21 | n/a | 32.23 | 8.50 |
| 32,768 | Vegas | n/a | 2,304 | 391.75 | 63.59 | 54.24 | 509.58 | 9.00 |
| 32,768 | TriForce | n/a | 2,304 | 2.44 | 40.53 | 157.91 | 200.88 | 36.00 |
|  |  |  |  |  |  |  |  |  |
| 65,536 | Highlight | BF16 | 4,592 | 29.69 | 10.41 | n/a | 40.11 | 64.00 |
| 65,536 | Highlight | MXFP4 | 4,592 | 25.03 | 10.34 | n/a | 35.37 | 17.00 |
| 65,536 | Vegas | n/a | 4,592 | 739.77 | 113.95 | 88.36 | 942.09 | 17.94 |
| 65,536 | TriForce | n/a | 4,592 | 4.17 | 54.12 | 318.90 | 377.20 | 71.75 |
|  |  |  |  |  |  |  |  |  |
| 130,560 | Highlight | BF16 | 9,152 | 44.65 | 11.31 | n/a | 55.96 | 127.50 |
| 130,560 | Highlight | MXFP4 | 9,152 | 29.91 | 11.14 | n/a | 41.05 | 33.87 |
| 130,560 | Vegas | n/a | 9,152 | 1,468.24 | 129.01 | 169.59 | 1,766.83 | 35.75 |
| 130,560 | TriForce | n/a | 9,152 | 4.50 | 57.79 | 656.61 | 718.90 | 143.00 |

<!-- selector-repeatability:matched:end -->

### Interpretation and scope

<!-- selector-repeatability:comparison:start -->
MXFP4 lowers Highlight's summed stages by 3.9–26.8% relative to BF16 at matched
budgets, with the largest reduction at 130,560 tokens. Its indexer cache uses
26.56% of BF16 storage, including scales: 8.50 / 17.00 / 33.87 MiB rather than
32.00 / 64.00 / 127.50 MiB. This is a 73.44% reduction in indexer storage, not
in total model or backbone KV memory.

<!-- selector-repeatability:comparison:end -->

TriForce has the lowest score-stage time; ranking and gathering dominate its
measured selection event. Highlight has the lowest summed stages in these
implementations. Refresh frequency differs: Highlight scores once per draft
round, Vegas refreshes after verification, and the released TriForce policy
selects once at prefill then updates its tail. Tail-update latency is not
measured. These event costs do not establish per-token speedups, complete
system superiority, or selector latency at saturated batches.

Inputs use the main grid's checkpoints, prompts and YaRN settings with dense
FA4. We replay seven queries from six greedy continuation tokens. Vegas uses
both boundary queries; Highlight and TriForce use the last prefill query on
the same prompt KV. Fixed-length captures may include EOS. They measure
timing, not acceptance or task quality.

Selector-quality evidence comes from [Chapter 2's attention-map and retrieval
analysis](https://github.com/huzama/PhDThesis/blob/76271e73a045ff853f33f0f608305cccd8b072a3/thesis/chapters/02-distilled-selectors.tex#L190)
(`fig:distilled:attention-maps`, `fig:distilled:attention-retrieval`). It supports
the learned attention-approximation argument; these timing measurements do
not establish equal selection quality against Vegas/TriForce at 7%, or equal
quality between BF16 and MXFP4 indexers.

## 3. Decode throughput

<!-- eos-throughput:start -->
### EOS-stopped Multi-LexSum throughput

Status: `running`; 70/216 process runs audited.

Both models draw from the same 32 frozen cases, detailed-summary instruction, thinking off, and EOS or a 512-token cap. All prefills finish before timing; the window ends at the first request finish. The remaining requests drain outside timing. Three fresh processes per cell; median pooled tokens/s. All methods and indexer formats use the same capacity-checked batch within each model/context.

Dense is tokens/s; other cells are ratios to its unrounded median. A cell stays empty until all three repetitions pass the raw-output audit.

Every new process measures one full batch: the first B frozen cases, identical across methods and repetitions. Retained 32-batch results: Qwen3-4B at 32,768; Qwen3-4B at 65,536. Each speedup uses a dense reference with the same batch protocol. The main 8B and FFN tables share their identical one-batch dense controls.

#### Qwen3-4B

| Input tokens | Batch | Indexer | Dense (tok/s) | Direct (×) | DV (×) | SV50 (×) | SV40 (×) | SV25 (×) |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 32,768 | 26 | BF16 | 861.73 | 2.493× | 2.396× | 2.793× | 2.883× |  |
| 32,768 | 26 | MXFP4 | 861.73 | 3.048× | 2.501× | 2.930× | 3.118× | 3.345× |
|  |  |  |  |  |  |  |  |  |
| 65,536 | 13 | BF16 | 450.90 | 2.516× | 2.483× | 2.807× | 2.931× |  |
| 65,536 | 13 | MXFP4 | 450.90 | 3.070× | 2.537× | 2.954× | 3.194× | 3.401× |
|  |  |  |  |  |  |  |  |  |
| 130,560 |  | BF16 |  |  |  |  |  |  |
| 130,560 |  | MXFP4 |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |

#### Qwen3-8B

| Input tokens | Batch | Indexer | Dense (tok/s) | Direct (×) | DV (×) | SV50 (×) | SV40 (×) | SV25 (×) |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 32,768 |  | BF16 |  |  |  |  |  |  |
| 32,768 |  | MXFP4 |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |
| 65,536 |  | BF16 |  |  |  |  |  |  |
| 65,536 |  | MXFP4 |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |
| 130,560 |  | BF16 |  |  |  |  |  |  |
| 130,560 |  | MXFP4 |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |

#### 8B FFN comparison

Dense BF16 verification for every FFN draft variant. BF16 indexer keys; backbone KV remains BF16.

Priority: one full batch per fresh process, three processes per cell. Use the first B frozen cases identically across all five methods. Dense controls are shared with the main 8B comparison. All FFN comparisons finish before remaining main-method jobs start.

| Input tokens | Batch | Dense (tok/s) | BF16/full (×) | INT4/full (×) | BF16/Highlight (×) | INT4/Highlight (×) |
|---:|---:|---:|---:|---:|---:|---:|
| 32,768 |  |  |  |  |  |  |
| 65,536 |  |  |  |  |  |  |
| 130,560 |  |  |  |  |  |  |

Useful tokens per verification count only emitted tokens through EOS/cap. Draft acceptance excludes terminal rounds and rounds shortened by the output cap; their counts are retained in the audit. Native pre-clipping counters are diagnostic only.

<!-- eos-throughput:end -->

**Historical EOS limitation:** the tables below measure forced continuation to 512 tokens.
The audited 8B/32K dense and dense-verifier samples reach EOS at token 8;
those measurements do not establish natural-generation speedups. The
[EOS-stopped Multi-LexSum suite](../experiments.html#eos-stopped-workload-suite)
is now frozen and validated for dense 8B at 32K/64K/128K, batch 1.
Audited matched-batch results appear above; unfinished cells remain empty.

### Historical measurement protocol

Tokens/s at matched prompts and batch sizes, with 512 fixed output tokens.
Pool tokens/time within each repetition, then report the median of three
repetitions. Hold each request after its first token until all prefills finish;
time decode until the first request finishes. Exclude warm-up, prefill, first
tokens and the shrinking-batch tail. Async scheduling and prefix caching are
off; chunked prefill is on. Reserve KV for the full output and six speculative
lookahead slots; reject any KV preemption.

Batches were chosen near the BF16 sparse runtime's KV capacity, then kept
fixed for MXFP4. They are not retuned to MXFP4's larger capacity. For example,
4B/32,768 at batch 26 uses about 98% of BF16 sparse KV capacity versus 79% for
dense, including output reserve. A 0.9 GPU-memory setting alone does not imply
KV saturation. Saved baselines and new direct timing use different physical
B200 hosts; the results are not a same-device paired BF16/MXFP4 experiment.

### Historical forced-continuation results

Dense is reported in tokens/s. Every other method is shown as a speedup
over the same model/context Dense reference: method tokens/s divided by
Dense tokens/s, using unrounded medians. These are ratios of medians,
not medians of paired repetition ratios. The reference is reused for both
indexer formats. Absolute throughput is retained in the
[appendix](#absolute-decode-throughput).

#### Qwen3-4B

| Prompt tokens | Batch | Indexer | Dense (tokens/s) | Direct Highlight (× Dense) | Dense verifier (× Dense) | SV50 (× Dense) | SV40 (× Dense) |
|---:|---:|---|---:|---:|---:|---:|---:|
| 32,768 | 26 | BF16 | 864.3 | 2.461× | 2.719× | 3.071× | 3.271× |
| 32,768 | 26 | MXFP4 | 864.3 | 3.077× | 2.830× | 3.363× | 3.515× |
|  |  |  |  |  |  |  |  |
| 65,536 | 13 | BF16 | 448.8 | 2.474× | 2.740× | 3.111× | 3.319× |
| 65,536 | 13 | MXFP4 | 448.8 | 3.018× | 2.722× | 3.193× | 3.439× |
|  |  |  |  |  |  |  |  |
| 130,560 | 6 | BF16 | 263.2 | 2.144× | 2.511× | 2.717× | 2.882× |
| 130,560 | 6 | MXFP4 | 263.2 | 2.646× | 2.529× | 2.879× | 2.970× |

#### Qwen3-8B

| Prompt tokens | Batch | Indexer | Dense (tokens/s) | Direct Highlight (× Dense) | Dense verifier (× Dense) | SV50 (× Dense) | SV40 (× Dense) |
|---:|---:|---|---:|---:|---:|---:|---:|
| 32,768 | 25 | BF16 | 806.0 | 2.401× | 2.492× | 2.843× | 2.927× |
| 32,768 | 25 | MXFP4 | 806.0 | 2.844× | 2.551× | 2.969× | 3.099× |
|  |  |  |  |  |  |  |  |
| 65,536 | 12 | BF16 | 418.6 | 2.304× | 2.409× | 2.680× | 2.885× |
| 65,536 | 12 | MXFP4 | 418.6 | 2.765× | 2.478× | 2.823× | 3.029× |
|  |  |  |  |  |  |  |  |
| 130,560 | 6 | BF16 | 253.7 | 1.991× | 2.100× | 2.313× | 2.448× |
| 130,560 | 6 | MXFP4 | 253.7 | 2.364× | 2.165× | 2.407× | 2.508× |

## 4. Task quality

Scores below use separate workloads from throughput. Dense is reused in each
format pair. Equal aggregate scores can hide different errors; paired BF16
changes and intervals are available in the audited evidence. The MXFP4 tables
report aggregate scores without paired uncertainty estimates.

### RULER

Mean retrieval score (%), six needle tasks × 32 rows = 192 rows per cell.
Batch 1, EOS stopping, 1,024-token generation cap. Haystack lengths exclude
37–74 additional prompt tokens. These tasks test short retrieval, not general
accuracy preservation.

| Model | Haystack tokens | Indexer | Dense | Direct Highlight | Dense verifier | SV50 | SV40 |
|---|---:|---|---:|---:|---:|---:|---:|
| 4B | 65,536 | BF16 | 86.98 | 86.98 | 86.98 | 86.98 | 86.98 |
| 4B | 65,536 | MXFP4 | 86.98 | 87.50 | 86.98 | 86.98 | 86.98 |
|  |  |  |  |  |  |  |  |
| 4B | 129,024 | BF16 | 68.75 | 68.75 | 68.75 | 68.75 | 69.27 |
| 4B | 129,024 | MXFP4 | 68.75 | 69.27 | 68.75 | 69.27 | 69.27 |
|  |  |  |  |  |  |  |  |
| 8B | 65,536 | BF16 | 86.98 | 86.46 | 86.98 | 87.50 | 88.02 |
| 8B | 65,536 | MXFP4 | 86.98 | 87.50 | 86.98 | 88.02 | 88.02 |
|  |  |  |  |  |  |  |  |
| 8B | 129,024 | BF16 | 75.52 | 72.40 | 75.52 | 76.56 | 76.56 |
| 8B | 129,024 | MXFP4 | 75.52 | 72.40 | 75.52 | 76.04 | 76.56 |

The dense verifier matches the dense aggregate score in all four settings
with either indexer format. This does not establish identical generated
sequences. At BF16 4B/129,024, Direct Highlight and SV50 each lose one row and
recover one versus dense: net 0.00 pp, paired 95% interval [-1.56, 1.56] pp.
The BF16 dense verifier matches all 192 dense row scores in that setting.

### LongBench v2

Accuracy (%), 60 questions per model, batch 1, original prompts of
65,893–114,176 tokens. Answer-only uses a 16,384-token cap. CoT allows 1,024
reasoning tokens plus 128 answer tokens. Answer-only names the prompting
protocol; Direct Highlight names the generation method. One question changes
accuracy by 1.67 percentage points.

CoT's final answer is a separate, short request containing the question,
choices and generated reasoning, with the document omitted. The 7% rule also
applies to that request. Its budget can overflow the reserved recent window,
so these scores do not isolate long-context reasoning quality. See the
[training-free diagnosis](#training-free-diagnosis) before interpreting this gap.

| Model | Protocol | Indexer | Dense | Direct Highlight | Dense verifier | SV50 | SV40 |
|---|---|---|---:|---:|---:|---:|---:|
| 4B | Answer-only | BF16 | 10.00 | 8.33 | 10.00 | 8.33 | 8.33 |
| 4B | Answer-only | MXFP4 | 10.00 | 8.33 | 10.00 | 10.00 | 8.33 |
|  |  |  |  |  |  |  |  |
| 4B | CoT | BF16 | 16.67 | 5.00 | 20.00 | 20.00 | 16.67 |
| 4B | CoT | MXFP4 | 16.67 | 1.67 | 20.00 | 15.00 | 16.67 |
|  |  |  |  |  |  |  |  |
| 8B | Answer-only | BF16 | 31.67 | 31.67 | 31.67 | 31.67 | 31.67 |
| 8B | Answer-only | MXFP4 | 31.67 | 31.67 | 31.67 | 31.67 | 31.67 |
|  |  |  |  |  |  |  |  |
| 8B | CoT | BF16 | 18.33 | 5.00 | 23.33 | 31.67 | 23.33 |
| 8B | CoT | MXFP4 | 18.33 | 6.67 | 25.00 | 28.33 | 26.67 |

Dense accuracy is weak on this subset. In the BF16 runs, CoT reasoning reaches
the cap on 7–10/60 questions for 4B and 24–27/60 for 8B. MXFP4 changes are mixed:
4B CoT SV50 decreases from 20.00% to 15.00%, while 8B CoT SV40 increases from
23.33% to 26.67%. These small-sample differences do not establish a general
quality advantage or lossless indexer quantization.

## 5. Findings and limitations

### Throughput tradeoffs

Ranges span the six model/context settings and use unrounded measurements.
Speedups compare each method with Direct Highlight using the same indexer
format. The last column compares MXFP4 with BF16 for the same method.

| Method | BF16 vs Direct Highlight | MXFP4 vs Direct Highlight | MXFP4 vs BF16 throughput |
|---|---:|---:|---:|
| Direct Highlight | 1.000× | 1.000× | +18.5% to +25.0% |
| Dense verifier | 1.038–1.171× | 0.896–0.956× | -0.6% to +4.1% |
| SV50 | 1.162–1.267× | 1.018–1.093× | +2.7% to +9.5% |
| SV40 | 1.219–1.344× | 1.061–1.142× | +2.5% to +7.5% |

Direct Highlight gains most from MXFP4. The indexer format has little effect
on dense-verifier throughput, which with MXFP4 is 4.4–10.4% below Direct
Highlight across all six settings.
SV50 and SV40 remain faster, by 1.8–9.3% and 6.1–14.2%, respectively. These are
observed throughput differences, not significance claims. The relative gains
are consistent with reducing selection cost in a method that refreshes scores
every step; stage timing alone does not establish the cause of the full
throughput difference.

### Training-free diagnosis

The [diagnostic artifacts](../../outputs/runs/selector-diagnosis-current.json)
contain 36 completed captures: both models, the first six existing 32,768-token
prompts and layers 2/18/35. Each uses the last prefill query, dense FA4 and the
existing YaRN setting. Normalize native attention separately per query head,
then average over heads. Values below are means over 18 captures per model.

The native oracle maximizes retained attention mass using the same shared
16-token blocks, budget and 64-token recency rule. It is an analysis bound,
not an implementable low-cost selector or a generation-quality result.
Highlight uses the validated production BF16/MXFP4 kernels. The 7% budget
floors to 2,288 tokens; 25% uses 8,192.

| Model | Budget | Native oracle mass % | Highlight BF16 mass % | Highlight MXFP4 mass % | BF16 output cosine | MXFP4 output cosine |
|---|---:|---:|---:|---:|---:|---:|
| 4B | 7% | 91.98 | 90.42 | 90.31 | 0.9808 | 0.9803 |
| 4B | 25% | 96.39 | 94.63 | 94.55 | 0.9958 | 0.9957 |
| 8B | 7% | 88.43 | 85.84 | 85.80 | 0.9712 | 0.9710 |
| 8B | 25% | 94.54 | 91.88 | 91.87 | 0.9943 | 0.9945 |

At 7%, BF16 ranking loses 1.55 / 2.60 percentage points of native attention
relative to the constrained oracle; the budget itself excludes 8.02 / 11.57
points for 4B/8B. Quantization adds 0.11 / 0.04 points of loss on these captures.
Output cosine compares concatenated attention-head outputs before the output
projection. These measurements cover one query per prompt at three layers;
they do not establish agreement throughout generation or at 128K.

The saved CoT outputs reveal a separate short-prompt failure. All counts are
out of 60; parseable means the existing extractor finds the required answer.

| Model | Dense parseable | BF16 Direct parseable | MXFP4 Direct parseable | MXFP4 answer prompt tokens, min–max | Selected tokens, min–max | Current block omitted on first decode, BF16 / MXFP4 |
|---|---:|---:|---:|---:|---:|---:|
| 4B | 60 | 6 | 5 | 325–1,425 | 16–96 | 48 / 48 |
| 8B | 60 | 14 | 14 | 399–1,576 | 16–96 | 25 / 25 |

The last column is derived from reconstructed prompt lengths and the production
selection rule. A 64-token recent window spans four or five blocks. If the
fractional budget is smaller, equal forced scores retain the lowest block IDs
and omit the current block. For example, 835 cached positions with a 48-token
budget select blocks 48–50 from reserved blocks 48–52, dropping current block
52. The production CPU reference reproduces this case. This is a short-budget
policy defect, not evidence that MXFP4 alone causes the CoT failures.

The authorized [final-answer replay](../../outputs/runs/cot-answer-replay-current.json)
uses all 60 saved MXFP4 Direct Highlight reasoning outputs per model. Only
final-answer attention changes to dense. Checkpoint, BF16 weights/KV, static
YaRN factor 4, template, greedy decoding, 128-token cap and scoring are fixed.
Three saved dense answers per model replayed exactly before measurement.
This tests the short-stage policy without regenerating the document reasoning.
The original scores above remain the original protocol's measurements.

<!-- cot-answer-replay:results:start -->
| Model | Original correct / 60 | Dense final-stage correct / 60 | Original parseable / 60 | Dense final-stage parseable / 60 |
|---|---:|---:|---:|---:|
| 4B | 1 | 10 | 5 | 60 |
| 8B | 4 | 17 | 14 | 59 |
<!-- cot-answer-replay:results:end -->

Changing only final-answer attention raises accuracy from 1.67% to 16.67% for
4B and from 6.67% to 28.33% for 8B. The 4B replay recovers nine answers and
loses none; 8B recovers fourteen and loses one. This identifies the short-stage
attention policy as a major source of the original CoT loss. It does not
separate the effect of recency overflow from that of the small overall budget,
and does not validate a particular minimum budget or general short-prompt rule.
No retraining or long-context regeneration was needed. These are separate
controls, not replacements for the original fixed-7% results or a production
policy change.


### Boundaries of the evidence

- **Quality:** retrieval and a 60-question LongBench subset do not establish
  general quality preservation. Sparse verification changes the target
  computation; quantization can also change selection and generated outputs.
- **Speed and quality:** scored tasks have no matched timing. Their scores
  and the fixed-output throughput results are separate evidence, not measured
  points on a speed–quality curve.
- **Capacity:** throughput compares fixed batches selected for BF16 capacity.
  It does not measure the maximum throughput or batch capacity enabled by MXFP4.
- **Selector comparison:** batch-1 event timing uses different refresh policies
  and implementation paths. It does not measure complete Vegas/TriForce systems,
  equal-quality selection, or amortized cost per generated token.
- **Repeatability:** throughput has three repetitions; selector captures have
  no independent process repetitions. Cross-host baseline reuse and timing
  variation limit interpretation of small differences.

## 6. Validation and reproducibility

### Sanity checks

| Scope | Check | Finding |
|---|---|---|
| BF16 dense verifier | Matched replay at one 4B/128K round | Bitwise-equal hidden states and logits |
| BF16 dense verifier | Forced rejection at draft positions 1, 3 and 6 | Tokens, accepted KV and rollback correct; rejected KV did not affect continuation |
| BF16 replay state | Restore saved prefixes | All six 128-token prefixes preserved |
| MXFP4 native scoring | Five GPU kernel checks | Passed; covers query encoding, paged scores, verification queries and graph replay |
| Selector timing | Per-capture numerical and selection checks | Passed for BF16 and MXFP4 under both budget protocols |

In the BF16 output audit, dense-verifier outputs agree with dense through EOS
in five of six throughput cells. At 4B/130,560, 126/144 repeated appearances
agree. All 18 differences involve one prompt, starting at output token 70.
The original numerical mismatch remains unresolved: matched compiled replay
did not reproduce its tie flip. These checks do not establish identical dense
sequences and do not extend output-agreement claims to sparse verification or
MXFP4. MXFP4 kernel checks validate the quantized computation against its
reference, not equality with BF16 attention scores.

### Runtime configuration

| Method or variant | Benchmark flags |
|---|---|
| Dense | `--mode dense` |
| Direct Highlight | `--mode direct` |
| Dense verifier | `--mode highlight --draft-select round --spec-tokens 6 --verify-fraction 0` |
| SV50 / SV40 | Same speculative flags, with `--verify-fraction .5` / `.4` |
| Indexer format | `--indexer-cache-format bf16` or `mxfp4` for Highlight methods |

The plugin uses separate scored/reused draft graphs and reserves six
lookahead slots. The evidence records contain exact configurations, prompt
manifests, source hashes and individual measurements. Numerical tables above
use the audited aggregates; repeated baseline cells represent reused evidence.

| Evidence | Contents |
|---|---|
| [BF16 method results](../../outputs/worktrees/highlight-spec/outputs/evidence/methods.json) | Throughput repetitions; per-task scores; paired changes and intervals; 392 source hashes |
| [BF16 audit script](../../outputs/worktrees/highlight-spec/outputs/evidence/methods-audit.py) | Method-result aggregation and checks |
| [Selector comparisons](../../outputs/worktrees/highlight-spec/outputs/evidence/selectors.json) | BF16 Highlight, Vegas and TriForce stages; budgets; memory; 123 source hashes |
| [MXFP4 results](../../outputs/worktrees/highlight-spec/outputs/evidence/mxfp4.json) | Quantized method and selector measurements; three throughput repetitions; 282 source hashes |
| [Replay and rollback checks](../../outputs/worktrees/highlight-spec/outputs/evidence/sanity.json) | BF16 correctness probes and numerical limitations |
| [Output agreement](../../outputs/worktrees/highlight-spec/outputs/evidence/eos-agreement.json) | BF16 dense-verifier versus dense sequences |

## 7. Appendix

### Absolute decode throughput

All entries are tokens/s: the same three-repetition medians used in the main
speedup tables. Dense is the same reused reference in each format pair.

#### Qwen3-4B

| Prompt tokens | Batch | Indexer | Dense | Direct Highlight | Dense verifier | SV50 | SV40 |
|---:|---:|---|---:|---:|---:|---:|---:|
| 32,768 | 26 | BF16 | 864.3 | 2,126.8 | 2,350.3 | 2,654.0 | 2,826.8 |
| 32,768 | 26 | MXFP4 | 864.3 | 2,659.5 | 2,446.2 | 2,906.6 | 3,038.1 |
|  |  |  |  |  |  |  |  |
| 65,536 | 13 | BF16 | 448.8 | 1,110.2 | 1,229.6 | 1,395.9 | 1,489.5 |
| 65,536 | 13 | MXFP4 | 448.8 | 1,354.4 | 1,221.7 | 1,433.1 | 1,543.4 |
|  |  |  |  |  |  |  |  |
| 130,560 | 6 | BF16 | 263.2 | 564.3 | 660.9 | 715.2 | 758.7 |
| 130,560 | 6 | MXFP4 | 263.2 | 696.5 | 665.5 | 757.8 | 781.7 |

#### Qwen3-8B

| Prompt tokens | Batch | Indexer | Dense | Direct Highlight | Dense verifier | SV50 | SV40 |
|---:|---:|---|---:|---:|---:|---:|---:|
| 32,768 | 25 | BF16 | 806.0 | 1,934.9 | 2,008.2 | 2,291.5 | 2,359.2 |
| 32,768 | 25 | MXFP4 | 806.0 | 2,292.1 | 2,055.8 | 2,392.9 | 2,497.5 |
|  |  |  |  |  |  |  |  |
| 65,536 | 12 | BF16 | 418.6 | 964.4 | 1,008.3 | 1,121.8 | 1,207.8 |
| 65,536 | 12 | MXFP4 | 418.6 | 1,157.5 | 1,037.4 | 1,181.9 | 1,268.1 |
|  |  |  |  |  |  |  |  |
| 130,560 | 6 | BF16 | 253.7 | 505.0 | 532.7 | 586.7 | 621.0 |
| 130,560 | 6 | MXFP4 | 253.7 | 599.8 | 549.2 | 610.5 | 636.3 |

### Paper token budgets

Native budget rules are preserved: Highlight floors 7% to 16-token blocks;
Vegas takes ceil(7%); TriForce keeps 4,096 tokens in chunks of eight. These
budgets differ across selectors, so these rows complement the matched-budget
comparison. Timing and memory definitions are unchanged.

<!-- selector-repeatability:paper:start -->
#### Qwen3-4B

| Context | Selector | Indexer | Tokens/head | Score (µs) | Rank/index (µs) | Gather (µs) | Sum of stages (µs) | Extra KV (MiB) |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 32,768 | Highlight | BF16 | 2,288 | 23.31 | 9.45 | n/a | 32.76 | 32.00 |
| 32,768 | Highlight | MXFP4 | 2,288 | 22.01 | 9.26 | n/a | 31.27 | 8.50 |
| 32,768 | Vegas | n/a | 2,294 | 391.10 | 63.11 | 54.24 | 508.45 | 8.96 |
| 32,768 | TriForce | n/a | 4,096 | 2.47 | 40.74 | 281.70 | 324.91 | 64.00 |
|  |  |  |  |  |  |  |  |  |
| 65,536 | Highlight | BF16 | 4,576 | 28.56 | 10.43 | n/a | 38.99 | 64.00 |
| 65,536 | Highlight | MXFP4 | 4,576 | 23.95 | 10.38 | n/a | 34.33 | 17.00 |
| 65,536 | Vegas | n/a | 4,588 | 759.89 | 118.48 | 88.77 | 967.14 | 17.92 |
| 65,536 | TriForce | n/a | 4,096 | 4.18 | 54.14 | 284.05 | 342.38 | 64.00 |
|  |  |  |  |  |  |  |  |  |
| 130,560 | Highlight | BF16 | 9,136 | 43.29 | 11.25 | n/a | 54.54 | 127.50 |
| 130,560 | Highlight | MXFP4 | 9,136 | 29.10 | 11.24 | n/a | 40.34 | 33.87 |
| 130,560 | Vegas | n/a | 9,140 | 1,470.65 | 127.85 | 169.54 | 1,768.04 | 35.70 |
| 130,560 | TriForce | n/a | 4,096 | 4.54 | 57.03 | 285.70 | 347.28 | 64.00 |

#### Qwen3-8B

| Context | Selector | Indexer | Tokens/head | Score (µs) | Rank/index (µs) | Gather (µs) | Sum of stages (µs) | Extra KV (MiB) |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 32,768 | Highlight | BF16 | 2,288 | 23.50 | 9.43 | n/a | 32.94 | 32.00 |
| 32,768 | Highlight | MXFP4 | 2,288 | 23.01 | 9.25 | n/a | 32.26 | 8.50 |
| 32,768 | Vegas | n/a | 2,294 | 391.73 | 63.59 | 54.23 | 509.55 | 8.96 |
| 32,768 | TriForce | n/a | 4,096 | 2.45 | 40.78 | 280.86 | 324.09 | 64.00 |
|  |  |  |  |  |  |  |  |  |
| 65,536 | Highlight | BF16 | 4,576 | 29.06 | 10.41 | n/a | 39.47 | 64.00 |
| 65,536 | Highlight | MXFP4 | 4,576 | 25.29 | 10.41 | n/a | 35.70 | 17.00 |
| 65,536 | Vegas | n/a | 4,588 | 739.70 | 113.98 | 88.29 | 941.97 | 17.92 |
| 65,536 | TriForce | n/a | 4,096 | 4.17 | 53.93 | 283.75 | 341.85 | 64.00 |
|  |  |  |  |  |  |  |  |  |
| 130,560 | Highlight | BF16 | 9,136 | 44.07 | 11.24 | n/a | 55.32 | 127.50 |
| 130,560 | Highlight | MXFP4 | 9,136 | 29.89 | 11.25 | n/a | 41.14 | 33.87 |
| 130,560 | Vegas | n/a | 9,140 | 1,468.03 | 129.08 | 169.47 | 1,766.59 | 35.70 |
| 130,560 | TriForce | n/a | 4,096 | 4.51 | 57.26 | 285.52 | 347.28 | 64.00 |

<!-- selector-repeatability:paper:end -->

### One-time selector initialization

Milliseconds, excluded from the main stage sums. Highlight builds auxiliary
indexer keys; TriForce builds chunk-key means. Vegas has no separately measured
auxiliary-key initialization. Matched and paper protocols are retained as
separate observations even when rounded values coincide.

<!-- selector-repeatability:initialization:start -->
| Model | Context | Budget | Highlight BF16 (ms) | Highlight MXFP4 (ms) | TriForce (ms) |
|---|---:|---|---:|---:|---:|
| 4B | 32,768 | Matched | 1.189 | 1.237 | 0.029 |
| 4B | 32,768 | Paper | 1.190 | 1.237 | 0.029 |
|  |  |  |  |  |  |
| 4B | 65,536 | Matched | 2.358 | 2.459 | 0.051 |
| 4B | 65,536 | Paper | 2.357 | 2.460 | 0.051 |
|  |  |  |  |  |  |
| 4B | 130,560 | Matched | 4.702 | 4.895 | 0.088 |
| 4B | 130,560 | Paper | 4.703 | 4.895 | 0.088 |
|  |  |  |  |  |  |
| 8B | 32,768 | Matched | 1.522 | 1.566 | 0.029 |
| 8B | 32,768 | Paper | 1.524 | 1.565 | 0.030 |
|  |  |  |  |  |  |
| 8B | 65,536 | Matched | 3.013 | 3.120 | 0.051 |
| 8B | 65,536 | Paper | 3.013 | 3.123 | 0.051 |
|  |  |  |  |  |  |
| 8B | 130,560 | Matched | 6.000 | 6.210 | 0.090 |
| 8B | 130,560 | Paper | 5.995 | 6.207 | 0.090 |

<!-- selector-repeatability:initialization:end -->

<!-- selector-repeatability:spread:start -->
<!-- selector-repeatability:spread:end -->

## 8. Historical experiments

INT4 drafting, early Spot/Vegas pilots, old batch sweeps and earlier
YaRN/thinking settings are development evidence. The [earlier Vegas throughput
comparison](../../outputs/worktrees/highlight-spec/outputs/evidence/chapter.json) used different runtimes and is
excluded from the main comparison. Earlier acceptance probes have an unresolved
serving/probe ranking gap. Detailed legacy notes and artifacts remain in the
[development archive](../../outputs/archive/development.tar.gz).
