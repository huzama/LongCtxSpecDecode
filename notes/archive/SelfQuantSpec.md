> Frozen before the unified HTML report. Read the [current study](../experiments.html#ffn-study).

# SelfQuantSpec: FFN quantization

**Batch-1 result:** all ten serving runs and numerical checks passed. Pooled INT4
decode throughput is 3.0% above the BF16 full-attention draft and 4.7% above the
BF16 Highlight draft. Every speculative variant remains slower than dense AR
at 32K, batch 1. The sample is mostly after EOS and does not establish drafting
quality.

## Method and protocol

Only draft gate/up/down weights change from BF16 to INT4. Draft activations,
attention, verifier weights and the shared backbone KV stay BF16. INT4 adds
packed FFN weights, not a second backbone KV cache. Highlight uses BF16
indexer keys, a 7% budget and the last 64 tokens inside that budget. Scores
refresh once per six-token draft round. Verification uses full attention.

Agreement and batch-1 timing: Qwen3-8B; six existing 32,768-token prompts;
greedy; thinking off;
YaRN factor 2; six drafts; 128 output tokens with EOS retained. Token IDs,
model tensors, runtime and code are frozen. QuantSpec's [released code][quantspec]
informs the FFN-only weight scope; its hierarchical KV design is outside
this experiment. MXFP4 indexer quantization is a separate component.

## EOS-stopped workload suite

**Finalized and validated for dense Qwen3-8B, batch 1.**
[Multi-LexSum](https://huggingface.co/datasets/allenai/multi_lexsum)
`v20230518`, validation split. Freeze the first ten cases in dataset order
with a short reference and enough source tokens for the largest input.
Reuse these cases at all three lengths, taking nested document prefixes.
Each input includes 87 instruction/template tokens. No padding or artificial
repetition; references stay outside the prompt. Selection precedes generation;
retain every case regardless of its output length.

Request a detailed **500–700-word summary** covering background, parties,
claims, proceedings and documented outcomes. Thinking off; greedy; seed 42;
BF16 dense attention and KV; vLLM 0.27.1; FA4. Stop at either configured EOS
token, with **512 maximum outputs and no minimum**. Static YaRN factors are
2 / 3 / 4. The largest input reserves 512 tokens within the 131,072-token
window. Each process uses one B200; physical GPUs 1 and 2 were used.

| Input tokens | Cases reaching 512 | EOS stops | Pilot decode tok/s |
|---:|---:|---:|---:|
| 32,768 | 10/10 | 0 | 185.54 |
| 65,536 | 10/10 | 0 | 169.30 |
| 130,560 | 10/10 | 0 | 137.10 |

All 30 outputs contain substantive summary text; inspection found no obvious
repetition loops. No EOS tokens were emitted, so there are no post-EOS tokens.
These are capped summary prefixes. Complete-summary quality and factual
accuracy are not established. The suite validates the generation workload;
speculative acceptance and matched-capacity throughput remain unmeasured here.

Rates pool 5,110 tokens after the first output token of each request per
context. Loading and prefill are excluded. One fresh process per context;
no workload warm-up, repetitions or KV saturation. These are exploratory
dense rates, not method speedups.

Prompt selection used a separate paired 32K pilot: the 200–250-word request
produced eight EOS stops and two capped outputs; the detailed request reached
512 in all ten cases. The final context suite uses a common pool long enough
for 128K, selected before generation.

Fifteen CPU tests passed. The complete suite audit verifies prompt/code
hashes, case order, nested source prefixes, shared settings, raw output
counts and EOS stopping. `highlight-spec` branch: `benchmarks/multilexsum.py`
provides `prepare-suite`, per-context `run` and `report`; tests are in
`tests/test_multilexsum.py`. Evidence: `outputs/runs/multilexsum-suite-*/`
contains the frozen suite, per-context plans/inputs/code, raw outputs,
timings, audits and readable summaries. Initial prompt checks remain under
`outputs/runs/multilexsum-pilot-*/`.

## EOS-stopped throughput campaign

**FFN comparisons now take priority on `b200-1`, idle GPUs 2/3 only.**
Every remaining timing run uses one full batch per fresh process and three
processes per cell, for both FFN and main comparisons. Inputs remain frozen.
Completed evidence is preserved. The queue revision passed 51 CPU tests and
reuses the four passed GPU checks for the unchanged measurement implementation.
Live status and audited results appear in
[experiments.html](../experiments.html#throughput).
The ten-case dense suite above is complete; it does not validate the new
32-case, batched, speculative grid.

Use both models at 32,768 / 65,536 / 130,560 input tokens. Main methods:
Dense, Direct Highlight, dense verification, SV50 and SV40, with BF16/MXFP4
indexers; add MXFP4 SV25 as an ablation. For 8B, compare BF16/INT4 draft FFNs
with full/Highlight attention and dense verification. All four FFN variants
use BF16 indexers. Run all 45 FFN/control processes first: five methods,
three contexts and three process repetitions. The main 8B comparison shares
these identical one-batch Dense AR controls, removing nine duplicate runs.
Other pending main-method jobs follow only after these finish.

Freeze the first 32 eligible validation cases before generation. Keep the
detailed prompt, EOS stopping and 512-token cap. Each new timing process
measures the first B cases once, identically across methods and repetitions.
The 60 completed main-method processes at 4B/32K and 4B/64K retain their
32 sequential cyclic batches. Every speedup uses a dense denominator measured
with the same batch protocol. No remaining run repeats those 32 batches.
The common batch is the smaller
of the previous matched batch and the measured BF16 SV50 / INT4 Highlight
capacity. Initialization, warm-up, prefill, first tokens and the shrinking
tail are excluded. Drafting, verification, sampling and scheduler overhead
remain inside the decode measurement. Draft CUDA graphs cover one step each.
No requests wait and no KV preemption occurs during timed decode. Async
scheduling and prefix caching are off. The common batch is limited by the
smallest compared KV capacity; other methods can have unused capacity. Saved
capacity records give each method's reserved fraction. This is a matched-batch
comparison, not an independent maximum-capacity sweep for every method.

Useful tokens per verification count only emitted tokens, including the
terminal round. Draft acceptance counts nonterminal rounds with room for all
six proposals plus the correction/bonus token. Exclude terminal or
cap-shortened rounds from acceptance and report their count; native counters
before EOS clipping remain diagnostic only.

`highlight-spec` branch: `benchmarks/eos_throughput.py` prepares inputs,
freezes source, runs GPU preflight, audits results and renders the bounded
report section in [experiments.html](../experiments.html#throughput).
`benchmarks/queue.py` requires the matching preflight proof and 60 idle
seconds before using physical GPU 2 or 3. A failed run or audit stops new
launches and preserves evidence. The local report-sync command updates only
that section. No new speedup is published before three audited repetitions.

## Batch-1 results

Agreement uses dense reference prefixes after 0, 16, 32, 48, 64, 80, 96 and
112 outputs: 48 states and 288 prediction positions per variant.

| Method | Draft attention | Agreement % | Accepted / 6 | Decode tok/s | Speedup vs dense AR |
|---|---|---:|---:|---:|---:|
| Dense AR | Full | n/a | n/a | 189.79 | 1.000× |
| BF16 draft | Full | 100.00 | 6.000 | 120.22 | 0.633× |
| INT4 draft | Full | 99.31 | 5.854 | 123.80 | 0.652× |
| BF16 draft | Highlight 7% | 99.65 | 5.979 | 101.84 | 0.537× |
| INT4 draft | Highlight 7% | 98.96 | 5.833 | 106.66 | 0.562× |

**Acceptance limitation:** all six continuations reach EOS at output token 8.
Thus 42/48 states are after EOS. On the six states before EOS, every variant
matches 36/36 predictions and accepts all six drafts. This small, short-answer
sample cannot distinguish useful drafting quality between variants.

Agreement counts teacher-forced next-token matches against BF16 dense.
Accepted / 6 counts the matching prefix of six freely generated drafts,
averaged over states; it excludes correction and bonus tokens. Every variant
starts with the same dense prefix KV and creates a temporary draft tail.
Six-position BF16 verification checks free drafts; the tail is then discarded.
These agreement columns come from the eager HF diagnostic. Serving uses
fused vLLM kernels and records its own acceptance separately.

| Serving draft | Accepted / proposed % | Accepted / round |
|---|---:|---:|
| BF16, full | 100.00 | 6.000 |
| INT4, full | 95.41 | 5.708 |
| BF16, Highlight | 95.13 | 5.658 |
| INT4, Highlight | 93.75 | 5.513 |

Serving acceptance pools the actual timed trajectories across both processes.
It differs from the fixed-prefix diagnostic above. Each INT4 run stores
2.63 GiB of additional packed FFN weights, scales and zero points.

### Serving timing

Same model, prompts, batch and token limits. Dense AR plus the four draft
variants above; two fresh processes per variant, swapping physical GPUs 1
and 2 on `b200-1`. Each worker waits for 60 seconds idle and takes a GPU lock.
This measures batch-1 decode, not throughput at KV saturation.

- Native vLLM/FA4 attention; INT4 uses Marlin, fused gate/up projection and
  fused SiLU multiplication. The target model remains unchanged.
- Standard vLLM execution: one CUDA graph per draft step, with verification
  separate. Exact graph/eager draft agreement gates each speculative run.
  No whole-draft graph optimization is included in this comparison.
- Time starts after prefill emits the first token and ends at completion.
  Count the remaining 127 emitted tokens per prompt, including verifier
  corrections and bonuses. Include scheduling, launches and control overhead.
  Exclude loading, prefill, compilation and warm-up; asynchronous scheduling
  and prefix caching are disabled.
- Pool emitted tokens / decode seconds across both processes and all prompts.
  Speedup = that rate / the pooled dense AR rate. Save each process's raw
  counts, token outputs, acceptance, graph validation and extra FFN weight bytes.

The earlier eager agreement harness supplies no timing numbers. The timing
workload deliberately continues after EOS; its speed describes that fixed
workload, not answer-quality performance.

## Matched large-batch throughput

**Status:** stopped at the user's request; all evidence retained. Completed
outputs checked reach EOS at token 8, so the forced 512-token workload does
not establish natural-generation acceptance or speedup. The protocol below
records the stopped experiment, pending an EOS-aware replacement.

Same five methods,
Qwen3-8B/32K, full BF16 verification and BF16 indexer. INT4 adds FFN weight
storage, so its capacity limits the shared batch. The INT4 Highlight probe
has 51,228 usable blocks; each request reserves 2,081 blocks for the prompt,
512 outputs and six lookahead positions. Batch 25 needs 52,025 blocks and
does not fit. Batch 24 needs 49,944 blocks, or 97.5% of the probed capacity.
Every method uses batch 24; record its actual reserved KV fraction.

Use all 31 existing throughput prompts in 31 cyclic batches per process.
Each prompt appears equally often. Two fresh processes per method, swapping
GPUs 1 and 2; pool emitted tokens/time. Hold completed prefills until the
whole batch is ready, then time until the first request finishes. Exclude
first tokens and the shrinking-batch tail. Async scheduling and prefix
caching are off; KV preemption invalidates the run. Validate graph/eager
draft agreement on the full active batch during warm-up. Keep one CUDA graph
per draft step. Recompute dense AR at the same batch and on the same GPUs.

| Method | Draft attention | Reserved KV % | Decode tok/s | Speedup vs dense AR |
|---|---|---:|---:|---:|
| Dense AR | Full | | | |
| BF16 draft | Full | | | |
| INT4 draft | Full | | | |
| BF16 draft | Highlight 7% | | | |
| INT4 draft | Highlight 7% | | | |

## Validation and evidence

All 399 Highlight backbone tensors equal the original Qwen3-8B tensors;
all 147 unquantized tensors in the INT4 checkpoint also match. Eight CPU tests,
216 packed-FFN checks, all dense controls and prefix-restoration checks passed.
Counts were independently recomputed from saved predictions.

The serving adapter passed 41 CPU regression tests. Native fused INT4 FFNs
passed all 36 layer checks against the audited implementation, with relative
L2 below 0.000741 and exact graph/eager outputs. Six-token graph/eager agreement is
checked separately before each speculative timing process.

All eight speculative timing processes passed six-token graph/eager agreement
and recorded per-step graph replays. Each variant differs from dense AR at
one of 768 emitted positions per process: prompt slot 0, output token 120,
after EOS. Outputs before EOS match; complete emitted sequences are not
identical. Pooled throughput and speedups were independently recomputed from
the raw counts and elapsed times.

INT4/BF16 FFN relative L2 ranges from 0.0218 to 0.1392 across the 216 layer/input
checks. Packed/dequantized INT4 relative L2 stays below 0.001085. Vector error
alone does not establish token acceptance or speed.

`highlight-spec` branch: `benchmarks/ffn_agreement.py` and
`benchmarks/ffn_timing.py`. Evidence is under `outputs/runs/ffn-agreement-*/`
and `outputs/runs/ffn-timing-*/`: frozen plans/code, raw predictions, numerical
checks, engine logs and results. INT4 checkpoint:
`RedHatAI/Qwen3-8B-quantized.w4a16`, revision
`32527053243a382bc56e941c964fc516a5014d39`.

## Historical pilot

Older Vegas fork; 8B/32K; batch 1; two prompts and two processes with GPUs
swapped; 128 outputs; six drafts; 7% draft attention; dense BF16 verification.
Greedy, thinking off, YaRN 2 and shared BF16 KV. These are not current-runtime
speed measurements.

| Draft FFN | Acceptance % | Decode tok/s | vs BF16 draft |
|---|---:|---:|---:|
| BF16 | 95.55 | 187.46 | 1.000× |
| INT4 full FFN | 89.08 | 200.42 | 1.069× |
| INT4 gate/up only | 84.17 | 180.55 | 0.963× |
| INT4 down only | 93.46 | 197.64 | 1.054× |

Acceptance pools accepted/proposed tokens; speed pools emitted tokens/time.
[Raw archive](../../outputs/archive/development.tar.gz):
`outputs/runs/ffn-int4-pilot-*/` and `ffn-projections-*/`.
Earlier protocol: commit `3754fd1b6`.

[quantspec]: https://github.com/SqueezeAILab/QuantSpec/blob/661366739d89c14269c6204dddcb164727e8be7b/Engine/model_quantspec.py
