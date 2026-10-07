# Experiments

All results are pending. Batch timing and complete CUDA graphs need runtime validation.
Run only when instructed by the user.

## Setup

- LongBench v2; Qwen3-4B/8B; 32K/64K/128K prompt tokens; BF16 target; greedy sampling; seed 42.
- One A6000, same physical GPU, sequential runs, no competing workload. Weights load locally.
- Same 90% GPU memory allowance and CUDA graph setting. Async scheduling and prefix caching off.
- Compare dense, Vegas (6 draft tokens, shared target weights, 7% cache selection), and [our method](Method.md).

## Batch selection

1. For each method/model/context, increase batch from 1 until a confirmed capacity failure. Bmax is the largest successful batch. Reserve all 512 output tokens plus lookahead; require no preemption and complete every request.
2. Use Bshared = min(dense Bmax, Vegas Bmax, ours Bmax) for same-workload speedups.
3. Sweep 1 through Bmax for each method. Bbest is its fastest measured batch; use the smaller batch for exact ties. Compare these results separately to include each method's memory cost.

Unused Vegas capacity at Bshared is acceptable. Bbest need not equal Bmax.

| Model | Context | Dense Bmax | Vegas Bmax | Ours Bmax | Bshared |
|---|---|---|---|---|---|
| Qwen3-4B | 32K | | | | |
| Qwen3-4B | 64K | | | | |
| Qwen3-4B | 128K | | | | |
| Qwen3-8B | 32K | | | | |
| Qwen3-8B | 64K | | | | |
| Qwen3-8B | 128K | | | | |

## Full-batch throughput

- Hold completed prefills until all B requests are ready, then decode together. Every active request runs each step; no replacement arrivals.
- Ignore EOS; generate 512 tokens. Time from the last prefill's first token through the first request finishing. Exclude prefill tokens and the later partial-batch tail. Drain all requests to validate capacity.
- Throughput = emitted target tokens / timed seconds. Acceptance uses the same timed steps. Prefill, loading and warmup are excluded. These are synchronous decode measurements, not production-server throughput.
- Fix a pool of N = max(8, largest Bmax across methods) questions per model/context. Run N batches of B consecutive questions, wrapping around and shifting the start by one. Each question appears equally often.
- Confirm Bshared and each Bbest with 3 fresh-engine repetitions after batch selection. Warm each engine; rotate method order. Aggregate tokens/time per repetition, report median and range, and compute ratios from medians.

### Same batch size

| Model | Context | Bshared | Dense tok/s | Vegas tok/s | Ours tok/s | Ours / dense | Ours / Vegas | Our acceptance |
|---|---|---|---|---|---|---|---|---|
| Qwen3-4B | 32K | | | | | | | |
| Qwen3-4B | 64K | | | | | | | |
| Qwen3-4B | 128K | | | | | | | |
| Qwen3-8B | 32K | | | | | | | |
| Qwen3-8B | 64K | | | | | | | |
| Qwen3-8B | 128K | | | | | | | |

### Each method's fastest batch

| Model | Context | Dense batch | Dense tok/s | Vegas batch | Vegas tok/s | Our batch | Our tok/s |
|---|---|---|---|---|---|---|---|
| Qwen3-4B | 32K | | | | | | |
| Qwen3-4B | 64K | | | | | | |
| Qwen3-4B | 128K | | | | | | |
| Qwen3-8B | 32K | | | | | | |
| Qwen3-8B | 64K | | | | | | |
| Qwen3-8B | 128K | | | | | | |

## Single-request latency

Batch 1, eight questions, stop at EOS or 512 tokens. Measure first-to-last-token
time, excluding the first token from the count. Use 3 repetitions and the same
aggregation rule. Record seconds and tok/s; acceptance covers the whole generation.
Keep these results separate from fixed-length throughput.

| Model | Context | Dense tok/s | Vegas tok/s | Ours tok/s | Speedup vs dense | Speedup vs Vegas | Our acceptance |
|---|---|---|---|---|---|---|---|
| Qwen3-4B | 32K | | | | | | |
| Qwen3-4B | 64K | | | | | | |
| Qwen3-4B | 128K | | | | | | |
| Qwen3-8B | 32K | | | | | | |
| Qwen3-8B | 64K | | | | | | |
| Qwen3-8B | 128K | | | | | | |
