# TODO

Method: [Method.md](Method.md).

## Goals

1. **Our method:** faster than dense and Vegas on LongBench v2. The target checks every drafted token using the whole cache.
2. **Sparse verify:** its speedup, and its wrong tokens per 1000 against dense.

## Decisions

| Item | Choice |
|---|---|
| Dataset | LongBench v2 only |
| Models | Qwen3-4B, Qwen3-8B |
| Draft | 4-bit weights, top-p 0.85 |
| Rejected | Sparse weights |
| Dropped | Layer skip, early exit, attention skip, draft trees, copy drafting, dynamic draft length |

## Measured

| Run | Result | Note |
|---|---|---|
| Our method, 4B, LongBench v2 | 1.13x / 1.10x / 1.03x over dense at 32K / 64K / 128K | Used the old attention-mass target and estimated decode time by subtracting separate runs; rerun with the current settings and timing |
| Qwen3-0.6B, 4K, batch 2 | Draft reads the whole cache: 105/106 tokens accepted; sharing target weights and loading a separate copy gave identical outputs | 64-token LongBench check on RTX 3090. First output difference at token 43. Dense alone also changes its prediction when processing the same prefix one token at a time versus all at once. |
| Qwen3-4B, 8K, batch 2, W4 | 62.5% draft acceptance | 64-token LongBench check on RTX 3090. First output difference at token 41. Dense gives two tokens the same highest score when decoding one token at a time; processing the prefix all at once chooses the W4 run's token. |

Checks passed on RTX 3090 with FlashAttention-2: 46 tests without the
`slow_test` marker, plus 3 model tests with fixed synthetic inputs and identical output
tokens. FlashAttention-3 and the full set of A6000 benchmarks remain untested
after this cleanup.

## Next

1. **Our method at top-p 0.85:** dense, Vegas, ours. 4B and 8B. 32K, 64K, 128K.
2. **Sparse verify:** choose the verify's top-p. No new code.
3. **Sparse verify:** build it and measure.
4. **Split-KV verify:** a faster verification pass at batch 1 by splitting the KV calculation across GPU blocks.
   - Problem: its attention uses 8 of the GPU's 84 compute units: 31 ms, against 7.5 ms for a dense step.
   - Tried: FlashAttention's own split. It used more GPU compute units but cut acceptance to 0.946.
   - Candidate: vLLM's Triton split-KV decode kernel (`triton_decode_attention.py`). Its row tile of 16 cannot hold 4B's 28 rows per KV head, so wrap it with a tile of 32.
   - Required result: matches the current kernel and reaches at least 0.98 acceptance with the whole cache selected on Qwen3-0.6B.
