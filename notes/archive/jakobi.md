> Frozen before the unified HTML report. Read the [current study](../experiments.html#jacobi-study).

# Jacobi drafting

**Prompt copying + one Jacobi update + dense verification reached 1.87× AR
speed at 4K.** Carry-over stayed near AR speed. Converged Jacobi and Lookahead
were slower on A6000. The copying result does not yet establish that the
Jacobi update is useful.

## Main result: 4K on B200

Same 24 RULER prompts per model; one fresh process per method. All outputs
match dense AR token for token, with 100% RULER accuracy.
[Audited results and frozen inputs](../../outputs/runs/seeded-draft-current.json).

Each speculative round uses up to six guesses, one causal Jacobi update and
one dense verification pass. Only verified predecessors enter history KV.

- **Random:** uniform guesses from non-special vocabulary tokens.
- **Carry:** unused, aligned predictions from the previous verification pass.
- **Prompt copying:** match the longest 1–4-token suffix of accepted history
  in the original prompt; take the latest occurrence with a continuation and
  copy up to six following tokens. Carry and copying use random fallback.

<!-- seeded-draft:results:start -->
| Model | Initialization | Decode s | Speedup | Accepted/round | Emitted/round |
|---|---|---:|---:|---:|---:|
| 4B | AR | 7.831 | 1.000× |  |  |
| 4B | Random | 8.443 | 0.927× | 1.185 | 2.165 |
| 4B | Carry | 7.989 | 0.980× | 1.376 | 2.291 |
| 4B | Prompt copying | 4.197 | 1.866× | 3.663 | 4.418 |
| 8B | AR | 8.108 | 1.000× |  |  |
| 8B | Random | 8.804 | 0.921× | 1.206 | 2.176 |
| 8B | Carry | 8.048 | 1.007× | 1.445 | 2.379 |
| 8B | Prompt copying | 4.341 | 1.868× | 3.711 | 4.464 |
<!-- seeded-draft:results:end -->

**Accepted/round** counts emitted draft tokens. **Emitted/round** also counts
verifier corrections or bonuses. Each speculative round uses two forward
passes; prompt copying therefore emits about 2.2 tokens per forward.

The 1.87× result includes copying, Jacobi refinement and verification. RULER
favors copying answers from the prompt; this is one process measurement per
method, not evidence of general generation speedup. The missing control is
**prompt copying followed directly by verification**, without the Jacobi pass.
That comparison has not run.

## Protocol and status

| Setting | Value |
|---|---|
| Models | Official Qwen3-4B and Qwen3-8B; pinned revisions |
| Computation | BF16 weights/KV, dense causal attention, one shared history KV cache; no training or sparse selection |
| Generation | Batch 1, greedy, thinking off, EOS or 256-token cap; seed 42 + prompt index |
| Inputs | Six RULER needle tasks, four prompts each; frozen token IDs and row order |
| Lengths | 4K bin: 3,742–4,541 prompt tokens. Long run: 129,024-token haystack, 129,073–129,079 prompt tokens |
| Runtime | Transformers Qwen3 and PyTorch FlexAttention; dense prefill in 1,024-token chunks |
| RoPE | Native at 4K; static YaRN factor 4 for the long run; total positions capped at 131,072 |
| Timing | Synchronized decode wall time summed over prompts; includes CPU/index setup, draft, verification and cache work; excludes prefill |
| Speedup | Matched AR decode time / method decode time, on the same hardware type and runtime |
| Validation | CPU acceptance/cache checks and GPU stock-greedy checks passed; reported timing excludes compilation |

The A6000 campaigns cover AR, converged Jacobi, Lookahead and random
one-update drafting at 4K and 128K. Long-context cells fill after audit. Carry
and prompt copying have only the completed B200 4K comparison.

Blank cells are unmeasured or awaiting audit. A6000 and B200 timings have
separate AR baselines. Source hashes, exact revisions, request outputs,
prefill times and memory measurements remain in the linked run artifacts.

## Supporting A6000 results

<details>
<summary>Converged Jacobi and Lookahead: 4K results and pending 128K audit</summary>

Jacobi refines random six-token blocks until certified. Lookahead uses the
[reference](https://github.com/hao-ai-lab/LookaheadDecoding#use-lookahead-decoding-in-your-own-code)
settings `N=5, W=7, G=7`, with a per-request token pool and shared prefix KV.
[Campaign and audit](../../outputs/runs/jakobi-current.json).
`Tokens/call` excludes the first token from prefill; peak includes weights,
KV and temporary tensors.

| Model | Haystack | Method | Agreement | RULER % | Prefill s | Decode s | Speedup | Tokens/call | Query tokens | Peak GiB |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 4B | 4,096 | AR | 24/24 | 100.00 | 20.051 | 16.933 | 1.000× | 1.000 | 433 | 8.353 |
| 4B | 4,096 | Jacobi | 24/24 | 100.00 | 20.369 | 62.493 | 0.271× | 1.024 | 2538 | 8.353 |
| 4B | 4,096 | Lookahead | 24/24 | 100.00 | 20.429 | 54.062 | 0.313× | 1.206 | 10424 | 8.353 |
| 4B | 129,024 | AR | 24/24 | 87.50 | 10383.734 | 13.583 | 1.000× | 1.000 | 295 | 25.509 |
| 4B | 129,024 | Jacobi | 24/24 | 87.50 | 10336.488 | 921.195 | 0.015× | 1.003 | 1764 | 25.509 |
| 4B | 129,024 | Lookahead | 24/24 | 87.50 | 10372.301 | 796.035 | 0.017× | 1.161 | 6788 | 25.509 |
| 8B | 4,096 | AR | 24/24 | 100.00 | 27.061 | 16.374 | 1.000× | 1.000 | 433 | 16.084 |
| 8B | 4,096 | Jacobi | 24/24 | 100.00 | 26.725 | 66.917 | 0.245× | 1.024 | 2538 | 16.084 |
| 8B | 4,096 | Lookahead | 24/24 | 100.00 | 27.048 | 57.858 | 0.283× | 1.213 | 10256 | 16.084 |
| 8B | 129,024 | AR | 24/24 | 83.33 | 10469.850 | 16.672 | 1.000× | 1.000 | 298 | 33.240 |
| 8B | 129,024 | Jacobi | 24/24 | 83.33 | 10471.674 | 937.173 | 0.018× | 1.000 | 1788 | 33.240 |
| 8B | 129,024 | Lookahead | 24/24 | 83.33 | 10492.088 | 815.452 | 0.020× | 1.151 | 6916 | 33.240 |

The first-update diagnostic uses complete six-token blocks only; it is
separate from accepted tokens per speculative round.

| Model | Haystack | Eligible Jacobi blocks | First-update matching prefix / 6 | First-update full matches % |
|---|---:|---:|---:|---:|
| 4B | 4,096 | 56 | 1.143 | 0.00 |
| 4B | 129,024 | 33 | 1.000 | 0.00 |
| 8B | 4,096 | 56 | 1.214 | 0.00 |
| 8B | 129,024 | 34 | 1.029 | 0.00 |

</details>

<details>
<summary>Random one-update drafting with dense verification</summary>

Same two-forward algorithm as the B200 random row, measured on srv09 A6000.
AR timings come from matched separate jobs in the original campaign.
[Campaign and audit](../../outputs/runs/parallel-draft-current.json).

<!-- parallel-draft:results:start -->
| Model | Haystack | Agreement | RULER % | Decode s | Speedup | Accepted/round | Emitted/round | Calls | Peak GiB |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4B | 4,096 | 24/24 | 100.00 | 14.874 | 1.138× | 1.186 | 2.176 | 398 | 8.353 |
| 4B | 129,024 | 24/24 | 87.50 | 14.257 | 0.953× | 1.034 | 2.034 | 290 | 25.509 |
| 8B | 4,096 | 24/24 | 100.00 | 15.769 | 1.038× | 1.206 | 2.176 | 398 | 16.085 |
| 8B | 129,024 | 24/24 | 83.33 | 17.472 | 0.954× | 1.041 | 2.041 | 292 | 33.240 |
<!-- parallel-draft:results:end -->

</details>

Code: [Jacobi and Lookahead](../../benchmarks/jakobi.py),
[one-update verification](../../benchmarks/parallel_draft.py),
[initialization comparison](../../benchmarks/seeded_draft.py).
