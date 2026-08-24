# Fork overrider design

How our method drops into the vegas fork as a third attention overrider. Verify stays stock. Per-page key metadata is updated from the keys each forward writes. At the first draft step of each propose, every layer scores its pages against its own current query and builds a pruned block table; later draft steps reuse it. No modified kernel anywhere. Read with [final/todo.md](../final/todo.md).

## How their loop works, read from the code

| Fact | Where |
|---|---|
| propose() runs g eager single-token forwards of the target model; an `in_propose` flag routes every attention call to `_draft_attention` | `proposer.py`, `attn_overrider/__init__.py` |
| Draft KV is written into the real cache via the slot mapping; rejection rolls back by subtracting rejected counts from seq_lens | `proposer.py` |
| The layer index is tracked by counting kernel calls modulo num_layers; their own TODO calls this fragile | `attn_overrider/__init__.py` |
| Verify: every layer passes a scores buffer into their patched FA kernel, then varlen_reduce and varlen_topk write a per-layer token-level page table | `vegas.py` `_verify_attention` |
| Prefill goes through the same path with a prefill branch, so the first draft mask exists when prefill ends. No cold start, but even prefill needs the patched kernel | `vegas.py` reduce_entry logic |
| Draft: k/v are viewed at page size 1; the page table is top-k slots plus recent tokens; the budget grows by one per draft step | `vegas.py` `_draft_attention` |
| Page tables and budgets are rebuilt every propose in eager Python, so per-propose adaptivity is CUDA-graph-safe by construction | `vegas.py` `enter_propose` |
| FlashAttentionMetadata is the only allowed backend | `proposer.py` `allowed_attn_types` |
| streamingllm overrider proves block-granular selection with the stock kernel: pruned block table, no scores | `streamingllm.py` |

## Our overrider

| Piece | Choice | Why |
|---|---|---|
| Scoring metadata | Per physical KV page: min and max key per kv head, fp16, indexed by the same physical block ids as the cache, one tensor per layer | The block table already maps logical to physical; no per-request bookkeeping. Read cost 6.25% of KV per scoring event (fp16, page 16) |
| Metadata update | Inside `_verify_attention`, kernel call untouched: read the slots this step wrote (slot mapping from the forward context), update min/max of touched pages. Prefill updates the same way, chunk by chunk | O(new tokens) per layer; the cache is never re-read |
| Selection | First draft step of each propose, per layer: score residual pages with that layer's current query (q is an argument of the intercepted call), take top-k pages plus sink and recent reserves, write a block-table row; steps 2..g reuse it | Fresh query, per-layer signal, once per burst; mirrors their `_metadata_initialized` pattern |
| Draft attention | Stock paged FA over the pruned block table at native page size | Contiguous reads instead of their page-size-1 gather |
| Budget | Per layer, zero allowed, config vector; their global ratio remains the default | Depth asymmetry: oracle at 1% density preserves 0.719 shallow vs 0.855 deep total mass (32K) |
| Parity bypass | Ratio 1.0 returns the full block table unchanged and must reproduce dense output | The correctness gate |

## Open choices for the design session

| # | Choice | Options |
|---|---|---|
| 1 | Head aggregation | Sum of per-head bound scores (matches our measured setup) vs per-head top-k union |
| 2 | Reserves | Sink and recent sizes; streamingllm uses one sink block plus a recency window |
| 3 | Metadata dtype | fp16 (floor 6.25% of KV) vs INT4 (1.6%) |
| 4 | Metadata allocation | Overrider-owned tensors vs registered in the cache config so preallocation accounts for them |
| 5 | Rescoring cadence | Once per propose vs also mid-burst; cheap for us, impossible for them |
| 6 | Zero-budget layers | Reserves-only attention vs skipping the attention read entirely (sublayer skip, later) |

## What stays theirs

Proposer, scheduler wiring, rejection sampler, CUDA-graph dispatch, and both baselines (vegas, streamingllm). We add one file and one dispatch branch.
