# SPDX-License-Identifier: Apache-2.0
"""GQA-packed verify attention for FA2.

FA2 packs the query heads of one KV head into the row dimension only when
``seqlen_q == 1``, so the multi-token verify pass reads the KV cache once per
query head: group-size times the bytes, at one block per query head and no
KV split. This module restores the packing for the uniform multi-query
decode shape by the cascade decomposition:

1. Prefix, non-causal: queries reshaped from ``[B*T, Hq, D]`` to
   ``[B*T*G, Hk, D]``, keys up to the last page boundary at or below
   ``seqused_k - T``. Every query attends every prefix key, so no mask is
   lost, and one block serves a whole KV head: the prefix is read once.
2. Tail, causal, original layout: the last pages only, at most
   ``T + page - 1`` keys. FA2's bottom-right causal alignment reproduces
   the mask exactly because the tail ends at ``seqused_k``.
3. ``merge_attn_states`` combines the two by their LSE. The merged LSE is
   the full-row LSE the score reduction consumes.

All arithmetic is static-shaped device work with no host reads, so the path
is safe under a FULL CUDA graph. Rows with an empty prefix take the tail
result alone, and padded rows (``seqused_k == 0``) read one garbage tail
key instead of zero so no NaN leaves the merge; their output is unused, as
on the unpacked path.
"""

import torch

from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

# Above this the verify is prefill-shaped and the plain call is fine.
MAX_PACKED_QUERIES = 16


def packed_verify_eligible(kwargs: dict, group: int) -> bool:
    """The uniform multi-query paged decode shape, with no feature FA2's
    packed layout cannot carry."""
    seqused_k = kwargs.get("seqused_k")
    queries = kwargs.get("max_seqlen_q", 0)
    window = kwargs.get("window_size")
    return (group > 1
            and seqused_k is not None
            and kwargs.get("block_table") is not None
            and kwargs.get("causal", False)
            and 1 < queries <= MAX_PACKED_QUERIES
            and kwargs["q"].shape[0] == seqused_k.shape[0] * queries
            and not kwargs.get("softcap", 0.0)
            and kwargs.get("alibi_slopes") is None
            and (window is None or tuple(window) == (-1, -1))
            and kwargs.get("s_aux") is None
            and kwargs.get("q_v") is None
            and kwargs.get("scores") is None)


def packed_verify_attention(fa_func, kwargs: dict):
    """Run the verify attention as packed prefix plus causal tail.

    Returns ``(out, lse)`` exactly like ``fa_func`` with
    ``return_softmax_lse=True``: ``out`` written into ``kwargs["out"]`` when
    given, ``lse`` of shape ``[Hq, B*T]`` over the full row.
    """
    q = kwargs["q"]
    key_cache, value_cache = kwargs["k"], kwargs["v"]
    block_table = kwargs["block_table"]
    seqused_k = kwargs["seqused_k"]
    queries = kwargs["max_seqlen_q"]
    batch = seqused_k.shape[0]
    heads, head_dim = q.shape[1], q.shape[2]
    page, kv_heads = key_cache.shape[1], key_cache.shape[2]
    group = heads // kv_heads
    scale = kwargs.get("softmax_scale")
    if scale is None:
        scale = head_dim**-0.5
    common = dict(
        k=key_cache, v=value_cache, softmax_scale=scale,
        return_softmax_lse=True,
        fa_version=kwargs.get("fa_version", 2),
    )

    # Split at the last page boundary at or below seqused_k - T, per row.
    prefix_pages = torch.div((seqused_k - queries).clamp_min(0), page,
                             rounding_mode="floor")
    prefix_lens = prefix_pages * page
    tail_lens = (seqused_k - prefix_lens).clamp_min(1)
    tail_width = (queries + page - 2) // page + 1
    page_idx = (prefix_pages.long().unsqueeze(1) +
                torch.arange(tail_width, device=q.device))
    page_idx.clamp_max_(block_table.shape[1] - 1)
    tail_table = block_table.gather(1, page_idx).clamp_min(0)

    # Prefix: group packed into rows, one KV read for all of a KV head.
    rows = queries * group
    q_packed = (q.view(batch, queries, kv_heads, group, head_dim)
                .transpose(2, 3)
                .reshape(batch * rows, kv_heads, head_dim))
    cu_packed = torch.arange(0, (batch + 1) * rows, rows,
                             device=q.device, dtype=torch.int32)
    # num_splits is inherited (the FULL-graph value, 1 on FA2). Letting the
    # FA2 heuristic split (num_splits=0) fills the SMs at batch 1 but moved
    # acceptance at theta 1 below the parity gate; parked until understood.
    prefix_out, prefix_lse = fa_func(
        q=q_packed, cu_seqlens_q=cu_packed, max_seqlen_q=rows,
        seqused_k=prefix_lens, max_seqlen_k=kwargs["max_seqlen_k"],
        causal=False, block_table=block_table,
        num_splits=kwargs.get("num_splits", 0), **common)
    prefix_out = (prefix_out.view(batch, queries, group, kv_heads, head_dim)
                  .transpose(2, 3)
                  .reshape(batch * queries, heads, head_dim))
    prefix_lse = (prefix_lse.view(kv_heads, batch, queries, group)
                  .permute(0, 3, 1, 2)
                  .reshape(heads, batch * queries))
    # Rows whose prefix is empty (short or padded): FA2's zero-key result is
    # undefined, and the tail already covers every key. Take the tail alone.
    empty = (prefix_lens == 0).repeat_interleave(queries)
    prefix_out.masked_fill_(empty[:, None, None], 0)
    prefix_lse.masked_fill_(empty[None, :], float("-inf"))

    # Tail: tiny, causal, bottom-right aligned to seqused_k.
    tail_out, tail_lse = fa_func(
        q=q, cu_seqlens_q=kwargs["cu_seqlens_q"], max_seqlen_q=queries,
        seqused_k=tail_lens, max_seqlen_k=tail_width * page,
        causal=True, block_table=tail_table, **common)

    out = kwargs.get("out")
    if out is None:
        out = torch.empty_like(q)
    lse = torch.empty_like(tail_lse)
    merge_attn_states(out, prefix_out, prefix_lse, tail_out, tail_lse, lse)
    return out, lse
