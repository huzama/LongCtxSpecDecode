# SPDX-License-Identifier: Apache-2.0
"""Fused verification-guided scoring for kernels that expose no scores.

Vegas ranks KV tokens by the attention scores of two query rows per request,
the first and the last of the verify pass, which its patched FA3 kernel writes
out as a by-product. This kernel recomputes the same quantity from the paged K
cache for any attention backend: one read of K, both rows' dot products for
every head, optional rematerialization of the softmax weight from the kernel's
log-sum-exp, and the mean over rows and heads written directly into the
per-token metric that top-k consumes. The per-token score buffer of the kernel
path (batch x heads x 2 x max_len) never exists on this path.

Numerics mirror varlen_reduce: fp32 accumulation, bf16 output, entry 0 averages
both rows over all heads, entry 1 the first row only, entry 2 the last row only
(prefill). Positions at or past valid_len are left untouched; top-k never
reads them.
"""

import torch

from vllm.triton_utils import tl, triton

_LOG2E = tl.constexpr(1.4426950408889634)  # globals in kernels must be constexpr


@triton.jit
def _c2q_metric_kernel(
    q_ptr, k_ptr, bt_ptr, cu_ptr, valid_ptr, entry_ptr, lse_ptr, out_ptr,
    stride_qt, stride_qh, stride_ks, stride_kh, stride_bt, stride_lse_h,
    stride_out, scale2,
    logical_ptr, selected_lens_ptr, stride_logical,
    SELECTED: tl.constexpr,
    KVH: tl.constexpr, G: tl.constexpr, GP: tl.constexpr, D: tl.constexpr,
    PAGE: tl.constexpr, TILE: tl.constexpr, USE_WEIGHT: tl.constexpr,
):
    b = tl.program_id(0)
    t0 = tl.program_id(1) * TILE
    valid = tl.load(valid_ptr + b)
    count_k = tl.load(selected_lens_ptr + b) if SELECTED else valid
    if t0 >= count_k:
        return

    entry = tl.load(entry_ptr + b)
    w_first = (entry != 2).to(tl.float32)
    w_last = (entry != 1).to(tl.float32)
    count = (w_first + w_last) * (KVH * G)
    tok_first = tl.load(cu_ptr + b)
    tok_last = tl.load(cu_ptr + b + 1) - 1

    t = t0 + tl.arange(0, TILE)
    if SELECTED:
        logical = tl.load(logical_ptr + b * stride_logical + t,
                          t < count_k, other=0)
        tmask = (t < count_k) & (logical < valid)
    else:
        logical = t
        tmask = t < valid
    phys = tl.load(bt_ptr + b * stride_bt + t // PAGE, mask=tmask, other=0)
    slot = phys * PAGE + t % PAGE
    d = tl.arange(0, D)
    gp = tl.arange(0, GP)
    gmask = gp < G

    acc = tl.zeros([TILE], dtype=tl.float32)
    for h in tl.static_range(KVH):
        k_tile = tl.load(
            k_ptr + slot[:, None] * stride_ks + h * stride_kh + d[None, :],
            mask=tmask[:, None], other=0.0,
        )  # [TILE, D]
        heads = h * G + gp
        q_rows = q_ptr + heads[:, None] * stride_qh + d[None, :]

        q_first = tl.load(q_rows + tok_first * stride_qt,
                          mask=gmask[:, None], other=0.0)
        s = tl.dot(q_first, tl.trans(k_tile))  # [GP, TILE] fp32
        if USE_WEIGHT:
            lse = tl.load(lse_ptr + heads * stride_lse_h + tok_first,
                          mask=gmask, other=float("inf"))
            s = tl.exp2(s * scale2 - lse[:, None] * _LOG2E)
        s = tl.where(gmask[:, None], s, 0.0)
        acc += tl.sum(s, axis=0) * w_first

        q_last = tl.load(q_rows + tok_last * stride_qt,
                         mask=gmask[:, None], other=0.0)
        s = tl.dot(q_last, tl.trans(k_tile))
        if USE_WEIGHT:
            lse = tl.load(lse_ptr + heads * stride_lse_h + tok_last,
                          mask=gmask, other=float("inf"))
            s = tl.exp2(s * scale2 - lse[:, None] * _LOG2E)
        s = tl.where(gmask[:, None], s, 0.0)
        acc += tl.sum(s, axis=0) * w_last

    tl.store(out_ptr + b * stride_out + logical,
             (acc / count).to(out_ptr.dtype.element_ty), mask=tmask)


def c2q_metric(
    q: torch.Tensor,            # [total_q, heads, dim]
    k_cache: torch.Tensor,      # [num_blocks, page, kv_heads, dim], contiguous
    block_table: torch.Tensor,  # [batch, max_blocks] int32
    cu_seqlens_q: torch.Tensor, # [batch + 1] int32
    valid_lens: torch.Tensor,   # [batch] int32
    reduce_entry: torch.Tensor, # [batch] int32
    lse: torch.Tensor,          # [heads, total_q] fp32; ignored in logit mode
    softmax_scale: float,
    use_weight: bool,
    output: torch.Tensor,       # [batch, max_len] bf16, written in place
    logical_indices: torch.Tensor | None = None,
    selected_lens: torch.Tensor | None = None,
) -> None:
    """Write the per-token metric of every request into ``output``.

    Static launch shape (batch rows, max_len tiles) so the call is CUDA-graph
    safe; programs past the scored cache length exit immediately. Optional
    logical indices scatter selected-cache scores back to their original
    positions; the caller initializes omitted scores before this launch.
    """
    assert k_cache.is_contiguous(), "paged K cache must be contiguous"
    assert (logical_indices is None) == (selected_lens is None)
    batch = output.shape[0]
    heads, dim = q.shape[1], q.shape[2]
    page, kv_heads = k_cache.shape[1], k_cache.shape[2]
    group = heads // kv_heads
    tile = 64
    grid = (batch, triton.cdiv(output.shape[1], tile))
    _c2q_metric_kernel[grid](
        q, k_cache, block_table, cu_seqlens_q, valid_lens, reduce_entry,
        lse if use_weight else q, output,
        q.stride(0), q.stride(1), k_cache.stride(1), k_cache.stride(2),
        block_table.stride(0), lse.stride(0) if use_weight else 0,
        output.stride(0), softmax_scale * 1.4426950408889634,
        logical_indices if logical_indices is not None else block_table,
        selected_lens if selected_lens is not None else valid_lens,
        logical_indices.stride(0) if logical_indices is not None else 0,
        SELECTED=logical_indices is not None,
        KVH=kv_heads, G=group, GP=max(16, triton.next_power_of_2(group)),
        D=dim, PAGE=page, TILE=tile, USE_WEIGHT=use_weight,
    )


@triton.jit
def _c2q_lse_tiles(
    Q, K, BT, CU, SEQ, PART,
    sqt, sqh, skt, skh, sbt,
    H: tl.constexpr, G: tl.constexpr, GP: tl.constexpr, D: tl.constexpr,
    PAGE: tl.constexpr, TILES: tl.constexpr, TILE: tl.constexpr,
    SCALE: tl.constexpr,
):
    b, kh, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    first = tl.load(CU + b)
    last = tl.load(CU + b + 1) - 1
    seq = tl.load(SEQ + b)
    rows = tl.arange(0, GP)
    head = kh * G + rows % G
    query = tl.where(rows < G, first, last)
    valid_q = (rows < 2 * G) & (last >= first) & (seq > 0)
    limit = tl.where(rows < G, seq - (last - first), seq)
    t = tile * TILE + tl.arange(0, TILE)
    d = tl.arange(0, D)
    page = tl.load(BT + b * sbt + t // PAGE, t < seq, other=0)
    slot = page * PAGE + t % PAGE
    q = tl.load(Q + query[:, None] * sqt + head[:, None] * sqh + d[None, :],
                valid_q[:, None], other=0.)
    k = tl.load(K + slot[:, None] * skt + kh * skh + d[None, :],
                (t < seq)[:, None], other=0.)
    score = tl.dot(q, tl.trans(k)) * SCALE
    score = tl.where(valid_q[:, None] & (t[None, :] < limit[:, None]),
                     score, -float("inf"))
    maximum = tl.max(score, axis=1)
    safe_max = tl.where(maximum == -float("inf"), 0., maximum)
    partial = tl.log(tl.sum(tl.exp(score - safe_max[:, None]), axis=1)) + safe_max
    row = head + (rows // G) * H
    tl.store(PART + (b * 2 * H + row) * TILES + tile, partial, rows < 2 * G)


@triton.jit
def _c2q_lse_reduce(PART, CU, OUT, stride_h,
                    H: tl.constexpr, TILES: tl.constexpr, BLOCK: tl.constexpr):
    b, row = tl.program_id(0), tl.program_id(1)
    t = tl.arange(0, BLOCK)
    values = tl.load(PART + (b * 2 * H + row) * TILES + t,
                     t < TILES, other=-float("inf"))
    maximum = tl.max(values, axis=0)
    safe_max = tl.where(maximum == -float("inf"), 0., maximum)
    lse = tl.log(tl.sum(tl.exp(values - safe_max), axis=0)) + safe_max
    lse = tl.where(maximum == -float("inf"), 0., lse)
    first = tl.load(CU + b)
    last = tl.load(CU + b + 1) - 1
    query = tl.where(row < H, first, last)
    tl.store(OUT + (row % H) * stride_h + query, lse, last >= first)


def c2q_lse(kwargs: dict, partial: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    """Full-cache LSE for the first/last queries, without reading V.

    Sparse attention's LSE cannot normalize full-cache selection scores. Only
    these two query columns are consumed by c2q_metric; other columns are unused.
    Workspaces are persistent and both launches are captured by the verify graph.
    """
    q, k = kwargs["q"], kwargs["k"]
    batch = kwargs["seqused_k"].shape[0]
    heads, dim = q.shape[1:]
    group = heads // k.shape[2]
    tiles = partial.shape[-1]
    scale = kwargs.get("softmax_scale")
    if scale is None:
        scale = dim ** -0.5
    _c2q_lse_tiles[(batch, k.shape[2], tiles)](
        q, k, kwargs["block_table"], kwargs["cu_seqlens_q"], kwargs["seqused_k"],
        partial, q.stride(0), q.stride(1), k.stride(1), k.stride(2),
        kwargs["block_table"].stride(0), H=heads, G=group,
        GP=max(16, triton.next_power_of_2(2 * group)), D=dim, PAGE=k.shape[1],
        TILES=tiles, TILE=128, SCALE=scale,
    )
    _c2q_lse_reduce[(batch, 2 * heads)](
        partial, kwargs["cu_seqlens_q"], output, output.stride(0),
        H=heads, TILES=tiles, BLOCK=triton.next_power_of_2(tiles),
    )
    return output
