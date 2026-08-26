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
    KVH: tl.constexpr, G: tl.constexpr, GP: tl.constexpr, D: tl.constexpr,
    PAGE: tl.constexpr, TILE: tl.constexpr, USE_WEIGHT: tl.constexpr,
):
    b = tl.program_id(0)
    t0 = tl.program_id(1) * TILE
    valid = tl.load(valid_ptr + b)
    if t0 >= valid:
        return

    entry = tl.load(entry_ptr + b)
    w_first = (entry != 2).to(tl.float32)
    w_last = (entry != 1).to(tl.float32)
    count = (w_first + w_last) * (KVH * G)
    tok_first = tl.load(cu_ptr + b)
    tok_last = tl.load(cu_ptr + b + 1) - 1

    t = t0 + tl.arange(0, TILE)
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

    tl.store(out_ptr + b * stride_out + t,
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
) -> None:
    """Write the per-token metric of every request into ``output``.

    Static launch shape (batch rows, max_len tiles) so the call is CUDA-graph
    safe; programs past a request's valid length exit immediately.
    """
    assert k_cache.is_contiguous(), "paged K cache must be contiguous"
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
        KVH=kv_heads, G=group, GP=max(16, triton.next_power_of_2(group)),
        D=dim, PAGE=page, TILE=tile, USE_WEIGHT=use_weight,
    )
