# SPDX-License-Identifier: Apache-2.0
"""Token-level KV selection on kernels that only page in groups of 16.

The draft pass of verification-guided decoding attends over an arbitrary set
of selected tokens. FA3 addresses them through a page table of single-token
pages; FA2 and other paged kernels require 16-token pages. This kernel copies
the selected tokens' K and V rows into page-aligned scratch so the stock kernel
attends over them unchanged. It copies positions [start, end) per request,
which lets the first draft step gather the whole selection and later steps
append only the newest token.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _gather_kernel(
    src_k, src_v, slots_ptr, start_ptr, end_ptr, dst_k, dst_v,
    stride_slots, stride_src, stride_dst_b, stride_dst_t,
    INNER: tl.constexpr, CHUNK: tl.constexpr, TOK: tl.constexpr,
):
    b = tl.program_id(0)
    t0 = tl.program_id(1) * TOK
    start = tl.load(start_ptr + b)
    end = tl.load(end_ptr + b)
    if t0 >= end:
        return
    if t0 + TOK <= start:
        return
    t = t0 + tl.arange(0, TOK)
    tmask = (t >= start) & (t < end)
    slot = tl.load(slots_ptr + b * stride_slots + t, mask=tmask, other=0)
    for c in tl.static_range(0, INNER, CHUNK):
        i = c + tl.arange(0, CHUNK)
        mask = tmask[:, None] & (i < INNER)[None, :]
        src = slot[:, None] * stride_src + i[None, :]
        dst = b * stride_dst_b + t[:, None] * stride_dst_t + i[None, :]
        tl.store(dst_k + dst, tl.load(src_k + src, mask=mask), mask=mask)
        tl.store(dst_v + dst, tl.load(src_v + src, mask=mask), mask=mask)


def gather_tokens(
    k_cache: torch.Tensor,  # [num_blocks, page, kv_heads, dim], contiguous
    v_cache: torch.Tensor,
    slots: torch.Tensor,    # [batch, width] int32 physical slot per position
    starts: torch.Tensor,   # [batch] int32, first position to copy
    ends: torch.Tensor,     # [batch] int32, one past the last position
    dst_k: torch.Tensor,    # [batch, width_padded, kv_heads, dim], contiguous
    dst_v: torch.Tensor,
) -> None:
    """Copy rows slots[b, p] for p in [starts[b], ends[b]) into dst[b, p]."""
    assert k_cache.is_contiguous() and dst_k.is_contiguous()
    batch, width = slots.shape[0], dst_k.shape[1]
    inner = dst_k.shape[2] * dst_k.shape[3]
    tok = 16
    grid = (batch, triton.cdiv(width, tok))
    _gather_kernel[grid](
        k_cache, v_cache, slots, starts, ends, dst_k, dst_v,
        slots.stride(0), k_cache.stride(1), dst_k.stride(0), dst_k.stride(1),
        INNER=inner, CHUNK=min(256, triton.next_power_of_2(inner)), TOK=tok,
    )
