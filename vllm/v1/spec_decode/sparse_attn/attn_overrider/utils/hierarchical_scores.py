# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vegas scores from the verifier's exact hierarchical-KV precision view."""

import triton
import triton.language as tl


@triton.jit
def _tile_keys(
    PAGES,
    META,
    KT,
    TABLE,
    start,
    head,
    slot,
    view,
    valid,
    H: tl.constexpr,
    STRIDE: tl.constexpr,
):
    # A 64-token tile stays inside one 128-token group and precision view.
    # Load each channel scale/minimum once, then broadcast across the tile.
    t, d = start + tl.arange(0, 64), tl.arange(0, 128)
    mask = (t < valid) & (slot >= 0)
    if start < view:
        physical = tl.load(TABLE + t // 16, mask, 0).to(tl.int64)
        off = physical[:, None] * STRIDE + ((t[:, None] % 16) * H + head) * 64
        off += tl.arange(0, 64)[None, :]
        packed = tl.load(PAGES + off, mask[:, None], 0).to(tl.int32)
        coarse = tl.interleave(packed & 15, packed >> 4).to(tl.float32)
        packed = tl.load(PAGES + off + 16 * H * 64, mask[:, None], 0).to(tl.int32)
        residual = tl.interleave(packed & 15, packed >> 4).to(tl.float32) - 8.0
        physical = tl.load(TABLE + (start // 128) * 8 + d // 16).to(tl.int64)
        moff = physical * (STRIDE // 2) + head * 16 + d % 16
        scale = tl.load(META + moff).to(tl.float32)
        minimum = tl.load(META + moff + 16 * H).to(tl.float32)
        keys = coarse * scale[None, :] + minimum[None, :]
        keys += residual * (scale[None, :] / 16.0)
    else:
        tail_off = (((slot * 263 + t[:, None] % 263) * H) + head) * 128 + d[None, :]
        keys = tl.load(KT + tail_off, mask[:, None], 0).to(tl.float32)
    return keys.to(tl.bfloat16)


@triton.jit
def _scores(
    Q,
    PREFILL_K,
    PAGES,
    META,
    KT,
    STATE,
    SLOTS,
    TABLE,
    CU,
    POS,
    PREFILL,
    VALID,
    ENTRY,
    LSE,
    OUT,
    sqt,
    sqh,
    skt,
    skh,
    sbt,
    slh,
    slt,
    sout,
    scale,
    STRIDE: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    GP: tl.constexpr,
    WEIGHT: tl.constexpr,
):
    b, tile = tl.program_id(0), tl.program_id(1)
    valid = tl.load(VALID + b)
    if tile * 64 >= valid:
        return
    t = tile * 64 + tl.arange(0, 64)
    d, g = tl.arange(0, 128), tl.arange(0, GP)
    first, last = tl.load(CU + b), tl.load(CU + b + 1) - 1
    entry, prefill = tl.load(ENTRY + b), tl.load(PREFILL + b)
    slot = tl.load(SLOTS + b)
    prefix = tl.load(STATE + slot * 4 + 1, slot >= 0, 0)
    first_view = tl.maximum(prefix, ((tl.load(POS + first) - 128) // 128) * 128)
    last_view = tl.maximum(prefix, ((tl.load(POS + last) - 128) // 128) * 128)
    # Only the two tiles in a transitioning group need separate query views.
    split = (
        (~prefill) & (entry == 0) & (tile * 64 >= first_view) & (tile * 64 < last_view)
    )
    selected_view = tl.where(entry == 2, last_view, first_view)
    token = tl.where(g < G, first, last)
    active = (g < 2 * G) & tl.where(g < G, entry != 2, entry != 1)
    acc = tl.zeros((64,), tl.float32)
    for kh in tl.range(H):
        heads = kh * G + g % G
        for view_index in range(1 + split.to(tl.int32)):
            selected = active & ((~split) | (g // G == view_index))
            if prefill:
                keys = tl.load(
                    PREFILL_K + (first + t[:, None]) * skt + kh * skh + d[None, :],
                    (t < valid)[:, None],
                    0,
                )
            else:
                view = tl.where(split & (view_index == 1), last_view, selected_view)
                keys = _tile_keys(
                    PAGES,
                    META,
                    KT,
                    TABLE + b * sbt,
                    tile * 64,
                    kh,
                    slot,
                    view,
                    valid,
                    H,
                    STRIDE,
                )
            query = tl.load(
                Q + token[:, None] * sqt + heads[:, None] * sqh + d[None, :],
                selected[:, None],
                0,
            )
            scores = tl.dot(query, tl.trans(keys))
            if WEIGHT:
                lse = tl.load(LSE + heads * slh + token * slt, selected, float("inf"))
                scores = tl.exp(scores * scale - lse[:, None])
            acc += tl.sum(tl.where(selected[:, None], scores, 0.0), axis=0)
    count = (1 + (entry == 0).to(tl.int32)) * H * G
    tl.store(OUT + b * sout + t, acc / count, t < valid)


def score_tokens(kwargs, valid, entry, lse, scale, use_weight, out):
    cache, q, k = kwargs["hierarchical_cache"], kwargs["q"], kwargs["k"]
    group = q.shape[1] // cache.kv_heads
    _scores[(out.shape[0], triton.cdiv(out.shape[1], 64))](
        q,
        k,
        cache.pages,
        cache.metadata,
        cache.tail[0],
        cache.state,
        kwargs["request_slots"],
        kwargs["block_table"],
        kwargs["cu_seqlens_q"],
        kwargs["positions"],
        kwargs["prefill_mask"],
        valid,
        entry,
        lse,
        out,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        kwargs["block_table"].stride(0),
        lse.stride(0),
        lse.stride(-1),
        out.stride(0),
        scale,
        cache.pages.stride(0),
        cache.kv_heads,
        group,
        max(16, triton.next_power_of_2(2 * group)),
        use_weight,
    )
