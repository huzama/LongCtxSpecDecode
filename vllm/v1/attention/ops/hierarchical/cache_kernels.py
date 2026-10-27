# SPDX-License-Identifier: Apache-2.0
"""Graph-safe transactions for six drafts and a shared 4+4-bit KV cache."""

import torch

from vllm.triton_utils import tl, triton
from vllm.triton_utils import tldevice as libdevice


@triton.jit
def _quantize(x, VALUE: tl.constexpr):
    axis: tl.constexpr = 1 if VALUE else 0
    minimum = tl.min(x, axis).to(tl.bfloat16).to(tl.float32)
    scale = ((tl.max(x, axis) - minimum) / 15).to(tl.bfloat16).to(tl.float32)
    scale = tl.where(scale > 0, scale, 1.0)
    if VALUE:
        lo, step = minimum[:, None], scale[:, None]
    else:
        lo, step = minimum[None, :], scale[None, :]
    coarse = tl.minimum(
        15.0, tl.maximum(0.0, libdevice.nearbyint(tl.div_rn(x - lo, step)))
    )
    residual = (
        tl.minimum(
            7.0,
            tl.maximum(
                -8.0,
                libdevice.nearbyint((x - (coarse * step + lo)) * tl.div_rn(16.0, step)),
            ),
        )
        + 8
    )
    pair = tl.broadcast_to((tl.arange(0, 64) * 2)[None, :], (128, 64))
    c = tl.gather(coarse.to(tl.uint8), pair, 1) | (
        tl.gather(coarse.to(tl.uint8), pair + 1, 1) << 4
    )
    r = tl.gather(residual.to(tl.uint8), pair, 1) | (
        tl.gather(residual.to(tl.uint8), pair + 1, 1) << 4
    )
    return c, r, scale, minimum


@triton.jit
def _store_group(
    PAGES,
    META,
    TABLE,
    x,
    group,
    head,
    H: tl.constexpr,
    STRIDE: tl.constexpr,
    VALUE: tl.constexpr,
):
    c, r, scale, minimum = _quantize(x, VALUE)
    t, d = tl.arange(0, 128), tl.arange(0, 64)
    physical = tl.load(TABLE + group * 8 + t // 16).to(tl.int64)
    plane: tl.constexpr = 16 * H * 64
    off = physical[:, None] * STRIDE + ((t[:, None] % 16) * H + head) * 64 + d[None, :]
    tl.store(PAGES + off + (2 if VALUE else 0) * plane, c)
    tl.store(PAGES + off + (3 if VALUE else 1) * plane, r)
    # t indexes channels for K metadata, tokens for V metadata.
    moff = physical * (STRIDE // 2) + (2 if VALUE else 0) * 16 * H + head * 16 + t % 16
    tl.store(META + moff, scale)
    tl.store(META + moff + 16 * H, minimum)


@triton.jit
def _prefill_groups(
    K,
    V,
    PAGES,
    META,
    TABLE,
    H: tl.constexpr,
    STRIDE: tl.constexpr,
    K_STRIDE: tl.constexpr,
    V_STRIDE: tl.constexpr,
    VALUE: tl.constexpr,
):
    group, head = tl.program_id(0), tl.program_id(1)
    t, d = tl.arange(0, 128), tl.arange(0, 128)
    token_stride: tl.constexpr = V_STRIDE if VALUE else K_STRIDE
    src = (group * 128 + t[:, None]) * token_stride + head * 128 + d[None, :]
    x = tl.load((V if VALUE else K) + src).to(tl.float32)
    _store_group(PAGES, META, TABLE, x, group, head, H, STRIDE, VALUE)


@triton.jit
def _prefill_recent(
    K,
    V,
    KT,
    VT,
    STATE,
    ERROR,
    N: tl.constexpr,
    PREFIX: tl.constexpr,
    SLOT: tl.constexpr,
    H: tl.constexpr,
    K_STRIDE: tl.constexpr,
    V_STRIDE: tl.constexpr,
):
    head = tl.program_id(0)
    t, d = tl.arange(0, 256), tl.arange(0, 128)
    p = PREFIX + t
    ksrc = p[:, None] * K_STRIDE + head * 128 + d[None, :]
    vsrc = p[:, None] * V_STRIDE + head * 128 + d[None, :]
    dst = (((SLOT * 263 + p[:, None] % 263) * H) + head) * 128 + d[None, :]
    tl.store(KT + dst, tl.load(K + ksrc, p[:, None] < N, 0), p[:, None] < N)
    tl.store(VT + dst, tl.load(V + vsrc, p[:, None] < N, 0), p[:, None] < N)
    if head == 0:
        tl.store(STATE + SLOT * 4, N)
        tl.store(STATE + SLOT * 4 + 1, PREFIX)
        tl.store(STATE + SLOT * 4 + 2, N)
        tl.store(STATE + SLOT * 4 + 3, PREFIX)
        tl.store(ERROR + SLOT, 0)


@triton.jit
def _prepare(
    KT,
    VT,
    PAGES,
    META,
    STATE,
    ERROR,
    QP,
    CU,
    SLOTS,
    COMMITTED,
    TABLE,
    H: tl.constexpr,
    STRIDE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
):
    req, head = tl.program_id(0), tl.program_id(1)
    value = tl.program_id(2) == 1
    slot = tl.load(SLOTS + req)
    start, end = tl.load(CU + req), tl.load(CU + req + 1)
    if slot >= 0 and end > start:
        base = tl.load(COMMITTED + req)
        first = tl.load(QP + start).to(tl.int32)
        length = tl.load(QP + end - 1).to(tl.int32) + 1
        previous_length = tl.load(STATE + slot * 4)
        previous_prefix = tl.load(STATE + slot * 4 + 1)
        previous_commit = tl.load(STATE + slot * 4 + 2)
        encoded = tl.load(STATE + slot * 4 + 3)
        prefix = tl.maximum(0, ((base - 128) // 128) * 128)
        needed = tl.maximum(prefix, ((length - 1 - 128) // 128) * 128)
        bad = (
            (tl.load(ERROR + slot) != 0)
            | (base < previous_commit)
            | (base > previous_length)
            | (first < base)
            | (first > previous_length)
            | (length > base + 7)
            | (end - start > 7)
            | (length - first != end - start)
            | (prefix < previous_prefix)
            | (needed > encoded + 128)
            | (needed > base)
            | (length - prefix > 263)
        )
        if head == 0 and not value:
            tl.store(ERROR + slot, bad.to(tl.int32))
        if not bad and needed > encoded:
            t, d = tl.arange(0, 128), tl.arange(0, 128)
            p = encoded + t
            src = (((slot * 263 + p[:, None] % 263) * H) + head) * 128 + d[None, :]
            if value:
                x = tl.load(VT + src).to(tl.float32)
                _store_group(
                    PAGES,
                    META,
                    TABLE + req * TABLE_STRIDE,
                    x,
                    encoded // 128,
                    head,
                    H,
                    STRIDE,
                    True,
                )
            else:
                x = tl.load(KT + src).to(tl.float32)
                _store_group(
                    PAGES,
                    META,
                    TABLE + req * TABLE_STRIDE,
                    x,
                    encoded // 128,
                    head,
                    H,
                    STRIDE,
                    False,
                )


@triton.jit
def _stage(
    K,
    V,
    KT,
    VT,
    STATE,
    ERROR,
    QP,
    CU,
    SLOTS,
    COMMITTED,
    H: tl.constexpr,
    K_STRIDE: tl.constexpr,
    V_STRIDE: tl.constexpr,
):
    req, head = tl.program_id(0), tl.program_id(1)
    slot = tl.load(SLOTS + req)
    start, end = tl.load(CU + req), tl.load(CU + req + 1)
    if slot >= 0 and end > start:  # noqa: SIM102 (guard the error load for padding)
        if tl.load(ERROR + slot) == 0:
            t, d = tl.arange(0, 8), tl.arange(0, 128)
            p = tl.load(QP + start + t, start + t < end, 0).to(tl.int32)
            ksrc = (start + t[:, None]) * K_STRIDE + head * 128 + d[None, :]
            vsrc = (start + t[:, None]) * V_STRIDE + head * 128 + d[None, :]
            dst = (((slot * 263 + p[:, None] % 263) * H) + head) * 128 + d[None, :]
            tl.store(
                KT + dst,
                tl.load(K + ksrc, start + t[:, None] < end, 0),
                start + t[:, None] < end,
            )
            tl.store(
                VT + dst,
                tl.load(V + vsrc, start + t[:, None] < end, 0),
                start + t[:, None] < end,
            )
            if head == 0:
                length = tl.load(QP + end - 1).to(tl.int32) + 1
                base = tl.load(COMMITTED + req)
                prefix = tl.maximum(0, ((base - 128) // 128) * 128)
                needed = tl.maximum(prefix, ((length - 1 - 128) // 128) * 128)
                encoded = tl.load(STATE + slot * 4 + 3)
                tl.store(STATE + slot * 4, length)
                tl.store(STATE + slot * 4 + 1, prefix)
                tl.store(STATE + slot * 4 + 2, base)
                tl.store(STATE + slot * 4 + 3, tl.maximum(encoded, needed))


@triton.jit
def load_keys(
    PAGES,
    META,
    KT,
    TABLE,
    pos,
    channels,
    head,
    slot,
    view_prefix,
    valid,
    H: tl.constexpr,
    STRIDE: tl.constexpr,
    BITS: tl.constexpr,
):
    """Load one KV head at arbitrary logical token positions and precision view."""
    old = valid & (pos >= 0) & (pos < view_prefix) & (slot >= 0)
    recent = valid & (pos >= view_prefix) & (slot >= 0)
    physical = tl.load(TABLE + pos // 16, old, 0).to(tl.int64)
    code = physical[:, None] * STRIDE + ((pos[:, None] % 16) * H + head) * 64
    code += channels[None, :] // 2
    packed = tl.load(PAGES + code, old[:, None], 0).to(tl.int32)
    shift = (channels[None, :] % 2) * 4
    coarse = ((packed >> shift) & 15).to(tl.float32)
    kpage = tl.load(
        TABLE + (pos[:, None] // 128) * 8 + channels[None, :] // 16, old[:, None], 0
    ).to(tl.int64)
    moff = kpage * (STRIDE // 2) + head * 16 + channels[None, :] % 16
    scale = tl.load(META + moff, old[:, None], 0).to(tl.float32)
    minimum = tl.load(META + moff + 16 * H, old[:, None], 0).to(tl.float32)
    key = coarse * scale + minimum
    if BITS == 8:
        packed = tl.load(PAGES + code + 16 * H * 64, old[:, None], 0).to(tl.int32)
        residual = ((packed >> shift) & 15).to(tl.float32) - 8.0
        key += residual * (scale / 16.0)
    tail = (((slot * 263 + pos[:, None] % 263) * H) + head) * 128 + channels[None, :]
    key += tl.load(KT + tail, recent[:, None], 0).to(tl.float32)
    return key.to(tl.bfloat16)


def _kv_contract(cache, key, value):
    if (
        cache.device.type != "cuda"
        or key.ndim != 3
        or key.shape != value.shape
        or key.shape[1:] != (cache.kv_heads, 128)
        or key.dtype != torch.bfloat16
        or value.dtype != key.dtype
        or key.device != cache.device
        or value.device != cache.device
        or key.stride()[-2:] != (128, 1)
        or value.stride()[-2:] != (128, 1)
    ):
        raise ValueError(
            "hierarchical updates require CUDA BF16 K/V with contiguous heads"
        )


def prefill(cache, slot, table, key, value):
    """Initialize or reuse a request slot, including prompts shorter than seven."""
    _kv_contract(cache, key, value)
    if not 0 <= slot < cache.max_requests or key.shape[0] < 1:
        raise ValueError("invalid prefill slot or length")
    if (
        table.ndim != 1
        or table.dtype not in (torch.int32, torch.int64)
        or not table.is_contiguous()
        or table.device != cache.device
        or table.numel() * 16 < key.shape[0]
    ):
        raise ValueError(
            "prefill requires a contiguous device page table with capacity"
        )
    n, h = key.shape[0], cache.kv_heads
    prefix = max(0, ((n - 128) // 128) * 128)
    if prefix:
        for value_plane in (False, True):
            _prefill_groups[(prefix // 128, h)](
                key,
                value,
                cache.pages,
                cache.metadata,
                table,
                h,
                cache.pages.stride(0),
                key.stride(0),
                value.stride(0),
                value_plane,
                num_warps=8,
                enable_fp_fusion=False,
            )
    _prefill_recent[(h,)](
        key,
        value,
        *cache.tail,
        cache.state,
        cache.error,
        n,
        prefix,
        slot,
        h,
        key.stride(0),
        value.stride(0),
        num_warps=8,
    )


def update(cache, key, value, positions, cu, slots, committed, table):
    """Commit the accepted prefix and stage this call's KV without encoding drafts.

    `committed` stays fixed across six draft steps. It is the anchor position
    during verification. Padding/overflow rows use slot=-1 and perform no writes.
    """
    _kv_contract(cache, key, value)
    n = len(slots)
    if (
        slots.ndim != 1
        or positions.shape != (len(key),)
        or cu.shape != (n + 1,)
        or committed.shape != (n,)
        or table.ndim != 2
        or table.shape[0] != n
    ):
        raise ValueError("incompatible hierarchical update metadata")
    for x in (positions, cu, slots, committed, table):
        if (
            x.device != cache.device
            or x.dtype not in (torch.int32, torch.int64)
            or x.stride(-1) != 1
        ):
            raise ValueError(
                "metadata must be integer device tensors with contiguous rows"
            )
    if not n:
        return
    _prepare[(n, cache.kv_heads, 2)](
        *cache.tail,
        cache.pages,
        cache.metadata,
        cache.state,
        cache.error,
        positions,
        cu,
        slots,
        committed,
        table,
        cache.kv_heads,
        cache.pages.stride(0),
        table.stride(0),
        num_warps=8,
        enable_fp_fusion=False,
    )
    _stage[(n, cache.kv_heads)](
        key,
        value,
        *cache.tail,
        cache.state,
        cache.error,
        positions,
        cu,
        slots,
        committed,
        cache.kv_heads,
        key.stride(0),
        value.stride(0),
        num_warps=4,
    )
    torch._assert_async(
        torch.all(cache.error == 0), "hierarchical KV transaction failed"
    )
