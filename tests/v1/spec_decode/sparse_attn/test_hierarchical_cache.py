# SPDX-License-Identifier: Apache-2.0
"""Independent quantization and transaction checks for the shared packed cache."""

import pytest
import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.hierarchical import PagedHierarchicalKV, PageLayout
from vllm.v1.attention.ops.hierarchical.cache_kernels import load_keys, prefill, update

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def quantize(x, axis):
    """Torch oracle: round metadata before codes, residuals use signed [-8, 7]."""
    x = x.float()
    minimum = x.amin(axis, keepdim=True).bfloat16().float()
    scale = ((x.amax(axis, keepdim=True) - minimum) / 15).bfloat16().float()
    scale = torch.where(scale > 0, scale, 1)
    coarse = ((x - minimum) / scale).round().clamp(0, 15)
    residual = ((x - (coarse * scale + minimum)) * (16 / scale)).round().clamp(-8, 7)
    return coarse, residual, scale, minimum


def reference_values(x, prefix, bits, value=False):
    result = x.clone()
    if prefix:
        groups = x[:prefix].reshape(-1, 128, x.shape[1], 128)
        c, r, scale, minimum = quantize(groups, -1 if value else 1)
        decoded = c * scale + minimum
        if bits == 8:
            decoded += r * (scale / 16)
        result[:prefix] = decoded.flatten(0, 1).bfloat16()
    return result


def read(cache, slot, table, positions, prefix, bits, value=False):
    """Diagnostic decoder independent of the Triton addressing implementation."""
    h = cache.kv_heads
    result = cache.tail[int(value)][slot, positions % 263].clone()
    old = positions < prefix
    pos = positions[old]
    if not len(pos):
        return result
    physical = table[pos // 16].long()
    plane = cache.layout.plane_bytes
    planes = [
        cache.pages[:, i * plane : (i + 1) * plane].reshape(-1, 16, h, 64)
        for i in range(4)
    ]

    def unpack(x):
        return torch.stack((x & 15, x >> 4), -1).flatten(-2).float()

    coarse = unpack(planes[2 * int(value)][physical, pos % 16])
    residual = unpack(planes[2 * int(value) + 1][physical, pos % 16]) - 8
    if value:
        scale, minimum = [
            cache.metadata[physical, i, pos % 16, :, None].float() for i in (2, 3)
        ]
    else:
        channel = torch.arange(128, device=pos.device)
        kp = table[(pos[:, None] // 128) * 8 + channel[None, :] // 16].long()
        scale, minimum = [
            cache.metadata[kp, i, channel[None, :] % 16].transpose(1, 2).float()
            for i in (0, 1)
        ]
    decoded = coarse * scale + minimum
    if bits == 8:
        decoded += residual * (scale / 16)
    result[old] = decoded.bfloat16()
    return result


@triton.jit
def _read_keys(
    PAGES,
    META,
    TAIL,
    TABLE,
    POS,
    OUTPUT,
    N,
    H: tl.constexpr,
    STRIDE: tl.constexpr,
    SLOT,
    PREFIX,
    BITS: tl.constexpr,
):
    head = tl.program_id(1)
    row = tl.program_id(0) * 32 + tl.arange(0, 32)
    pos = tl.load(POS + row, row < N, -1)
    d = tl.arange(0, 128)
    key = load_keys(
        PAGES, META, TAIL, TABLE, pos, d, head, SLOT, PREFIX, row < N, H, STRIDE, BITS
    )
    tl.store(
        OUTPUT + (row[:, None] * H + head) * 128 + d[None, :], key, row[:, None] < N
    )


def assert_values(cache, slot, table, kv, length, views):
    # Unsorted, duplicated token IDs must retain token-level Vegas selection.
    positions = torch.cat(
        (
            torch.randperm(length, device="cuda"),
            torch.tensor([0, length - 1, 0], device="cuda"),
        )
    )
    for prefix in views:
        for bits in (4, 8):
            for value, source in enumerate(kv):
                expected = reference_values(source[:length], prefix, bits, bool(value))
                actual = read(cache, slot, table, positions, prefix, bits, bool(value))
                torch.testing.assert_close(actual, expected[positions], rtol=0, atol=0)
                if not value:
                    output = torch.empty_like(actual)
                    _read_keys[(triton.cdiv(len(positions), 32), cache.kv_heads)](
                        cache.pages,
                        cache.metadata,
                        cache.tail[0],
                        table,
                        positions,
                        output,
                        len(positions),
                        cache.kv_heads,
                        cache.pages.stride(0),
                        slot,
                        prefix,
                        bits,
                        enable_fp_fusion=False,
                    )
                    torch.testing.assert_close(output, actual, rtol=0, atol=0)


def test_layout_native_allocation_and_rebinding():
    layout = PageLayout(8)
    assert layout.page_bytes == 33792
    spec = layout.vllm_spec()
    assert spec.dtype == torch.uint8 and spec.page_size_bytes == layout.page_bytes
    assert layout.page_bytes / (2 * 16 * 8 * 128 * 2) == 0.515625
    cache = PagedHierarchicalKV.allocate(0, 8, 2)
    tail_ptrs = [x.data_ptr() for x in cache.tail]
    # Exercise nonzero storage offset and padded page stride without copying.
    raw = torch.full((3, layout.page_bytes + 128), 42, dtype=torch.uint8)
    pages = raw[:, 64 : 64 + layout.page_bytes]
    cache.bind_pages(pages)
    assert [x.data_ptr() for x in cache.tail] == tail_ptrs
    assert (
        cache.metadata.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
    )
    cache.metadata.zero_()
    assert (raw[:, :64] == 42).all() and (raw[:, -64:] == 42).all()
    assert (pages[:, : 4 * layout.plane_bytes] == 42).all()
    with pytest.raises(ValueError):
        PageLayout(8, head_dim=64)


def test_reference_quantizer_constant_and_residual_range():
    constant = torch.full((128, 2, 128), -3.5, dtype=torch.bfloat16)
    for axis in (0, -1):
        c, r, scale, minimum = quantize(constant, axis)
        assert not c.any() and not r.any() and (scale == 1).all()
        torch.testing.assert_close(
            c * scale + minimum, constant.float(), rtol=0, atol=0
        )
    ramp = torch.linspace(-2, 3, 128).bfloat16()
    c, r, scale, minimum = quantize(ramp, 0)
    assert c.min() == 0 and c.max() == 15 and r.min() >= -8 and r.max() <= 7
    assert (r != 0).any()
    assert (c * scale + minimum + r * (scale / 16) - ramp).abs().max() < (
        c * scale + minimum - ramp
    ).abs().max()


@CUDA
@pytest.mark.parametrize("length", [1, 7, 127, 128, 255, 256, 257, 511, 512])
def test_prefill_shuffled_pages_and_slot_reuse(length):
    torch.manual_seed(31)
    cache = PagedHierarchicalKV.allocate(96, 2, 2, "cuda")
    cache.pages.fill_(42)
    cache.state[1].fill_(789)
    table = torch.randperm(96, device="cuda", dtype=torch.int32)[:32]
    data = [
        torch.randn(512, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2)
    ]
    # Reset a genuinely used slot, even for a one-token replacement request.
    prefill(cache, 0, table, data[0], data[1])
    cache.error[0] = 1
    prefill(cache, 0, table, data[0][:length], data[1][:length])
    prefix = max(0, ((length - 128) // 128) * 128)
    assert cache.state[0].tolist() == [length, prefix, length, prefix]
    assert not cache.error.any() and (cache.state[1] == 789).all()
    assert_values(cache, 0, table, data, length, [prefix])


@CUDA
def test_fused_qkv_token_strides():
    torch.manual_seed(53)
    cache = PagedHierarchicalKV.allocate(32, 2, 1, "cuda")
    table = torch.randperm(32, device="cuda", dtype=torch.int32)
    fused = torch.randn(1024, 16, 128, device="cuda", dtype=torch.bfloat16)
    # Different token strides catch accidentally reusing K's stride for V.
    key, value = fused[:512, 8:10], fused[::2, 12:14]
    assert not key.is_contiguous() and not value.is_contiguous()
    prefill(cache, 0, table, key[:383], value[:383])
    assert_values(cache, 0, table, (key, value), 383, [128])
    slots = torch.zeros(1, device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, 7], device="cuda", dtype=torch.int32)
    for base in (383, 386):
        positions = torch.arange(base, base + 7, device="cuda")
        committed = torch.tensor([base], device="cuda", dtype=torch.int32)
        update(
            cache,
            key[base : base + 7],
            value[base : base + 7],
            positions,
            cu,
            slots,
            committed,
            table[None],
        )
        prefix, encoded = cache.state[0, [1, 3]].tolist()
        assert_values(
            cache, 0, table, (key, value), base + 7, sorted({prefix, encoded})
        )


@CUDA
def test_invalid_transaction_blocks_both_planes_and_stays_failed(monkeypatch):
    # Inspect the device error flag without poisoning this process's CUDA context.
    monkeypatch.setattr(torch, "_assert_async", lambda *args: None)
    cache = PagedHierarchicalKV.allocate(32, 8, 1, "cuda")
    cache.pages.zero_()
    for tail in cache.tail:
        tail.zero_()
    table = torch.arange(32, device="cuda", dtype=torch.int32)
    source = [
        torch.randn(383, 8, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2)
    ]
    prefill(cache, 0, table, *source)
    state = cache.state.clone()
    storage = [cache.pages, *cache.tail]
    snapshots = [tensor.clone() for tensor in storage]
    kv = [tensor[:7] for tensor in source]
    cu = torch.tensor([0, 7], device="cuda", dtype=torch.int32)
    slots = torch.zeros(1, device="cuda", dtype=torch.int32)
    committed = torch.tensor([383], device="cuda", dtype=torch.int32)
    # First violates the committed prefix; the otherwise valid retry must also
    # remain blocked. Neither K nor V may encode a group after either failure.
    for first in (382, 383):
        update(
            cache,
            *kv,
            torch.arange(first, first + 7, device="cuda"),
            cu,
            slots,
            committed,
            table[None],
        )
        assert cache.error.item() == 1
        assert torch.equal(cache.state, state)
        assert all(torch.equal(a, b) for a, b in zip(storage, snapshots))


@CUDA
@pytest.mark.parametrize("graph", [False, True])
def test_rejections_rollover_reordering_and_padding(graph):
    torch.manual_seed(43)
    h, requests, capacity = 2, 3, 1024
    cache = PagedHierarchicalKV.allocate(requests * 64, h, requests, "cuda")
    pages = torch.randperm(requests * 64, device="cuda", dtype=torch.int32)
    pages = pages.reshape(requests, 64)
    data = [
        [
            torch.randn(capacity, h, 128, device="cuda", dtype=torch.bfloat16)
            for _ in range(2)
        ]
        for _ in range(requests)
    ]
    lengths = [249, 254, 381]
    for slot, length in enumerate(lengths):
        prefill(cache, slot, pages[slot], *[x[:length] for x in data[slot]])
    slots = torch.tensor([0, 1, 2, -1], device="cuda", dtype=torch.int32)
    # Padding intentionally contains invalid table entries and token positions.
    table = torch.full((4, 64), -1, device="cuda", dtype=torch.int32)
    cu = torch.arange(5, device="cuda", dtype=torch.int32) * 7
    committed = torch.zeros(4, device="cuda", dtype=torch.int32)
    positions = torch.full((28,), -1, device="cuda", dtype=torch.int64)
    kv = [
        torch.zeros(28, h, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2)
    ]

    def inputs(order):
        slots[:3] = torch.tensor(order, device="cuda")
        for row, slot in enumerate(order):
            base = lengths[slot]
            committed[row] = base
            table[row] = pages[slot]
            positions[row * 7 : (row + 1) * 7] = torch.arange(
                base, base + 7, device="cuda"
            )
            for dst, source in zip(kv, data[slot]):
                dst[row * 7 : (row + 1) * 7] = source[base : base + 7]

    def verify():
        update(cache, *kv, positions, cu, slots, committed, table)

    inputs([0, 1, 2])
    if graph:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            verify()
        torch.cuda.current_stream().wait_stream(stream)
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            verify()
        # Warmup/capture must not prepare the first replay's compression boundary.
        for slot, length in enumerate(lengths):
            prefill(cache, slot, pages[slot], *[x[:length] for x in data[slot]])
    poison = [
        torch.full((4, h, 128), 1000, device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    ]
    draft_cu = torch.arange(5, device="cuda", dtype=torch.int32)
    for turn in range(24):
        order = [turn % 3, (turn + 1) % 3, (turn + 2) % 3]
        inputs(order)
        for step in range(6):
            update(
                cache,
                *poison,
                committed.long() + step,
                draft_cu,
                slots,
                committed,
                table,
            )
        captured.replay() if graph else verify()
        assert not cache.error.any()
        for row, slot in enumerate(order):
            base = lengths[slot]
            prefix = max(0, ((base - 128) // 128) * 128)
            encoded = max(prefix, ((base + 6 - 128) // 128) * 128)
            assert cache.state[slot].tolist() == [base + 7, prefix, base, encoded]
            assert_values(
                cache,
                slot,
                pages[slot],
                data[slot],
                base + 7,
                sorted({prefix, encoded}),
            )
            # Zero, partial and full acceptance; rejected poisoned drafts must
            # never appear in a later immutable compressed group.
            lengths[slot] += 1 + (0, 3, 6)[(turn + row) % 3]
