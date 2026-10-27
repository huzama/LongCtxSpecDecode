# SPDX-License-Identifier: Apache-2.0
"""Packed native FA4 against an independent quantization/attention oracle."""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.ops.hierarchical.fa4 import NativeAttentionWorkspace, attention


def make_cache():
    torch.manual_seed(53)
    heads, requests, length = 2, 2, 520
    plane = 16 * heads * 64
    page_bytes = 4 * plane + 4 * 16 * heads * 2
    pages = torch.zeros((80, page_bytes), device="cuda", dtype=torch.uint8)
    meta = (
        pages[:, 4 * plane :]
        .view(torch.bfloat16)
        .view(80, 4, heads, 16)
        .transpose(2, 3)
    )
    planes = [
        pages.as_strided(
            (80, 16, heads, 64), (page_bytes, heads * 64, 64, 1), i * plane
        )
        for i in range(4)
    ]
    table = torch.randperm(80, device="cuda").view(2, 40).int()
    source = [
        torch.randn((requests, length, heads, 128), device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    ]
    tail = [
        torch.zeros((requests, 263, heads, 128), device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    ]
    coarse, full = [x.clone() for x in source], [x.clone() for x in source]
    state = torch.tensor(
        [[516, 256, 509, 384], [390, 256, 386, 256]], device="cuda", dtype=torch.int32
    )
    for b, (n, prefix, _, encoded) in enumerate(state.tolist()):
        p = torch.arange(prefix, n, device="cuda")
        for kind in range(2):
            tail[kind][b, p % 263] = source[kind][b, p]
            for group in range(encoded // 128):
                values = source[kind][b, group * 128 : (group + 1) * 128].float()
                axis = 0 if kind == 0 else 2
                minimum = values.amin(axis).bfloat16().float()
                scale = ((values.amax(axis) - minimum) / 15).bfloat16().float()
                scale = torch.where(scale > 0, scale, 1.0)
                step, low = (
                    (scale[None], minimum[None])
                    if kind == 0
                    else (scale[..., None], minimum[..., None])
                )
                c = ((values - low) / step).round().clamp(0, 15)
                r = ((values - (c * step + low)) * (16 / step)).round().clamp(-8, 7)
                coarse[kind][b, group * 128 : (group + 1) * 128] = (
                    c * step + low
                ).bfloat16()
                full[kind][b, group * 128 : (group + 1) * 128] = (
                    c * step + low + r * (step / 16)
                ).bfloat16()
                physical = table[b, group * 8 : (group + 1) * 8].long()
                for code, index in ((c, kind * 2), (r + 8, kind * 2 + 1)):
                    code = code.byte()
                    packed = code[..., ::2] | (code[..., 1::2] << 4)
                    planes[index][physical] = packed.view(8, 16, heads, 64)
                if kind == 0:
                    meta[physical, 0] = (
                        scale.transpose(0, 1).reshape(8, 16, heads).bfloat16()
                    )
                    meta[physical, 1] = (
                        minimum.transpose(0, 1).reshape(8, 16, heads).bfloat16()
                    )
                else:
                    meta[physical, 2] = scale.reshape(8, 16, heads).bfloat16()
                    meta[physical, 3] = minimum.reshape(8, 16, heads).bfloat16()
    cache = SimpleNamespace(
        pages=pages,
        metadata=meta,
        tail=tail,
        state=state,
        device=pages.device,
        kv_heads=heads,
    )
    return cache, table, source, coarse, full


def reference(cache, source, coarse, full, q, positions, cu, slots, bits, ids, counts):
    outputs, normalizers = [], []
    for b in range(len(cu) - 1):
        slot = int(slots[b])
        length, prefix = cache.state[slot, :2].tolist()
        for index in range(int(cu[b]), int(cu[b + 1])):
            position = int(positions[index])
            view = prefix + (128 if position >= prefix + 256 else 0)
            selected = (
                torch.arange(length, device="cuda")
                if ids is None
                else ids[b, : int(counts[b])].long()
            )
            selected = selected[
                (selected >= 0) & (selected <= position) & (selected < length)
            ]
            arrays = []
            for kind in range(2):
                approx = (coarse if bits == 4 else full)[kind][slot, selected]
                values = torch.where(
                    (selected < view)[:, None, None],
                    approx,
                    source[kind][slot, selected],
                )
                arrays.append(
                    values.float().repeat_interleave(
                        q.shape[1] // cache.kv_heads, dim=1
                    )
                )
            keys, values = arrays
            logits = torch.einsum("hd,khd->hk", q[index].float(), keys) * 128**-0.5
            normalizers.append(logits.logsumexp(-1))
            outputs.append(torch.einsum("hk,khd->hd", logits.softmax(-1), values))
    return torch.stack(outputs), torch.stack(normalizers, 1)


@pytest.fixture(autouse=True)
def require_native_overlay():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100 required")
    native = pytest.importorskip("flash_attn.cute.flash_fwd_sm100")
    if not hasattr(native.FlashAttentionForwardSm100, "make_paged_kv_manager"):
        pytest.skip("isolated FA4 hierarchical overlay required")


@pytest.mark.parametrize("maxq", [1, 7])
@pytest.mark.parametrize("splits", [1, 2])
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("sparse", [False, True])
def test_native_packed_attention_and_graph(maxq, splits, bits, sparse):
    cache, table, source, coarse, full = make_cache()
    # Request order deliberately differs from persistent request-slot order.
    slots = torch.tensor([1, 0], device="cuda", dtype=torch.int32)
    table = table.flip(0)
    positions = (
        torch.tensor([389, 515], device="cuda")
        if maxq == 1
        else torch.cat(
            (
                torch.arange(386, 390, device="cuda"),
                torch.arange(509, 516, device="cuda"),
            )
        )
    )
    cu = torch.tensor(
        [0, 1, 2] if maxq == 1 else [0, 4, 11], device="cuda", dtype=torch.int32
    )
    # Match Q slices from a fused QKV projection without a contiguous copy.
    q = torch.randn((len(positions), 2, 8, 128), device="cuda", dtype=torch.bfloat16)[
        :, 0
    ]
    ids = torch.stack([torch.randperm(390, device="cuda")[:97] for _ in range(2)]).int()
    ids[1, :7] = torch.arange(509, 516, device="cuda")
    ids[0, 3], ids[1, 8] = -1, 999  # Invalid selected IDs must not reach packed memory.
    counts = torch.tensor([73, 97], device="cuda", dtype=torch.int32)
    selected, used = (ids, counts) if sparse else (None, None)
    ws = NativeAttentionWorkspace(2, 2, 8, maxq, 520, "cuda", num_splits=splits)

    def run():
        return attention(
            cache,
            q,
            positions,
            cu,
            slots,
            table,
            bits=bits,
            selected_ids=selected,
            selected_counts=used,
            workspace=ws,
        )

    def check(actual, lse):
        expected, expected_lse = reference(
            cache, source, coarse, full, q, positions, cu, slots, bits, selected, used
        )
        torch.testing.assert_close(actual.float(), expected, atol=0.006, rtol=0.025)
        torch.testing.assert_close(lse, expected_lse, atol=2e-4, rtol=2e-5)

    actual, lse = run()
    torch.cuda.synchronize()
    check(actual, lse)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, lse = run()
    q.mul_(0.875)
    if sparse:
        ids.copy_(ids.flip(1))
        counts.sub_(20)
    if maxq == 7:
        # A replay can remove the second precision view without recompiling.
        positions[4:].sub_(4)
    graph.replay()
    torch.cuda.synchronize()
    check(actual, lse)


def test_overlay_preserves_ordinary_paged_bf16():
    from flash_attn.cute.interface import flash_attn_varlen_func

    torch.manual_seed(71)
    keys = torch.randn(80, 16, 2, 128, device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    table = torch.randperm(80, device="cuda").view(2, 40).int()
    q = torch.randn(14, 8, 128, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, 7, 14], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([517, 390], device="cuda", dtype=torch.int32)
    actual, lse = flash_attn_varlen_func(
        q,
        keys,
        values,
        cu_seqlens_q=cu,
        seqused_k=lengths,
        page_table=table,
        max_seqlen_q=7,
        max_seqlen_k=520,
        causal=True,
        num_splits=2,
        pack_gqa=True,
        return_lse=True,
    )
    expected, expected_lse = [], []
    for b, length in enumerate(lengths.tolist()):
        k, v = [
            x[table[b].long()]
            .reshape(-1, 2, 128)[:length]
            .float()
            .repeat_interleave(4, dim=1)
            for x in (keys, values)
        ]
        for row in range(7):
            score = torch.einsum("hd,khd->hk", q[b * 7 + row].float(), k)
            score = score * 128**-0.5
            mask = torch.arange(length, device="cuda") > length - 7 + row
            score.masked_fill_(mask[None], -float("inf"))
            expected.append(torch.einsum("hk,khd->hd", score.softmax(-1), v))
            expected_lse.append(score.logsumexp(-1))
    torch.testing.assert_close(
        actual.float(), torch.stack(expected), atol=0.006, rtol=0.025
    )
    torch.testing.assert_close(lse, torch.stack(expected_lse, 1), atol=2e-4, rtol=2e-5)
