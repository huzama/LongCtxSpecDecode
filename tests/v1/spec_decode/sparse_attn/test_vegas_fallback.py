# SPDX-License-Identifier: Apache-2.0
"""Numerical and lifecycle stress tests for the actual Vegas fallback kernels."""

import pytest
import torch

from vllm.v1.spec_decode.sparse_attn.attn_overrider.utils.vegas_fallback import (
    GatheredDraftKV,
    gather_tokens,
    score_tokens,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def score_case(dim, group, dtype, page, length=255):
    torch.manual_seed(41)
    heads, kv_heads = 2 * group, 2
    lengths = [17, 65, length, 0]
    q_lengths = [1, 4, 19, 0]
    cu = torch.tensor([0, 1, 5, 24, 24], device="cuda", dtype=torch.int32)
    q = torch.randn(24, heads, dim, device="cuda", dtype=dtype) / 2
    blocks = (max(lengths) + 3 + page - 1) // page
    k = torch.randn(blocks * 4, page, kv_heads, dim, device="cuda", dtype=dtype)
    # Strided page-table rows and query tensors exercise real backend layouts.
    table = torch.stack(
        [torch.randperm(blocks * 4, device="cuda")[:blocks] for _ in lengths]
    ).to(torch.int32)
    table = torch.stack((table, table), dim=1)[:, 0]
    table[-1].fill_(-1)
    valid = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    entry = torch.tensor([1, 0, 2, 0], device="cuda", dtype=torch.int32)
    q = torch.stack((q, q), dim=1)[:, 0]
    lse = torch.empty(heads, 24, device="cuda", dtype=torch.float32)
    keys = []
    for b, (n, nq) in enumerate(zip(lengths[:-1], q_lengths[:-1])):
        full = n + nq - 1 if b == 1 else n
        key = k[table[b].long()].reshape(-1, kv_heads, dim)[:full]
        key = key.repeat_interleave(group, dim=1).float()
        logits = torch.einsum("qhd,thd->hqt", q[cu[b] : cu[b + 1]].float(), key)
        logits *= dim**-0.5
        mask = (
            torch.arange(full, device="cuda")[None, :]
            > (full - nq + torch.arange(nq, device="cuda"))[:, None]
        )
        logits.masked_fill_(mask[None], -torch.inf)
        lse[:, cu[b] : cu[b + 1]] = logits.logsumexp(-1)
        keys.append(key)
    out = torch.full(
        (4, max(lengths) + 17), -123.0, device="cuda", dtype=torch.bfloat16
    )
    return q, k, table, cu, valid, entry, lse, dim**-0.5, out, keys


def reference(case, weight):
    q, _, _, cu, valid, entry, lse, scale, _, keys = case
    refs = []
    for b, key in enumerate(keys):
        rows = {
            0: [int(cu[b]), int(cu[b + 1]) - 1],
            1: [int(cu[b])],
            2: [int(cu[b + 1]) - 1],
        }[int(entry[b])]
        raw = torch.einsum("qhd,thd->qht", q[rows].float(), key[: valid[b]])
        if weight:
            raw = torch.exp(raw * scale - lse[:, rows].T[:, :, None])
        refs.append(raw.mean((0, 1)))
    return refs


@pytest.mark.parametrize("dim,group", [(8, 1), (64, 1), (80, 3), (128, 8), (256, 2)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("page", [16, 32])
@pytest.mark.parametrize("weight", [False, True])
def test_scores_against_full_attention(dim, group, dtype, page, weight):
    case = score_case(dim, group, dtype, page)
    q, k, table, cu, valid, entry, lse, scale, out, *_ = case
    score_tokens(q, k, table, cu, valid, entry, lse, scale, weight, out)
    for b, expected in enumerate(reference(case, weight)):
        n = len(expected)
        torch.testing.assert_close(out[b, :n].float(), expected, rtol=0.005, atol=2e-5)
        assert (out[b, n:] == -123).all()
    assert (out[-1] == -123).all()


@pytest.mark.parametrize("length", [4097, 32769, 131073])
def test_long_context_scores(length):
    case = score_case(64, 2, torch.bfloat16, 16, length)
    q, k, table, cu, valid, entry, lse, scale, out, *_ = case
    score_tokens(q, k, table, cu, valid, entry, lse, scale, True, out)
    for b, expected in enumerate(reference(case, True)):
        torch.testing.assert_close(
            out[b, : len(expected)].float(), expected, rtol=0.005, atol=2e-7
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("page", [16, 32])
def test_gather_ranges_and_untouched_storage(dtype, page):
    torch.manual_seed(9)
    k = torch.randn(64, page, 3, 80, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    slots = torch.stack(
        [torch.randperm(64 * page, device="cuda")[:97] for _ in range(5)]
    ).to(torch.int32)
    slots[-1].fill_(-1)
    starts = torch.tensor([0, 15, 16, 64, 0], device="cuda", dtype=torch.int32)
    ends = torch.tensor([17, 16, 65, 97, 0], device="cuda", dtype=torch.int32)
    dk = torch.full((5, 128, 3, 80), -123.0, device="cuda", dtype=dtype)
    dv = torch.full_like(dk, 123.0)
    gather_tokens(k, v, slots, starts, ends, dk, dv)
    for b in range(5):
        lo, hi = int(starts[b]), int(ends[b])
        for src, dst, sentinel in ((k, dk, -123.0), (v, dv, 123.0)):
            expected = src.view(-1, 3, 80)[slots[b, lo:hi].long()]
            torch.testing.assert_close(dst[b, lo:hi], expected, rtol=0, atol=0)
            assert (dst[b, :lo] == sentinel).all() and (dst[b, hi:] == sentinel).all()


def test_gather_rounds_layers_and_batch_reordering():
    torch.manual_seed(12)
    cache = GatheredDraftKV(3, 4, 97, 16, 2, 64, torch.bfloat16, "cuda")
    k = torch.randn(64, 16, 2, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    for batch in (4, 1, 3, 2, 4):
        cache.begin_propose()
        slots = torch.stack(
            [torch.randperm(1024, device="cuda")[:97] for _ in range(batch)]
        ).to(torch.int32)
        used = torch.arange(batch, device="cuda", dtype=torch.int32) + 15
        saved = {}
        for step in range(6):
            for layer in range(3):
                # Distinct layer data catches accidental cross-layer aliasing.
                source_k, source_v = k + layer + step, v - layer - step
                kwargs = dict(k=source_k, v=source_v, max_seqlen_k=1024)
                cache.attention_kwargs(kwargs, layer, slots, used)
                assert kwargs["max_seqlen_k"] == 112
                for b in range(batch):
                    n = int(used[b])
                    for name, source in (("k", source_k), ("v", source_v)):
                        pages = kwargs[name][kwargs["block_table"][b].long()]
                        actual = pages.reshape(-1, 2, 64)[:n]
                        newest = source.view(-1, 2, 64)[slots[b, n - 1].long()]
                        key = (layer, b, name)
                        expected = (
                            torch.cat((saved[key], newest[None]))
                            if step
                            else source.view(-1, 2, 64)[slots[b, :n].long()]
                        )
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        saved[key] = expected.clone()
            used += 1


def test_score_and_gather_cuda_graph_replay():
    case = score_case(80, 3, torch.bfloat16, 16)
    q, k, table, cu, valid, entry, lse, scale, out, *_ = case
    slots = torch.arange(97, device="cuda", dtype=torch.int32)[None].repeat(4, 1)
    starts = torch.zeros(4, device="cuda", dtype=torch.int32)
    ends = torch.tensor([17, 65, 97, 0], device="cuda", dtype=torch.int32)
    dk = torch.zeros(4, 112, 2, 80, device="cuda", dtype=k.dtype)
    dv = torch.zeros_like(dk)

    def run():
        score_tokens(q, k, table, cu, valid, entry, lse, scale, False, out)
        gather_tokens(k, k, slots, starts, ends, dk, dv)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for repeat in range(20):
        q.mul_(-1)
        slots.copy_((slots + 13) % (k.shape[0] * k.shape[1]))
        ends[0] = 15 + repeat % 3
        graph.replay()
        torch.cuda.synchronize()
        for b, expected in enumerate(reference(case, False)):
            torch.testing.assert_close(
                out[b, : len(expected)].float(), expected, rtol=0.005, atol=2e-5
            )
        for b in range(3):
            n = int(ends[b])
            expected = k.view(-1, 2, 80)[slots[b, :n].long()]
            torch.testing.assert_close(dk[b, :n], expected, rtol=0, atol=0)


def test_draft_tail_stops_at_context_limit_and_ignores_padding(monkeypatch):
    from vllm.v1.spec_decode.sparse_attn.attn_overrider import BaseAttnOverrider
    from vllm.v1.spec_decode.sparse_attn.attn_overrider.vegas import VegasAttnOverrider

    class Probe(VegasAttnOverrider):
        def __del__(self):
            pass

    probe = Probe.__new__(Probe)
    probe.num_layers, probe.batch_size, probe.curr_layer = 1, 4, 0
    probe._hierarchical = False
    probe.num_spec_tokens, probe.max_model_len, probe.block_size = 6, 32, 16
    probe._metadata_initialized = False
    probe._page_table = torch.full((1, 4, 16), -1, device="cuda", dtype=torch.int32)
    probe._page_table[0, :3, :2] = torch.tensor([0, 14], device="cuda")
    probe._topk_budget = torch.tensor([2, 2, 2, 0], device="cuda", dtype=torch.int32)
    probe._topk_valid_lens = torch.tensor(
        [29, 17, 31, 0], device="cuda", dtype=torch.int32
    )
    probe._draft_kv = GatheredDraftKV(1, 4, 16, 16, 1, 8, torch.bfloat16, "cuda")
    # Exactly two pages per request: no spare table columns to hide an overrun.
    table = torch.tensor(
        [[0, 1], [2, 3], [4, 5], [-1, -1]], device="cuda", dtype=torch.int32
    )
    k = torch.arange(6 * 16 * 8, device="cuda", dtype=torch.float32)
    k = k.to(torch.bfloat16).view(6, 16, 1, 8)
    v = -k
    captured = []
    monkeypatch.setattr(
        BaseAttnOverrider,
        "_original_attn_func",
        lambda **kw: captured.append(kw["seqused_k"].clone()),
        raising=False,
    )
    for lengths, expected in (
        ([31, 18, 1, 0], [4, 3, 2, 0]),
        ([32, 19, 1, 0], [5, 4, 2, 0]),
        ([1, 20, 1, 0], [5, 5, 2, 0]),
        ([1, 21, 1, 0], [5, 6, 2, 0]),
    ):
        probe._draft_attention(
            k=k,
            v=v,
            cu_seqlens_q=torch.tensor([0, 1, 2, 3, 3], device="cuda"),
            seqused_k=torch.tensor(lengths, device="cuda", dtype=torch.int32),
            block_table=table,
            max_seqlen_k=32,
        )
        assert captured[-1].tolist() == expected
    assert probe._page_table[0, 0, :5].tolist() == [0, 14, 29, 30, 31]
    assert probe._page_table[0, 2, :2].tolist() == [64, 78]
    assert (probe._page_table[0, 3] == -1).all()
