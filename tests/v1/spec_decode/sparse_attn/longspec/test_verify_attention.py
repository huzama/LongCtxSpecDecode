# SPDX-License-Identifier: Apache-2.0
"""Packed verify attention against the single causal FA2 call: outputs and
LSE match, page-boundary and short-prefix rows included, padded rows stay
finite, and the eligibility gate admits only the shape the path handles."""

import pytest
import torch

from vllm.v1.spec_decode.sparse_attn.longspec.verify_attention import (
    packed_verify_attention,
    packed_verify_eligible,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")

HEADS, KV_HEADS, HEAD_DIM, PAGE = 32, 8, 128, 16
QUERIES = 7


def _fa():
    # The overrider patches the backend module, not this dispatcher.
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

    return flash_attn_varlen_func


def _case(seq_lens, seed=0):
    """Paged cache and verify-shaped queries; seq_lens include the queries."""
    torch.manual_seed(seed)
    device = "cuda"
    batch = len(seq_lens)
    max_pages = (max(seq_lens) + PAGE - 1) // PAGE
    num_pages = batch * max_pages + 1
    key_cache = torch.randn(
        num_pages, PAGE, KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    value_cache = torch.randn_like(key_cache)
    perm = torch.randperm(num_pages - 1, device=device)[: batch * max_pages]
    block_table = (perm.to(torch.int32) + 1).reshape(batch, max_pages)
    q = torch.randn(
        batch * QUERIES, HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    kwargs = dict(
        q=q,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=torch.arange(
            0, (batch + 1) * QUERIES, QUERIES, device=device, dtype=torch.int32
        ),
        max_seqlen_q=QUERIES,
        seqused_k=torch.tensor(seq_lens, device=device, dtype=torch.int32),
        max_seqlen_k=max(seq_lens),
        causal=True,
        block_table=block_table,
        return_softmax_lse=True,
        fa_version=2,
    )
    return kwargs


def _compare(seq_lens, seed=0):
    kwargs = _case(seq_lens, seed)
    ref_out, ref_lse = _fa()(**kwargs)
    out, lse = packed_verify_attention(_fa(), dict(kwargs))
    torch.testing.assert_close(out.float(), ref_out.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse, ref_lse, atol=1e-3, rtol=1e-3)


def test_matches_reference_across_lengths():
    # Empty prefix, one page, page boundaries both sides, long rows.
    _compare([QUERIES, 8, 22, 23, 24, 39, 1000, 4096 + 13])


def test_matches_reference_batch_one_long():
    _compare([16384 + 5], seed=1)


def test_uses_out_buffer():
    kwargs = _case([300, 77])
    out_buf = torch.empty_like(kwargs["q"])
    kwargs["out"] = out_buf
    out, _ = packed_verify_attention(_fa(), kwargs)
    assert out.data_ptr() == out_buf.data_ptr()
    del kwargs["out"]
    ref_out, _ = _fa()(**kwargs)
    torch.testing.assert_close(out.float(), ref_out.float(), atol=2e-2, rtol=2e-2)


def test_padded_row_stays_finite():
    # A padded graph row: no valid sequence, block table zeroed. Real rows
    # must match the reference; the padded row must not produce NaN.
    kwargs = _case([500, QUERIES, 64])
    kwargs["seqused_k"][1] = 0
    kwargs["block_table"][1] = 0
    out, lse = packed_verify_attention(_fa(), dict(kwargs))
    assert torch.isfinite(out.float()).all()
    ref_out, ref_lse = _fa()(**kwargs)
    keep = torch.ones_like(out, dtype=torch.bool)
    keep[QUERIES : 2 * QUERIES] = False
    torch.testing.assert_close(
        out.float()[keep], ref_out.float()[keep], atol=2e-2, rtol=2e-2
    )


def test_eligibility_gate():
    kwargs = _case([300])
    assert packed_verify_eligible(kwargs, group=4)
    assert not packed_verify_eligible(kwargs, group=1)
    assert not packed_verify_eligible({**kwargs, "causal": False}, 4)
    assert not packed_verify_eligible({**kwargs, "max_seqlen_q": 1}, 4)
    assert not packed_verify_eligible({**kwargs, "max_seqlen_q": 64}, 4)
    assert not packed_verify_eligible({**kwargs, "block_table": None}, 4)
    assert not packed_verify_eligible({**kwargs, "softcap": 30.0}, 4)
    assert not packed_verify_eligible({**kwargs, "s_aux": object()}, 4)
    assert not packed_verify_eligible({**kwargs, "window_size": [128, 0]}, 4)
    assert not packed_verify_eligible({**kwargs, "scores": torch.empty(0)}, 4)
    # A ragged batch: total queries no longer batch * max_seqlen_q.
    ragged = _case([300, 200])
    ragged["q"] = ragged["q"][:-1]
    assert not packed_verify_eligible(ragged, 4)


def _require_fa4():
    from vllm.vllm_flash_attn import is_fa_version_supported

    if not is_fa_version_supported(4):
        pytest.skip("FA4 requires Blackwell and flash-attn-4")


@pytest.mark.parametrize("queries", [1, 7])
def test_fa4_paged_graph_replays_changed_inputs(queries):
    _require_fa4()
    kwargs = _case([137, 255, 4097], seed=3)
    batch = 3
    kwargs["q"] = kwargs["q"][: batch * queries]
    kwargs["max_seqlen_q"] = queries
    kwargs["cu_seqlens_q"] = torch.arange(
        0, (batch + 1) * queries, queries, device="cuda", dtype=torch.int32
    )
    kwargs["out"] = torch.empty_like(kwargs["q"])
    kwargs["fa_version"] = 4
    # vLLM passes unit descales even with BF16 inputs.
    for name in ("q_descale", "k_descale", "v_descale"):
        kwargs[name] = torch.ones(batch, KV_HEADS, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _fa()(**kwargs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out, lse = _fa()(**kwargs)
    assert out.data_ptr() == kwargs["out"].data_ptr()
    for lengths in ([137, 255, 4097], [131, 241, 4089]):
        kwargs["q"].normal_()
        kwargs["seqused_k"].copy_(torch.tensor(lengths, device="cuda"))
        kwargs["block_table"].copy_(kwargs["block_table"].roll(1, dims=1))
        graph.replay()
        reference = {key: value for key, value in kwargs.items() if key != "out"}
        ref_out, ref_lse = _fa()(**{**reference, "fa_version": 2})
        torch.testing.assert_close(out.float(), ref_out.float(), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(lse, ref_lse, atol=1e-3, rtol=1e-3)


def test_fa4_variable_length_prefill():
    _require_fa4()
    lengths = [19, 37, 65]
    total = sum(lengths)
    cu = torch.tensor([0, 19, 56, total], device="cuda", dtype=torch.int32)
    q = torch.randn(total, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(total, KV_HEADS, HEAD_DIM, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    kwargs = dict(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=max(lengths),
        max_seqlen_k=max(lengths),
        causal=True,
        return_softmax_lse=True,
    )
    ref_out, ref_lse = _fa()(**kwargs, fa_version=2)
    out, lse = _fa()(**kwargs, fa_version=4)
    torch.testing.assert_close(out.float(), ref_out.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse, ref_lse, atol=1e-3, rtol=1e-3)


def test_full_lse_for_sparse_scoring():
    from vllm.v1.spec_decode.sparse_attn.longspec.kernels.c2q_scores import c2q_lse

    kwargs = _case([7, 139, 4099], seed=17)
    batch = 3
    partial = torch.empty(batch, 2 * HEADS, (4099 + 127) // 128, device="cuda")
    output = torch.zeros(HEADS, batch * QUERIES, device="cuda")
    actual = c2q_lse(kwargs, partial, output)
    for b, length in enumerate(kwargs["seqused_k"].tolist()):
        pages = kwargs["block_table"][b].long()
        keys = kwargs["k"][pages].flatten(0, 1)[:length].float()
        keys = keys.repeat_interleave(HEADS // KV_HEADS, dim=1)
        for offset, limit in ((0, length - QUERIES + 1), (QUERIES - 1, length)):
            query = kwargs["q"][b * QUERIES + offset].float()
            scores = torch.einsum("hd,thd->ht", query, keys[:limit]) / HEAD_DIM**0.5
            expected = scores.logsumexp(dim=1)
            torch.testing.assert_close(
                actual[:, b * QUERIES + offset], expected, atol=1e-4, rtol=1e-4
            )


@pytest.mark.parametrize("ratio", [0.5, 1.0])
def test_sparse_verifier_selection_causality_and_graph_replay(ratio):
    from types import SimpleNamespace

    from vllm.v1.spec_decode.sparse_attn.longspec.sparse_verifier import SparseVerifier

    _require_fa4()
    kwargs = _case([1007, 519], seed=23)
    kwargs["fa_version"] = 4
    spec = SimpleNamespace(
        sparse_attn_verify_ratio=ratio, sparse_attn_sink=4, sparse_attn_recent=64
    )
    verifier = SparseVerifier(2, 2, 1024, HEADS, QUERIES, spec, "cuda")
    metric = torch.rand(2, 2, 1024, device="cuda", dtype=torch.bfloat16)
    known = torch.tensor([995, 510], device="cuda", dtype=torch.int32)
    verifier.remember(kwargs, known)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        verifier.prepare(kwargs, metric)
        _fa()(**verifier.attention_kwargs(kwargs, 0))
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        verifier.prepare(kwargs, metric)
        out, lse = _fa()(**verifier.attention_kwargs(kwargs, 0))
    original_k = kwargs["k"].clone()
    original_v = kwargs["v"].clone()
    for lengths in ([1007, 519], [1008, 523]):
        kwargs["seqused_k"].copy_(torch.tensor(lengths, device="cuda"))
        kwargs["q"].normal_()
        graph.replay()
        tables, used = verifier.table.cpu(), verifier.used.cpu()
        for b, length in enumerate(lengths):
            prefix = length - QUERIES
            count = max(int(prefix * ratio + 0.999999), 68 + prefix - int(known[b]))
            assert used[0, b] == count + QUERIES
            logical = torch.arange(length, device="cuda")
            physical = kwargs["block_table"][b, logical // PAGE] * PAGE + logical % PAGE
            selected = tables[0, b, : used[0, b]].to("cuda").long()
            assert selected.unique().numel() == selected.numel()
            torch.testing.assert_close(selected[-QUERIES:], physical[-QUERIES:].long())
            mandatory = torch.cat((physical[:4], physical[int(known[b]) - 64 :]))
            assert torch.isin(mandatory, selected).all()
            k = kwargs["k"].flatten(0, 1)[selected]
            v = kwargs["v"].flatten(0, 1)[selected]
            # Independent, explicitly causal attention on the chosen physical KV.
            q = kwargs["q"][b * QUERIES : (b + 1) * QUERIES].float()
            k = k.float().repeat_interleave(HEADS // KV_HEADS, dim=1)
            v = v.float().repeat_interleave(HEADS // KV_HEADS, dim=1)
            logits = torch.einsum("qhd,khd->hqk", q, k) / HEAD_DIM**0.5
            positions = torch.arange(selected.numel(), device="cuda")
            limits = selected.numel() - QUERIES + torch.arange(QUERIES, device="cuda")
            logits.masked_fill_(
                positions[None, None, :] > limits[None, :, None], -float("inf")
            )
            expected = torch.einsum("hqk,khd->qhd", logits.softmax(-1), v)
            torch.testing.assert_close(
                out[b * QUERIES : (b + 1) * QUERIES].float(),
                expected,
                atol=2e-2,
                rtol=2e-2,
            )
        if ratio == 1:
            reference, _ = _fa()(**kwargs)
            torch.testing.assert_close(
                out.float(), reference.float(), atol=2e-2, rtol=2e-2
            )
    torch.testing.assert_close(kwargs["k"], original_k)
    torch.testing.assert_close(kwargs["v"], original_v)
    # A different request in the row must not inherit stale scores.
    kwargs["block_table"].copy_(kwargs["block_table"].roll(1, dims=0))
    verifier.prepare(kwargs, metric)
    torch.testing.assert_close(verifier.used[0], kwargs["seqused_k"])


def test_selected_scoring_graph_matches_causal_reference_and_excludes_old_tokens():
    from types import SimpleNamespace

    from vllm.v1.spec_decode.sparse_attn.longspec.sparse_verifier import SparseVerifier

    _require_fa4()
    kwargs = _case([1007, 519], seed=37)
    kwargs["fa_version"] = 4
    spec = SimpleNamespace(
        sparse_attn_verify_ratio=0.5,
        sparse_attn_sink=4,
        sparse_attn_recent=64,
        sparse_attn_verify_score_scope="selected",
    )
    verifier = SparseVerifier(1, 2, 1024, HEADS, QUERIES, spec, "cuda")
    assert verifier.partial is None and verifier.lse is None
    metric = torch.rand(1, 2, 1024, device="cuda", dtype=torch.bfloat16)
    valid = torch.empty(2, device="cuda", dtype=torch.int32)
    entry = torch.zeros_like(valid)
    verifier.remember(kwargs, kwargs["seqused_k"] - QUERIES + 1)

    def step():
        valid.copy_(kwargs["seqused_k"] - QUERIES + 1)
        verifier.prepare(kwargs, metric)
        out, lse = _fa()(**verifier.attention_kwargs(kwargs, 0))
        verifier.refresh_selected(kwargs, 0, lse, valid, entry, metric[0])
        verifier.remember(kwargs, valid)
        return out

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = step()
    previous = None
    for lengths in ([1007, 519], [1008, 523]):
        kwargs["seqused_k"].copy_(torch.tensor(lengths, device="cuda"))
        kwargs["q"].normal_()
        graph.replay()
        chosen = []
        for b, length in enumerate(lengths):
            count = int(verifier.used[0, b])
            logical = verifier.logical[0, b, :count].long()
            physical = verifier.table[0, b, :count].long()
            expected_slots = (
                kwargs["block_table"][b, logical // PAGE] * PAGE + logical % PAGE
            )
            torch.testing.assert_close(physical, expected_slots.long())
            assert logical.unique().numel() == count
            if previous is not None:
                # Only the previous retained history or newly appended positions.
                old, old_length = previous[b]
                assert torch.isin(logical[logical < old_length], old).all()
            chosen.append((logical.clone(), length))
            k = kwargs["k"].flatten(0, 1)[physical].float()
            k = k.repeat_interleave(HEADS // KV_HEADS, dim=1)
            v = kwargs["v"].flatten(0, 1)[physical].float()
            v = v.repeat_interleave(HEADS // KV_HEADS, dim=1)
            q = kwargs["q"][b * QUERIES : (b + 1) * QUERIES].float()
            scores = torch.einsum("qhd,khd->hqk", q, k) / HEAD_DIM**0.5
            positions = length - QUERIES + torch.arange(QUERIES, device="cuda")
            scores.masked_fill_(
                logical[None, None, :] > positions[None, :, None], -float("inf")
            )
            weights = scores.softmax(-1)
            expected_out = torch.einsum("hqk,khd->qhd", weights, v)
            torch.testing.assert_close(
                out[b * QUERIES : (b + 1) * QUERIES].float(),
                expected_out,
                atol=2e-2,
                rtol=2e-2,
            )
            expected = torch.full((1024,), -1.0, device="cuda")
            historical = logical < valid[b]
            values = weights[:, [0, QUERIES - 1], :].mean(dim=(0, 1))
            expected[logical[historical]] = values[historical]
            torch.testing.assert_close(
                metric[0, b].float(), expected, atol=2e-4, rtol=2e-2
            )
        previous = chosen
    # New rows must bootstrap from full history, never another request's scores.
    kwargs["block_table"].copy_(kwargs["block_table"].roll(1, dims=0))
    graph.replay()
    torch.testing.assert_close(verifier.used[0], kwargs["seqused_k"])
