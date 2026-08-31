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
    # Straight from the kernel package: the overrider tests patch the
    # backend module's attribute, and the reference must dodge that.
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    return flash_attn_varlen_func


def _case(seq_lens, seed=0):
    """Paged cache and verify-shaped queries; seq_lens include the queries."""
    torch.manual_seed(seed)
    device = "cuda"
    batch = len(seq_lens)
    max_pages = (max(seq_lens) + PAGE - 1) // PAGE
    num_pages = batch * max_pages + 1
    key_cache = torch.randn(num_pages, PAGE, KV_HEADS, HEAD_DIM,
                            device=device, dtype=torch.bfloat16)
    value_cache = torch.randn_like(key_cache)
    perm = torch.randperm(num_pages - 1, device=device)[:batch * max_pages]
    block_table = (perm.to(torch.int32) + 1).reshape(batch, max_pages)
    q = torch.randn(batch * QUERIES, HEADS, HEAD_DIM,
                    device=device, dtype=torch.bfloat16)
    kwargs = dict(
        q=q, k=key_cache, v=value_cache,
        cu_seqlens_q=torch.arange(0, (batch + 1) * QUERIES, QUERIES,
                                  device=device, dtype=torch.int32),
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
    torch.testing.assert_close(out.float(), ref_out.float(),
                               atol=2e-2, rtol=2e-2)
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
    torch.testing.assert_close(out.float(), ref_out.float(),
                               atol=2e-2, rtol=2e-2)


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
    keep[QUERIES:2 * QUERIES] = False
    torch.testing.assert_close(out.float()[keep], ref_out.float()[keep],
                               atol=2e-2, rtol=2e-2)


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
    assert not packed_verify_eligible(
        {**kwargs, "window_size": [128, 0]}, 4)
    assert not packed_verify_eligible(
        {**kwargs, "scores": torch.empty(0)}, 4)
    # A ragged batch: total queries no longer batch * max_seqlen_q.
    ragged = _case([300, 200])
    ragged["q"] = ragged["q"][:-1]
    assert not packed_verify_eligible(ragged, 4)
