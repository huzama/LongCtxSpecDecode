# SPDX-License-Identifier: Apache-2.0
"""Actual FA2/FA4 attention and gathered-KV parity; VEGAS_TEST_FA selects backend."""

import os

import pytest
import torch

from vllm.v1.attention.backends import fa_utils
from vllm.v1.spec_decode.sparse_attn.attn_overrider.utils.vegas_fallback import (
    GatheredDraftKV,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def attention(kwargs):
    version = int(os.environ.get("VEGAS_TEST_FA", "2"))
    return fa_utils.flash_attn_varlen_func(**kwargs, fa_version=version)


def reference(q, k, v, table, cu, used):
    outputs, normalizers = [], []
    for b in range(len(used)):
        query = q[cu[b] : cu[b + 1]].float()
        keys = k[table[b].long()].flatten(0, 1)[: used[b]].float()
        values = v[table[b].long()].flatten(0, 1)[: used[b]].float()
        group = q.shape[1] // k.shape[2]
        keys, values = (x.repeat_interleave(group, dim=1) for x in (keys, values))
        logits = torch.einsum("qhd,khd->hqk", query, keys) * q.shape[-1] ** -0.5
        mask = (
            torch.arange(len(keys), device=q.device)[None]
            > (len(keys) - len(query) + torch.arange(len(query), device=q.device))[
                :, None
            ]
        )
        logits.masked_fill_(mask[None], -torch.inf)
        outputs.append(torch.einsum("hqk,khd->qhd", logits.softmax(-1), values))
        normalizers.append(logits.logsumexp(-1))
    return torch.cat(outputs), torch.cat(normalizers, dim=1)


@pytest.mark.parametrize("dim", [64, 80, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_paged_attention_and_lse(dim, dtype):
    torch.manual_seed(33)
    q = torch.randn(27, 6, dim, device="cuda", dtype=dtype)
    page = 128 if dim == 256 and os.environ.get("VEGAS_TEST_FA") == "4" else 16
    k = torch.randn(32, page, 2, dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    table = torch.stack([torch.randperm(32, device="cuda")[:8] for _ in range(3)]).int()
    cu = torch.tensor([0, 1, 8, 27], device="cuda", dtype=torch.int32)
    used = torch.tensor([31, 65, 127], device="cuda", dtype=torch.int32)
    out = torch.empty_like(q)
    actual, lse = attention(
        dict(
            q=q,
            k=k,
            v=v,
            block_table=table,
            cu_seqlens_q=cu,
            seqused_k=used,
            max_seqlen_q=19,
            max_seqlen_k=127,
            causal=True,
            out=out,
            return_softmax_lse=True,
        )
    )
    expected, expected_lse = reference(q, k, v, table, cu, used)
    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.004)
    torch.testing.assert_close(lse, expected_lse, rtol=2e-5, atol=2e-5)


def test_gathered_attention_through_six_draft_steps():
    torch.manual_seed(72)
    q = torch.randn(2, 8, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(32, 16, 2, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    slots = torch.stack(
        [torch.randperm(512, device="cuda")[:97] for _ in range(2)]
    ).int()
    used = torch.tensor([17, 65], device="cuda", dtype=torch.int32)
    cu = torch.arange(3, device="cuda", dtype=torch.int32)
    cache = GatheredDraftKV(2, 2, 97, 16, 2, 128, k.dtype, "cuda")
    cache.begin_propose()
    for step in range(6):
        for layer in range(2):
            kwargs = dict(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu,
                max_seqlen_q=1,
                max_seqlen_k=512,
                causal=True,
            )
            cache.attention_kwargs(kwargs, layer, slots, used)
            actual = attention(kwargs)
            # Independent token-page reference, not the gathered scratch.
            expected, _ = reference(
                q, k.view(-1, 1, 2, 128), v.view(-1, 1, 2, 128), slots, cu, used
            )
            torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.004)
        used += 1
