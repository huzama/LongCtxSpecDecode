# SPDX-License-Identifier: Apache-2.0
"""The fused recompute must reproduce varlen_reduce's metric from raw scores.

Reference: the two rows' raw QK scores per head, rematerialized weights in
weight mode, averaged per varlen_reduce's entry rules, in fp32, cast to bf16.
"""
import pytest
import torch

from vllm.v1.spec_decode.sparse_attn.longspec.kernels.c2q_scores import (
    c2q_metric,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")


def _reference(q, k_cache, block_table, cu, valid, entry, lse, scale,
               use_weight):
    batch = valid.numel()
    page = k_cache.shape[1]
    heads, kvh = q.shape[1], k_cache.shape[2]
    group = heads // kvh
    out = torch.zeros(batch, int(valid.max()), dtype=torch.float32,
                      device=q.device)
    for b in range(batch):
        n = int(valid[b])
        pages = block_table[b, : (n + page - 1) // page].long()
        keys = k_cache[pages].reshape(-1, kvh, k_cache.shape[-1])[:n]
        keys = keys.repeat_interleave(group, dim=1)  # [n, heads, d]
        rows = {0: [int(cu[b]), int(cu[b + 1]) - 1],
                1: [int(cu[b])], 2: [int(cu[b + 1]) - 1]}[int(entry[b])]
        acc = torch.zeros(n, dtype=torch.float32, device=q.device)
        for tok in rows:
            s = torch.einsum("hd,nhd->hn", q[tok].float(), keys.float())
            if use_weight:
                s = torch.exp(scale * s - lse[:, tok].float()[:, None])
            acc += s.sum(0)
        out[b, :n] = acc / (heads * len(rows))
    return out


@pytest.mark.parametrize("use_weight", [False, True])
@pytest.mark.parametrize("entry_mode", [0, 1, 2])
def test_metric_matches_reference(use_weight, entry_mode):
    torch.manual_seed(0)
    device = "cuda"
    batch, heads, kvh, d, page = 3, 8, 2, 64, 16
    lens_q = torch.tensor([1, 4, 7], dtype=torch.int32)
    cu = torch.cat([torch.zeros(1, dtype=torch.int32), lens_q.cumsum(0)]).to(device)
    total_q = int(lens_q.sum())
    q = torch.randn(total_q, heads, d, device=device, dtype=torch.bfloat16)
    num_blocks = 40
    k_cache = torch.randn(num_blocks, page, kvh, d, device=device,
                          dtype=torch.bfloat16)
    valid = torch.tensor([37, 101, 250], dtype=torch.int32, device=device)
    max_blocks = (int(valid.max()) + page - 1) // page
    block_table = torch.stack([
        torch.randperm(num_blocks, device=device)[:max_blocks]
        for _ in range(batch)]).to(torch.int32)
    entry = torch.full((batch,), entry_mode, dtype=torch.int32, device=device)
    lse = torch.randn(heads, total_q, device=device, dtype=torch.float32) + 5
    scale = d ** -0.5
    out = torch.zeros(batch, 512, device=device, dtype=torch.bfloat16)

    c2q_metric(q, k_cache, block_table, cu, valid, entry, lse, scale,
               use_weight, out)
    ref = _reference(q, k_cache, block_table, cu, valid, entry, lse, scale,
                     use_weight)
    for b in range(batch):
        n = int(valid[b])
        torch.testing.assert_close(out[b, :n].float(), ref[b, :n],
                                   rtol=2e-2, atol=2e-2)
        assert torch.all(out[b, n:] == 0)  # untouched past the valid length
