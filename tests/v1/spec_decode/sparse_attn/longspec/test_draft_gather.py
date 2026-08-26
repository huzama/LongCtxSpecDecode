# SPDX-License-Identifier: Apache-2.0
"""Gather must copy exactly the requested position range of each request."""
import pytest
import torch

from vllm.v1.spec_decode.sparse_attn.longspec.kernels.draft_gather import (
    gather_tokens,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")


def test_full_then_incremental_gather():
    torch.manual_seed(0)
    device = "cuda"
    batch, page, kvh, d, width = 2, 16, 4, 64, 50
    num_slots = 512
    k = torch.randn(num_slots // page, page, kvh, d, device=device,
                    dtype=torch.bfloat16)
    v = torch.randn_like(k)
    slots = torch.stack([torch.randperm(num_slots, device=device)[:width]
                         for _ in range(batch)]).to(torch.int32)
    padded = (width + page - 1) // page * page
    dst_k = torch.zeros(batch, padded, kvh, d, device=device, dtype=k.dtype)
    dst_v = torch.zeros_like(dst_k)

    used = torch.tensor([20, 33], dtype=torch.int32, device=device)
    zeros = torch.zeros(batch, dtype=torch.int32, device=device)
    gather_tokens(k, v, slots, zeros, used, dst_k, dst_v)
    k_flat, v_flat = k.view(-1, kvh, d), v.view(-1, kvh, d)
    for b in range(batch):
        n = int(used[b])
        assert torch.equal(dst_k[b, :n], k_flat[slots[b, :n].long()])
        assert torch.equal(dst_v[b, :n], v_flat[slots[b, :n].long()])
        assert torch.all(dst_k[b, n:] == 0)

    # Append one token per request: only position used-1 is written.
    used2 = used + 1
    starts = used2 - 1
    k_flat[slots[0, 20].long()] += 1  # change the source so the copy is visible
    gather_tokens(k, v, slots, starts, used2, dst_k, dst_v)
    for b in range(batch):
        n = int(used2[b])
        assert torch.equal(dst_k[b, n - 1], k_flat[slots[b, n - 1].long()])
        assert torch.all(dst_k[b, n:] == 0)
