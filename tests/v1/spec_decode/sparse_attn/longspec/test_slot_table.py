# SPDX-License-Identifier: Apache-2.0
"""Slot conversion must follow the block table and append the recent range,
per layer or shared across layers."""
import pytest
import torch

from vllm.v1.spec_decode.sparse_attn.longspec.kernels.slot_table import (
    index_to_slots,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")


def _reference(table, block_table, used, valid, seqlens, page):
    out = table.clone()
    layers, batch = table.shape[:2]
    for l in range(layers):
        for b in range(batch):
            n = int(used[l, b]) if used.dim() == 2 else int(used[b])
            idx = table[l, b, :n].long()
            out[l, b, :n] = (block_table[b, idx // page] * page +
                             idx % page).to(torch.int32)
            recent = torch.arange(int(valid[b]), int(seqlens[b]),
                                  device=table.device)
            slots = (block_table[b, recent // page] * page +
                     recent % page).to(torch.int32)
            out[l, b, n:n + recent.numel()] = slots
    return out


@pytest.mark.parametrize("per_layer", [True, False])
def test_matches_reference(per_layer):
    torch.manual_seed(0)
    device, page = "cuda", 16
    layers, batch_max, batch, width = 3, 4, 3, 64
    valid = torch.tensor([100, 40, 7], dtype=torch.int32, device=device)
    seqlens = valid + torch.tensor([5, 2, 0], dtype=torch.int32,
                                   device=device)
    block_table = torch.randperm(1024, device=device)[:batch * 8].view(
        batch, 8).to(torch.int32)
    table = torch.zeros(layers, batch_max, width, dtype=torch.int32,
                        device=device)
    # Selected counts never exceed the scored prefix.
    cap = torch.full((batch_max,), 30, dtype=torch.int32, device=device)
    cap[:batch] = torch.minimum(cap[:batch], valid)
    if per_layer:
        used = torch.randint(0, 30, (layers, batch_max), dtype=torch.int32,
                             device=device)
        used = torch.minimum(used, cap[None, :])
    else:
        used = torch.randint(0, 30, (batch_max,), dtype=torch.int32,
                             device=device)
        used = torch.minimum(used, cap)
    for l in range(layers):
        for b in range(batch):
            n = int(used[l, b]) if per_layer else int(used[b])
            table[l, b, :n] = torch.randperm(
                int(valid[b]), device=device)[:n].to(torch.int32)
    expected = _reference(table[:, :batch], block_table, used, valid,
                          seqlens, page)
    index_to_slots(table[:, :batch], block_table, used, valid, seqlens, page)
    assert torch.equal(table[:, :batch], expected)
    assert torch.all(table[:, batch:] == 0)
