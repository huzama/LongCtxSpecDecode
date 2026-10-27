# SPDX-License-Identifier: Apache-2.0
"""Vegas must score the same per-query precision view used by verification."""

import pytest
import torch

from tests.v1.spec_decode.sparse_attn.test_hierarchical_cache import reference_values
from vllm.v1.attention.ops.hierarchical.cache import PagedHierarchicalKV
from vllm.v1.attention.ops.hierarchical.cache_kernels import prefill, update
from vllm.v1.spec_decode.sparse_attn.attn_overrider.utils.hierarchical_scores import (
    score_tokens,
)
from vllm.v1.spec_decode.sparse_attn.attn_overrider.vegas import _index_to_slot_kernel

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("weight", [False, True])
@pytest.mark.parametrize("h,group", [(2, 3), (8, 4), (8, 8)])
@pytest.mark.parametrize(
    "initial,base,queries",
    [
        (True, 0, 7),
        (True, 0, 257),
        (False, 249, 7),
        (False, 255, 7),
        (False, 383, 7),
        (False, 383, 1),
    ],
)
def test_scores_match_verifier_across_quantization_boundary(
    weight, h, group, initial, base, queries
):
    torch.manual_seed(25)
    length = queries if initial else base + queries
    # Fused-QKV slices retain a token stride larger than H*D.
    keys = torch.randn(length, 2, h, 128, device="cuda", dtype=torch.bfloat16)[:, 0]
    values = torch.randn_like(keys)
    cache = PagedHierarchicalKV.allocate(64, h, 1, "cuda")
    table = torch.stack(
        (torch.randperm(64, device="cuda")[:32], torch.full((32,), -1, device="cuda"))
    ).int()
    cu = torch.tensor([0, queries, queries], device="cuda", dtype=torch.int32)
    slots = torch.tensor([0, -1], device="cuda", dtype=torch.int32)
    positions = torch.arange(queries, device="cuda", dtype=torch.int32)
    if not initial:
        positions += base
        prefill(cache, 0, table[0], keys[:base], values[:base])
        update(
            cache,
            keys[base:],
            values[base:],
            positions,
            cu,
            slots,
            torch.tensor([base, 0], device="cuda", dtype=torch.int32),
            table,
        )
    else:
        prefill(cache, 0, table[0], keys, values)
    q = torch.randn(queries, 2, h * group, 128, device="cuda", dtype=torch.bfloat16)[
        :, 0
    ]
    rows = [queries - 1] if initial else ([0] if queries == 1 else [0, queries - 1])
    lse = torch.full((h * group, queries), -torch.inf, device="cuda")

    def reference(valid_length):
        expected = []
        for row in rows:
            position = int(positions[row])
            prefix = 0 if initial else max(0, ((position - 128) // 128) * 128)
            view = reference_values(keys, prefix, 8).repeat_interleave(group, dim=1)
            raw = torch.einsum(
                "hd,khd->hk", q[row].float(), view[: position + 1].float()
            )
            lse[:, row] = (raw * 128**-0.5).logsumexp(-1)
            raw = raw[:, :valid_length]
            expected.append(
                torch.exp(raw * 128**-0.5 - lse[:, row, None]) if weight else raw
            )
        return torch.stack(expected).mean((0, 1))

    valid_length = queries if initial else base + 1
    valid = torch.tensor([valid_length, 0], device="cuda", dtype=torch.int32)
    mode = 2 if initial else (1 if queries == 1 else 0)
    entry = torch.tensor([mode, 0], device="cuda", dtype=torch.int32)
    out = torch.full((2, length + 8), -123.0, device="cuda", dtype=torch.bfloat16)
    kwargs = dict(
        hierarchical_cache=cache,
        q=q,
        k=keys if initial else keys[base:],
        request_slots=slots,
        block_table=table,
        cu_seqlens_q=cu,
        positions=positions,
        prefill_mask=torch.tensor([initial, False], device="cuda"),
    )

    def run():
        score_tokens(kwargs, valid, entry, lse, 128**-0.5, weight, out)

    expected = reference(valid_length)
    run()
    torch.testing.assert_close(
        out[0, :valid_length].float(),
        expected,
        rtol=0.005,
        atol=2e-4 if not weight else 2e-5,
    )
    assert (out[0, valid_length:] == -123).all() and (out[1] == -123).all()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    before = out.clone()
    out.fill_(-123)
    graph.replay()
    torch.testing.assert_close(out, before, rtol=0, atol=0)
    if not initial and base == 383 and queries == 7:
        # The same graph must alternate between a shared view and two views.
        # The old BF16 group remains live until acceptance advances the prefix.
        for shifted_base in (base - 7, base):
            positions.copy_(torch.arange(queries, device="cuda") + shifted_base)
            valid_length = shifted_base + 1
            valid[0] = valid_length
            q.neg_()
            expected = reference(valid_length)
            out.fill_(-123)
            graph.replay()
            torch.testing.assert_close(
                out[0, :valid_length].float(),
                expected,
                rtol=0.005,
                atol=2e-4 if not weight else 2e-5,
            )
            assert (out[0, valid_length:] == -123).all() and (out[1] == -123).all()


def test_packed_selection_preserves_token_ids_and_recent_tail():
    indices = torch.full((2, 2, 12), -1, device="cuda", dtype=torch.int32)
    indices[:, 0, :3] = torch.tensor([95, 1, 33], device="cuda")
    indices[:, 1, :2] = torch.tensor([70, 4], device="cuda")
    table = torch.randperm(32, device="cuda").int().view(2, 16)
    budget = torch.tensor([3, 2], device="cuda", dtype=torch.int32)
    valid = torch.tensor([100, 75], device="cuda", dtype=torch.int32)
    end = torch.tensor([106, 78], device="cuda", dtype=torch.int32)
    _index_to_slot_kernel[(2, 2)](
        indices,
        table,
        budget,
        valid,
        end,
        indices.stride(0),
        indices.stride(1),
        table.stride(0),
        16,
        BLOCK_SIZE=32,
        LOGICAL=True,
    )
    assert indices[0, 0, :9].tolist() == [95, 1, 33, 100, 101, 102, 103, 104, 105]
    assert indices[1, 1, :5].tolist() == [70, 4, 75, 76, 77]
