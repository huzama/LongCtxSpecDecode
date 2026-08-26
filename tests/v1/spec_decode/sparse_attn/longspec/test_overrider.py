# SPDX-License-Identifier: Apache-2.0
"""The overrider's verify and draft hooks, driven with a fake attention op:
per-layer budgets, padded rows, the tail bookkeeping, attention skips."""
from math import ceil
from types import SimpleNamespace

import pytest
import torch

from vllm.config.speculative import SpeculativeConfig
from vllm.v1.spec_decode.sparse_attn.attn_overrider import BaseAttnOverrider
from vllm.v1.spec_decode.sparse_attn.longspec import CoverageAttnOverrider

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")

LAYERS, B_MAX, MAX_LEN, PAGE, G = 3, 4, 256, 16, 2
HEADS, KV_HEADS, DIM = 4, 2, 32
SINK, RECENT, RATIO = 2, 4, 0.5


def _vllm_config(**spec_kwargs):
    spec = SpeculativeConfig(
        method="sparse_attn", num_speculative_tokens=G,
        sparse_attn_algorithm="coverage", sparse_attn_ratio=RATIO,
        sparse_attn_min_tokens=0, sparse_attn_sink=SINK,
        sparse_attn_recent=RECENT, **spec_kwargs)
    model = SimpleNamespace(
        get_num_layers=lambda _: LAYERS,
        get_num_attention_heads=lambda _: HEADS,
        get_num_kv_heads=lambda _: KV_HEADS,
        get_head_size=lambda: DIM,
        dtype=torch.bfloat16, max_model_len=MAX_LEN, enforce_eager=True)
    return SimpleNamespace(
        speculative_config=spec, model_config=model, parallel_config=None,
        cache_config=SimpleNamespace(block_size=PAGE),
        scheduler_config=SimpleNamespace(max_num_seqs=B_MAX))


@pytest.fixture
def harness():
    ov = CoverageAttnOverrider(_vllm_config(sparse_attn_skip_attn_layers=[1]),
                               torch.device("cuda"))
    real = BaseAttnOverrider._original_attn_func
    calls = []

    def fake(*args, **kwargs):
        calls.append(dict(kwargs))
        out = kwargs["out"]
        if kwargs.get("return_softmax_lse"):
            q = kwargs["q"]
            lse = torch.zeros(q.shape[1], q.shape[0], device=q.device,
                              dtype=torch.float32)
            return out, lse
        return out

    BaseAttnOverrider._original_attn_func = fake
    yield ov, calls
    BaseAttnOverrider._original_attn_func = real


def _caches(device):
    torch.manual_seed(0)
    k = torch.randn(64, PAGE, KV_HEADS, DIM, device=device,
                    dtype=torch.bfloat16)
    v = torch.randn_like(k)
    block_table = torch.stack([torch.randperm(64, device=device)[:8]
                               for _ in range(B_MAX)]).to(torch.int32)
    block_table[3] = -1  # graph-padded row
    return k, v, block_table


def _verify(ov, k, v, block_table, device):
    seqused_k = torch.tensor([100, 40, 7, 0], dtype=torch.int32, device=device)
    cu = torch.tensor([0, 3, 6, 7, 7], dtype=torch.int32, device=device)
    for _ in range(LAYERS):
        q = torch.randn(7, HEADS, DIM, device=device, dtype=torch.bfloat16)
        ov._attention(q=q, k=k, v=v, out=torch.zeros_like(q), cu_seqlens_q=cu,
                      seqused_k=seqused_k, block_table=block_table,
                      softmax_scale=DIM ** -0.5, max_seqlen_q=3,
                      max_seqlen_k=100, causal=True)


def test_verify_then_draft(harness):
    ov, calls = harness
    device = torch.device("cuda")
    k, v, block_table = _caches(device)

    _verify(ov, k, v, block_table, device)
    assert ov.curr_layer == 0
    assert ov._valid_lens.tolist() == [98, 38, 7, 0]
    assert torch.all(ov._used[:, 3] == 0)
    valid = [98, 38, 7]
    seqused = [100, 40, 7]
    for b in range(3):
        s_eff = min(SINK, valid[b])
        r_eff = min(RECENT, valid[b] - s_eff)
        k_max = min(ceil(seqused[b] * RATIO), valid[b])
        for layer in range(LAYERS):
            used = int(ov._used[layer, b])
            kk = used - s_eff - r_eff
            assert 0 <= kk <= k_max
            idx = ov._table[layer, b, :used].long()
            assert idx.unique().numel() == used and idx.max() < valid[b]
            reserved = set(range(s_eff)) | set(range(valid[b] - r_eff, valid[b]))
            assert reserved <= set(idx.tolist())
    assert ov.stats()["request_rounds"] == 3
    used_sel = ov._used.clone()
    assert len(calls) == LAYERS
    calls.clear()

    ov.enter_propose()
    cu = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device)
    seqused_draft = torch.tensor([99, 39, 8], dtype=torch.int32, device=device)
    for step in range(G):
        for layer in range(LAYERS):
            q = torch.randn(3, HEADS, DIM, device=device, dtype=torch.bfloat16)
            out = torch.ones_like(q)
            before = len(calls)
            ret = ov._attention(q=q, k=k, v=v, out=out, cu_seqlens_q=cu,
                                seqused_k=seqused_draft,
                                block_table=block_table[:3],
                                softmax_scale=DIM ** -0.5, max_seqlen_q=1,
                                max_seqlen_k=99, causal=True)
            if layer == 1:  # attention skipped: zero output, no kernel
                assert len(calls) == before and torch.all(ret == 0)
                continue
            call = calls[-1]
            expected = used_sel[layer, :3] + 1 + step
            assert torch.equal(call["seqused_k"], expected)
            assert call["k"].data_ptr() == ov._draft_kv._k.data_ptr()
            assert torch.equal(call["block_table"],
                               ov._draft_kv._block_table[layer, :3])
    ov.exit_propose()
    assert ov.curr_layer == 0


def test_skipped_layer_keeps_counter_aligned(harness, monkeypatch):
    ov, calls = harness
    device = torch.device("cuda")
    k, v, block_table = _caches(device)
    _verify(ov, k, v, block_table, device)
    used_sel = ov._used.clone()
    calls.clear()

    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 1, 2, 3], dtype=torch.int32,
                                     device=device),
        seq_lens=torch.tensor([99, 39, 8], dtype=torch.int32, device=device),
        block_table=block_table[:3])
    monkeypatch.setattr(
        "vllm.v1.spec_decode.sparse_attn.longspec.overrider.get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"layer": metadata}))

    ov.enter_propose()
    ov.note_skipped_layer()  # layer 0 bypassed on the first draft step
    assert ov.curr_layer == 1
    assert torch.equal(ov._used[:, :3], used_sel[:, :3] + 1)
    q = torch.randn(3, HEADS, DIM, device=device, dtype=torch.bfloat16)
    ov._attention(q=q, k=k, v=v, out=torch.zeros_like(q),
                  cu_seqlens_q=metadata.query_start_loc,
                  seqused_k=metadata.seq_lens, block_table=block_table[:3],
                  softmax_scale=DIM ** -0.5, max_seqlen_q=1, max_seqlen_k=99,
                  causal=True)  # layer 1: attn-skip
    ov._attention(q=q, k=k, v=v, out=torch.zeros_like(q),
                  cu_seqlens_q=metadata.query_start_loc,
                  seqused_k=metadata.seq_lens, block_table=block_table[:3],
                  softmax_scale=DIM ** -0.5, max_seqlen_q=1, max_seqlen_k=99,
                  causal=True)  # layer 2
    assert torch.equal(calls[-1]["seqused_k"], used_sel[2, :3] + 1)
    ov.exit_propose()
