"""GPU regressions for packed loads and short/long paged attention shapes.

Run through Slurm. The independent FP32 oracle defines numerical tolerances;
graph replay must exactly match the same compiled eager kernel.
"""
import copy

import pytest
import torch
import triton as tr
import triton.language as tl

from highlight_spec.hierarchical_cache import reference_attention
from highlight_spec.hierarchical_paged import PagedHierarchicalKV
from highlight_spec.hierarchical_paged_kernels import (
    attention_workspace, _unpack_codes, attention, prefill,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU allocation required")


@pytest.fixture(params=['triton', 'fa4'])
def workspace_factory(request):
    from functools import partial
    if request.param == 'fa4' and torch.cuda.get_device_capability() != (10, 0):
        pytest.skip('native hierarchical attention requires SM100')
    return partial(attention_workspace, backend=request.param)


@tr.jit
def _unpack_test(SRC, DST, CONTIGUOUS: tl.constexpr):
    channel = tl.arange(0, 64) if CONTIGUOUS else tl.arange(0, 128)//2
    off = tl.arange(0, 4)[:, None]*64+channel[None, :]
    values = _unpack_codes(SRC, off, True, CONTIGUOUS)
    tl.store(DST+tl.arange(0, 4)[:, None]*128+tl.arange(0, 128)[None, :], values)


@pytest.mark.parametrize('contiguous', [True, False])
def test_vectorized_unpack_exhaustive(contiguous):
    packed = torch.arange(256, device="cuda").to(torch.uint8).reshape(4, 64)
    got = torch.empty((4, 128), device="cuda", dtype=torch.float32)
    _unpack_test[(1,)](packed, got, contiguous, num_warps=4)
    expected = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).float()
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_attention_rejects_invalid_device_metadata_before_launch(workspace_factory):
    cache = PagedHierarchicalKV.allocate(16, 8, 1, "cuda")
    ws = workspace_factory(1, 8, 32, 1, 128, "cuda")
    q = torch.zeros((1, 32, 128), dtype=torch.bfloat16, device="cuda")
    qp = torch.zeros(1, device="cuda", dtype=torch.int64)
    qsl = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    slots = torch.zeros(1, device="cuda", dtype=torch.int32)
    table = torch.arange(16, device="cuda", dtype=torch.int32)[None]
    for bad_qp in (qp.cpu(), qp.float()):
        with pytest.raises(ValueError, match="metadata"):
            attention(cache, q, bad_qp, qsl, slots, table, workspace=ws)
    with pytest.raises(ValueError, match="metadata"):
        attention(cache, q, qp, qsl, slots, table[:, ::2], workspace=ws)
    with pytest.raises(ValueError, match="output"):
        attention(cache, q, qp, qsl, slots, table, workspace=ws, output=torch.empty_like(q, device="cpu"))


def test_single_query_cannot_see_future_cache_entries(workspace_factory):
    torch.manual_seed(17)
    cache = PagedHierarchicalKV.allocate(16, 8, 1, 'cuda')
    table = torch.arange(16, device='cuda', dtype=torch.int32)[None]
    kv = [torch.randn((129, 8, 128), device='cuda', dtype=torch.bfloat16) for _ in range(2)]
    prefill(cache, 0, table[0], *kv)
    q = torch.randn((1, 32, 128), device='cuda', dtype=torch.bfloat16)
    qp = torch.tensor([63], device='cuda')
    qsl = torch.tensor([0, 1], device='cuda', dtype=torch.int32)
    slots = torch.zeros(1, device='cuda', dtype=torch.int32)
    ws = workspace_factory(1, 8, 32, 1, 129, 'cuda')
    got = attention(cache, q, qp, qsl, slots, table, workspace=ws)
    positions = torch.arange(129, device='cuda')
    expected = reference_attention(q, *kv, qp, positions)
    torch.testing.assert_close(got, expected, rtol=.025, atol=.004)


@pytest.mark.parametrize("length,hq,nq,bits,sparse", [
    (1, 8, 1, 4, False), (1, 8, 1, 8, True),
    (127, 16, 7, 8, False), (255, 32, 7, 8, True),
    (256, 32, 1, 4, True), (257, 64, 7, 8, False),
    (4097, 32, 1, 8, False), (4097, 64, 7, 8, True),
    (4097, 32, 1, 4, True),
])
def test_shapes_against_independent_reference_and_graph(workspace_factory, length, hq, nq, bits, sparse):
    torch.manual_seed(15)
    batch, h, blocks = 2, 8, tr.cdiv(length, 16)
    # Padded page strides exercise the allocator contract as well as nibble loads.
    page_bytes = 4*16*h*64+4*16*h*2
    raw = torch.zeros((batch*blocks, page_bytes+256), device="cuda", dtype=torch.uint8)
    cache = PagedHierarchicalKV(raw[:, 128:128+page_bytes], h, batch)
    table = torch.randperm(batch*blocks, device="cuda", dtype=torch.int32).reshape(batch, blocks)
    for slot in range(batch):
        kv = [torch.randn((length, h, 128), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        prefill(cache, slot, table[slot], *kv)
    selected = torch.arange(0, blocks, 3, device="cuda", dtype=torch.int32)
    selected = torch.unique(torch.cat((selected, torch.arange(max(0,blocks-4), blocks,
                                                           device="cuda", dtype=torch.int32))))
    # The native logical mask must also support unsorted selected pages.
    selected = selected[torch.randperm(len(selected), device='cuda')]
    ids = selected[None].repeat(batch, 1)
    counts = torch.full((batch,), len(selected), device="cuda", dtype=torch.int32)
    q = torch.randn((batch*nq, hq, 128), device="cuda", dtype=torch.bfloat16)
    qp = torch.arange(length-nq, length, device="cuda").repeat(batch)
    qsl = torch.arange(0, (batch+1)*nq, nq, device="cuda", dtype=torch.int32)
    slots = torch.arange(batch, device="cuda", dtype=torch.int32)
    positions = ((selected.long()[:, None]*16+torch.arange(16, device="cuda")).flatten()
                 if sparse else torch.arange(length, device="cuda"))
    positions = positions[positions < length]
    ws = workspace_factory(batch, h, hq, nq, len(selected)*16 if sparse else length, "cuda")
    assert ws.views == (1 if nq==1 else 2)
    output = torch.empty_like(q)
    def call():
        return attention(cache, q, qp, qsl, slots, table, bits=bits, workspace=ws, output=output,
                         selected_ids=ids if sparse else None, selected_counts=counts if sparse else None)
    for _ in range(3): call()
    expected_eager = output.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): call()
    output.fill_(float("nan"))
    graph.replay()
    torch.testing.assert_close(output, expected_eager, rtol=0, atol=0)
    for slot in range(batch):
        sl = slice(slot*nq, (slot+1)*nq)
        kv = cache.read(slot, table[slot], bits, positions)
        expected = reference_attention(q[sl], *kv, qp[sl], positions)
        torch.testing.assert_close(output[sl], expected, rtol=.025, atol=.004)
    # The serving adapter copies a maximum-capacity workspace for smaller batches.
    smaller = copy.copy(ws)
    smaller.shape = (1, *ws.shape[1:])
    got = attention(cache, q[:nq], qp[:nq], qsl[:2], slots[:1], table[:1], bits=bits,
                    workspace=smaller, selected_ids=ids[:1] if sparse else None,
                    selected_counts=counts[:1] if sparse else None)
    torch.testing.assert_close(got, expected_eager[:nq], rtol=0, atol=0)
    assert not raw[:, :128].any() and not raw[:, -128:].any()
