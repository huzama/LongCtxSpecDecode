"""Paged addressing must preserve the independent contiguous reference."""
import pytest
import torch

from highlight_spec.hierarchical_cache import HierarchicalKV
from highlight_spec.hierarchical_paged import PageLayout, PagedHierarchicalKV


@pytest.mark.parametrize("length", [127, 255, 256, 511, 1024])
def test_shuffled_pages_and_both_precision_views(length):
    torch.manual_seed(9)
    cache = PagedHierarchicalKV.allocate(160, 2, 3)
    pages = torch.randperm(160).reshape(2, 80)
    for slot in (0, 1):
        ref = HierarchicalKV(1200, 2, 128)
        kv = [torch.randn((length, 2, 128)).bfloat16()+slot for _ in range(2)]
        ref.prefill(*kv)
        if length == 255:
            ref.append(*[torch.randn((7, 2, 128)).bfloat16() for _ in range(2)])
        cache.import_reference(slot, pages[slot], ref)
        pos = torch.tensor([ref.length-1, 0, length//2, 0])
        for view in sorted({ref.prefix, ref.encoded_prefix}):
            for bits in (4, 8):
                for a, b in zip(cache.read(slot, pages[slot], bits, pos, precision_prefix=view),
                                ref.read(bits, pos, precision_prefix=view)):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
        if slot == 0:
            before = [x.clone() for x in cache.read(0, pages[0])]
    for a, b in zip(cache.read(0, pages[0]), before):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_page_padding_aliases_one_allocation():
    layout = PageLayout(8)
    assert layout.page_bytes == 33792
    assert layout.page_bytes/(2*16*8*128*2) == .515625
    raw = torch.zeros((20, layout.page_bytes+256), dtype=torch.uint8)
    pages = raw[:, 128:128+layout.page_bytes]
    cache = PagedHierarchicalKV(pages, 8, 2)
    for p in cache.planes: p.fill_(35)
    cache.metadata.fill_(1.5)
    assert not raw[:, :128].any() and not raw[:, -128:].any()
    assert cache.metadata.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
    for p in cache.planes:
        assert p.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()


def test_invalid_page_contracts():
    with pytest.raises(ValueError): PageLayout(8, block_size=128)
    with pytest.raises(ValueError): PagedHierarchicalKV(torch.empty((10, 100), dtype=torch.uint8), 8, 2)
    cache = PagedHierarchicalKV.allocate(32, 1, 1)
    ref = HierarchicalKV(256, 1, 128)
    ref.prefill(*[torch.zeros((256, 1, 128), dtype=torch.bfloat16) for _ in range(2)])
    with pytest.raises(ValueError): cache.import_reference(0, torch.zeros(16, dtype=torch.int32), ref)


def test_split_budget_tracks_occupancy_and_keeps_single_request_parallelism():
    from highlight_spec.hierarchical_paged_kernels import attention_splits
    assert attention_splits(1, 8, 1, 32768, 84) == 64
    assert attention_splits(1, 8, 7, 32768, 84) == 32
    assert attention_splits(25, 8, 7, 32768, 84) == 4
    assert attention_splits(25, 8, 1, 2304, 84) == 4
    assert attention_splits(128, 8, 7, 32768, 84) == 1
    assert attention_splits(1, 8, 7, 127, 84) == 1
    for bad in ((0,8,7,32768,84), (1,0,7,32768,84), (1,8,6,32768,84),
                (1,8,7,0,84), (1,8,7,32768,0)):
        with pytest.raises(ValueError): attention_splits(*bad)
