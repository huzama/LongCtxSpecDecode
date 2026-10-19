"""Native loader conversion and cache-transaction regressions on SM100."""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason='SM100 GPU allocation required')


def test_native_packed_conversion_exhaustive():
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from highlight_spec.hierarchical_native_loader import signed_codes

    @cute.kernel
    def convert(src: cute.Tensor, dst: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        vals = cute.make_rmem_tensor((8,), cutlass.BFloat16)
        vals.store(signed_codes(src[tid]))
        for i in cutlass.range_constexpr(8):
            dst[tid, i] = vals[i]

    @cute.jit
    def launch(src: cute.Tensor, dst: cute.Tensor, stream):
        convert(src, dst).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    packed = torch.arange(256, device='cuda').to(torch.uint8)
    output = torch.empty((64, 8), device='cuda', dtype=torch.bfloat16)
    words = packed.view(torch.uint32)
    fn = cute.compile(launch, from_dlpack(words), from_dlpack(output),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options='--enable-tvm-ffi')
    fn(words, output)
    expected = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(64, 8).float()-8
    torch.testing.assert_close(output.float(), expected, rtol=0, atol=0)


def test_native_transactions_and_precision_boundaries(tmp_path):
    from benchmarks.hierarchical_cache import validate_paged
    validate_paged(tmp_path, attention_backend='fa4')


@pytest.mark.parametrize('bits,sparse,queries',[(8,False,1),(4,False,1),(8,False,7),(8,True,7)])
def test_native_large_physical_cache_addresses(bits,sparse,queries):
    """Full-capacity allocations can place BF16 metadata beyond 2**31 elements."""
    from highlight_spec.hierarchical_paged import PagedHierarchicalKV
    from highlight_spec import hierarchical_paged_kernels as reference
    from highlight_spec.hierarchical_native import NativeAttentionWorkspace,attention
    torch.manual_seed(91)
    length=511
    cache=PagedHierarchicalKV.allocate(131072,8,1,'cuda')
    blocks=(length+15)//16
    table=torch.arange(131072-blocks,131072,device='cuda',dtype=torch.int32)[None]
    assert int(table[0,0])*cache.metadata.stride(0)>2**31
    k,v=[torch.randn(length,8,128,device='cuda',dtype=torch.bfloat16) for _ in range(2)]
    reference.prefill(cache,0,table[0],k,v)
    torch.cuda.synchronize()
    q=torch.randn(queries,32,128,device='cuda',dtype=torch.bfloat16)
    positions=torch.arange(length-queries,length,device='cuda',dtype=torch.int64)
    starts=torch.tensor([0,queries],device='cuda',dtype=torch.int32)
    slots=torch.zeros(1,device='cuda',dtype=torch.int32)
    ids=torch.arange(blocks,device='cuda',dtype=torch.int32)[None] if sparse else None
    counts=torch.tensor([blocks],device='cuda',dtype=torch.int32) if sparse else None
    ws=NativeAttentionWorkspace(1,8,32,queries,length,'cuda')
    legacy=reference.PagedAttentionWorkspace(1,8,32,queries,length,'cuda')
    expected=reference.attention(cache,q,positions,starts,slots,table,bits=bits,
        selected_ids=ids,selected_counts=counts,workspace=legacy)
    torch.cuda.synchronize()
    actual=attention(cache,q,positions,starts,slots,table,bits=bits,
        selected_ids=ids,selected_counts=counts,workspace=ws)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual,expected,rtol=.025,atol=.004)
