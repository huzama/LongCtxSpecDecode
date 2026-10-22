"""Capture/replay and compression-boundary checks for calibrated dispatch."""
from unittest.mock import patch

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10,0), reason='B200 required')


def test_multi_split_precision_transitions(tmp_path):
    from benchmarks.hierarchical_cache import validate_paged
    with patch('highlight_spec.attention_policy.choose_splits',return_value=4):
        validate_paged(tmp_path, attention_backend='fa4')


@pytest.mark.parametrize('bits,queries', [(4,1),(8,1),(8,7)])
def test_smaller_batches_share_arena_and_replay(bits,queries):
    from highlight_spec.hierarchical_native import NativeAttentionWorkspace, attention
    from highlight_spec.hierarchical_paged import PagedHierarchicalKV
    from highlight_spec.hierarchical_paged_kernels import prefill
    from highlight_spec.hierarchical_cache import reference_attention
    b,length=3,1023
    cache=PagedHierarchicalKV.allocate(192,8,b,'cuda')
    table=torch.arange(192,device='cuda',dtype=torch.int32).view(b,64)
    torch.manual_seed(17)
    for slot in range(b):
        kv=[torch.randn(length,8,128,device='cuda',dtype=torch.bfloat16) for _ in range(2)]
        prefill(cache,slot,table[slot],*kv)
    with patch('highlight_spec.attention_policy.choose_splits',side_effect=lambda n,*args: {1:8,2:4,3:2}[n]):
        base=NativeAttentionWorkspace(b,8,32,queries,length,'cuda',cache_format=f'hierarchical{bits}')
    for n in (3,1,2,3):
        ws=base.for_requests(n)
        q=torch.randn(n*queries,32,128,device='cuda',dtype=torch.bfloat16)
        qp=torch.arange(length-queries,length,device='cuda').repeat(n)
        qsl=torch.arange(0,(n+1)*queries,queries,device='cuda',dtype=torch.int32)
        slots=torch.arange(n,device='cuda',dtype=torch.int32)
        out=torch.empty_like(q)
        def call(): return attention(cache,q,qp,qsl,slots,table[:n],bits=bits,workspace=ws,output=out)
        for _ in range(3):call()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):call()
        q.normal_();graph.replay();replay=out.clone();call()
        torch.testing.assert_close(out,replay,rtol=0,atol=0)
        chosen=torch.arange(length,device='cuda')
        expected=reference_attention(q[:queries],*cache.read(0,table[0],bits,chosen),qp[:queries],chosen)
        torch.testing.assert_close(out[:queries],expected,rtol=.025,atol=.004)
