"""Dispatch must distinguish real work and reserve capture-safe scratch."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from highlight_spec import attention_policy as policy


def test_draft_and_verifier_have_independent_bounds():
    layouts = policy.attention_layouts(66048, True, True, 4, .07, .5)
    assert layouts[True, 1, True] == (4, 4608)
    assert layouts[False, 1, True] == layouts[False, 7, True] == (8, 33024)
    dense = policy.attention_layouts(66048, True, True, 8, .07, 0.)
    assert dense[True, 1, True] == (8, 4608)
    assert dense[False, 7, False] == (8, 66048)
    direct = policy.attention_layouts(66048, True, False, 4, .07, 0.)
    assert direct == {(False, q, True): (8, 4608) for q in (1, 7)}
    full = policy.attention_layouts(66048, False, True, 4, .07, 0.)
    assert full[True, 1, False] == (4, 66048)


def test_calibration_is_specific_to_workload(monkeypatch):
    args = (19, 8, 32, 1, 66048, 'hierarchical8', False, 148)
    key = policy.workload(*args)
    monkeypatch.setattr(policy, 'calibrated', lambda: {key: 8})
    assert policy.choose_splits(*args) == 8
    # Adjacent lengths inside one bucket share a captured workload.
    assert policy.choose_splits(19,8,32,1,66049,'hierarchical8',False,148) == 8
    for offset,value in ((0,9),(3,7),(4,32768),(5,'hierarchical4'),(6,True),(7,132)):
        other = list(args);other[offset] = value
        assert policy.workload(*other) != key
        assert policy.choose_splits(*other) != 8


def test_stale_calibration_is_rejected(monkeypatch):
    from highlight_spec import attention_tuning
    monkeypatch.setattr(attention_tuning, 'KERNEL_SHA256', {'hierarchical_native.py': 'wrong'})
    policy.calibrated.cache_clear()
    with pytest.raises(RuntimeError, match='stale'):
        policy.calibrated()
    policy.calibrated.cache_clear()


@pytest.mark.parametrize('queries', [1, 7])
def test_arena_handles_different_batch_split_products(queries):
    from highlight_spec import hierarchical_native as native
    empty = torch.empty
    def allocate(*args, **kwargs):
        kwargs['device'] = 'cpu'
        return empty(*args, **kwargs)
    def choose(n, *unused): return {1:8, 2:4, 3:1, 4:2, 5:4}[n]
    with patch.object(native, '_check_runtime'), patch.object(torch.cuda,'get_device_capability',return_value=(10,0)), \
            patch.object(torch.cuda,'get_device_properties',return_value=SimpleNamespace(multi_processor_count=148)), \
            patch.object(torch,'empty',side_effect=allocate), patch.object(policy,'choose_splits',side_effect=choose):
        base = native.NativeAttentionWorkspace(5,8,32,queries,32768,'cpu',cache_format='hierarchical8')
    for n in (5,2,1,3,4):
        ws = base.for_requests(n)
        assert ws.shape[0] == n and ws.splits == choose(n)
        assert ws.partial.data_ptr() == base.partial.data_ptr()
        if ws.splits > 1:
            needed = ws.splits*n*ws.views*queries*32
            assert ws.partial.numel() >= needed*128 and ws.lse.numel() >= needed
    with pytest.raises(ValueError): base.for_requests(6)


def test_fp8_explicit_splits_are_preserved(monkeypatch):
    from highlight_spec.native_fp8_attention import flash_attn_varlen_func
    from vllm.vllm_flash_attn.cute import interface
    q = torch.empty(2,32,128,dtype=torch.float8_e4m3fn)
    k = torch.empty(1,16,8,128,dtype=torch.float8_e4m3fn)
    calls = []
    def launch(**kwargs): calls.append(kwargs);return (q,None,None,None)
    monkeypatch.setattr(interface,'_flash_attn_fwd',launch)
    monkeypatch.setattr(policy,'choose_splits',lambda *args:8)
    for explicit,expected in ((0,8),(1,1),(4,4)):
        flash_attn_varlen_func(q=q,k=k,v=k,cu_seqlens_q=torch.arange(3),
            max_seqlen_q=1,max_seqlen_k=65536,fa_version=4,num_splits=explicit,
            split_config=(4608,148))
        assert calls[-1]['num_splits'] == expected
        assert calls[-1]['max_seqlen_k'] == 65536


def test_serving_context_routes_all_method_stages(monkeypatch):
    from highlight_spec import hierarchical_backend as backend
    from highlight_spec.hierarchical_config import METHODS, engine_options
    def factory(n,kvh,qh,q,length,device,**kwargs):
        return SimpleNamespace(shape=(n,kvh,qh,q,length),**kwargs)
    monkeypatch.setattr(backend,'attention_workspace',factory)
    for name,method in METHODS.items():
        options=engine_options(name,batch=19,context=65536,generation=512,
            checkpoint='unused',rope=None,scheduler=None,attention_backend='fa4')
        config=SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=19),
            device_config=SimpleNamespace(device='cpu'),
            model_config=SimpleNamespace(max_model_len=66048,hf_config=SimpleNamespace(**options['hf_overrides'])),
            attention_config=SimpleNamespace(flash_attn_version=4),speculative_config=options['speculative_config'])
        c=backend.HierarchicalContext(config,8,32,object() if method.sparse_draft else None)
        for drafting in ([False,True] if method.speculative else [False]):
            sparse=c.sparse_decode(drafting)
            for q in ([1] if drafting else [1,7]):
                ws=c.workspace(2,q,sparse,drafting)
                expected_fraction=(.07 if drafting or not method.speculative else method.verify_fraction) if sparse else 1.
                assert ws.shape == (2,8,32,q,policy.token_bound(66048,expected_fraction))
                assert ws.cache_format == f'hierarchical{method.draft_bits if drafting else 8}'
