"""Explicit opt-in compiler and GPU checks; ordinary CPU tests never claim a GPU."""
import os

import pytest
import torch

from highlight_spec.selectors import vegas_reference_scores
from highlight_spec.vegas_fa4 import FusedVegasAttention, _backend


def test_private_launcher_does_not_patch_installed_runtime():
    pytest.importorskip("vllm.vllm_flash_attn.cute.interface")
    from vllm.vllm_flash_attn.cute import interface
    original = interface._flash_attn_fwd
    original_class = interface.FlashAttentionForwardSm100
    launcher, _ = _backend()
    assert interface._flash_attn_fwd is original
    assert interface.FlashAttentionForwardSm100 is original_class
    assert launcher is not original
    assert launcher.compile_cache is not original.compile_cache
    assert launcher.__globals__["FlashAttentionForwardSm100"] is not original_class


@pytest.mark.skipif(os.environ.get("SELECTOR_COMPILE_FA4") != "1", reason="explicit CPU compilation opt-in")
@pytest.mark.parametrize("qlen,heads,kvheads", [(1, 32, 8), (7, 32, 8), (139, 4, 4)])
def test_compile_fused_collector_without_gpu(qlen, heads, kvheads):
    from vllm.vllm_flash_attn.cute.interface import _make_compile_only_tensor_spec as spec
    launch, marker = _backend()
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode():
        launch(spec((1, qlen, heads, 128), torch.bfloat16),
               spec((1, 259, kvheads, 128), torch.bfloat16),
               spec((1, 259, kvheads, 128), torch.bfloat16),
               out=spec((1, qlen, heads, 128), torch.bfloat16),
               causal=True, num_splits=1, _arch=100,
               score_mod=marker,
               aux_tensors=[spec((1, 2, heads, 259 - qlen + 1), torch.bfloat16), spec((1, 2), torch.int32, 4)],
               compile_only=True)


@pytest.mark.skipif(os.environ.get("SELECTOR_TEST_GPU") != "1", reason="requires an idle authorized GPU")
@pytest.mark.parametrize("phase,qlen,kvheads", [("verify", 1, 1), ("verify", 7, 2), ("prefill", 139, 4)])
def test_fused_scores_attention_and_replay(phase, qlen, kvheads):
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"2", "3"}:
        pytest.fail("select exactly physical GPU 2 or 3 before this test")
    from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd
    torch.manual_seed(31)
    q = torch.randn(2, qlen, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 259, kvheads, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    prefix = k.shape[1] if phase == "prefill" else k.shape[1] - qlen + 1
    fused = FusedVegasAttention(q, k, v, prefix_tokens=prefix, phase=phase)
    expected_out = _flash_attn_fwd(q, k, v, causal=True, num_splits=1)[0]
    expected_scores = vegas_reference_scores(q, k, prefix_tokens=prefix, phase=phase)
    # Refill with NaN to expose missing writes and index wrapping/padding races.
    for _ in range(3):
        fused.scores.fill_(float("nan"))
        actual = fused.attention().clone()
        torch.testing.assert_close(actual, expected_out, atol=0.02, rtol=0.02)
        torch.testing.assert_close(fused.reduce(), expected_scores, atol=0.03, rtol=0.01)
