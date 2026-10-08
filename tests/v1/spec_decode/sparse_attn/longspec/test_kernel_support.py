# SPDX-License-Identifier: Apache-2.0
"""Capability detection must answer without a GPU and never raise."""

from vllm.v1.spec_decode.sparse_attn.longspec.portable import kernel_support


def test_detection_is_total():
    fa = kernel_support.flash_attn_version()
    assert fa in (2, 3, 4)
    assert isinstance(kernel_support.kernel_collects_scores(fa), bool)
    assert isinstance(kernel_support.supports_token_pages(fa), bool)


def test_fa2_never_collects_scores():
    assert kernel_support.kernel_collects_scores(2) is False
    assert kernel_support.supports_token_pages(2) is False


def test_fa4_uses_recomputed_scores_and_gathered_cache():
    assert kernel_support.kernel_collects_scores(4) is False
    assert kernel_support.supports_token_pages(4) is False


def test_flash_version_is_not_cached_across_engine_configs(monkeypatch):
    monkeypatch.setattr(kernel_support, "get_flash_attn_version", lambda: 2)
    assert kernel_support.flash_attn_version() == 2
    monkeypatch.setattr(kernel_support, "get_flash_attn_version", lambda: 4)
    assert kernel_support.flash_attn_version() == 4


def test_fa4_detection_does_not_call_vendor_version_gate(monkeypatch):
    import sys
    from types import ModuleType
    from types import SimpleNamespace as NS

    import vllm.config
    import vllm.platforms
    from vllm.v1.attention.backends import fa_utils

    monkeypatch.setattr(fa_utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        vllm.platforms,
        "current_platform",
        NS(
            is_xpu=lambda: False,
            is_rocm=lambda: False,
            get_device_capability=lambda: NS(major=10),
        ),
    )
    monkeypatch.setattr(
        vllm.config,
        "get_current_vllm_config_or_none",
        lambda: NS(attention_config=NS(flash_attn_version=4)),
    )
    interface = ModuleType("flash_attn.cute.interface")
    interface._flash_attn_fwd = lambda: None
    monkeypatch.setitem(sys.modules, "flash_attn.cute.interface", interface)
    # No vendor module is needed at all in this branch.
    monkeypatch.setitem(sys.modules, "vllm.vllm_flash_attn.flash_attn_interface", None)
    assert fa_utils.get_flash_attn_version() == 4
