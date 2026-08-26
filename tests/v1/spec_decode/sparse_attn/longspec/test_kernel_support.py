# SPDX-License-Identifier: Apache-2.0
"""Capability detection must answer without a GPU and never raise."""
from vllm.v1.spec_decode.sparse_attn.longspec.portable import kernel_support


def test_detection_is_total():
    fa = kernel_support.flash_attn_version()
    assert fa in (2, 3)
    assert isinstance(kernel_support.kernel_collects_scores(fa), bool)
    assert isinstance(kernel_support.supports_token_pages(fa), bool)


def test_fa2_never_collects_scores():
    assert kernel_support.kernel_collects_scores(2) is False
    assert kernel_support.supports_token_pages(2) is False
