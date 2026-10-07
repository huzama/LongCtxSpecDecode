# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Method defaults and rejected configuration values."""

import pytest

from vllm.config.speculative import SpeculativeConfig


def config(algorithm="coverage", **kwargs):
    return SpeculativeConfig(
        method="sparse_attn",
        num_speculative_tokens=6,
        sparse_attn_algorithm=algorithm,
        **kwargs,
    )


@pytest.mark.parametrize(
    "algorithm,ratio,floor",
    [("coverage", 0.15, 0), ("vegas", 0.05, 256), ("streamingllm", 0.05, 256)],
)
def test_defaults(algorithm, ratio, floor):
    spec = config(algorithm)
    assert (spec.sparse_attn_ratio, spec.sparse_attn_min_tokens) == (ratio, floor)
    assert (spec.sparse_attn_theta, spec.sparse_attn_sink, spec.sparse_attn_recent) == (
        0.85,
        4,
        64,
    )
    assert spec.sparse_attn_collect_stats is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("sparse_attn_theta", 0),
        ("sparse_attn_theta", 1.5),
        ("sparse_attn_ratio", 0),
        ("sparse_attn_ratio", 1.5),
        ("sparse_attn_min_tokens", -1),
        ("sparse_attn_algorithm", "longspec"),
    ],
)
def test_invalid(field, value):
    with pytest.raises(ValueError):
        SpeculativeConfig(
            method="sparse_attn", num_speculative_tokens=6, **{field: value}
        )


def test_explicit_budget():
    spec = config(sparse_attn_ratio=1, sparse_attn_min_tokens=17)
    assert spec.sparse_attn_ratio == 1 and spec.sparse_attn_min_tokens == 17


def test_fp8_cache_rejected_before_initialization():
    from types import SimpleNamespace

    from vllm.v1.spec_decode.sparse_attn.attn_overrider import build_attention_overrider

    cfg = SimpleNamespace(
        speculative_config=config(), cache_config=SimpleNamespace(cache_dtype="fp8")
    )
    with pytest.raises(ValueError, match="non-FP8"):
        build_attention_overrider(cfg, None)
