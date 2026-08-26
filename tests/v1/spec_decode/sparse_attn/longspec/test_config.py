# SPDX-License-Identifier: Apache-2.0
"""Coverage knobs: field ranges on the fork's config, cross-field rules on
the typed view."""
from types import SimpleNamespace

import pytest

from vllm.config.speculative import SpeculativeConfig
from vllm.v1.spec_decode.sparse_attn.longspec.config import CoverageConfig


def _spec(**kwargs) -> SpeculativeConfig:
    return SpeculativeConfig(method="sparse_attn", num_speculative_tokens=6,
                             sparse_attn_algorithm="coverage", **kwargs)


def _vllm_config(spec, num_layers=4, enforce_eager=False):
    model = SimpleNamespace(get_num_layers=lambda _: num_layers,
                            enforce_eager=enforce_eager)
    return SimpleNamespace(speculative_config=spec, model_config=model,
                           parallel_config=None)


def test_defaults_and_bounds():
    spec = _spec()
    assert (spec.sparse_attn_coverage, spec.sparse_attn_sink,
            spec.sparse_attn_recent) == (0.9, 4, 64)
    assert _spec(sparse_attn_ratio=1.0).sparse_attn_ratio == 1.0
    assert _spec(sparse_attn_min_tokens=0).sparse_attn_min_tokens == 0
    with pytest.raises(ValueError):
        _spec(sparse_attn_coverage=0.0)
    with pytest.raises(ValueError):
        _spec(sparse_attn_coverage=1.5)


def test_typed_view():
    spec = _spec(sparse_attn_skip_attn_layers=[1], sparse_attn_skip_layers=[2],
                 sparse_attn_min_tokens=0)
    cfg = CoverageConfig.from_vllm_config(_vllm_config(spec, enforce_eager=True))
    assert cfg.theta == 0.9 and cfg.min_tokens == 0 and cfg.num_layers == 4
    assert cfg.skip_attn_layers == {1} and cfg.skip_layers == {2}
    assert cfg.num_spec_tokens == 6


def test_mask_validation():
    with pytest.raises(ValueError, match="sparse_attn_skip_attn_layers"):
        CoverageConfig.from_vllm_config(
            _vllm_config(_spec(sparse_attn_skip_attn_layers=[4])))
    with pytest.raises(ValueError, match="overlap"):
        CoverageConfig.from_vllm_config(_vllm_config(
            _spec(sparse_attn_skip_attn_layers=[1], sparse_attn_skip_layers=[1]),
            enforce_eager=True))
    with pytest.raises(ValueError, match="enforce_eager"):
        CoverageConfig.from_vllm_config(
            _vllm_config(_spec(sparse_attn_skip_layers=[1])))
