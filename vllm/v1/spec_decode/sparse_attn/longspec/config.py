# SPDX-License-Identifier: Apache-2.0
"""Typed view of the coverage knobs on the speculative config.

The pydantic fields carry the per-field ranges; this view carries the
cross-field rules: skip masks index real layers, no layer is in both masks,
and whole-layer skips need eager execution because the compiled layer loop
bakes Python conditionals in at trace time.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CoverageConfig:
    theta: float
    sink: int
    recent: int
    min_tokens: int
    ratio: float
    num_spec_tokens: int
    num_layers: int
    skip_attn_layers: frozenset[int]
    skip_layers: frozenset[int]

    @classmethod
    def from_vllm_config(cls, vllm_config) -> "CoverageConfig":
        spec = vllm_config.speculative_config
        model = vllm_config.model_config
        num_layers = model.get_num_layers(vllm_config.parallel_config)
        skip_attn = frozenset(spec.sparse_attn_skip_attn_layers)
        skip = frozenset(spec.sparse_attn_skip_layers)
        _check_layers("sparse_attn_skip_attn_layers", skip_attn, num_layers)
        _check_layers("sparse_attn_skip_layers", skip, num_layers)
        both = skip_attn & skip
        if both:
            raise ValueError(
                "sparse_attn_skip_attn_layers and sparse_attn_skip_layers "
                f"overlap on layers {sorted(both)}")
        if skip and not model.enforce_eager:
            raise ValueError(
                "sparse_attn_skip_layers needs enforce_eager=True: the "
                "compiled layer loop cannot skip layers at runtime")
        return cls(
            theta=spec.sparse_attn_coverage,
            sink=spec.sparse_attn_sink,
            recent=spec.sparse_attn_recent,
            min_tokens=spec.sparse_attn_min_tokens,
            ratio=spec.sparse_attn_ratio,
            num_spec_tokens=spec.num_speculative_tokens,
            num_layers=num_layers,
            skip_attn_layers=skip_attn,
            skip_layers=skip,
        )


def _check_layers(field: str, layers: frozenset[int], num_layers: int) -> None:
    bad = sorted(l for l in layers if l < 0 or l >= num_layers)
    if bad:
        raise ValueError(
            f"{field} has layers {bad} outside [0, {num_layers})")
