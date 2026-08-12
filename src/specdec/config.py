"""Shared configuration dataclasses. Experiment configs compose these."""

from dataclasses import dataclass


@dataclass
class ModelSpec:
    """A plain HuggingFace causal LM."""

    name_or_path: str
    dtype: str = "bfloat16"
    device: str = "cuda"


@dataclass
class SelectorSpec:
    """A selector-retrofitted checkpoint served by the selector repo (SELECTOR_ROOT).

    topp_mass is the coverage target p of the dual rule; sink_boundary,
    recency_window, and k_min are in tokens (the selector converts to blocks).
    max_token_length sizes the selection buffers and must be >= the longest
    prompt + generation this model will see in the run.
    """

    checkpoint: str
    topp_mass: float = 0.95
    sink_boundary: int = 128
    recency_window: int = 256
    k_min: int = 64
    temperature: float = 1.0
    max_token_length: int = 66560
    dtype: str = "bfloat16"
    device: str = "cuda"
