"""Plain HuggingFace model loading and layer accounting."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from specdec.config import ModelSpec


def resolve_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"dtype={name!r} is not a torch.dtype attribute")
    return dtype


def load_causal_lm(spec: ModelSpec):
    tokenizer = AutoTokenizer.from_pretrained(spec.name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        spec.name_or_path, dtype=resolve_dtype(spec.dtype)
    )
    model = model.to(spec.device).eval()
    return model, tokenizer


def num_layers(model) -> int:
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", None) or cfg
    return int(text_cfg.num_hidden_layers)


def exit_layers_from_fracs(n_layers: int, fracs: list[float]) -> list[int]:
    """Fractions of depth to exit-layer counts, deduplicated, clamped to [1, L]."""
    layers = sorted({max(1, min(n_layers, round(f * n_layers))) for f in fracs})
    return layers
