# SPDX-License-Identifier: Apache-2.0
"""Load a separately quantized weight copy for the draft forward.

The draft stack must issue exactly the target's attention calls: KV cache
binding, layer names and the overrider's call-order bookkeeping all key on
the target's Attention instances. Those instances are grafted into the
loaded copy, and its own attention registrations are removed before KV
sizing, so the copy contributes only decoder projections and norms.
Embeddings and lm_head are shared with the target; both are unquantized
in the supported INT4 and NVFP4 checkpoints of the same base model.
"""

from dataclasses import replace
from typing import Literal

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

_PREFIX = "sparse_attn_draft"


def _share_target_modules(target: nn.Module, draft: nn.Module, *,
                          ffn_only: bool, scope: str = "ffn") -> None:
    """Preserve target modules; replace only the draft's references."""
    for target_layer, draft_layer in zip(target.model.layers, draft.model.layers):
        if ffn_only:
            if scope == "gate_up":
                draft_layer.mlp.down_proj = target_layer.mlp.down_proj
            elif scope == "down":
                draft_layer.mlp.gate_up_proj = target_layer.mlp.gate_up_proj
            draft_layer.self_attn = target_layer.self_attn
            draft_layer.input_layernorm = target_layer.input_layernorm
            draft_layer.post_attention_layernorm = target_layer.post_attention_layernorm
        else:
            draft_layer.self_attn.attn = target_layer.self_attn.attn
    draft.model.embed_tokens = target.model.embed_tokens
    draft.lm_head = target.lm_head
    if ffn_only:
        draft.model.norm = target.model.norm


def _validate_int4_ffns(model: nn.Module) -> None:
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        CompressedTensorsWNA16,
    )

    for index, layer in enumerate(model.model.layers):
        for name in ("gate_up_proj", "down_proj"):
            scheme = getattr(getattr(layer.mlp, name), "scheme", None)
            if (
                not isinstance(scheme, CompressedTensorsWNA16)
                or scheme.pack_factor != 8
            ):
                raise ValueError(
                    f"FFN-only drafting requires compressed-tensors INT4 W4A16; "
                    f"layer {index} {name} is {type(scheme).__name__}")


def load_draft_model(vllm_config: VllmConfig, target: nn.Module,
                     checkpoint: str, *,
                     scope: Literal["all", "ffn", "gate_up", "down"] = "all",
                     ) -> nn.Module:
    from vllm.compilation.backends import set_model_tag
    from vllm.model_executor.model_loader import get_model

    if scope not in ("all", "ffn", "gate_up", "down"):
        raise ValueError(f"Unknown draft weights scope: {scope}")
    ffn_only = scope != "all"
    if ffn_only and (
        vllm_config.model_config.dtype != torch.bfloat16
        or vllm_config.model_config.quantization is not None
        or vllm_config.cache_config.cache_dtype not in ("auto", "bfloat16")
        or vllm_config.model_config.hf_config.model_type != "qwen3"
    ):
        raise ValueError("FFN-only drafting requires a BF16 Qwen3 target and BF16 KV")
    # The target's repository commit cannot resolve in a different checkpoint.
    # For reproducible draft weights, pass a pinned local snapshot path.
    # quantization=None re-resolves from the checkpoint's own config;
    # quant_config=None makes VllmConfig recompute it for this copy.
    model_config = replace(vllm_config.model_config, model=checkpoint,
                           quantization=None, revision=None, code_revision=None)
    draft_config = replace(vllm_config, model_config=model_config,
                           quant_config=None)
    registry = vllm_config.compilation_config.static_forward_context
    before = set(registry)
    try:
        with set_model_tag(_PREFIX):
            model = get_model(vllm_config=draft_config, prefix=_PREFIX)
    finally:
        # Only target attention instances participate in KV sizing and lookup.
        for name in set(registry) - before:
            del registry[name]

    target_layers = list(target.model.layers)
    draft_layers = list(model.model.layers)
    if len(target_layers) != len(draft_layers):
        raise ValueError(
            f"sparse_attn_draft_weights: {checkpoint} has "
            f"{len(draft_layers)} layers, the target has "
            f"{len(target_layers)}; the architectures must match")
    for field in ("hidden_size", "intermediate_size", "vocab_size",
                  "num_attention_heads", "num_key_value_heads", "hidden_act"):
        target_value = getattr(vllm_config.model_config.hf_config, field)
        draft_value = getattr(model_config.hf_config, field)
        if target_value != draft_value:
            raise ValueError(
                f"sparse_attn_draft_weights: {checkpoint} has {field}="
                f"{draft_value}, the target has {target_value}; sharing "
                "attention, embeddings and lm_head needs equal shapes")
    if ffn_only:
        _validate_int4_ffns(model)
    _share_target_modules(target, model, ffn_only=ffn_only, scope=scope)
    logger.info(
        "Draft weights from %s (quantization %s, scope %s); %s, embeddings "
        "and lm_head shared with the target", checkpoint,
        model_config.quantization, scope,
        "attention projections, attention and norms" if ffn_only else "attention")
    return model
