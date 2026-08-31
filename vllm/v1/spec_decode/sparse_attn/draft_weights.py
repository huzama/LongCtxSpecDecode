# SPDX-License-Identifier: Apache-2.0
"""Load a separately quantized weight copy for the draft forward.

The draft stack must issue exactly the target's attention calls: KV cache
binding, layer names and the overrider's call-order bookkeeping all key on
the target's Attention instances. Those instances are grafted into the
loaded copy, and its own attention registrations are removed before KV
sizing, so the copy contributes only decoder projections and norms.
Embeddings and lm_head are shared with the target; both are unquantized
in W4A16 checkpoints of the same base model.
"""

from dataclasses import replace

import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

_PREFIX = "sparse_attn_draft"


def load_draft_model(vllm_config: VllmConfig, target: nn.Module,
                     checkpoint: str) -> nn.Module:
    from vllm.compilation.backends import set_model_tag
    from vllm.model_executor.model_loader import get_model

    # quantization=None re-resolves from the checkpoint's own config;
    # quant_config=None makes VllmConfig recompute it for this copy.
    model_config = replace(vllm_config.model_config, model=checkpoint,
                           quantization=None)
    draft_config = replace(vllm_config, model_config=model_config,
                           quant_config=None)
    registry = vllm_config.compilation_config.static_forward_context
    before = set(registry)
    with set_model_tag(_PREFIX):
        model = get_model(vllm_config=draft_config, prefix=_PREFIX)
    # The copy's attention layers must reach neither KV sizing nor the
    # runtime layer lookup; the target's instances replace them below.
    for name in set(registry) - before:
        del registry[name]

    target_layers = list(target.model.layers)
    draft_layers = list(model.model.layers)
    if len(target_layers) != len(draft_layers):
        raise ValueError(
            f"sparse_attn_draft_weights: {checkpoint} has "
            f"{len(draft_layers)} layers, the target has "
            f"{len(target_layers)}; the architectures must match")
    for field in ("hidden_size", "vocab_size", "num_attention_heads",
                  "num_key_value_heads"):
        target_value = getattr(vllm_config.model_config.hf_config, field)
        draft_value = getattr(model_config.hf_config, field)
        if target_value != draft_value:
            raise ValueError(
                f"sparse_attn_draft_weights: {checkpoint} has {field}="
                f"{draft_value}, the target has {target_value}; sharing "
                "attention, embeddings and lm_head needs equal shapes")
    for target_layer, draft_layer in zip(target_layers, draft_layers):
        draft_layer.self_attn.attn = target_layer.self_attn.attn
    model.model.embed_tokens = target.model.embed_tokens
    model.lm_head = target.lm_head
    logger.info(
        "Draft weights from %s (quantization %s); attention, embeddings "
        "and lm_head shared with the target", checkpoint,
        model_config.quantization)
    return model
