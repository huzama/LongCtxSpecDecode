# SPDX-License-Identifier: Apache-2.0
"""Static whole-layer skip for the draft pass.

Wraps the forward of the masked decoder layers so that, inside a propose,
the layer returns its inputs unchanged (the fused add-norm convention makes
that the identity). The verify pass and the profile run, which carries no
attention metadata, run the original. Every skip tells the overrider, whose
layer counter is derived from attention call order. Eager execution only:
the compiled layer loop bakes the condition in at trace time.
"""

from collections.abc import Iterable

import torch.nn as nn

from vllm.forward_context import get_forward_context


def install_layer_skip(model: nn.Module, layers: Iterable[int],
                       overrider) -> None:
    """``model`` is the causal LM; its decoder stack is ``model.model``."""
    stack = model.model
    start, end = stack.start_layer, stack.end_layer
    for index in sorted(layers):
        if not start <= index < end:
            raise ValueError(
                f"sparse_attn_skip_layers: layer {index} is not on this rank "
                f"(layers [{start}, {end}))")
        layer = stack.layers[index]
        layer.forward = _skipping_forward(layer.forward, overrider)


def _skipping_forward(original, overrider):

    def forward(positions, hidden_states, residual):
        if overrider.in_propose and _has_attention_metadata():
            overrider.note_skipped_layer()
            return hidden_states, residual
        return original(positions, hidden_states, residual)

    return forward


def _has_attention_metadata() -> bool:
    try:
        return get_forward_context().attn_metadata is not None
    except AssertionError:
        return False
