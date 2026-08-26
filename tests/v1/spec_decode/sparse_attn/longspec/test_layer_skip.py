# SPDX-License-Identifier: Apache-2.0
"""Whole-layer skip bypasses masked layers only inside a propose with
attention metadata, and reports every skip to the overrider."""
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.spec_decode.sparse_attn.longspec import layer_skip


class _Layer:

    def __init__(self):
        self.calls = 0

    def forward(self, positions, hidden_states, residual):
        self.calls += 1
        return hidden_states + 1, residual


class _Overrider:

    def __init__(self, in_propose):
        self.in_propose = in_propose
        self.skipped = 0

    def note_skipped_layer(self):
        self.skipped += 1


def _model(num_layers=3):
    stack = SimpleNamespace(start_layer=0, end_layer=num_layers,
                            layers=[_Layer() for _ in range(num_layers)])
    return SimpleNamespace(model=stack)


def _run(model, index):
    h, r = torch.zeros(2), torch.ones(2)
    return model.model.layers[index].forward(None, h, r)


def _context(monkeypatch, metadata):
    monkeypatch.setattr(layer_skip, "get_forward_context",
                        lambda: SimpleNamespace(attn_metadata=metadata))


def test_skip_inside_propose(monkeypatch):
    _context(monkeypatch, {"layer": object()})
    model, ov = _model(), _Overrider(in_propose=True)
    layer_skip.install_layer_skip(model, [1], ov)
    h, r = _run(model, 1)
    assert torch.equal(h, torch.zeros(2)) and torch.equal(r, torch.ones(2))
    assert ov.skipped == 1 and model.model.layers[1].calls == 0
    h, _ = _run(model, 0)
    assert torch.equal(h, torch.ones(2)) and ov.skipped == 1


def test_original_outside_propose_or_without_metadata(monkeypatch):
    _context(monkeypatch, {"layer": object()})
    model, ov = _model(), _Overrider(in_propose=False)
    layer_skip.install_layer_skip(model, [2], ov)
    h, _ = _run(model, 2)
    assert torch.equal(h, torch.ones(2)) and ov.skipped == 0
    ov.in_propose = True
    _context(monkeypatch, None)
    h, _ = _run(model, 2)
    assert torch.equal(h, torch.ones(2)) and ov.skipped == 0

    def raising():
        raise AssertionError("no forward context")

    monkeypatch.setattr(layer_skip, "get_forward_context", raising)
    h, _ = _run(model, 2)
    assert torch.equal(h, torch.ones(2)) and ov.skipped == 0


def test_rejects_layers_off_this_rank():
    with pytest.raises(ValueError, match="sparse_attn_skip_layers"):
        layer_skip.install_layer_skip(_model(), [3], _Overrider(True))
