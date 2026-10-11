"""Plugin hooks: the speculator and its lookahead slots apply to this method only."""

from types import SimpleNamespace

import pytest

from highlight_spec import SPECULATOR, register, uses_highlight


def spec(method, model):
    return SimpleNamespace(
        method=method, model=model, use_gemma4_mtp=lambda: False, use_eagle=lambda: False
    )


def test_uses_highlight():
    assert uses_highlight(spec("custom_class", SPECULATOR))
    assert not uses_highlight(spec("custom_class", "other.Proposer"))
    assert not uses_highlight(spec("eagle", SPECULATOR))
    assert not uses_highlight(None)


def patched():
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.worker.gpu import spec_decode

    return (
        VllmConfig._get_v2_model_runner_unsupported_features,
        spec_decode.init_speculator,
        Scheduler.__init__,
    )


def test_register_is_idempotent():
    register()
    first = patched()
    register()
    assert patched() == first
    assert all(getattr(f, "highlight", False) for f in first)


def test_other_methods_reach_vllm():
    from vllm.v1.worker.gpu import spec_decode

    register()
    config = SimpleNamespace(speculative_config=spec("ngram", None))
    with pytest.raises(NotImplementedError):
        spec_decode.init_speculator(config, "cpu")
