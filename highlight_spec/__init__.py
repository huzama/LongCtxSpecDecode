"""vLLM plugin: lossless self-speculative decoding drafted by Highlight selection.

Loaded in every vLLM process through the ``vllm.general_plugins`` entry point,
so ``register`` stays idempotent and light. An engine opts in with
``speculative_config={"method": "custom_class", "model": SPECULATOR,
"num_speculative_tokens": k}``; any other configuration is untouched.
"""

from __future__ import annotations

import sys

SPECULATOR = "highlight_spec.speculator.HighlightSpeculator"


def uses_highlight(speculative_config) -> bool:
    return (
        speculative_config is not None
        and speculative_config.method == "custom_class"
        and speculative_config.model == SPECULATOR
    )


def register() -> None:
    _admit_v2_runner()
    _register_speculator()
    _reserve_draft_slots()
    from vllm.model_executor.models import ModelRegistry
    if "NativeFP8Qwen3ForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("NativeFP8Qwen3ForCausalLM",
            "highlight_spec.native_fp8:NativeFP8Qwen3ForCausalLM")
    if "HierarchicalQwen3ForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("HierarchicalQwen3ForCausalLM",
            "highlight_spec.hierarchical_backend:HierarchicalQwen3ForCausalLM")


def _admit_v2_runner() -> None:
    """The V2 runner allowlists speculative methods; admit this speculator only."""
    from vllm.config import VllmConfig

    base = VllmConfig._get_v2_model_runner_unsupported_features
    if getattr(base, "highlight", False):
        return

    def unsupported_features(self) -> list[str]:
        unsupported = base(self)
        spec = self.speculative_config
        if uses_highlight(spec):
            unsupported.remove(f"speculative method '{spec.method}'")
        return unsupported

    unsupported_features.highlight = True
    VllmConfig._get_v2_model_runner_unsupported_features = unsupported_features


def _register_speculator() -> None:
    """The V2 runner builds speculators by method and has no custom_class branch."""
    from vllm.v1.worker.gpu import spec_decode

    base = spec_decode.init_speculator
    if getattr(base, "highlight", False):
        return

    def init_speculator(vllm_config, device):
        if uses_highlight(vllm_config.speculative_config):
            from highlight_spec.speculator import HighlightSpeculator

            return HighlightSpeculator(vllm_config, device)
        return base(vllm_config, device)

    init_speculator.highlight = True
    spec_decode.init_speculator = init_speculator
    # The runner binds the name at import; rebind it if that already happened.
    runner = sys.modules.get("vllm.v1.worker.gpu.model_runner")
    if runner is not None:
        runner.init_speculator = init_speculator


def _reserve_draft_slots() -> None:
    """Drafts write KV up to k-1 positions past the verified tokens.

    The scheduler reserves those slots only for speculators it knows; without
    them a draft write lands in a block the request does not own.
    """
    from vllm.v1.core.sched.scheduler import Scheduler

    base = Scheduler.__init__
    if getattr(base, "highlight", False):
        return

    def __init__(self, vllm_config, *args, **kwargs):
        base(self, vllm_config, *args, **kwargs)
        if uses_highlight(vllm_config.speculative_config):
            self.num_lookahead_tokens = self.num_spec_tokens

    __init__.highlight = True
    Scheduler.__init__ = __init__
