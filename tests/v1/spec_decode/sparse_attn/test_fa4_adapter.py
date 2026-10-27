# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests: FA4 adaptation must not silently drop attention options."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import vllm.config
import vllm.platforms
from vllm.config.attention import AttentionConfig
from vllm.v1.attention.backends import fa_utils
from vllm.v1.spec_decode.sparse_attn.attn_overrider.utils import vegas_fallback


@pytest.fixture
def fa4(monkeypatch):
    calls = []
    interface = ModuleType("flash_attn.cute.interface")

    def forward(**kwargs):
        calls.append(kwargs)
        return kwargs["out"], "lse", "unused"

    interface._flash_attn_fwd = forward
    monkeypatch.setitem(sys.modules, "flash_attn.cute.interface", interface)
    return calls


def arguments(dtype=torch.bfloat16):
    q = torch.empty(7, 8, 80, dtype=dtype)
    k = torch.empty(8, 16, 2, 80, dtype=dtype)
    return dict(
        q=q,
        k=k,
        v=k,
        max_seqlen_q=7,
        max_seqlen_k=128,
        cu_seqlens_q=torch.tensor([0, 7], dtype=torch.int32),
        seqused_k=torch.tensor([128], dtype=torch.int32),
        block_table=torch.arange(8, dtype=torch.int32)[None],
        out=torch.empty_like(q),
        causal=True,
        softmax_scale=80**-0.5,
        fa_version=4,
        window_size=[-1, -1],
        num_splits=3,
        q_descale=torch.ones(1, 2),
        k_descale=torch.ones(1, 2),
        v_descale=torch.ones(1, 2),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("return_lse", [False, True])
def test_fa4_mapping(fa4, dtype, return_lse):
    args = arguments(dtype)
    actual = fa_utils._flash_attn4_varlen_func(**args, return_softmax_lse=return_lse)
    assert (actual[0] if return_lse else actual) is args["out"]
    if return_lse:
        assert actual[1] == "lse"
    (call,) = fa4
    for name in ("q", "k", "v", "out", "cu_seqlens_q", "seqused_k"):
        assert call[name] is args[name]
    assert call["page_table"] is args["block_table"]
    assert call["causal"] and call["num_splits"] == 3
    assert call["softmax_scale"] == args["softmax_scale"]
    assert call["window_size_left"] is None and call["window_size_right"] is None
    assert all(call[name] is None for name in ("q_descale", "k_descale", "v_descale"))


@pytest.mark.parametrize(
    "name,value",
    [
        ("dropout_p", 0.1),
        ("return_attn_probs", True),
        ("alibi_slopes", torch.ones(8)),
        ("scheduler_metadata", torch.ones(1)),
        ("cp_world_size", 2),
        ("cp_rank", 1),
        ("cp_tot_seqused_k", torch.ones(1)),
    ],
)
def test_fa4_rejects_unsupported_options(fa4, name, value):
    with pytest.raises(NotImplementedError, match=name):
        fa_utils._flash_attn4_varlen_func(**arguments(), **{name: value})
    assert not fa4


def test_fa4_rejects_unknown_options(fa4):
    with pytest.raises(TypeError, match="unknown"):
        fa_utils._flash_attn4_varlen_func(**arguments(), unknown=True)


@pytest.mark.parametrize("page", [16, 128])
def test_fa4_head256_page_requirement(fa4, page):
    args = arguments()
    args.update(
        q=torch.empty(7, 8, 256, dtype=torch.bfloat16),
        k=torch.empty(8, page, 2, 256, dtype=torch.bfloat16),
        v=torch.empty(8, page, 2, 256, dtype=torch.bfloat16),
    )
    if page == 16:
        with pytest.raises(ValueError, match="block_size=128"):
            fa_utils._flash_attn4_varlen_func(**args)
        assert not fa4
    else:
        assert fa_utils._flash_attn4_varlen_func(**args) is args["out"]


@pytest.mark.parametrize(
    "major,alibi",
    [
        (10, False),
        (9, False),
        (12, False),
        (10, True),
    ],
)
def test_fa4_selection_bypasses_vendor_probe(monkeypatch, fa4, major, alibi):
    monkeypatch.setattr(
        vllm.platforms,
        "current_platform",
        SimpleNamespace(
            is_xpu=lambda: False,
            is_rocm=lambda: False,
            get_device_capability=lambda: SimpleNamespace(major=major),
        ),
    )
    config = SimpleNamespace(attention_config=AttentionConfig(flash_attn_version=4))
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: config)
    monkeypatch.setitem(sys.modules, "vllm.vllm_flash_attn.flash_attn_interface", None)
    if major == 10 and not alibi:
        assert fa_utils.get_flash_attn_version() == 4
    else:
        with pytest.raises(ValueError, match="FA4"):
            fa_utils.get_flash_attn_version(requires_alibi=alibi)


@pytest.mark.parametrize("version", [2, 4])
def test_only_fa3_can_expose_scores(monkeypatch, version):
    def unexpected_import(*args):
        pytest.fail("FA2/FA4 must not import the patched FA3 op")

    monkeypatch.setattr(vegas_fallback.importlib, "import_module", unexpected_import)
    assert not vegas_fallback.kernel_collects_scores(version)


@pytest.mark.parametrize("expected", [False, True])
def test_fa3_detection_reads_the_loaded_schema(monkeypatch, expected):
    # Use PyTorch's actual OpOverloadPacket, not a mock of its schema interface.
    namespace = f"vegas_test_schema_{int(expected)}"
    library = torch.library.Library(namespace, "DEF")
    library.define(
        "fwd(Tensor q, Tensor? scores=None) -> Tensor"
        if expected
        else "fwd(Tensor q, Tensor k) -> Tensor"
    )
    packet = getattr(torch.ops, namespace).fwd
    monkeypatch.setattr(vegas_fallback.importlib, "import_module", lambda _: None)
    monkeypatch.setattr(torch.ops, "_vllm_fa3_C", SimpleNamespace(fwd=packet))
    assert vegas_fallback.kernel_collects_scores(3) is expected


@pytest.mark.parametrize("version", [2, 3])
def test_vendor_dispatch_is_unchanged(monkeypatch, version):
    if not hasattr(fa_utils, "_flash_attn_varlen_func"):
        pytest.skip("CUDA dispatch is defined on CUDA platforms")
    calls = []
    sentinel = object()

    def vendor(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(fa_utils, "_flash_attn_varlen_func", vendor)
    actual = fa_utils.flash_attn_varlen_func("q", fa_version=version, scores="scores")
    assert actual is sentinel
    assert calls == [(("q",), dict(fa_version=version, scores="scores"))]


@pytest.mark.parametrize(
    "option,value",
    [
        ("causal", False),
        ("block_table", None),
        ("softcap", 1.0),
        ("alibi_slopes", torch.ones(8)),
        ("window_size", [128, 0]),
        ("s_aux", torch.ones(8)),
    ],
)
def test_vegas_rejects_attention_it_cannot_rescore(option, value):
    from vllm.v1.spec_decode.sparse_attn.attn_overrider.vegas import VegasAttnOverrider

    class Probe(VegasAttnOverrider):
        def __del__(self):
            pass  # This probe never installs the global attention hook.

    probe = Probe.__new__(Probe)
    probe._attn_score_buffer = None
    args = arguments()
    args[option] = value
    with pytest.raises(NotImplementedError, match="Vegas score recomputation"):
        probe._verify_attention(**args)


def test_vegas_rejects_fp8_before_installing_hooks():
    from vllm.v1.spec_decode.sparse_attn.attn_overrider import build_attention_overrider

    config = SimpleNamespace(
        speculative_config=SimpleNamespace(sparse_attn_algorithm="vegas"),
        cache_config=SimpleNamespace(cache_dtype="fp8_e4m3"),
    )
    with pytest.raises(ValueError, match="unquantized"):
        build_attention_overrider(config, torch.device("cpu"))
