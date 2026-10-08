# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# Track whether upstream flash-attn is available on ROCm.
# Set during module initialization and never modified afterwards.
# This module-level flag avoids repeated import attempts and ensures
# consistent behavior (similar to IS_AITER_FOUND in _aiter_ops.py).
_ROCM_FLASH_ATTN_AVAILABLE = False

if current_platform.is_cuda():
    from vllm._custom_ops import reshape_and_cache_flash
    from vllm.vllm_flash_attn import (  # type: ignore[attr-defined]
        flash_attn_varlen_func as _flash_attn_varlen_func,
    )
    from vllm.vllm_flash_attn import (
        get_scheduler_metadata,
    )

    def flash_attn_varlen_func(*args: Any, **kwargs: Any):
        if kwargs.get("fa_version", 2) == 4:
            return _flash_attn4_varlen_func(*args, **kwargs)
        return _flash_attn_varlen_func(*args, **kwargs)

elif current_platform.is_xpu():
    from vllm import _custom_ops as ops
    from vllm._xpu_ops import xpu_ops

    reshape_and_cache_flash = ops.reshape_and_cache_flash
    flash_attn_varlen_func = xpu_ops.flash_attn_varlen_func  # type: ignore[assignment]
    get_scheduler_metadata = xpu_ops.get_scheduler_metadata  # type: ignore[assignment]
elif current_platform.is_rocm():
    try:
        from flash_attn import flash_attn_varlen_func  # type: ignore[no-redef]

        # Mark that upstream flash-attn is available on ROCm
        _ROCM_FLASH_ATTN_AVAILABLE = True
    except ImportError:

        def flash_attn_varlen_func(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef,misc]
            raise ImportError(
                "ROCm platform requires upstream flash-attn "
                "to be installed. Please install flash-attn first."
            )

    # ROCm doesn't use scheduler metadata (FA3 feature), provide stub
    def get_scheduler_metadata(*args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        return None

    # ROCm uses the C++ custom op for reshape_and_cache
    from vllm import _custom_ops as ops

    reshape_and_cache_flash = ops.reshape_and_cache_flash


def _flash_attn4_varlen_func(
    q, k, v, max_seqlen_q, cu_seqlens_q, max_seqlen_k, **kwargs
):
    """Adapt FA4 paged attention and LSE to the vLLM inference interface."""
    from flash_attn.cute.interface import _flash_attn_fwd

    unsupported = {
        "dropout_p": 0.0,
        "return_attn_probs": False,
        "alibi_slopes": None,
        "scheduler_metadata": None,
        "cp_world_size": 1,
        "cp_rank": 0,
        "cp_tot_seqused_k": None,
    }
    for name, default in unsupported.items():
        value = kwargs.pop(name, default)
        supported = value is None if default is None else value == default
        if not supported:
            raise NotImplementedError(f"FA4 adapter does not support {name}")
    kwargs.pop("fa_version", 4)
    # This flag controls backward; inference has no backward pass.
    kwargs.pop("deterministic", False)
    return_lse = kwargs.pop("return_softmax_lse", False)
    window = kwargs.pop("window_size", None) or (-1, -1)
    if len(window) != 2:
        raise ValueError("window_size must contain two bounds")
    mapped = {
        "cu_seqlens_k": kwargs.pop("cu_seqlens_k", None),
        "seqused_k": kwargs.pop("seqused_k", None),
        "qv": kwargs.pop("q_v", None),
        "page_table": kwargs.pop("block_table", None),
        "softmax_scale": kwargs.pop("softmax_scale", None),
        "causal": kwargs.pop("causal", False),
        "softcap": kwargs.pop("softcap", 0.0),
        "learnable_sink": kwargs.pop("s_aux", None),
        "num_splits": kwargs.pop("num_splits", 0),
        "out": kwargs.pop("out", None),
        "q_descale": kwargs.pop("q_descale", None),
        "k_descale": kwargs.pop("k_descale", None),
        "v_descale": kwargs.pop("v_descale", None),
    }
    # The vLLM backend supplies descales even for BF16. FA4 accepts them only
    # for FP8 inputs; as with FA2, they have no role for unquantized attention.
    for name, tensor in (("q_descale", q), ("k_descale", k), ("v_descale", v)):
        if tensor.dtype in (torch.float16, torch.bfloat16):
            mapped[name] = None
    if kwargs:
        raise TypeError(f"Unsupported FA4 arguments: {sorted(kwargs)}")
    result = _flash_attn_fwd(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        window_size_left=None if window[0] < 0 else window[0],
        window_size_right=None if window[1] < 0 else window[1],
        return_lse=return_lse,
        **mapped,
    )
    out, lse = result[:2]
    return (out, lse) if return_lse else out


def get_flash_attn_version(requires_alibi: bool = False) -> int | None:
    # import here to avoid circular dependencies
    from vllm.platforms import current_platform

    if current_platform.is_xpu():
        return 2
    if current_platform.is_rocm():
        # ROCm doesn't use vllm_flash_attn; return None to skip fa_version arg
        return None
    if not torch.cuda.is_available():
        return None
    # FA4 is installed separately. Do not ask the vendored FA2/FA3 wheel
    # whether it supports version 4; an unmodified wheel rejects that value.
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    if config is not None and config.attention_config.flash_attn_version == 4:
        capability = current_platform.get_device_capability()
        if capability is None or capability.major != 10 or requires_alibi:
            raise ValueError("FA4 requires Blackwell and does not support ALiBi here")
        try:
            from flash_attn.cute.interface import _flash_attn_fwd  # noqa: F401
        except ImportError as exc:
            raise ImportError("FA4 requires flash-attn-4==4.0.0b33") from exc
        return 4
    try:
        from vllm.vllm_flash_attn.flash_attn_interface import (
            fa_version_unsupported_reason,
            is_fa_version_supported,
        )

        device_capability = current_platform.get_device_capability()

        assert device_capability is not None

        # 1. default version depending on platform
        fa_version = (
            3 if (device_capability.major == 9 and is_fa_version_supported(3)) else 2
        )

        # 2. override if passed by environment or config
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
        if (
            vllm_config is not None
            and vllm_config.attention_config.flash_attn_version is not None
        ):
            fa_version = vllm_config.attention_config.flash_attn_version

        # 3. fallback for unsupported combinations
        if device_capability.major == 10 and fa_version == 3:
            logger.warning_once(
                "Cannot use FA version 3 on Blackwell platform, "
                "defaulting to FA version 2."
            )
            fa_version = 2

        if requires_alibi and fa_version == 3:
            logger.warning_once(
                "Cannot use FA version 3 with ALiBi, defaulting to FA version 2."
            )
            fa_version = 2

        if not is_fa_version_supported(fa_version):
            logger.error(
                "Cannot use FA version %d is not supported due to %s",
                fa_version,
                fa_version_unsupported_reason(fa_version),
            )

        assert is_fa_version_supported(fa_version)
        return fa_version
    except (ImportError, AssertionError):
        return None


def flash_attn_supports_fp8() -> bool:
    return (
        get_flash_attn_version() == 3
        and current_platform.is_device_capability_family(90)
    )


def flash_attn_supports_sinks() -> bool:
    if current_platform.is_xpu():
        return True
    else:
        return get_flash_attn_version() == 3


def flash_attn_supports_mla():
    from vllm.platforms import current_platform

    if current_platform.is_cuda():
        try:
            from vllm.vllm_flash_attn.flash_attn_interface import (
                is_fa_version_supported,
            )

            return is_fa_version_supported(
                3
            ) and current_platform.is_device_capability_family(90)
        except (ImportError, AssertionError):
            pass
    return False


def is_flash_attn_varlen_func_available() -> bool:
    """Check if flash_attn_varlen_func is available.

    This function determines whether the flash_attn_varlen_func imported at module
    level is a working implementation or a stub.

    Platform-specific sources:
    - CUDA: vllm.vllm_flash_attn.flash_attn_varlen_func
    - XPU: xpu_ops.flash_attn_varlen_func
    - ROCm: upstream flash_attn.flash_attn_varlen_func (if available)

    Note: This is separate from the AITER flash attention backend (rocm_aiter_fa.py)
    which uses rocm_aiter_ops.flash_attn_varlen_func. The condition to use AITER is
    handled separately via _aiter_ops.is_aiter_found_and_supported().

    Returns:
        bool: True if a working flash_attn_varlen_func implementation is available.
    """
    if current_platform.is_cuda() or current_platform.is_xpu():
        # CUDA and XPU always have flash_attn_varlen_func available
        return True

    if current_platform.is_rocm():
        # Use the flag set during module import to check if
        # upstream flash-attn was successfully imported
        return _ROCM_FLASH_ATTN_AVAILABLE

    return False
