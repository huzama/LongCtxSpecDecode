"""Forward FP8 descales missing from vLLM 0.27.1's FA4 Python bridge.

The upstream CuTe kernel already implements these scales. This adapter changes
only argument forwarding, not quantization, cache storage or attention math.
"""
import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func as _native_varlen


def flash_attn_varlen_func(*args, **kwargs):
    q = kwargs.get("q", args[0] if args else None)
    if kwargs.get("fa_version") != 4 or q.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        return _native_varlen(*args, **kwargs)
    if args:
        raise ValueError("native FP8 adapter expects named attention arguments")
    from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd
    values = dict(kwargs)
    values.pop("fa_version")
    if values.pop("alibi_slopes", None) is not None:
        raise ValueError("FA4 does not support ALiBi")
    if values.pop("scheduler_metadata", None) is not None:
        raise ValueError("FA4 does not consume FA3 scheduler metadata")
    window = values.pop("window_size", None) or (-1, -1)
    values["window_size_left"] = window[0] if window[0] >= 0 else None
    values["window_size_right"] = window[1] if window[1] >= 0 else None
    values["page_table"] = values.pop("block_table", None)
    values["learnable_sink"] = values.pop("s_aux", None)
    return_lse = values.pop("return_softmax_lse", False)
    values["return_lse"] = return_lse
    out, lse, _, _ = _flash_attn_fwd(**values)
    return (out, lse) if return_lse else out


def install():
    # Only the explicitly selected native FP8 model calls this, in its worker.
    # BF16 and other FA versions continue through the original dispatcher.
    from vllm.v1.attention.backends import flash_attn
    flash_attn.flash_attn_varlen_func = flash_attn_varlen_func
