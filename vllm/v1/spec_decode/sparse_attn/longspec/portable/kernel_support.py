# SPDX-License-Identifier: Apache-2.0
"""Capability detection for the attention kernels the sparse overriders use.

Verification-guided drafting needs two things from the paged attention kernel:
the attention scores of two query rows per request, and a page table that can
address single tokens. The reference build gets both from a patched
FlashAttention-3, which exists only on sm90. The functions here answer, for
the binary actually loaded, whether each feature is present, so the overrider
can choose a strategy instead of assuming one.
"""

import importlib
from functools import cache

import torch

from vllm.v1.attention.backends.fa_utils import get_flash_attn_version


def _op_argument_names(module: str, op: str) -> list[str]:
    try:
        importlib.import_module(f"vllm.vllm_flash_attn.{module}")
        schema = getattr(getattr(torch.ops, module), op)._schema
        return [arg.name for arg in schema.arguments]
    except Exception:
        return []


@cache
def flash_attn_version() -> int:
    """FlashAttention major version the backend uses on this platform."""
    return get_flash_attn_version() or 2


@cache
def kernel_collects_scores(fa_version: int) -> bool:
    """Whether the loaded kernel op writes per-token QK scores into a buffer.

    Only the FA3 op of the vegas flash-attention fork does, and the Python
    interface forwards the scores argument to FA3 alone, so FA2 builds always
    answer no even if their kernel were patched.
    """
    if fa_version != 3:
        return False
    return "scores" in _op_argument_names("_vllm_fa3_C", "fwd")


@cache
def supports_token_pages(fa_version: int) -> bool:
    """Whether paged attention accepts a page size of one token. FA3 only."""
    return fa_version == 3
