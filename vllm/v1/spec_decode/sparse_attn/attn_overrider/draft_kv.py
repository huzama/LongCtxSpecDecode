# SPDX-License-Identifier: Apache-2.0
"""How the draft pass sees the selected tokens.

Both strategies take the same per-layer slot table (one physical KV slot per
selected position) and per-request used counts, and rewrite the attention
kwargs so the paged kernel attends over exactly those tokens. Token pages need
a kernel that accepts a page size of one; the gather works with any paged
kernel by copying the selection into page-aligned scratch.
"""

import torch

from .utils.draft_gather import gather_tokens
from .utils.kernel_support import supports_token_pages


class TokenPagedDraftKV:
    """View the cache at page size 1 and hand the slot table to the kernel."""

    def begin_propose(self) -> None:
        pass

    def attention_kwargs(self, kwargs: dict, layer: int, slots: torch.Tensor,
                         used: torch.Tensor) -> None:
        k, v = kwargs["k"], kwargs["v"]
        kwargs["k"] = k.view(-1, 1, k.shape[-2], k.shape[-1])
        kwargs["v"] = v.view(-1, 1, v.shape[-2], v.shape[-1])
        kwargs["seqused_k"] = used
        kwargs["block_table"] = slots

    def scratch_bytes(self) -> int:
        return 0


class GatheredDraftKV:
    """Copy the selection into per-layer page-aligned scratch, incrementally.

    The first draft step of a propose gathers each layer's whole selection;
    every later step copies only the newest token, whose KV the model wrote
    into the cache just before attention. Scratch is per layer because each
    layer's gathered set must persist across the draft steps of one propose.
    """

    def __init__(self, num_layers: int, max_batch_size: int, width: int,
                 page: int, device: torch.device):
        self._num_layers = num_layers
        self._max_batch_size = max_batch_size
        self._page = page
        self._pages = (width + page - 1) // page
        self._device = device
        self._k = None
        self._v = None
        self._block_table = None
        self._fresh = [False] * num_layers
        self._starts = torch.zeros(max_batch_size, device=device,
                                   dtype=torch.int32)
        self._zeros = torch.zeros(max_batch_size, device=device,
                                  dtype=torch.int32)

    def _allocate(self, like: torch.Tensor) -> None:
        kv_heads, dim = like.shape[-2], like.shape[-1]
        shape = (self._num_layers, self._max_batch_size,
                 self._pages * self._page, kv_heads, dim)
        self._k = torch.zeros(*shape, dtype=like.dtype, device=like.device)
        self._v = torch.zeros_like(self._k)
        self._block_table = torch.arange(
            self._num_layers * self._max_batch_size * self._pages,
            device=like.device, dtype=torch.int32,
        ).view(self._num_layers, self._max_batch_size, self._pages)

    def begin_propose(self) -> None:
        self._fresh = [False] * self._num_layers

    def attention_kwargs(self, kwargs: dict, layer: int, slots: torch.Tensor,
                         used: torch.Tensor) -> None:
        k, v = kwargs["k"], kwargs["v"]
        if self._k is None:
            self._allocate(k)
        batch = slots.shape[0]
        if self._fresh[layer]:
            torch.sub(used, 1, out=self._starts[:batch]).clamp_(min=0)
            starts = self._starts[:batch]
        else:
            starts = self._zeros[:batch]
            self._fresh[layer] = True
        gather_tokens(k, v, slots, starts, used,
                      self._k[layer, :batch], self._v[layer, :batch])
        kv_heads, dim = k.shape[-2], k.shape[-1]
        kwargs["k"] = self._k.view(-1, self._page, kv_heads, dim)
        kwargs["v"] = self._v.view(-1, self._page, kv_heads, dim)
        kwargs["seqused_k"] = used
        kwargs["block_table"] = self._block_table[layer, :batch]
        if kwargs.get("max_seqlen_k") is not None:
            kwargs["max_seqlen_k"] = min(kwargs["max_seqlen_k"],
                                         self._pages * self._page)

    def scratch_bytes(self) -> int:
        return 0 if self._k is None else 2 * self._k.nbytes


def build_draft_kv(mode: str, fa_version: int, num_layers: int,
                   max_batch_size: int, width: int, page: int,
                   device: torch.device):
    """Resolve the configured mode against what the loaded kernel accepts."""
    available = supports_token_pages(fa_version)
    if mode == "token_pages" and not available:
        raise ValueError(
            "sparse_attn_draft_kv='token_pages' needs a kernel that accepts "
            f"page size 1 (FA3); the loaded kernel is FA{fa_version}. Use "
            "'gather' or 'auto'.")
    if mode == "gather" or (mode == "auto" and not available):
        return GatheredDraftKV(num_layers, max_batch_size, width, page, device)
    return TokenPagedDraftKV()
