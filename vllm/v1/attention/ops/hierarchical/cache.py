# SPDX-License-Identifier: Apache-2.0
"""Shared four-bit coarse/residual planes and a bounded BF16 recent tail."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PageLayout:
    kv_heads: int
    block_size: int = 16
    head_dim: int = 128
    recent_capacity: int = 263

    def __post_init__(self):
        if (
            self.kv_heads < 1
            or self.block_size != 16
            or self.head_dim != 128
            or self.recent_capacity != 263
        ):
            raise ValueError(
                "hierarchical KV requires pages=16, head_dim=128, drafts=6"
            )

    @property
    def plane_bytes(self):
        return 16 * self.kv_heads * 64

    @property
    def page_bytes(self):
        # KC, KR, VC, VR, then BF16 K scale/minimum and V scale/minimum.
        return 4 * self.plane_bytes + 4 * 16 * self.kv_heads * 2

    def vllm_spec(self):
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        return FullAttentionSpec(
            block_size=16,
            num_kv_heads=self.kv_heads,
            head_size=128,
            dtype=torch.uint8,
            page_size_padded=self.page_bytes,
        )


class PagedHierarchicalKV:
    def __init__(self, pages, kv_heads, max_requests):
        self.layout = PageLayout(kv_heads)
        if max_requests < 1:
            raise ValueError("positive request capacity required")
        self.kv_heads = kv_heads
        self.device = pages.device
        self.max_requests = max_requests
        self.bind_pages(pages)
        self.tail = [
            torch.empty(
                (max_requests, 263, kv_heads, 128),
                dtype=torch.bfloat16,
                device=self.device,
            )
            for _ in range(2)
        ]
        # length, physical prefix, committed length, encoded prefix.
        self.state = torch.zeros(
            (max_requests, 4), dtype=torch.int32, device=self.device
        )
        self.error = torch.zeros(max_requests, dtype=torch.int32, device=self.device)

    def bind_pages(self, pages):
        """Bind the native vLLM allocation without replacing profiled tail buffers."""
        if (
            pages.ndim != 2
            or pages.dtype != torch.uint8
            or pages.shape[1] != self.layout.page_bytes
            or pages.stride(1) != 1
            or pages.stride(0) < self.layout.page_bytes
            or pages.stride(0) % 2
            or pages.storage_offset() % 2
            or pages.device != self.device
        ):
            raise ValueError(
                "hierarchical pages require disjoint, BF16-aligned byte rows"
            )
        self.pages = pages
        metadata = pages[:, 4 * self.layout.plane_bytes :].view(torch.bfloat16)
        # Each head's sixteen entries are contiguous. K metadata spans the
        # eight physical pages belonging to one logical 128-token group.
        self.metadata = metadata.view(-1, 4, self.kv_heads, 16).transpose(2, 3)

    @classmethod
    def allocate(cls, num_pages, kv_heads, max_requests, device="cpu"):
        pages = torch.empty(
            (num_pages, PageLayout(kv_heads).page_bytes),
            dtype=torch.uint8,
            device=device,
        )
        return cls(pages, kv_heads, max_requests)
