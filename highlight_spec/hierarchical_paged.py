"""Paged coarse/refinement storage for one attention layer.

Sixteen-token physical pages preserve Highlight's selection units. Key groups
span eight logical pages; their 128 channel scales/minima are distributed over
those pages, so metadata is stored once. Both precision planes share the same
allocation. A 263-position BF16 ring per live request holds recent/provisional
KV. The ring avoids copying the recent buffer at compression boundaries.

Host-side import/read functions are independent diagnostics, not serving paths.
GPU updates use device request slots and never read lengths back to the CPU.
"""
from dataclasses import dataclass

import torch

from .hierarchical_cache import unpack_nibbles


@dataclass(frozen=True)
class PageLayout:
    kv_heads: int
    block_size: int = 16
    head_dim: int = 128
    recent_capacity: int = 263

    def __post_init__(self):
        if (self.kv_heads < 1 or self.block_size != 16 or self.head_dim != 128 or
                self.recent_capacity != 263):
            raise ValueError("paged cache requires 16-token pages, 128 channels and six drafts")

    @property
    def plane_bytes(self):
        return self.block_size * self.kv_heads * (self.head_dim // 2)

    @property
    def page_bytes(self):
        # KC, KR, VC, VR, then BF16 KS, Kmin, VS, Vmin.
        return 4 * self.plane_bytes + 4 * self.block_size * self.kv_heads * 2

    def vllm_spec(self):
        from vllm.v1.kv_cache_interface import FullAttentionSpec
        return FullAttentionSpec(block_size=16, num_kv_heads=self.kv_heads,
                                 head_size=128, dtype=torch.uint8,
                                 page_size_padded=self.page_bytes)


class PagedHierarchicalKV:
    def __init__(self, pages, kv_heads, max_requests):
        self.layout = PageLayout(kv_heads)
        if max_requests < 1:
            raise ValueError("positive request capacity required")
        self.kv_heads = kv_heads
        self.device = pages.device
        self.max_requests = max_requests
        self.bind_pages(pages)
        self.tail = [torch.empty((max_requests, 263, kv_heads, 128),
                                dtype=torch.bfloat16, device=pages.device) for _ in range(2)]
        # length, physical prefix, committed length, encoded prefix.
        self.state = torch.zeros((max_requests, 4), dtype=torch.int32, device=pages.device)
        self.error = torch.zeros(max_requests, dtype=torch.int32, device=pages.device)

    def bind_pages(self, pages):
        """Bind vLLM's allocation while retaining the already-profiled tail."""
        layout, kv_heads = self.layout, self.kv_heads
        if (pages.ndim != 2 or pages.dtype != torch.uint8 or
                pages.shape[1] != layout.page_bytes or pages.stride(1) != 1 or
                pages.stride(0) < layout.page_bytes or pages.stride(0) % 2 or
                pages.device != self.device):
            raise ValueError("invalid paged storage; pages must own aligned, disjoint byte ranges")
        self.pages = pages
        nb, stride = pages.shape[0], pages.stride(0)
        self.planes = [pages.as_strided((nb, 16, kv_heads, 64),
            (stride, kv_heads*64, 64, 1), pages.storage_offset()+i*layout.plane_bytes)
            for i in range(4)]
        metadata = pages[:, 4*layout.plane_bytes:].view(torch.bfloat16)
        # Each attention CTA consumes one KV head. Store its sixteen metadata
        # values together, retaining the diagnostic [page, kind, lane, head]
        # view and the same number of bytes per page.
        self.metadata = metadata.view(nb, 4, kv_heads, 16).transpose(2, 3)

    @classmethod
    def allocate(cls, num_pages, kv_heads, max_requests, device="cpu"):
        return cls(torch.empty((num_pages, PageLayout(kv_heads).page_bytes),
                               dtype=torch.uint8, device=device), kv_heads, max_requests)

    def memory_bytes(self):
        return dict(pages=self.pages.numel(),
                    recent=sum(t.numel()*t.element_size() for t in self.tail),
                    state=self.state.numel()*4+self.error.numel()*4)

    def _check_table(self, slot, table, length):
        count = (length+15)//16
        if (not 0 <= slot < self.max_requests or table.ndim != 1 or
                table.dtype not in (torch.int32, torch.int64) or
                table.device != self.device or table.numel() < count):
            raise ValueError("invalid request slot or page table")
        used = table[:count].long()
        if bool(((used < 0) | (used >= len(self.pages))).any()) or used.unique().numel() != count:
            raise ValueError("request pages must be distinct and in range")

    @torch.no_grad()
    def import_reference(self, slot, table, source):
        """Populate a shuffled physical table from independent reference storage."""
        self._check_table(slot, table, source.length)
        if source.kv_heads != self.kv_heads or source.head_dim != 128:
            raise ValueError("reference layout mismatch")
        n, h = source.encoded_prefix, self.kv_heads
        if n:
            physical = table[:n//16].long()
            for dest, src in zip(self.planes, (source.coarse[0], source.refinement[0],
                                              source.coarse[1], source.refinement[1])):
                dest[physical] = src[:n//128].reshape(-1, 16, h, 64)
            for i, src in enumerate((source.scales[0], source.minima[0])):
                self.metadata[physical, i] = src[:n//128].permute(0, 2, 1).reshape(-1, 16, h)
            for i, src in enumerate((source.scales[1], source.minima[1]), start=2):
                self.metadata[physical, i] = src[:n//128].reshape(-1, 16, h)
        pos = torch.arange(source.prefix, source.length, device=self.device)
        for dst, src in zip(self.tail, source.tail):
            dst[slot, pos % 263] = src[:len(pos)]
        self.state[slot].copy_(source.state)
        self.error[slot] = 0

    @torch.no_grad()
    def read(self, slot, table, bits=8, positions=None, *, precision_prefix=None):
        """Slow independent view of paged storage, used only for validation."""
        length, prefix, _, encoded = map(int, self.state[slot].tolist())
        self._check_table(slot, table, length)
        if bits not in (4, 8):
            raise ValueError("precision must be 4 or 8")
        if positions is None:
            positions = torch.arange(length, device=self.device)
        if (positions.ndim != 1 or positions.dtype != torch.int64 or
                positions.device != self.device or
                bool(((positions < 0) | (positions >= length)).any())):
            raise ValueError("invalid positions")
        view = prefix if precision_prefix is None else precision_prefix
        if view % 128 or not prefix <= view <= encoded:
            raise ValueError("unprepared precision view")
        old = positions < view
        p = positions[old]
        physical = table[p//16].long()
        offset = p % 16
        channel = torch.arange(128, device=self.device)
        # A key's channel metadata is on one of the group's eight pages.
        kp = table[(p[:, None]//128)*8+channel[None, :]//16].long()
        result = []
        for kind in range(2):
            output = torch.empty((len(positions), self.kv_heads, 128),
                                 dtype=torch.bfloat16, device=self.device)
            c = unpack_nibbles(self.planes[kind*2][physical, offset]).float()
            r = unpack_nibbles(self.planes[kind*2+1][physical, offset]).float()-8
            if kind == 0:
                scale, minimum = [self.metadata[kp, i, channel[None, :] % 16].transpose(1, 2).float()
                                  for i in (0, 1)]
            else:
                scale, minimum = [self.metadata[physical, i, offset, :, None].float() for i in (2, 3)]
            values = c*scale+minimum
            if bits == 8:
                values += r*(scale/16)
            output[old] = values.bfloat16()
            output[~old] = self.tail[kind][slot, positions[~old] % 263]
            result.append(output)
        return tuple(result)

    def update(self, key, value, query_positions, query_start_loc, slots, committed, block_table):
        """Commit/crop the previous target result, then stage this call's KV.

        ``committed`` is fixed across all six draft steps, and comes from the
        verifier's accepted length. During verification it is the anchor's
        position. Padded batch rows have slot=-1 and perform no writes.
        """
        from .hierarchical_paged_kernels import update
        return update(self, key, value, query_positions, query_start_loc, slots, committed, block_table)
