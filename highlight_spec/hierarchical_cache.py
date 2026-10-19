"""Shared coarse/refinement KV storage, independent of attention block selection.

The numerical scheme follows QuantSpec: asymmetric four-bit coarse codes and
signed four-bit residuals at 1/16 of the coarse scale. Keys group across 128
positions; values group across 128 channels. BF16 scales/minima are stored once.
The combined view is an eight-bit approximation, not ordinary INT8 or FP8.

This module also provides the deliberately slow, independent attention oracle.
Materialized tensors from ``read`` are diagnostic temporaries, never a second
persistent backbone cache. Production kernels must read the packed planes.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CacheConfig:
    group_size: int = 128
    selection_block_size: int = 16
    draft_tokens: int = 6

    def __post_init__(self):
        if self.group_size != 128 or self.selection_block_size != 16:
            raise ValueError("initial implementation fixes groups at 128 and selection blocks at 16")
        if not 1 <= self.draft_tokens < self.group_size:
            raise ValueError("draft length must fit inside one quantization group")


def pack_nibbles(codes):
    if codes.dtype != torch.uint8 or codes.shape[-1] % 2:
        raise ValueError("packing requires uint8 codes with an even final dimension")
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


def unpack_nibbles(packed):
    if packed.dtype != torch.uint8:
        raise ValueError("packed codes must be uint8")
    return torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)


def quantize_groups(x, axis):
    """Reference quantizer. Reduction axis describes one complete group."""
    values = x.float()
    minimum = values.amin(axis, keepdim=True).to(torch.bfloat16)
    scale = ((values.amax(axis, keepdim=True) - minimum.float()) / 15).to(torch.bfloat16)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    coarse = ((values - minimum.float()) / scale.float()).round().clamp(0, 15)
    residual = ((values - (coarse * scale.float() + minimum.float())) *
                (16 / scale.float())).round().clamp(-8, 7)
    return (coarse.to(torch.uint8), (residual + 8).to(torch.uint8),
            scale.squeeze(axis), minimum.squeeze(axis))


class HierarchicalKV:
    """One layer, one request; persistent prefix planes plus a bounded BF16 tail.

    All tensors use [position, KV head, channel] order. ``append`` stages
    provisional values. ``commit`` is called only after target verification;
    ``crop`` rolls back provisional entries. Compressed groups are immutable.
    This explicit transaction boundary prevents rejected drafts entering groups.
    """
    def __init__(self, capacity, kv_heads, head_dim, *, device="cpu", config=None, encoder="reference"):
        self.config = config or CacheConfig()
        g = self.config.group_size
        if capacity < 1 or kv_heads < 1 or head_dim < g or head_dim % g:
            raise ValueError("positive capacity/heads and head dimension divisible by 128 required")
        self.capacity, self.kv_heads, self.head_dim = capacity, kv_heads, head_dim
        self.device = torch.device(device)
        if encoder not in ("reference", "triton") or (encoder == "triton" and self.device.type != "cuda"):
            raise ValueError("invalid cache encoder/device")
        self.encoder = encoder
        groups = (capacity + g - 1) // g
        shape = (groups, g, kv_heads, head_dim // 2)
        self.coarse = [torch.empty(shape, dtype=torch.uint8, device=device) for _ in range(2)]
        self.refinement = [torch.empty(shape, dtype=torch.uint8, device=device) for _ in range(2)]
        kmeta = (groups, kv_heads, head_dim)
        vmeta = (groups, g, kv_heads, head_dim // g)
        self.scales = [torch.empty(s, dtype=torch.bfloat16, device=device) for s in (kmeta, vmeta)]
        self.minima = [torch.empty_like(s) for s in self.scales]
        tail_shape = (2 * g + self.config.draft_tokens + 1, kv_heads, head_dim)
        self.tail = [torch.empty(tail_shape, dtype=torch.bfloat16, device=device) for _ in range(2)]
        self.device = self.tail[0].device
        self.length = self.committed = self.prefix = self.encoded_prefix = 0
        self.state = torch.zeros(4, dtype=torch.int32, device=device)

    def _sync_state(self):
        # Stable device storage lets attention graph replays see buffer rollover.
        self.state.copy_(torch.tensor([self.length, self.prefix, self.committed, self.encoded_prefix],
                                     dtype=torch.int32, device=self.device))

    def _validate(self, key, value):
        if (key.ndim != 3 or key.shape != value.shape or
                key.shape[1:] != (self.kv_heads, self.head_dim) or
                key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16 or
                key.device != self.tail[0].device or value.device != key.device):
            raise ValueError("K/V must be matching BF16 [tokens, KV heads, channels] tensors")

    @torch.no_grad()
    def _encode(self, key, value, first_group):
        g = self.config.group_size
        count = key.shape[0] // g
        if key.shape[0] % g or count == 0:
            raise ValueError("encode requires complete nonempty groups")
        if self.encoder == "triton":
            from .hierarchical_kernels import encode
            encode(self, key, value, first_group)
            return
        k = key.reshape(count, g, self.kv_heads, self.head_dim)
        v = value.reshape(count, g, self.kv_heads, self.head_dim // g, g)
        dest = slice(first_group, first_group + count)
        for i, (x, axis) in enumerate(((k, 1), (v, 4))):
            c, r, scale, minimum = quantize_groups(x, axis)
            c, r = (a.reshape(count, g, self.kv_heads, self.head_dim) for a in (c, r))
            self.coarse[i][dest].copy_(pack_nibbles(c))
            self.refinement[i][dest].copy_(pack_nibbles(r))
            self.scales[i][dest].copy_(scale)
            self.minima[i][dest].copy_(minimum)

    @torch.no_grad()
    def prefill(self, key, value):
        self._validate(key, value)
        n, g = key.shape[0], self.config.group_size
        if self.length or not 0 < n <= self.capacity:
            raise ValueError("prefill requires an empty cache and a valid nonempty input")
        # Keep between G and 2G-1 recent tokens, matching the reference policy.
        prefix = max(0, ((n - g) // g) * g)
        if prefix:
            self._encode(key[:prefix], value[:prefix], 0)
        for target, source in zip(self.tail, (key, value)):
            target[:n - prefix].copy_(source[prefix:])
        self.prefix, self.length, self.committed, self.encoded_prefix = prefix, n, n, prefix
        self._sync_state()

    @torch.no_grad()
    def append(self, key, value):
        self._validate(key, value)
        n, offset = key.shape[0], self.length - self.prefix
        if n == 0 or self.length + n > self.capacity or offset + n > self.tail[0].shape[0]:
            raise ValueError("append exceeds cache capacity or provisional tail capacity")
        if (self.length+n > self.prefix+2*self.config.group_size and
                self.committed < self.prefix+self.config.group_size):
            raise ValueError("cannot compress a group containing provisional entries")
        for target, source in zip(self.tail, (key, value)):
            target[offset:offset + n].copy_(source)
        self.length += n
        # A fixed-size verification block may cross the next serial transition.
        # Encode only the older, already committed group; retain its BF16 copy
        # in the bounded tail until acceptance decides whether to advance.
        if self.length > self.prefix+2*self.config.group_size:
            end = self.prefix+self.config.group_size
            if self.encoded_prefix < end:
                self._encode(self.tail[0][:self.config.group_size],
                             self.tail[1][:self.config.group_size], self.prefix//self.config.group_size)
                self.encoded_prefix = end
        self._sync_state()

    def crop(self, length):
        if not self.committed <= length <= self.length:
            raise ValueError("rollback may only remove provisional entries")
        self.length = length
        self._sync_state()

    def precision_prefix(self, query_position):
        """Compressed extent seen by serial attention at this query position."""
        g = self.config.group_size
        prefix = max(self.prefix, ((query_position-g)//g)*g)
        if prefix > self.encoded_prefix:
            raise ValueError("query requires an unprepared compressed group")
        return prefix

    @torch.no_grad()
    def commit(self, length=None):
        length = self.length if length is None else length
        self.crop(length)
        g = self.config.group_size
        while self.length - self.prefix >= 2 * g:
            if self.encoded_prefix < self.prefix+g:
                self._encode(self.tail[0][:g], self.tail[1][:g], self.prefix // g)
                self.encoded_prefix = self.prefix+g
            remaining = self.length - self.prefix - g
            for t in self.tail:
                t[:remaining].copy_(t[g:g + remaining].clone())
            self.prefix += g
        self.committed = self.length
        self._sync_state()

    @torch.no_grad()
    def read(self, bits=8, positions=None, *, precision_prefix=None):
        if bits not in (4, 8):
            raise ValueError("precision must be 4 or 8")
        if positions is None:
            positions = torch.arange(self.length, device=self.device)
        if (positions.ndim != 1 or positions.dtype != torch.int64 or
                positions.device != self.tail[0].device or
                bool(((positions < 0) | (positions >= self.length)).any())):
            raise ValueError("invalid cache positions")
        precision_prefix = self.prefix if precision_prefix is None else precision_prefix
        if not self.prefix <= precision_prefix <= self.encoded_prefix or precision_prefix % self.config.group_size:
            raise ValueError("invalid precision view")
        old = positions < precision_prefix
        old_pos, tail_pos = positions[old], positions[~old] - self.prefix
        g = self.config.group_size
        group, offset = old_pos // g, old_pos % g
        result = []
        for i in range(2):
            out = torch.empty((len(positions), self.kv_heads, self.head_dim),
                              dtype=torch.bfloat16, device=self.device)
            if len(old_pos):
                coarse = unpack_nibbles(self.coarse[i][group, offset]).float()
                if i == 0:
                    scale, minimum = self.scales[i][group], self.minima[i][group]
                else:
                    scale = self.scales[i][group, offset].repeat_interleave(g, -1)
                    minimum = self.minima[i][group, offset].repeat_interleave(g, -1)
                decoded = coarse * scale.float() + minimum.float()
                if bits == 8:
                    residual = unpack_nibbles(self.refinement[i][group, offset]).float() - 8
                    decoded += residual * (scale.float() / 16)
                out[old] = decoded.to(torch.bfloat16)
            out[~old] = self.tail[i][tail_pos]
            result.append(out)
        return tuple(result)

    def memory_bytes(self):
        def size(items):
            return sum(t.numel() * t.element_size() for t in items)
        return dict(coarse=size(self.coarse), refinement=size(self.refinement),
                    metadata=size(self.scales + self.minima), recent=size(self.tail),
                    state=size([self.state]),
                    total=size(self.coarse + self.refinement + self.scales + self.minima + self.tail + [self.state]))


def reference_attention(query, key, value, query_positions, key_positions, scale=None):
    """FP32 GQA oracle with explicit absolute-position causal masking."""
    if (query.ndim != 3 or key.shape != value.shape or key.ndim != 3 or
            query.shape[-1] != key.shape[-1] or query.shape[1] % key.shape[1] or
            query_positions.numel() != query.shape[0] or key_positions.numel() != key.shape[0]):
        raise ValueError("incompatible attention dimensions")
    repeats = query.shape[1] // key.shape[1]
    k, v = (x.float().repeat_interleave(repeats, dim=1) for x in (key, value))
    scores = torch.einsum("qhd,khd->hqk", query.float(), k) * (scale or query.shape[-1] ** -.5)
    mask = key_positions[None, :] <= query_positions[:, None]
    if not bool(mask.any(-1).all()):
        raise ValueError("every query needs at least one visible key")
    scores.masked_fill_(~mask[None], -torch.inf)
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v).to(query.dtype)
