# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vegas scoring and selected KV access without the patched FA3 kernel."""

import importlib

import torch
import triton
import triton.language as tl


def kernel_collects_scores(fa_version: int) -> bool:
    if fa_version != 3:
        return False
    try:
        importlib.import_module("vllm.vllm_flash_attn._vllm_fa3_C")
        schema = torch.ops._vllm_fa3_C.fwd.default._schema
    except (ImportError, AttributeError):
        return False
    return any(arg.name == "scores" for arg in schema.arguments)


@triton.jit
def _score_kernel(
    Q,
    K,
    BT,
    CU,
    VALID,
    ENTRY,
    LSE,
    OUT,
    sqt,
    sqh,
    skt,
    skh,
    sbt,
    slh,
    slt,
    sout,
    scale,
    KVH: tl.constexpr,
    G: tl.constexpr,
    GP: tl.constexpr,
    D: tl.constexpr,
    DP: tl.constexpr,
    PAGE: tl.constexpr,
    TILE: tl.constexpr,
    WEIGHT: tl.constexpr,
):
    b, tile = tl.program_id(0), tl.program_id(1)
    valid = tl.load(VALID + b)
    if tile * TILE >= valid:
        return
    entry = tl.load(ENTRY + b)
    first, last = tl.load(CU + b), tl.load(CU + b + 1) - 1
    t = tile * TILE + tl.arange(0, TILE)
    page = tl.load(BT + b * sbt + t // PAGE, t < valid, other=0)
    slots = page * PAGE + t % PAGE
    d, g = tl.arange(0, DP), tl.arange(0, GP)
    acc = tl.zeros((TILE,), tl.float32)
    for kh in tl.static_range(KVH):
        keys = tl.load(
            K + slots[:, None] * skt + kh * skh + d[None, :],
            (t < valid)[:, None] & (d < D)[None, :],
            other=0.0,
        )
        heads = kh * G + g
        for row in tl.static_range(2):
            active = (g < G) & ((entry != 2) if row == 0 else (entry != 1))
            token = first if row == 0 else last
            query = tl.load(
                Q + token * sqt + heads[:, None] * sqh + d[None, :],
                active[:, None] & (d < D)[None, :],
                other=0.0,
            )
            scores = tl.dot(query, tl.trans(keys))
            if WEIGHT:
                lse = tl.load(
                    LSE + heads * slh + token * slt, active, other=float("inf")
                )
                scores = tl.exp(scores * scale - lse[:, None])
            acc += tl.sum(tl.where(active[:, None], scores, 0.0), axis=0)
    count = (1 + (entry == 0).to(tl.int32)) * KVH * G
    tl.store(OUT + b * sout + t, acc / count, t < valid)


def score_tokens(q, k, block_table, cu, valid, entry, lse, scale, use_weight, out):
    """Average first/last-query QK or attention weights over all query heads.

    LSE comes from full verification, including keys beyond the scored prefix.
    Zero-length padded requests never dereference their block-table entries.
    """
    assert k.is_contiguous() and q.stride(-1) == 1
    assert q.shape[1] % k.shape[2] == 0
    group = q.shape[1] // k.shape[2]
    _score_kernel[(out.shape[0], triton.cdiv(out.shape[1], 64))](
        q,
        k,
        block_table,
        cu,
        valid,
        entry,
        lse,
        out,
        q.stride(0),
        q.stride(1),
        k.stride(1),
        k.stride(2),
        block_table.stride(0),
        lse.stride(0),
        lse.stride(-1),
        out.stride(0),
        scale,
        KVH=k.shape[2],
        G=group,
        GP=max(16, triton.next_power_of_2(group)),
        D=q.shape[-1],
        DP=max(16, triton.next_power_of_2(q.shape[-1])),
        PAGE=k.shape[1],
        TILE=64,
        WEIGHT=use_weight,
    )


@triton.jit
def _gather_kernel(
    K,
    V,
    SLOTS,
    START,
    END,
    DK,
    DV,
    ss,
    sk,
    sdb,
    sdt,
    INNER: tl.constexpr,
    CHUNK: tl.constexpr,
):
    b, tile = tl.program_id(0), tl.program_id(1)
    start, end = tl.load(START + b), tl.load(END + b)
    if tile * 16 >= end or (tile + 1) * 16 <= start:
        return
    t = tile * 16 + tl.arange(0, 16)
    valid = (t >= start) & (t < end)
    slots = tl.load(SLOTS + b * ss + t, valid, other=0)
    for offset in tl.static_range(0, INNER, CHUNK):
        c = offset + tl.arange(0, CHUNK)
        mask = valid[:, None] & (c < INNER)[None, :]
        src = slots[:, None] * sk + c[None, :]
        dst = b * sdb + t[:, None] * sdt + c[None, :]
        tl.store(DK + dst, tl.load(K + src, mask, other=0.0), mask)
        tl.store(DV + dst, tl.load(V + src, mask, other=0.0), mask)


def gather_tokens(k, v, slots, starts, ends, dst_k, dst_v):
    """Copy physical slots into scratch only over each request's [start, end)."""
    assert k.is_contiguous() and v.is_contiguous() and k.shape == v.shape
    assert (
        dst_k.is_contiguous() and dst_v.is_contiguous() and dst_k.shape == dst_v.shape
    )
    assert k.dtype == v.dtype == dst_k.dtype == dst_v.dtype
    inner = k.shape[-2] * k.shape[-1]
    _gather_kernel[(slots.shape[0], triton.cdiv(dst_k.shape[1], 16))](
        k,
        v,
        slots,
        starts,
        ends,
        dst_k,
        dst_v,
        slots.stride(0),
        k.stride(1),
        dst_k.stride(0),
        dst_k.stride(1),
        INNER=inner,
        CHUNK=min(256, triton.next_power_of_2(inner)),
    )


class GatheredDraftKV:
    """Per-layer scratch: gather once per round, then append the new draft KV."""

    def __init__(self, layers, batch, width, page, heads, dim, dtype, device):
        self.page = page
        pages = triton.cdiv(width, page)
        # Allocate during model load so this storage reduces the KV-cache budget.
        self.k = torch.zeros(
            layers, batch, pages * page, heads, dim, dtype=dtype, device=device
        )
        self.v = torch.zeros_like(self.k)
        self.table = torch.arange(
            layers * batch * pages, dtype=torch.int32, device=device
        ).view(layers, batch, pages)
        self.starts = torch.zeros(batch, dtype=torch.int32, device=device)
        self.fresh = [False] * layers

    def begin_propose(self):
        self.fresh = [False] * len(self.fresh)

    def attention_kwargs(self, kwargs, layer, slots, used):
        batch = slots.shape[0]
        starts = self.starts[:batch]
        if self.fresh[layer]:
            torch.sub(used, 1, out=starts).clamp_(min=0)
        else:
            starts.zero_()
            self.fresh[layer] = True
        gather_tokens(
            kwargs["k"],
            kwargs["v"],
            slots,
            starts,
            used,
            self.k[layer, :batch],
            self.v[layer, :batch],
        )
        heads, dim = self.k.shape[-2:]
        kwargs["k"] = self.k.view(-1, self.page, heads, dim)
        kwargs["v"] = self.v.view(-1, self.page, heads, dim)
        kwargs["block_table"] = self.table[layer, :batch]
        kwargs["seqused_k"] = used
        kwargs["max_seqlen_k"] = min(kwargs["max_seqlen_k"], self.k.shape[2])
