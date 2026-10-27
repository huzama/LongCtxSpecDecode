# SPDX-License-Identifier: Apache-2.0
"""Native FA4 attention over coarse/refinement KV with arbitrary token selection.

FA4 owns tensor-core compute, softmax and split reduction. Only its paged loader
is replaced. Buffers and compiled callables are prepared before graph capture.
"""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _prepare_queries(
    Q,
    QP,
    CU,
    SLOTS,
    STATE,
    COUNTS,
    NQ,
    NQP,
    INFO,
    SQ,
    SK,
    H: tl.constexpr,
    MAXQ: tl.constexpr,
    VIEWS: tl.constexpr,
    SPARSE: tl.constexpr,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
):
    batch, head = tl.program_id(0), tl.program_id(1)
    req, view = batch // VIEWS, batch % VIEWS
    slot = tl.load(SLOTS + req)
    start, end = tl.load(CU + req), tl.load(CU + req + 1)
    prefix = tl.load(STATE + slot * 4 + 1, slot >= 0, 0)
    length = tl.load(STATE + slot * 4, slot >= 0, 0)
    rows, d = tl.arange(0, 8), tl.arange(0, 128)
    positions = tl.load(QP + start + rows, start + rows < end, -1).to(tl.int32)
    first_count = tl.sum(
        ((start + rows < end) & (positions < prefix + 256)).to(tl.int32), 0
    )
    if VIEWS == 1:
        view = (tl.load(QP + start, start < end, -1) >= prefix + 256).to(tl.int32)
        offset, count = 0, end - start
    else:
        offset = tl.where(view == 0, 0, first_count)
        count = tl.where(view == 0, first_count, end - start - first_count)
    count = tl.where(slot >= 0, count, 0)
    src = start + offset + rows
    values = tl.load(
        Q + src[:, None] * QS0 + head * QS1 + d[None, :], rows[:, None] < count, 0
    )
    tl.store(
        NQ + ((batch * MAXQ + rows[:, None]) * H + head) * 128 + d[None, :],
        values,
        rows[:, None] < MAXQ,
    )
    if head == 0:
        positions = tl.load(QP + src, rows < count, -1).to(tl.int32)
        tl.store(NQP + batch * MAXQ + rows, positions, rows < MAXQ)
        tl.store(INFO + batch * 4, slot)
        tl.store(INFO + batch * 4 + 1, prefix + view * 128)
        tl.store(INFO + batch * 4 + 2, length)
        tl.store(INFO + batch * 4 + 3, req)
        tl.store(SQ + batch, count)
        n = tl.load(COUNTS + req) if SPARSE else length
        if not SPARSE:
            # Each precision view contains consecutive query positions. Its
            # final position supplies the native bottom-right causal bound.
            last = tl.load(QP + start + offset + count - 1, count > 0, -1)
            n = tl.minimum(n, last + 1)
        tl.store(SK + batch, tl.where(count > 0, n, 0))


@tr.jit
def _scatter(
    SRC,
    LSE,
    OUT,
    OUTLSE,
    QP,
    CU,
    SLOTS,
    STATE,
    H: tl.constexpr,
    MAXQ: tl.constexpr,
    VIEWS: tl.constexpr,
    LS0: tl.constexpr,
    LS1: tl.constexpr,
    LS2: tl.constexpr,
    OUTLS: tl.constexpr,
):
    req, head = tl.program_id(0), tl.program_id(1)
    start, end = tl.load(CU + req), tl.load(CU + req + 1)
    slot = tl.load(SLOTS + req)
    prefix = tl.load(STATE + slot * 4 + 1, slot >= 0, 0)
    rows, d = tl.arange(0, 8), tl.arange(0, 128)
    qp = tl.load(QP + start + rows, start + rows < end, -1).to(tl.int32)
    first_count = tl.sum(((start + rows < end) & (qp < prefix + 256)).to(tl.int32), 0)
    if VIEWS == 1:
        batch, local = req, rows
    else:
        view = (qp >= prefix + 256).to(tl.int32)
        batch = req * VIEWS + view
        local = rows - tl.where(view == 1, first_count, 0)
    valid = (start + rows < end) & (slot >= 0)
    values = tl.load(
        SRC + (((batch * MAXQ + local)[:, None] * H + head) * 128) + d[None, :],
        valid[:, None],
        0,
    )
    tl.store(
        OUT + ((start + rows[:, None]) * H + head) * 128 + d[None, :],
        values,
        start + rows[:, None] < end,
    )
    normalizer = tl.load(
        LSE + batch * LS0 + head * LS1 + local * LS2, valid, -float("inf")
    )
    tl.store(OUTLSE + head * OUTLS + start + rows, normalizer, start + rows < end)


@lru_cache(maxsize=1)
def _check_runtime():
    from flash_attn.cute import flash_fwd_sm100

    path = Path(flash_fwd_sm100.__file__)
    marker = path.parents[2] / "hierarchical_loader_hook.json"
    if not marker.is_file():
        raise RuntimeError(
            "Use the isolated overlay from tools/patch_fa4_hierarchical.py"
        )
    manifest = json.loads(marker.read_text())
    expected = "24ff53a4d10d92ac0f86f3484cc978462686b795b62fd15c13d7c3cb001889b3"
    if (
        manifest.get("source_sha256") != expected
        or manifest.get("hook_version") != 2
        or hashlib.sha256(path.read_bytes()).hexdigest()
        != manifest.get("patched_sha256")
        or not hasattr(
            flash_fwd_sm100.FlashAttentionForwardSm100, "make_paged_kv_manager"
        )
    ):
        raise RuntimeError(
            "FA4 hierarchical overlay changed; revalidate the loader hook"
        )


class NativeAttentionWorkspace:
    def __init__(
        self,
        requests,
        kv_heads,
        query_heads,
        max_queries,
        max_positions,
        device,
        *,
        num_splits=None,
    ):
        if (
            min(requests, kv_heads, query_heads, max_positions) < 1
            or query_heads % kv_heads
            or not 1 <= max_queries <= 7
        ):
            raise ValueError(
                "native hierarchical attention requires 1–7 queries and valid heads"
            )
        if torch.cuda.get_device_capability(device) != (10, 0):
            raise ValueError("native hierarchical attention requires SM100")
        if num_splits is not None and (
            type(num_splits) is not int or not 1 <= num_splits <= 128
        ):
            raise ValueError("num_splits must be in [1,128]")
        _check_runtime()
        from flash_attn.cute.interface import num_splits_heuristic

        self.capacity = requests
        self.shape = (requests, kv_heads, query_heads, max_queries, max_positions)
        self.views = 1 if max_queries == 1 else 2
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        # Usually one precision view is populated per logical request. Count
        # requests for FA4's heuristic; partial storage still covers both views.
        self.split_counts = {
            n: num_splits
            or max(
                1,
                num_splits_heuristic(
                    n * kv_heads, sms, tr.cdiv(max_positions, 128), 128
                ),
            )
            for n in range(1, requests + 1)
        }
        self.splits = self.split_counts[requests]
        batches = requests * self.views
        self.q = torch.empty(
            (batches, max_queries, query_heads, 128),
            dtype=torch.bfloat16,
            device=device,
        )
        self.out = torch.empty_like(self.q)
        self.normalizers = torch.empty(
            (batches, query_heads, max_queries), dtype=torch.float32, device=device
        )
        split_rows = max(n * s if s > 1 else 0 for n, s in self.split_counts.items())
        elements = split_rows * self.views * max_queries * query_heads
        self.partial = torch.empty(elements * 128, dtype=torch.float32, device=device)
        self.partial_lse = torch.empty(elements, dtype=torch.float32, device=device)
        self.info = torch.empty((batches, 4), dtype=torch.int32, device=device)
        self.qpos = torch.empty(
            (batches, max_queries), dtype=torch.int32, device=device
        )
        self.seq_q = torch.empty(batches, dtype=torch.int32, device=device)
        self.seq_k = torch.empty_like(self.seq_q)
        # Native mask bounds derive from this shape even though its page IDs
        # are unused by the custom loader.
        self.placeholder_table = torch.zeros(
            (batches, tr.cdiv(max_positions, 16)), dtype=torch.int32, device=device
        )
        self.result = torch.empty(
            (requests * max_queries, query_heads, 128),
            dtype=torch.bfloat16,
            device=device,
        )
        self.result_lse = torch.empty(
            (query_heads, requests * max_queries), dtype=torch.float32, device=device
        )
        self.compiled = {}

    def for_requests(self, requests):
        import copy

        if requests not in self.split_counts:
            raise ValueError("request batch exceeds attention workspace capacity")
        result = copy.copy(self)
        result.shape = (requests, *self.shape[1:])
        result.splits = self.split_counts[requests]
        return result

    def memory_bytes(self):
        return sum(
            x.numel() * x.element_size()
            for x in vars(self).values()
            if isinstance(x, torch.Tensor)
        )


def _compile(workspace, args, bits, sparse):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from flash_attn.cute.utils import AuxData

    from .fa4_loader import PackedAttentionSm100

    # SM100 broadcasts one query/batch across eight keys and expects packed bits.
    @cute.jit
    def logical_mask(batch, head, query, key, seqlen, data):
        b, q = batch[0], query[0]
        req, length = data[6][b, 3], data[6][b, 2]
        query_position = data[7][b, q]
        result = cute.make_rmem_tensor((1,), cutlass.Uint32)
        result[0] = cutlass.Uint32(0)
        for i in cutlass.range_constexpr(8):
            pos = cutlass.Int32(-1)
            if key[i] < data[5].shape[1]:
                pos = data[5][req, key[i]]
            keep = (pos >= 0) & (pos < length) & (pos <= query_position)
            result[0] = result[0] | (
                cutlass.Uint32(1 << i) if keep else cutlass.Uint32(0)
            )
        return result.load()

    logical_mask.__vec_size__ = 8
    _, kvh, qh, maxq, _ = workspace.shape
    fa = PackedAttentionSm100(
        128,
        128,
        qhead_per_kvhead=qh // kvh,
        is_causal=not sparse,
        is_local=False,
        is_split_kv=workspace.splits > 1,
        pack_gqa=True,
        m_block_size=128,
        n_block_size=128,
        q_stage=1,
        is_static_persistent=False,
        mask_mod=logical_mask if sparse else None,
        has_aux_tensors=True,
        paged_kv_non_tma=True,
        is_varlen_q=True,
        use_2cta_instrs=False,
        use_clc_scheduler=False,
        bits=bits,
        sparse=sparse,
    )

    def convert(value):
        if isinstance(value, torch.Tensor):
            return from_dlpack(
                value,
                assumed_align=4 if value.dtype in (torch.int32, torch.int64) else 16,
            )
        if isinstance(value, AuxData):
            return AuxData(tuple(convert(t) for t in value.tensors), value.scalars)
        return value

    return cute.compile(
        fa,
        *(convert(x) for x in args),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def attention(
    cache,
    query,
    query_positions,
    query_start_loc,
    slots,
    block_table,
    *,
    bits=8,
    selected_ids=None,
    selected_counts=None,
    workspace,
    output=None,
):
    """Return BF16 output and FP32 LSE [query_heads,total_queries].

    Query positions are contiguous within each request, with at most the
    workspace's maximum query count (1–7). Selection contains logical TOKEN IDs.
    Returned workspace views are overwritten by the next call using it.
    """
    from flash_attn.cute.interface import _flash_attn_fwd_combine
    from flash_attn.cute.utils import AuxData

    n, h, hq, maxq, _ = workspace.shape
    if (
        cache.device.type != "cuda"
        or bits not in (4, 8)
        or h != cache.kv_heads
        or slots.shape != (n,)
        or n > workspace.capacity
        or query.ndim != 3
        or query.shape[1:] != (hq, 128)
        or query.dtype != torch.bfloat16
        or query.device != cache.device
        or query.stride(-1) != 1
        or len(query) > n * maxq
        or query_positions.shape != (len(query),)
        or query_start_loc.shape != (n + 1,)
        or block_table.ndim != 2
        or block_table.shape[0] != n
        or workspace.q.device != cache.device
    ):
        raise ValueError("incompatible hierarchical attention inputs/workspace")
    if cache.pages.data_ptr() % 16 or cache.pages.stride(0) % 16:
        raise ValueError("packed pages and their strides require 16-byte alignment")
    sparse = selected_ids is not None
    if sparse and (
        selected_counts is None
        or selected_ids.ndim != 2
        or selected_ids.shape[0] != n
        or selected_counts.shape != (n,)
    ):
        raise ValueError("sparse attention requires logical token IDs and counts")
    metadata = (query_positions, query_start_loc, slots, block_table)
    if sparse:
        metadata += (selected_ids, selected_counts)
    for tensor in metadata:
        if (
            tensor.device != cache.device
            or tensor.dtype not in (torch.int32, torch.int64)
            or tensor.stride(-1) != 1
        ):
            raise ValueError(
                "metadata must be integer device tensors with contiguous rows"
            )
    output = workspace.result[: len(query)] if output is None else output
    if (
        output.shape != query.shape
        or output.dtype != query.dtype
        or output.device != query.device
        or not output.is_contiguous()
    ):
        raise ValueError("invalid attention output buffer")
    batches = n * workspace.views
    q, native_out = workspace.q[:batches], workspace.out[:batches]
    native_lse = workspace.normalizers[:batches]
    info, qpos = workspace.info[:batches], workspace.qpos[:batches]
    sq, sk = workspace.seq_q[:batches], workspace.seq_k[:batches]
    ids, counts = (selected_ids, selected_counts) if sparse else (block_table, slots)
    _prepare_queries[(batches, hq)](
        query,
        query_positions,
        query_start_loc,
        slots,
        cache.state,
        counts,
        q,
        qpos,
        info,
        sq,
        sk,
        hq,
        maxq,
        workspace.views,
        sparse,
        query.stride(0),
        query.stride(1),
        num_warps=4,
    )
    # These views supply native shape/dtype metadata only. The packed manager
    # reads the cache and bounded tail through aux_data instead.
    kshape, vshape = (x[:, :16] for x in cache.tail)
    aux = AuxData(
        (
            cache.pages.view(torch.uint32),
            cache.metadata,
            *cache.tail,
            block_table,
            ids,
            info,
            qpos,
        )
    )
    splits = workspace.splits
    if splits > 1:
        elements = splits * batches * maxq * hq
        result = workspace.partial[: elements * 128].view(
            splits, batches, maxq, hq, 128
        )
        lse = workspace.partial_lse[:elements].view(splits, batches, hq, maxq)
    else:
        result, lse = native_out, native_lse
    # Match the pinned FA4 __call__ signature, with stream supplied at compile.
    args = (
        q,
        kshape,
        vshape,
        result,
        lse,
        128**-0.5,
        None,
        None,
        sq,
        sk,
        workspace.placeholder_table[:batches],
        None,
        None,
        None,
        None,
        None,
        aux,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        maxq,
    )

    def signature(value):
        if isinstance(value, torch.Tensor):
            return (tuple(value.shape), tuple(value.stride()), value.dtype)
        if isinstance(value, AuxData):
            return tuple(signature(t) for t in value.tensors)
        return value

    key = (bits, sparse, tuple(signature(x) for x in args))
    if key not in workspace.compiled:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm hierarchical attention before CUDA graph capture")
        workspace.compiled[key] = _compile(workspace, args, bits, sparse)
    workspace.compiled[key](*args)
    if splits > 1:
        _flash_attn_fwd_combine(
            result,
            lse.transpose(-1, -2),
            native_out,
            native_lse.transpose(-1, -2),
            seqused=sq,
        )
    normalizers = workspace.result_lse[:, : len(query)]
    _scatter[(n, hq)](
        native_out,
        native_lse,
        output,
        normalizers,
        query_positions,
        query_start_loc,
        slots,
        cache.state,
        hq,
        maxq,
        workspace.views,
        *native_lse.stride(),
        normalizers.stride(0),
        num_warps=4,
    )
    return output, normalizers
