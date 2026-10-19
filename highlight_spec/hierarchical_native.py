"""Native Blackwell attention over the existing hierarchical KV allocation.

FlashAttention owns tensor-core compute, online softmax and split reduction.
The only format-specific attention code is the tile loader and logical-position
mask. Small query workspaces handle both precision views in a verification
round. All buffers are allocated before CUDA graph capture.
"""
from functools import lru_cache
import hashlib
from pathlib import Path

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _prepare_queries(Q, QP, QSL, SLOTS, STATE, COUNTS, NQ, NQP, INFO, SQ, SK,
                     H: tl.constexpr, MAXQ: tl.constexpr, VIEWS: tl.constexpr,
                     SPARSE: tl.constexpr):
    batch, head = tl.program_id(0), tl.program_id(1)
    req, view = batch//VIEWS, batch%VIEWS
    slot = tl.load(SLOTS+req)
    start, end = tl.load(QSL+req), tl.load(QSL+req+1)
    prefix = tl.load(STATE+slot*4+1, slot >= 0, 0)
    length = tl.load(STATE+slot*4, slot >= 0, 0)
    rows, d = tl.arange(0, 8), tl.arange(0, 128)
    qp = tl.load(QP+start+rows, start+rows < end, -1).to(tl.int32)
    count0 = tl.sum(((start+rows < end) & (qp < prefix+256)).to(tl.int32), 0)
    if VIEWS == 1:
        view = (tl.load(QP+start, start < end, -1) >= prefix+256).to(tl.int32)
        offset, count = 0, end-start
    else:
        offset = tl.where(view == 0, 0, count0)
        count = tl.where(view == 0, count0, end-start-count0)
    count = tl.where(slot >= 0, count, 0)
    srcrow = start+offset+rows
    value = tl.load(Q+(srcrow[:, None]*H+head)*128+d[None, :], rows[:, None] < count, 0)
    tl.store(NQ+((batch*MAXQ+rows[:, None])*H+head)*128+d[None, :], value,
             rows[:, None] < MAXQ)
    if head == 0:
        positions = tl.load(QP+srcrow, rows < count, -1).to(tl.int32)
        tl.store(NQP+batch*MAXQ+rows, positions, rows < MAXQ)
        tl.store(INFO+batch*4, slot)
        tl.store(INFO+batch*4+1, prefix+view*128)
        tl.store(INFO+batch*4+2, length)
        tl.store(INFO+batch*4+3, req)
        tl.store(SQ+batch, count)
        n = tl.load(COUNTS+req)*16 if SPARSE else length
        if not SPARSE and VIEWS == 1:
            # Current-token decode has n == length. Also preserve causality
            # when the diagnostic API asks for an earlier single query.
            n = tl.minimum(n, tl.load(QP+start, start < end, -1)+1)
        tl.store(SK+batch, tl.where(count > 0, n, 0))


@tr.jit
def _scatter_output(SRC, OUT, QP, QSL, SLOTS, STATE,
                    H: tl.constexpr, MAXQ: tl.constexpr, VIEWS: tl.constexpr):
    req, head = tl.program_id(0), tl.program_id(1)
    start, end = tl.load(QSL+req), tl.load(QSL+req+1)
    slot = tl.load(SLOTS+req)
    prefix = tl.load(STATE+slot*4+1, slot >= 0, 0)
    rows, d = tl.arange(0, 8), tl.arange(0, 128)
    qp = tl.load(QP+start+rows, start+rows < end, -1).to(tl.int32)
    count0 = tl.sum(((start+rows < end) & (qp < prefix+256)).to(tl.int32), 0)
    if VIEWS == 1:
        srcrow = req*MAXQ+rows
    else:
        view = (qp >= prefix+256).to(tl.int32)
        srcrow = (req*VIEWS+view)*MAXQ+rows-tl.where(view == 1, count0, 0)
    value = tl.load(SRC+(srcrow[:, None]*H+head)*128+d[None, :],
                    (start+rows[:, None] < end) & (slot >= 0), 0)
    tl.store(OUT+((start+rows[:, None])*H+head)*128+d[None, :], value,
             start+rows[:, None] < end)


class NativeAttentionWorkspace:
    backend = 'fa4'

    def __init__(self, requests, kv_heads, query_heads, max_queries, max_positions, device):
        if (min(requests, kv_heads, query_heads, max_positions) < 1
                or query_heads % kv_heads or max_queries not in (1, 7)):
            raise ValueError('paged attention requires one or seven queries per request')
        if torch.cuda.get_device_capability(device) != (10, 0):
            raise ValueError('native hierarchical attention requires SM100')
        _check_runtime()
        from vllm.vllm_flash_attn.cute.interface import num_splits_heuristic
        self.capacity = requests
        self.shape = (requests, kv_heads, query_heads, max_queries, max_positions)
        self.views = 1 if max_queries == 1 else 2
        b = requests*self.views
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.splits = max(1, num_splits_heuristic(
            requests*kv_heads, sms, tr.cdiv(max_positions, 128), 128))
        self.q = torch.empty((b, max_queries, query_heads, 128), device=device, dtype=torch.bfloat16)
        self.out = torch.empty_like(self.q)
        split_buffers = self.splits if self.splits > 1 else 0
        self.partial = torch.empty((split_buffers, *self.q.shape), device=device, dtype=torch.float32)
        self.lse = torch.empty((split_buffers, b, query_heads, max_queries), device=device, dtype=torch.float32)
        self.info = torch.empty((b, 4), device=device, dtype=torch.int32)
        self.qpos = torch.empty((b, max_queries), device=device, dtype=torch.int32)
        self.seq_q = torch.empty(b, device=device, dtype=torch.int32)
        self.seq_k = torch.empty_like(self.seq_q)
        self.compiled = {}

    def memory_bytes(self):
        return sum(getattr(self, name).numel()*getattr(self, name).element_size()
                   for name in ('q', 'out', 'partial', 'lse', 'info', 'qpos', 'seq_q', 'seq_k'))


@lru_cache(maxsize=1)
def _check_runtime():
    # The adapter inherits upstream internals. Reject an unreviewed upstream
    # change instead of silently mixing incompatible producer/consumer code.
    from vllm.vllm_flash_attn.cute import flash_fwd_sm100
    expected = 'de30d2eb91584cc7ea0683d63a0e5a18853351ec0ec3c13909c1d4eaca7fe37e'
    actual = hashlib.sha256(Path(flash_fwd_sm100.__file__).read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError('FA4 source changed; revalidate the hierarchical pipeline adapter')


def _compile(workspace, args, bits, sparse):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from vllm.vllm_flash_attn.cute.utils import AuxData
    from .hierarchical_native_pipeline import HierarchicalAttentionSm100

    @cute.jit
    def logical_mask(batch, head, query, key, seqlen, data):
        # FA4 supplies 32 consecutive key indices and consumes one packed mask.
        # Resolve one logical page per 16 columns, sharing each metadata read
        # across the page instead of reloading it for every score element.
        b, q, first = batch[0], query[0], key[0]
        req = data[6][b, 3]
        qp = data[7][b, q]
        length = data[6][b, 2]
        result = cutlass.Uint32(0)
        for group in cutlass.range_constexpr(2):
            k = first + group*16
            pos = k
            valid = cutlass.Boolean(True)
            if cutlass.const_expr(sparse):
                if k//16 < data[5].shape[1]:
                    pos = data[5][req, k//16]*16+k%16
                else:
                    valid = cutlass.Boolean(False)
            count = min(16, min(qp+1, length)-pos)
            count = max(0, count) if valid & (pos >= 0) else cutlass.Int32(0)
            mask = cutlass.Uint32(0xffff) >> (16-count)
            result = result | (mask << (group*16))
        packed = cute.make_rmem_tensor((1,), cutlass.Uint32)
        packed[0] = result
        return packed.load()

    logical_mask.__vec_size__ = 32

    _, kvh, qh, maxq, _ = workspace.shape
    fa = HierarchicalAttentionSm100(128, 128, qhead_per_kvhead=qh//kvh,
        is_causal=False, is_local=False, is_split_kv=workspace.splits > 1,
        pack_gqa=True, m_block_size=128, n_block_size=128, q_stage=1,
        is_persistent=False, mask_mod=logical_mask if (sparse or maxq > 1) else None, has_aux_tensors=True,
        paged_kv_non_tma=True, is_varlen_q=True,
        hierarchical_bits=bits, hierarchical_sparse=sparse)
    def convert(x):
        if isinstance(x, torch.Tensor):
            return from_dlpack(x, assumed_align=4 if x.dtype in (torch.int32, torch.int64) else 16)
        if isinstance(x, AuxData):
            return AuxData(tuple(convert(t) for t in x.tensors), x.scalars)
        return x
    return cute.compile(fa, *(convert(x) for x in args),
                        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                        options='--enable-tvm-ffi')


def attention(cache, query, query_positions, query_start_loc, slots, block_table, *,
              bits=8, selected_ids=None, selected_counts=None, workspace, output=None):
    from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd_combine
    from vllm.vllm_flash_attn.cute.utils import AuxData
    n, h, hq, maxq, _ = workspace.shape
    if (cache.device.type != 'cuda' or bits not in (4, 8) or h != cache.kv_heads
            or slots.shape != (n,) or n > workspace.capacity
            or query.ndim != 3 or query.shape[1:] != (hq, 128)
            or query.dtype != torch.bfloat16 or query.device != cache.device
            or not query.is_contiguous() or len(query) > n*maxq
            or query_positions.shape != (len(query),) or query_start_loc.shape != (n+1,)
            or block_table.ndim != 2 or block_table.shape[0] != n
            or workspace.q.device != cache.device):
        raise ValueError('incompatible paged attention inputs/workspace')
    if cache.pages.data_ptr() % 16 or cache.pages.stride(0) % 16:
        raise ValueError('native packed loads require 16-byte aligned pages and strides')
    views, b = workspace.views, n*workspace.views
    sparse = selected_ids is not None
    if sparse and (selected_counts is None or selected_ids.ndim != 2
                   or selected_ids.shape[0] != n or selected_counts.shape != (n,)):
        raise ValueError('sparse attention requires logical block ids and counts')
    metadata = (query_positions, query_start_loc, slots, block_table)
    if sparse:
        metadata += (selected_ids, selected_counts)
    for tensor in metadata:
        if (tensor.device != cache.device or tensor.dtype not in (torch.int32, torch.int64)
                or tensor.stride(-1) != 1):
            raise ValueError('attention metadata must have contiguous integer device rows')
    output = torch.empty_like(query) if output is None else output
    if (output.shape != query.shape or output.dtype != query.dtype
            or output.device != query.device or not output.is_contiguous()):
        raise ValueError('invalid output buffer')
    ids, counts = (selected_ids, selected_counts) if sparse else (block_table, slots)
    q, out = workspace.q[:b], workspace.out[:b]
    partial, lse = workspace.partial[:, :b], workspace.lse[:, :b]
    info, qp = workspace.info[:b], workspace.qpos[:b]
    sq, sk = workspace.seq_q[:b], workspace.seq_k[:b]
    _prepare_queries[(b, hq)](query, query_positions, query_start_loc, slots,
        cache.state, counts, q, qp, info, sq, sk, hq, maxq, views, sparse, num_warps=4)
    # These existing BF16 ring views provide native K/V shape/type metadata.
    # The native page loader is overridden and never reads these as context KV.
    kshape, vshape = [x[:, :16] for x in cache.tail]
    aux = AuxData((cache.pages.view(torch.uint32), cache.metadata, *cache.tail, block_table, ids, info, qp))
    args = (q, kshape, vshape, partial if workspace.splits > 1 else out,
            lse if workspace.splits > 1 else None, 128**-.5,
            None, None, sq, sk, None, block_table, None, None, None, None, None, aux, None)
    def signature(x):
        if isinstance(x, torch.Tensor):
            return (tuple(x.shape), tuple(x.stride()), x.dtype)
        if isinstance(x, AuxData):
            return tuple(signature(t) for t in x.tensors)
        return x
    key = (bits, sparse, tuple(signature(x) for x in args))
    if key not in workspace.compiled:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('native hierarchical attention must be warmed before graph capture')
        workspace.compiled[key] = _compile(workspace, args, bits, sparse)
    workspace.compiled[key](*args)
    if workspace.splits > 1:
        _flash_attn_fwd_combine(partial, lse.transpose(-1, -2), out, seqused=sq)
    _scatter_output[(n, hq)](out, output, query_positions, query_start_loc, slots, cache.state,
                            hq, maxq, views, num_warps=4)
    return output
