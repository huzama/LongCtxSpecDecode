"""Graph-safe cache transactions for shared, paged hierarchical KV."""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice


@tr.jit
def _quantize(x, VALUE: tl.constexpr):
    axis: tl.constexpr = 1 if VALUE else 0
    minimum = tl.min(x, axis).to(tl.bfloat16).to(tl.float32)
    scale = ((tl.max(x, axis)-minimum)/15).to(tl.bfloat16).to(tl.float32)
    scale = tl.where(scale > 0, scale, 1.)
    if VALUE:
        lo, step = minimum[:, None], scale[:, None]
    else:
        lo, step = minimum[None, :], scale[None, :]
    coarse = tl.minimum(15., tl.maximum(0., libdevice.nearbyint(tl.div_rn(x-lo, step))))
    residual = tl.minimum(7., tl.maximum(-8., libdevice.nearbyint((x-(coarse*step+lo))*tl.div_rn(16., step))))+8
    pair = tl.broadcast_to((tl.arange(0, 64)*2)[None, :], (128, 64))
    c = tl.gather(coarse.to(tl.uint8), pair, 1) | (tl.gather(coarse.to(tl.uint8), pair+1, 1) << 4)
    r = tl.gather(residual.to(tl.uint8), pair, 1) | (tl.gather(residual.to(tl.uint8), pair+1, 1) << 4)
    return c, r, scale, minimum


@tr.jit
def _store_group(PAGES, META, TABLE, x, group, head, H: tl.constexpr,
                 STRIDE: tl.constexpr, VALUE: tl.constexpr):
    c, r, scale, minimum = _quantize(x, VALUE)
    t, d = tl.arange(0, 128), tl.arange(0, 64)
    physical = tl.load(TABLE+group*8+t//16).to(tl.int64)
    plane: tl.constexpr = 16*H*64
    off = physical[:, None]*STRIDE+((t[:, None]%16)*H+head)*64+d[None, :]
    tl.store(PAGES+off+(2 if VALUE else 0)*plane, c)
    tl.store(PAGES+off+(3 if VALUE else 1)*plane, r)
    # For K, t represents channel; for V it represents token. Both metadata
    # groups occupy eight physical pages without replication.
    moff = physical*(STRIDE//2)+(2 if VALUE else 0)*16*H+(t%16)*H+head
    tl.store(META+moff, scale)
    tl.store(META+moff+16*H, minimum)


@tr.jit
def _prefill_groups(K, V, PAGES, META, TABLE, H: tl.constexpr, STRIDE: tl.constexpr,
                    VALUE: tl.constexpr):
    group, head = tl.program_id(0), tl.program_id(1)
    t, d = tl.arange(0, 128), tl.arange(0, 128)
    x = tl.load((V if VALUE else K)+((group*128+t[:, None])*H+head)*128+d[None, :]).to(tl.float32)
    _store_group(PAGES, META, TABLE, x, group, head, H, STRIDE, VALUE)


@tr.jit
def _prefill_recent(K, V, KT, VT, STATE, ERROR, N: tl.constexpr,
                    PREFIX: tl.constexpr, SLOT: tl.constexpr, H: tl.constexpr):
    head = tl.program_id(0)
    t, d = tl.arange(0, 256), tl.arange(0, 128)
    p = PREFIX+t
    src = (p[:, None]*H+head)*128+d[None, :]
    dst = (((SLOT*263+p[:, None]%263)*H)+head)*128+d[None, :]
    tl.store(KT+dst, tl.load(K+src, p[:, None] < N, 0), p[:, None] < N)
    tl.store(VT+dst, tl.load(V+src, p[:, None] < N, 0), p[:, None] < N)
    if head == 0:
        tl.store(STATE+SLOT*4, N)
        tl.store(STATE+SLOT*4+1, PREFIX)
        tl.store(STATE+SLOT*4+2, N)
        tl.store(STATE+SLOT*4+3, PREFIX)
        tl.store(ERROR+SLOT, 0)


def prefill(cache, slot, table, key, value):
    _kv_contract(cache, key, value)
    if not 0 <= slot < cache.max_requests or key.shape[0] < 1 or table.numel()*16 < key.shape[0]:
        raise ValueError("invalid prefill slot or capacity")
    if table.ndim != 1 or not table.is_contiguous() or table.device != cache.device:
        raise ValueError("prefill requires a contiguous device page table")
    n, h = key.shape[0], cache.kv_heads
    prefix = max(0, ((n-128)//128)*128)
    for value_plane in (False, True):
        if prefix:
            _prefill_groups[(prefix//128, h)](key, value, cache.pages, cache.metadata, table,
                h, cache.pages.stride(0), value_plane, num_warps=8, enable_fp_fusion=False)
    _prefill_recent[(h,)](key, value, *cache.tail, cache.state, cache.error, n, prefix, slot, h,
                          num_warps=8)


@tr.jit
def _prepare(KT, VT, PAGES, META, STATE, ERROR, QP, QSL, SLOTS, COMMITTED, TABLE,
             H: tl.constexpr, STRIDE: tl.constexpr, TABLE_STRIDE: tl.constexpr,
             VALUE: tl.constexpr):
    req, head = tl.program_id(0), tl.program_id(1)
    slot = tl.load(SLOTS+req)
    start, end = tl.load(QSL+req), tl.load(QSL+req+1)
    if slot >= 0 and end > start:
        base = tl.load(COMMITTED+req)
        first = tl.load(QP+start).to(tl.int32)
        length = tl.load(QP+end-1).to(tl.int32)+1
        previous_length = tl.load(STATE+slot*4)
        previous_prefix = tl.load(STATE+slot*4+1)
        previous_commit = tl.load(STATE+slot*4+2)
        encoded = tl.load(STATE+slot*4+3)
        prefix = tl.maximum(0, ((base-128)//128)*128)
        needed = tl.maximum(prefix, ((length-1-128)//128)*128)
        bad = ((tl.load(ERROR+slot) != 0) | (base < previous_commit) | (base > previous_length) |
               (first < base) | (length > base+7) | (end-start > 7) |
               (length-first != end-start) | (prefix < previous_prefix) |
               (needed > encoded+128) | (needed > base) |
               (length-prefix > 263))
        if head == 0 and not VALUE:
            tl.store(ERROR+slot, bad.to(tl.int32))
        if not bad and needed > encoded:
            t, d = tl.arange(0, 128), tl.arange(0, 128)
            p = encoded+t
            src = (((slot*263+p[:, None]%263)*H)+head)*128+d[None, :]
            x = tl.load((VT if VALUE else KT)+src).to(tl.float32)
            _store_group(PAGES, META, TABLE+req*TABLE_STRIDE, x, encoded//128, head, H, STRIDE, VALUE)


@tr.jit
def _stage(K, V, KT, VT, STATE, ERROR, QP, QSL, SLOTS, COMMITTED, H: tl.constexpr):
    req, head = tl.program_id(0), tl.program_id(1)
    slot = tl.load(SLOTS+req)
    start, end = tl.load(QSL+req), tl.load(QSL+req+1)
    if slot >= 0 and end > start:
        if tl.load(ERROR+slot) == 0:
            t, d = tl.arange(0, 8), tl.arange(0, 128)
            p = tl.load(QP+start+t, start+t < end, 0).to(tl.int32)
            src = ((start+t[:, None])*H+head)*128+d[None, :]
            dst = (((slot*263+p[:, None]%263)*H)+head)*128+d[None, :]
            tl.store(KT+dst, tl.load(K+src, start+t[:, None] < end, 0), start+t[:, None] < end)
            tl.store(VT+dst, tl.load(V+src, start+t[:, None] < end, 0), start+t[:, None] < end)
            if head == 0:
                length = tl.load(QP+end-1).to(tl.int32)+1
                base = tl.load(COMMITTED+req)
                prefix = tl.maximum(0, ((base-128)//128)*128)
                needed = tl.maximum(prefix, ((length-1-128)//128)*128)
                encoded = tl.load(STATE+slot*4+3)
                tl.store(STATE+slot*4, length)
                tl.store(STATE+slot*4+1, prefix)
                tl.store(STATE+slot*4+2, base)
                tl.store(STATE+slot*4+3, tl.maximum(encoded, needed))


def _kv_contract(cache, key, value):
    if (cache.device.type != "cuda" or key.ndim != 3 or key.shape != value.shape or
            key.shape[1:] != (cache.kv_heads, 128) or key.dtype != torch.bfloat16 or
            value.dtype != key.dtype or key.device != cache.device or value.device != cache.device or
            not key.is_contiguous() or not value.is_contiguous()):
        raise ValueError("paged update requires contiguous CUDA BF16 K/V")


def update(cache, key, value, query_positions, query_start_loc, slots, committed, block_table):
    _kv_contract(cache, key, value)
    n = len(slots)
    if (query_positions.shape != (len(key),) or query_start_loc.shape != (n+1,) or
            committed.shape != (n,) or block_table.ndim != 2 or block_table.shape[0] != n):
        raise ValueError("incompatible batched update metadata")
    for x in (query_positions, query_start_loc, slots, committed, block_table):
        if x.device != cache.device or x.dtype not in (torch.int32, torch.int64) or x.stride(-1) != 1:
            raise ValueError("metadata must be integer device tensors with contiguous rows")
    for value_plane in (False, True):
        _prepare[(n, cache.kv_heads)](*cache.tail, cache.pages, cache.metadata, cache.state, cache.error,
            query_positions, query_start_loc, slots, committed, block_table,
            cache.kv_heads, cache.pages.stride(0), block_table.stride(0), value_plane,
            num_warps=8, enable_fp_fusion=False)
    _stage[(n, cache.kv_heads)](key, value, *cache.tail, cache.state, cache.error, query_positions,
                               query_start_loc, slots, committed, cache.kv_heads, num_warps=4)


@tr.jit
def _unpack_codes(PAGES, offset, mask, CONTIGUOUS: tl.constexpr = True):
    packed = tl.load(PAGES+offset, mask, 0).to(tl.int32)
    if CONTIGUOUS:
        return tl.interleave(packed & 15, packed >> 4).to(tl.float32)
    else:
        # SM100 lowers direct channel addressing more efficiently than interleave.
        shift = (tl.arange(0, 128)[None, :] % 2)*4
        return ((packed >> shift) & 15).to(tl.float32)


@tr.jit
def _attention(Q, QP, QSL, SLOTS, TABLE, IDS, COUNTS, STATE, PAGES, META, KT, VT, PART, LSE,
               H: tl.constexpr, RATIO: tl.constexpr, STRIDE: tl.constexpr,
               TABLE_STRIDE: tl.constexpr, IDS_STRIDE: tl.constexpr,
               BITS: tl.constexpr, SPARSE: tl.constexpr, SPLITS: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, SCALE: tl.constexpr, VIEWS: tl.constexpr,
               CONTIGUOUS: tl.constexpr):
    rh, split, view = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    req, head = rh//H, rh%H
    slot = tl.load(SLOTS+req)
    start, end = tl.load(QSL+req), tl.load(QSL+req+1)
    prefix = tl.load(STATE+slot*4+1, slot >= 0, 0)
    length = tl.load(STATE+slot*4, slot >= 0, 0)
    if VIEWS == 1:
        # A one-token step has exactly one precision view, even at rollover.
        view = (tl.load(QP+start, start < end, -1) >= prefix+256).to(tl.int32)
    row, d = tl.arange(0, BM), tl.arange(0, 128)
    qrow, qhead = start+row//RATIO, head*RATIO+row%RATIO
    q = tl.load(Q+(qrow[:, None]*(H*RATIO)+qhead[:, None])*128+d[None, :],
                qrow[:, None] < end, 0)
    qp = tl.load(QP+qrow, qrow < end, -1).to(tl.int32)
    visible = (slot >= 0) & (qrow < end) & ((qp >= prefix+256) == (view == 1))
    active = tl.sum(visible.to(tl.int32), 0) > 0
    view_prefix = prefix+view*128
    n = tl.load(COUNTS+req)*16 if SPARSE else length
    tiles = tl.cdiv(n, SPLITS*BN)
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    denominator = tl.zeros((BM,), tl.float32)
    accum = tl.zeros((BM, 128), tl.float32)
    for tile in range(tl.where(active, tiles, 0)):
        offsets = (split*tiles+tile)*BN+tl.arange(0, BN)
        if SPARSE:
            logical = tl.load(IDS+req*IDS_STRIDE+offsets//16, offsets < n, 0)
            pos = logical*16+offsets%16
        else:
            pos = offsets
        valid = (offsets < n) & (pos >= 0) & (pos < length)
        old = valid & (pos < view_prefix)
        recent = valid & (pos >= view_prefix)
        physical = tl.load(TABLE+req*TABLE_STRIDE+pos//16, old, 0).to(tl.int64)
        channel = tl.arange(0, 64) if CONTIGUOUS else d//2
        code_offset = physical[:, None]*STRIDE+((pos[:, None]%16)*H+head)*64+channel[None, :]
        plane: tl.constexpr = 16*H*64
        kc = _unpack_codes(PAGES, code_offset, old[:, None], CONTIGUOUS)
        kpage = tl.load(TABLE+req*TABLE_STRIDE+(pos[:, None]//128)*8+d[None, :]//16,
                        old[:, None], 0).to(tl.int64)
        km = kpage*(STRIDE//2)+(d[None, :]%16)*H+head
        ks = tl.load(META+km, old[:, None], 0).to(tl.float32)
        kz = tl.load(META+km+16*H, old[:, None], 0).to(tl.float32)
        k = kc*ks+kz
        if BITS == 8:
            kr = _unpack_codes(PAGES, code_offset+plane, old[:, None], CONTIGUOUS)-8
            k += kr*(ks/16.)
        tail_offset = (((slot*263+pos[:, None]%263)*H)+head)*128+d[None, :]
        k += tl.load(KT+tail_offset, recent[:, None], 0).to(tl.float32)
        score = tl.dot(q, tl.trans(k.to(tl.bfloat16)))*SCALE
        score = tl.where(valid[None, :] & (pos[None, :] <= qp[:, None]) & visible[:, None],
                         score, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(score, 1))
        safe_max = tl.where(next_max == -float("inf"), 0., next_max)
        alpha = tl.exp(maximum-safe_max)
        prob = tl.exp(score-safe_max[:, None])
        vc = _unpack_codes(PAGES, code_offset+2*plane, old[:, None], CONTIGUOUS)
        vm = physical*(STRIDE//2)+2*16*H+(pos%16)*H+head
        vs = tl.load(META+vm, old, 0).to(tl.float32)
        vz = tl.load(META+vm+16*H, old, 0).to(tl.float32)
        v = vc*vs[:, None]+vz[:, None]
        if BITS == 8:
            vr = _unpack_codes(PAGES, code_offset+3*plane, old[:, None], CONTIGUOUS)-8
            v += vr*(vs[:, None]/16.)
        v += tl.load(VT+tail_offset, recent[:, None], 0).to(tl.float32)
        accum = accum*alpha[:, None]+tl.dot(prob.to(tl.bfloat16), v.to(tl.bfloat16))
        denominator = denominator*alpha+tl.sum(prob, 1)
        maximum = next_max
    if active:
        stored_view = 0 if VIEWS == 1 else view
        section = (req*VIEWS+stored_view)*H+head
        off = (((section*SPLITS+split)*BM+row[:, None])*128+d[None, :])
        tl.store(PART+off, accum/tl.maximum(denominator[:, None], 1.e-30))
        tl.store(LSE+((section*SPLITS+split)*BM+row), maximum+tl.log(denominator))


@tr.jit
def _merge(PART, LSE, OUT, QP, QSL, SLOTS, STATE, H: tl.constexpr, RATIO: tl.constexpr,
           SPLITS: tl.constexpr, BM: tl.constexpr, BS: tl.constexpr, VIEWS: tl.constexpr):
    rh, row = tl.program_id(0), tl.program_id(1)
    req, head = rh//H, rh%H
    slot = tl.load(SLOTS+req)
    start, end = tl.load(QSL+req), tl.load(QSL+req+1)
    qrow = start+row//RATIO
    if qrow < end and slot >= 0:
        prefix = tl.load(STATE+slot*4+1, slot >= 0, 0)
        view = 0 if VIEWS == 1 else (tl.load(QP+qrow) >= prefix+256).to(tl.int32)
        section = (req*VIEWS+view)*H+head
        s, d = tl.arange(0, BS), tl.arange(0, 128)
        lse = tl.load(LSE+(section*SPLITS+s)*BM+row, s < SPLITS, -float("inf"))
        maximum = tl.max(lse, 0)
        weights = tl.exp(lse-tl.where(maximum == -float("inf"), 0., maximum))
        weights /= tl.maximum(tl.sum(weights, 0), 1.e-30)
        part = tl.load(PART+((section*SPLITS+s[:, None])*BM+row)*128+d[None, :], s[:, None] < SPLITS, 0)
        result = tl.sum(part*weights[:, None], 0)
        target = (qrow*(H*RATIO)+head*RATIO+row%RATIO)*128+d
        tl.store(OUT+target, result)
    elif qrow < end:
        # Inactive requests have no initialized partials and must not read them.
        d = tl.arange(0, 128)
        target = (qrow*(H*RATIO)+head*RATIO+row%RATIO)*128+d
        tl.store(OUT+target, 0.)


def attention_splits(requests, kv_heads, max_queries, max_positions, multiprocessors):
    """Bound split-KV scratch by occupancy, instead of growing it with batch.

    Eight CTAs per SM provide latency hiding for packed loads. Single-request
    decode retains the old split ceiling; large batches already supply CTAs.
    This uses host shape/device properties only and is safe before capture.
    """
    if (min(requests, kv_heads, max_positions, multiprocessors) < 1
            or max_queries not in (1, 7)):
        raise ValueError("invalid paged attention launch shape")
    ceiling = min(64 if max_queries == 1 else 32,
                  tr.cdiv(max_positions, 128 if max_queries == 1 else 512))
    occupancy = tr.cdiv(8*multiprocessors, requests*kv_heads)
    return max(1, min(ceiling, occupancy))


class PagedAttentionWorkspace:
    backend = 'triton'

    def __init__(self, requests, kv_heads, query_heads, max_queries, max_positions, device):
        if (min(requests, kv_heads, query_heads, max_positions) < 1
                or query_heads % kv_heads or max_queries not in (1, 7)):
            raise ValueError("paged attention requires one or seven queries per request")
        self.shape = (requests, kv_heads, query_heads, max_queries, max_positions)
        device_info = torch.cuda.get_device_properties(device)
        self.splits = attention_splits(requests, kv_heads, max_queries, max_positions,
                                      device_info.multi_processor_count)
        # Resolve hardware/shape choices before capture. SM100 needs different
        # packed-load lowering and tile sizes than Ampere or SM120.
        self.contiguous_codes = device_info.major != 10
        self.tile, self.warps = 32, 4
        if not self.contiguous_codes:
            self.tile = 64
            if requests*kv_heads < device_info.multi_processor_count and max_positions <= 8192:
                self.tile, self.warps = 128, 8
        self.views = 1 if max_queries == 1 else 2
        self.rows = max(16, tr.next_power_of_2(max_queries*(query_heads//kv_heads)))
        self.partial = torch.empty((requests, self.views, kv_heads, self.splits, self.rows, 128),
                                   dtype=torch.float32, device=device)
        self.lse = torch.empty(self.partial.shape[:-1], dtype=torch.float32, device=device)


def attention_workspace(requests, kv_heads, query_heads, max_queries, max_positions, device,
                        *, backend='triton'):
    if backend == 'fa4':
        from .hierarchical_native import NativeAttentionWorkspace
        cls = NativeAttentionWorkspace
    elif backend == 'triton':
        cls = PagedAttentionWorkspace
    else:
        raise ValueError('hierarchical attention backend must be triton or fa4')
    return cls(requests, kv_heads, query_heads, max_queries, max_positions, device)


def attention(cache, query, query_positions, query_start_loc, slots, block_table, *, bits=8,
              selected_ids=None, selected_counts=None, workspace, output=None):
    if workspace.backend == 'fa4':
        from .hierarchical_native import attention as native_attention
        return native_attention(cache, query, query_positions, query_start_loc, slots, block_table,
            bits=bits, selected_ids=selected_ids, selected_counts=selected_counts,
            workspace=workspace, output=output)
    n, h, hq, maxq, maxpos = workspace.shape
    if (cache.device.type != "cuda" or bits not in (4, 8) or h != cache.kv_heads
            or slots.shape != (n,) or n > workspace.partial.shape[0] or
            query.ndim != 3 or query.shape[1:] != (hq, 128) or query.dtype != torch.bfloat16 or
            query.device != cache.device or not query.is_contiguous() or len(query) > n*maxq or
            query_positions.shape != (len(query),) or query_start_loc.shape != (n+1,) or
            block_table.ndim != 2 or block_table.shape[0] != n or workspace.partial.device != cache.device):
        raise ValueError("incompatible paged attention inputs/workspace")
    sparse = selected_ids is not None
    if sparse and (selected_counts is None or selected_ids.ndim != 2
                   or selected_ids.shape[0] != n or selected_counts.shape != (n,)):
        raise ValueError("sparse attention requires logical block ids and counts")
    metadata = (query_positions, query_start_loc, slots, block_table)
    if sparse:
        metadata += (selected_ids, selected_counts)
    for tensor in metadata:
        if (tensor.device != cache.device or tensor.dtype not in (torch.int32, torch.int64)
                or tensor.stride(-1) != 1):
            raise ValueError("attention metadata must have contiguous integer device rows")
    if not sparse:
        selected_ids, selected_counts = block_table, slots  # unused pointers
    output = torch.empty_like(query) if output is None else output
    if (output.shape != query.shape or output.dtype != query.dtype or output.device != query.device
            or not output.is_contiguous()):
        raise ValueError("invalid output buffer")
    _attention[(n*h, workspace.splits, workspace.views)](query, query_positions, query_start_loc, slots,
        block_table, selected_ids, selected_counts, cache.state, cache.pages, cache.metadata, *cache.tail,
        workspace.partial, workspace.lse, h, hq//h, cache.pages.stride(0), block_table.stride(0),
        selected_ids.stride(0), bits, sparse, workspace.splits, workspace.rows, workspace.tile,
        128**-.5, workspace.views, workspace.contiguous_codes,
        num_warps=workspace.warps, num_stages=1, enable_fp_fusion=False)
    _merge[(n*h, maxq*(hq//h))](workspace.partial, workspace.lse, output, query_positions,
        query_start_loc, slots, cache.state, h, hq//h, workspace.splits, workspace.rows,
        tr.next_power_of_2(workspace.splits), workspace.views, num_warps=4)
    return output
