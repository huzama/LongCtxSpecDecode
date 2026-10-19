"""Fused packed-KV kernels for the Qwen3 128-channel GQA layout.

These are correctness-first Triton kernels, not a claim of FA4-equivalent
performance. Attention uses tensor-core matmuls, split-KV reduction and absolute
causality. Refinement loads are compiled out of the four-bit specialization.
No dequantized full-prefix tensor is allocated.
"""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice


@tr.jit
def _encode(X, C, R, S, Z, FIRST: tl.constexpr, H: tl.constexpr, VALUE: tl.constexpr):
    group, head = tl.program_id(0), tl.program_id(1)
    t, d = tl.arange(0, 128), tl.arange(0, 128)
    x = tl.load(X + ((group * 128 + t[:, None]) * H + head) * 128 + d[None, :]).to(tl.float32)
    if VALUE:
        minimum = tl.min(x, 1).to(tl.bfloat16).to(tl.float32)
        scale = ((tl.max(x, 1) - minimum) / 15).to(tl.bfloat16).to(tl.float32)
        scale = tl.where(scale > 0, scale, 1.)
        lo, step = minimum[:, None], scale[:, None]
        meta = ((FIRST + group) * 128 + t) * H + head
    else:
        minimum = tl.min(x, 0).to(tl.bfloat16).to(tl.float32)
        scale = ((tl.max(x, 0) - minimum) / 15).to(tl.bfloat16).to(tl.float32)
        scale = tl.where(scale > 0, scale, 1.)
        lo, step = minimum[None, :], scale[None, :]
        meta = ((FIRST + group) * H + head) * 128 + d
    coarse = tl.minimum(15., tl.maximum(0., libdevice.nearbyint(tl.div_rn(x-lo, step))))
    residual = tl.minimum(7., tl.maximum(-8., libdevice.nearbyint((x-(coarse*step+lo))*tl.div_rn(16., step))))+8
    pair = tl.broadcast_to((tl.arange(0, 64)*2)[None, :], (128, 64))
    cu = tl.gather(coarse.to(tl.uint8), pair, 1) | (tl.gather(coarse.to(tl.uint8), pair+1, 1) << 4)
    cr = tl.gather(residual.to(tl.uint8), pair, 1) | (tl.gather(residual.to(tl.uint8), pair+1, 1) << 4)
    out = (((FIRST+group)*128+t[:, None])*H+head)*64+tl.arange(0, 64)[None, :]
    tl.store(C+out, cu)
    tl.store(R+out, cr)
    tl.store(S+meta, scale)
    tl.store(Z+meta, minimum)


def encode(cache, key, value, first_group):
    if cache.head_dim != 128 or not key.is_contiguous() or not value.is_contiguous():
        raise ValueError("GPU encoding requires contiguous 128-channel Qwen3 K/V")
    for i, x in enumerate((key, value)):
        _encode[(key.shape[0]//128, cache.kv_heads)](
            x, cache.coarse[i], cache.refinement[i], cache.scales[i], cache.minima[i],
            first_group, cache.kv_heads, i == 1, num_warps=8, enable_fp_fusion=False)


@tr.jit(do_not_specialize=["N"])
def _attention(Q, QP, POS, STATE, KC, KR, KS, KZ, VC, VR, VS, VZ, KT, VT, PART, LSE,
               N, NQ: tl.constexpr, H: tl.constexpr, RATIO: tl.constexpr,
               BITS: tl.constexpr, SPLITS: tl.constexpr, SCALE: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr):
    head, split = tl.program_id(0), tl.program_id(1)
    view = tl.program_id(2)
    row, d = tl.arange(0, BM), tl.arange(0, 128)
    qrow, qhead = row // RATIO, head*RATIO + row % RATIO
    q = tl.load(Q+(qrow[:, None]*(H*RATIO)+qhead[:, None])*128+d[None, :],
                row[:, None] < NQ*RATIO, 0)
    qp = tl.load(QP+qrow, row < NQ*RATIO, -1)
    prefix = tl.load(STATE+1)
    length = tl.load(STATE)
    if NQ == 1:
        view = (tl.load(QP) >= prefix+256).to(tl.int32)
    row_visible = (row < NQ*RATIO) & ((qp >= prefix+256) == (view == 1))
    # The second view is needed only at a compression boundary. Each query
    # reads precisely one view; the BF16 recent buffer stays intact on rollback.
    active = tl.sum(row_visible.to(tl.int32), 0) > 0
    view_prefix = prefix+view*128
    tiles = tl.cdiv(N, SPLITS*BN)
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    denominator = tl.zeros((BM,), tl.float32)
    accum = tl.zeros((BM, 128), tl.float32)
    for tile in range(tl.where(active, tiles, 0)):
        offsets = (split*tiles+tile)*BN+tl.arange(0, BN)
        pos = tl.load(POS+offsets, offsets < N, -1).to(tl.int32)
        valid = (offsets < N) & (pos >= 0) & (pos < length)
        old = valid & (pos < view_prefix)
        tail = valid & (pos >= view_prefix)
        code_offset = (pos[:, None]*H+head)*64+d[None, :]//2
        shift = (d[None, :] % 2)*4
        kc = ((tl.load(KC+code_offset, old[:, None], 0).to(tl.int32) >> shift) & 15).to(tl.float32)
        ks_offset = ((pos[:, None]//128)*H+head)*128+d[None, :]
        ks = tl.load(KS+ks_offset, old[:, None], 0).to(tl.float32)
        kz = tl.load(KZ+ks_offset, old[:, None], 0).to(tl.float32)
        k = kc*ks+kz
        if BITS == 8:
            kr = ((tl.load(KR+code_offset, old[:, None], 0).to(tl.int32) >> shift) & 15).to(tl.float32)-8
            k += kr*(ks/16.)
        tail_offset = ((pos[:, None]-prefix)*H+head)*128+d[None, :]
        k += tl.load(KT+tail_offset, tail[:, None], 0).to(tl.float32)
        scores = tl.dot(q, tl.trans(k.to(tl.bfloat16))) * SCALE
        scores = tl.where(valid[None, :] & (pos[None, :] <= qp[:, None]) &
                          row_visible[:, None], scores, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(scores, 1))
        safe_max = tl.where(next_max == -float("inf"), 0., next_max)
        alpha = tl.exp(maximum-safe_max)
        prob = tl.exp(scores-safe_max[:, None])
        # Decode values only after QK and softmax have consumed the key
        # temporaries. Keeping both tiles live caused extensive register spills.
        vc = ((tl.load(VC+code_offset, old[:, None], 0).to(tl.int32) >> shift) & 15).to(tl.float32)
        vs_offset = pos*H+head
        vs = tl.load(VS+vs_offset, old, 0).to(tl.float32)
        vz = tl.load(VZ+vs_offset, old, 0).to(tl.float32)
        v = vc*vs[:, None]+vz[:, None]
        if BITS == 8:
            vr = ((tl.load(VR+code_offset, old[:, None], 0).to(tl.int32) >> shift) & 15).to(tl.float32)-8
            v += vr*(vs[:, None]/16.)
        v += tl.load(VT+tail_offset, tail[:, None], 0).to(tl.float32)
        accum = accum*alpha[:, None]+tl.dot(prob.to(tl.bfloat16), v.to(tl.bfloat16))
        denominator = denominator*alpha+tl.sum(prob, 1)
        maximum = next_max
    # Store normalized partials; masked splits carry -inf log normalizer.
    result = accum/tl.maximum(denominator[:, None], 1.e-30)
    base = (((view*H+head)*SPLITS+split)*BM+row[:, None])*128+d[None, :]
    tl.store(PART+base, result)
    tl.store(LSE+((view*H+head)*SPLITS+split)*BM+row, maximum+tl.log(denominator))


@tr.jit
def _merge(PART, LSE, OUT, QP, STATE, H: tl.constexpr, RATIO: tl.constexpr, NQ: tl.constexpr,
           SPLITS: tl.constexpr, BM: tl.constexpr, BS: tl.constexpr):
    head, row = tl.program_id(0), tl.program_id(1)
    view = (tl.load(QP+row//RATIO) >= tl.load(STATE+1)+256).to(tl.int32)
    head = view*H+head
    s, d = tl.arange(0, BS), tl.arange(0, 128)
    lse = tl.load(LSE+(head*SPLITS+s)*BM+row, s < SPLITS, -float("inf"))
    weights = tl.exp(lse-tl.max(lse, 0))
    weights /= tl.sum(weights, 0)
    part = tl.load(PART+((head*SPLITS+s[:, None])*BM+row)*128+d[None, :], s[:, None] < SPLITS, 0)
    out = tl.sum(part*weights[:, None], 0)
    target = ((row//RATIO)*(H*RATIO)+(head%H)*RATIO+row%RATIO)*128+d
    tl.store(OUT+target, out)


class AttentionWorkspace:
    """Reusable scratch for one layer call; never stores persistent KV values."""
    def __init__(self, kv_heads, query_heads, queries, positions, device):
        if query_heads % kv_heads or queries < 1 or positions < 1:
            raise ValueError("invalid attention workspace shape")
        self.shape = (kv_heads, query_heads, queries, positions)
        # Single-query drafts need more CTAs at small selection budgets.
        # Keep the measured lower-overhead split policy for block verification.
        self.splits = (min(64, max(1, tr.cdiv(positions, 128))) if queries == 1
                       else min(32, max(1, tr.cdiv(positions, 512))))
        self.rows = max(16, tr.next_power_of_2(queries*(query_heads//kv_heads)))
        self.partial = torch.empty((2, kv_heads, self.splits, self.rows, 128), dtype=torch.float32, device=device)
        self.lse = torch.empty((2, kv_heads, self.splits, self.rows), dtype=torch.float32, device=device)
        self.output = torch.empty((queries, query_heads, 128), dtype=torch.bfloat16, device=device)


def attention(cache, query, query_positions, positions, bits=8, workspace=None, scale=None):
    if cache.capacity*cache.kv_heads*128 >= 2**31:
        raise ValueError("packed attention requires 32-bit-safe cache offsets")
    if (bits not in (4, 8) or query.ndim != 3 or query.dtype != torch.bfloat16 or
            query.device != cache.device or query.shape[-1] != 128 or
            not query.is_contiguous() or not positions.is_contiguous() or positions.ndim != 1 or
            positions.dtype != torch.int64 or query_positions.dtype != torch.int64 or
            positions.device != query.device or query_positions.device != query.device):
        raise ValueError("invalid packed attention inputs")
    nq, hq, _ = query.shape
    if query_positions.shape != (nq,) or hq % cache.kv_heads:
        raise ValueError("invalid GQA/query-position shape")
    ws = workspace or AttentionWorkspace(cache.kv_heads, hq, nq, len(positions), query.device)
    if ws.shape != (cache.kv_heads, hq, nq, len(positions)):
        raise ValueError("workspace shape differs from attention inputs")
    ws.compiled_attention = _attention[(cache.kv_heads, ws.splits, 1 if nq == 1 else 2)](
        query, query_positions, positions, cache.state,
        cache.coarse[0], cache.refinement[0], cache.scales[0], cache.minima[0],
        cache.coarse[1], cache.refinement[1], cache.scales[1], cache.minima[1], *cache.tail,
        ws.partial, ws.lse, len(positions), nq, cache.kv_heads, hq//cache.kv_heads,
        bits, ws.splits, scale or 128**-.5, ws.rows, 128 if nq > 1 else 64, num_warps=8, num_stages=1,
        enable_fp_fusion=False)
    ws.compiled_merge = _merge[(cache.kv_heads, nq*(hq//cache.kv_heads))](
        ws.partial, ws.lse, ws.output, query_positions, cache.state, cache.kv_heads, hq//cache.kv_heads, nq,
        ws.splits, ws.rows, tr.next_power_of_2(ws.splits), num_warps=4)
    return ws.output
