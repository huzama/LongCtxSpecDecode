"""Decode hierarchical KV tiles into FlashAttention's BF16 shared memory.

The storage and quantization rules are unchanged. Reconstruction is tile-local;
no second KV cache or full-context dequantization buffer is allocated.
"""
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Float32, const_expr
from cutlass.cute.nvgpu import cpasync
from cutlass import pipeline
from quack.cute_dsl_utils import ParamsBase


@cute.jit
def signed_codes(word: cutlass.Uint32):
    """Convert eight offset-binary nibbles using CUTLASS's packed INT4 path."""
    packed = cute.make_rmem_tensor((1,), cutlass.Uint32)
    packed[0] = word ^ cutlass.Uint32(0x88888888)
    # Explicitly expose eight elements. A singleton layout has zero stride;
    # recast_tensor alone retains its length and silently broadcasts one code.
    values = cute.make_tensor(cute.recast_ptr(packed.iterator, dtype=cutlass.Int4),
                              cute.make_layout((8,)))
    return values.load().to(cutlass.BFloat16)


@dataclass
class HierarchicalKVManager(ParamsBase):
    data: tuple
    scratch: tuple
    batch: Int32
    head: Int32
    thread: Int32
    seqlen: Int32
    tile_n: cutlass.Constexpr[int]
    threads: cutlass.Constexpr[int]
    bits: cutlass.Constexpr[int]
    sparse: cutlass.Constexpr[bool]

    @cute.jit
    def load_page_table(self, n_block: Int32):
        # Logical sparse IDs and physical page mappings are resolved together
        # below; the native traversal calls this hook before each K/V pair.
        pass

    @cute.jit
    def load_KV(self, n_block: Int32, sX: cute.Tensor, kind: cutlass.Constexpr[str]):
        pages, meta, kt, vt, table, selected, info, qpos = self.data
        slot, prefix, length, req = (info[self.batch, i] for i in range(4))
        heads = kt.shape[2]
        plane = 16 * heads * 64
        # Preserve the native UMMA/swizzled shared layout. Native V operands
        # are transposed; both loader views below expose (token, channel).
        shared = cute.make_tensor(sX.iterator, cute.make_layout(
            (sX.shape[0][0], (sX.shape[0][1], sX.shape[2])),
            stride=(sX.stride[0][0], (sX.stride[0][1], sX.stride[2]))))
        if const_expr(kind == "V"):
            shared = cute.make_tensor(shared.iterator, cute.select(shared.layout, mode=[1, 0]))
        packed_tile, metadata_tile, logical_tile = self.scratch
        ready = pipeline.NamedBarrier(barrier_id=11, num_threads=self.threads)
        # All producers must finish reading the previous packed tile before
        # any producer reuses it. Only these load warps participate.
        ready.arrive_and_wait()
        async_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            cutlass.Uint32, num_bits_per_copy=128)
        predicate = cute.make_rmem_tensor((4,), cutlass.Boolean)
        for row in cutlass.range_constexpr(self.tile_n // (self.threads // 4)):
            t = row * (self.threads // 4) + self.thread // 4
            index = n_block * self.tile_n + t
            logical = index // 16
            valid = (n_block >= 0) & (index < self.seqlen) & (slot >= 0)
            if const_expr(self.sparse):
                if valid:
                    logical = selected[req, logical]
            pos = logical * 16 + index % 16
            old = valid & (pos >= 0) & (pos < prefix) & (pos < length)
            # Dense allocations exceed 4 GiB per layer. Promote the physical
            # index before multiplying by a tensor stride, including the
            # BF16 metadata view whose element offsets can exceed 2**31.
            physical = Int64(0)
            if old:
                physical = Int64(table[req, pos // 16])
            d = (self.thread % 4) * 4
            off = (pos % 16 * heads + self.head) * 16 + d
            if const_expr(kind == "V"):
                off = off + 2 * plane // 4
            predicate.fill(old)
            for precision in cutlass.range_constexpr(2 if self.bits == 8 else 1):
                src_ptr = pages.iterator + physical*pages.stride[0] + off + precision*plane//4
                src = cute.make_tensor(cute.make_ptr(cutlass.Uint32, src_ptr.toint(),
                    cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((4,)))
                dst_ptr = packed_tile.iterator + t*16 + d + precision*self.tile_n*16
                dst = cute.make_tensor(cute.make_ptr(cutlass.Uint32, dst_ptr.toint(),
                    cute.AddressSpace.smem, assumed_align=16), cute.make_layout((4,)))
                cute.copy(async_atom, src, dst, pred=predicate)
        cute.arch.cp_async_commit_group()

        if const_expr(self.sparse):
            # Resolve each reconstruction row once while its metadata is staged.
            index = n_block*self.tile_n+self.thread
            valid = (n_block >= 0) & (index < self.seqlen) & (slot >= 0)
            logical = index//16
            if const_expr(self.sparse):
                if valid:
                    logical = selected[req,logical]
            pos = logical*16+index%16
            logical_tile[self.thread] = pos if valid & (pos >= 0) & (pos < length) else Int32(-1)

        # Collect each scale/minimum once per tile (once per selected page for
        # sparse K). Reconstruction then uses shared reads, avoiding a serial
        # global-metadata dependency for every decoded token.
        if const_expr(kind == "K"):
            if const_expr(self.sparse):
                pg = self.thread//16 if const_expr(self.sparse) else 0
                lane = self.thread%16 if const_expr(self.sparse) else self.thread
                index = n_block*self.tile_n+pg*16
                logical = index//16
                valid = (n_block >= 0) & (index < self.seqlen) & (slot >= 0)
                if const_expr(self.sparse):
                    if valid:
                        logical = selected[req,logical]
                pos = logical*16
                old = valid & (pos >= 0) & (pos < prefix) & (pos < length)
                # Each producer group resolves one selected page and then loads
                # its channels. Keep the shared layout used by reconstruction.
                if old:
                    for group in cutlass.range_constexpr(8 if self.sparse else 1):
                        d = lane+group*16
                        mp = Int64(table[req,(pos//128)*8+d//16])
                        metadata_tile[pg,0,d]=meta[mp,0,d%16,self.head]
                        metadata_tile[pg,1,d]=meta[mp,1,d%16,self.head]
                else:
                    for group in cutlass.range_constexpr(8 if self.sparse else 1):
                        d = lane+group*16
                        metadata_tile[pg,0,d]=cutlass.BFloat16(0.)
                        metadata_tile[pg,1,d]=cutlass.BFloat16(0.)
            else:
                d = self.thread
                for pg in cutlass.range_constexpr(8 if self.sparse else 1):
                    index = n_block*self.tile_n + pg*16
                    logical = index//16
                    valid = (n_block >= 0) & (index < self.seqlen) & (slot >= 0)
                    if const_expr(self.sparse):
                        if valid:
                            logical = selected[req, logical]
                    pos = logical*16
                    old = valid & (pos >= 0) & (pos < prefix) & (pos < length)
                    staged_scale, staged_minimum = cutlass.BFloat16(0.), cutlass.BFloat16(0.)
                    if old:
                        mp = Int64(table[req, (pos//128)*8+d//16])
                        staged_scale = meta[mp, 0, d%16, self.head]
                        staged_minimum = meta[mp, 1, d%16, self.head]
                    metadata_tile[pg, 0, d] = staged_scale
                    metadata_tile[pg, 1, d] = staged_minimum
        else:
            t = self.thread
            index = n_block*self.tile_n + t
            logical = index//16
            valid = (n_block >= 0) & (index < self.seqlen) & (slot >= 0)
            if const_expr(self.sparse):
                if valid:
                    logical = selected[req, logical]
            pos = logical*16+index%16
            old = valid & (pos >= 0) & (pos < prefix) & (pos < length)
            staged_scale, staged_minimum = cutlass.BFloat16(0.), cutlass.BFloat16(0.)
            if old:
                physical = Int64(table[req, pos//16])
                staged_scale = meta[physical, 2, pos%16, self.head]
                staged_minimum = meta[physical, 3, pos%16, self.head]
            metadata_tile[0, 0, t] = staged_scale
            metadata_tile[0, 1, t] = staged_minimum
        cute.arch.cp_async_wait_group(0)
        ready.arrive_and_wait()

        # Match FA4's eight-BF16-element vector copy per producer lane.
        # One aligned uint32 read supplies eight packed codes. Decode in
        # registers, then issue a 128-bit shared store into the native layout.
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16,
                                   num_bits_per_copy=128)
        tiled = cute.make_tiled_copy_tv(atom,
            cute.make_ordered_layout((self.threads // 8, 8), order=(1, 0)),
            cute.make_layout((1, 8)))
        thread_copy = tiled.get_slice(self.thread)
        dest = thread_copy.partition_D(shared)
        coords = thread_copy.partition_S(cute.make_identity_tensor((self.tile_n, 128)))
        fragment = cute.make_rmem_tensor((8,), cutlass.BFloat16)
        # A runtime row loop bounds live registers within the producer budget.
        for row in cutlass.range(cute.size(dest, mode=[1]), unroll=1):
            t = coords[0, row, 0][0]
            if const_expr(self.sparse):
                pos = logical_tile[t]
                valid = pos >= 0
            else:
                index = n_block * self.tile_n + t
                logical = index // 16
                valid = (n_block >= 0) & (index < self.seqlen) & (slot >= 0)
                if const_expr(self.sparse):
                    if valid:
                        logical = selected[req, logical]
                pos = logical * 16 + index % 16
                valid = valid & (pos >= 0) & (pos < length)
            old = valid & (pos < prefix)
            for k in cutlass.range_constexpr(cute.size(dest, mode=[2])):
                d0 = coords[0, row, k][1]
                fragment.fill(0.)
                if old:
                    # CUTLASS dispatches Int4 -> BF16 to its native packed
                    # eight-value conversion. XOR converts our offset-binary
                    # codes to signed INT4 without changing their bit width.
                    coarse = signed_codes(packed_tile[t, d0//8, 0])
                    coarse_f32 = coarse.to(Float32) + 8.
                    scales = cute.make_rmem_tensor((8,), cutlass.BFloat16)
                    minima = cute.make_rmem_tensor((8,), cutlass.BFloat16)
                    if const_expr(kind == "K"):
                        pg = t//16 if const_expr(self.sparse) else 0
                        scale_src = cute.make_tensor(metadata_tile.iterator + pg*256+d0,
                                                     cute.make_layout((8,)))
                        min_src = cute.make_tensor(metadata_tile.iterator + pg*256+128+d0,
                                                   cute.make_layout((8,)))
                        cute.copy(atom, scale_src, scales)
                        cute.copy(atom, min_src, minima)
                    else:
                        scales.fill(metadata_tile[0, 0, t])
                        minima.fill(metadata_tile[0, 1, t])
                    scale_f32 = scales.load().to(Float32)
                    value = coarse_f32 * scale_f32 + minima.load().to(Float32)
                    if const_expr(self.bits == 8):
                        residual = signed_codes(packed_tile[t, d0//8, 1])
                        value = value + residual.to(Float32) * (scale_f32 * (1. / 16.))
                    fragment.store(value.to(cutlass.BFloat16))
                elif valid:
                    tail = kt if const_expr(kind == "K") else vt
                    tail_ptr = tail.iterator + slot*tail.stride[0] + (pos%263)*tail.stride[1] + self.head*128 + d0
                    tail_src = cute.make_tensor(cute.make_ptr(cutlass.BFloat16, tail_ptr.toint(),
                        cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((8,)))
                    cute.copy(atom, tail_src, fragment)
                cute.copy(atom, fragment, dest[None, row, k])
