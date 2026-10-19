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
    preselected: cutlass.Constexpr[bool]

    @cute.jit
    def load_page_table(self, n_block: Int32):
        # Logical sparse IDs and physical page mappings are resolved together
        # below; the native traversal calls this hook before each K/V pair.
        pass

    @cute.jit
    def load_KV(self, n_block: Int32, sX: cute.Tensor, kind: cutlass.Constexpr[str]):
        pages, meta, kt, vt, table, selected, info, qpos, physical_ids = self.data
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
        if const_expr(self.sparse and kind == "K"):
            # One descriptor per selected page, shared by K/V loads and all
            # reconstruction lanes. The existing 512-byte scratch holds eight
            # logical bases, eight physical pages and 8x8 key-metadata pages.
            if self.thread < 64:
                pg, shard = self.thread//8, self.thread%8
                index = n_block*self.tile_n+pg*16
                valid = (n_block >= 0) & (index < self.seqlen) & (slot >= 0)
                logical = Int32(-1)
                if valid:
                    logical = selected[req, index//16]
                pos = logical*16
                valid = valid & (pos >= 0) & (pos < length)
                old = valid & (pos < prefix)
                physical, metadata_page = Int32(0), Int32(0)
                if old:
                    if const_expr(self.preselected):
                        physical = Int32(physical_ids[req, index//16])
                    else:
                        physical = Int32(table[req, logical])
                    metadata_page = Int32(table[req, (pos//128)*8+shard])
                if shard == 0:
                    logical_tile[pg] = pos if valid else Int32(-1)
                    logical_tile[8+pg] = physical
                logical_tile[16+pg*8+shard] = metadata_page
            ready.arrive_and_wait()
        async_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            cutlass.Uint32, num_bits_per_copy=128)
        predicate = cute.make_rmem_tensor((4,), cutlass.Boolean)
        for row in cutlass.range_constexpr(self.tile_n // (self.threads // 4)):
            t = row * (self.threads // 4) + self.thread // 4
            if const_expr(self.sparse):
                base = logical_tile[t//16]
                pos = base+t%16
                valid = (base >= 0) & (pos < length)
                old = valid & (pos < prefix)
                physical = Int64(logical_tile[8+t//16])
            else:
                index = n_block * self.tile_n + t
                pos = index
                valid = (n_block >= 0) & (index < self.seqlen) & (slot >= 0)
                old = valid & (pos >= 0) & (pos < prefix) & (pos < length)
                # Promote before multiplying by a possibly >4 GiB stride.
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

        # Collect each scale/minimum once per tile (once per selected page for
        # sparse K). Reconstruction then uses shared reads, avoiding a serial
        # global-metadata dependency for every decoded token.
        if const_expr(kind == "K"):
            if const_expr(self.sparse):
                pg, lane = self.thread//16, self.thread%16
                pos = logical_tile[pg]
                old = (pos >= 0) & (pos < prefix)
                d = lane*8
                sparse_meta_page = Int64(logical_tile[16+pg*8+d//16])
                # Eight adjacent BF16 metadata values form one native 128-bit
                # asynchronous copy. The head-contiguous page layout supplies
                # these vectors without scalar gather/store instructions.
                metadata_atom = cute.make_copy_atom(
                    cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                    cutlass.BFloat16, num_bits_per_copy=128)
                metadata_predicate = cute.make_rmem_tensor((8,), cutlass.Boolean)
                metadata_predicate.fill(old)
                for field in cutlass.range_constexpr(2):
                    src_ptr = meta.iterator + sparse_meta_page*meta.stride[0] + field*meta.stride[1] + self.head*meta.stride[3]+d%16
                    src = cute.make_tensor(cute.make_ptr(cutlass.BFloat16, src_ptr.toint(),
                        cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((8,)))
                    dst_ptr = metadata_tile.iterator + pg*256+field*128+d
                    dst = cute.make_tensor(cute.make_ptr(cutlass.BFloat16, dst_ptr.toint(),
                        cute.AddressSpace.smem, assumed_align=16), cute.make_layout((8,)))
                    cute.copy(metadata_atom, src, dst, pred=metadata_predicate)
                cute.arch.cp_async_commit_group()
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
            if const_expr(self.sparse):
                base = logical_tile[t//16]
                pos = base+t%16
                valid = (base >= 0) & (pos < length)
                physical = Int64(logical_tile[8+t//16])
            else:
                index = n_block*self.tile_n+t
                pos = index
                valid = (n_block >= 0) & (index < self.seqlen) & (slot >= 0)
                physical = Int64(0)
                if valid & (pos >= 0) & (pos < prefix) & (pos < length):
                    physical = Int64(table[req, pos//16])
            old = valid & (pos >= 0) & (pos < prefix) & (pos < length)
            staged_scale, staged_minimum = cutlass.BFloat16(0.), cutlass.BFloat16(0.)
            if old:
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
                base = logical_tile[t//16]
                pos = base+t%16
                valid = (base >= 0) & (pos < length)
            else:
                pos = n_block*self.tile_n+t
                valid = (n_block >= 0) & (pos < self.seqlen) & (slot >= 0)
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
