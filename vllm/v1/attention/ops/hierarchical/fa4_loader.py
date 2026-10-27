# SPDX-License-Identifier: Apache-2.0
"""Reconstruct packed hierarchical KV directly in FA4's shared-memory stages."""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, const_expr, pipeline
from cutlass.cute.nvgpu import cpasync
from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100
from flash_attn.cute.utils import AuxData
from quack.cute_dsl_utils import ParamsBase


@cute.jit
def _signed_codes(word: cutlass.Uint32):
    packed = cute.make_rmem_tensor((1,), cutlass.Uint32)
    packed[0] = word ^ cutlass.Uint32(0x88888888)
    values = cute.make_tensor(
        cute.recast_ptr(packed.iterator, dtype=cutlass.Int4), cute.make_layout((8,))
    )
    return values.load().to(cutlass.BFloat16)


@dataclass
class PackedKVManager(ParamsBase):
    data: tuple
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
        # Each row can name an unrelated token, so resolve it in load_KV.
        pass

    @cute.jit
    def key_metadata(self, meta, table, req, group, d, scales, minima):
        mp = Int64(table[req, group * 8 + d // 16])
        base = (
            mp * meta.stride[0] + (d % 16) * meta.stride[2] + self.head * meta.stride[3]
        )
        atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128
        )
        for field in cutlass.range_constexpr(2):
            ptr = meta.iterator + base + field * meta.stride[1]
            src = cute.make_tensor(
                cute.make_ptr(
                    cutlass.BFloat16,
                    ptr.toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                cute.make_layout((8,)),
            )
            dst = scales if const_expr(field == 0) else minima
            cute.copy(atom, src, dst)

    @cute.jit
    def load_KV(self, n_block: Int32, sX: cute.Tensor, kind: cutlass.Constexpr[str]):
        pages, meta, kt, vt, table, selected, info, qpos = self.data[:8]
        packed, staged_meta = self.data[8:]
        slot, prefix, length, req = (info[self.batch, i] for i in range(4))
        heads = kt.shape[2]
        plane_words = 16 * heads * 16
        shared = cute.make_tensor(
            sX.iterator,
            cute.make_layout(
                (sX.shape[0][0], (sX.shape[0][1], sX.shape[2])),
                stride=(sX.stride[0][0], (sX.stride[0][1], sX.stride[2])),
            ),
        )
        if const_expr(kind == "V"):
            shared = cute.make_tensor(
                shared.iterator, cute.select(shared.layout, mode=[1, 0])
            )
        atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128
        )
        tiled = cute.make_tiled_copy_tv(
            atom,
            cute.make_ordered_layout((self.threads // 8, 8), order=(1, 0)),
            cute.make_layout((1, 8)),
        )
        thread_copy = tiled.get_slice(self.thread)
        dest = thread_copy.partition_D(shared)
        coords = thread_copy.partition_S(cute.make_identity_tensor((self.tile_n, 128)))
        async_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            cutlass.Uint32,
            num_bits_per_copy=128,
        )
        predicate = cute.make_rmem_tensor((4,), cutlass.Boolean)
        for copy_row in cutlass.range_constexpr(self.tile_n // (self.threads // 4)):
            t = copy_row * (self.threads // 4) + self.thread // 4
            index = n_block * self.tile_n + t
            valid = (n_block >= 0) & (index < self.seqlen) & (slot >= 0)
            pos = index
            if const_expr(self.sparse):
                pos = Int32(-1)
                if valid & (index < selected.shape[1]):
                    pos = selected[req, index]
            old = valid & (pos >= 0) & (pos < prefix) & (pos < length)
            physical, safe_pos = Int64(0), Int32(0)
            if old:
                physical, safe_pos = Int64(table[req, pos // 16]), pos
            word = (self.thread % 4) * 4
            off = ((safe_pos % 16) * heads + self.head) * 16 + word
            if const_expr(kind == "V"):
                off = off + 2 * plane_words
            predicate.fill(old)
            for precision in cutlass.range_constexpr(2 if self.bits == 8 else 1):
                packed_src_ptr = (
                    pages.iterator
                    + physical * pages.stride[0]
                    + off
                    + precision * plane_words
                )
                packed_src = cute.make_tensor(
                    cute.make_ptr(
                        cutlass.Uint32,
                        packed_src_ptr.toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    cute.make_layout((4,)),
                )
                packed_dst_ptr = (
                    packed.iterator + precision * self.tile_n * 16 + t * 16 + word
                )
                packed_dst = cute.make_tensor(
                    cute.make_ptr(
                        cutlass.Uint32,
                        packed_dst_ptr.toint(),
                        cute.AddressSpace.smem,
                        assumed_align=16,
                    ),
                    cute.make_layout((4,)),
                )
                cute.copy(async_atom, packed_src, packed_dst, pred=predicate)
        cute.arch.cp_async_commit_group()
        if const_expr(kind == "K" and not self.sparse):
            if self.thread < 16:
                channel = self.thread * 8
                valid_group = (
                    (n_block >= 0)
                    & (slot >= 0)
                    & (n_block * self.tile_n < self.seqlen)
                    & (n_block * self.tile_n < prefix)
                )
                metadata_page = Int64(0)
                if valid_group:
                    metadata_page = Int64(table[req, n_block * 8 + channel // 16])
                metadata_atom = cute.make_copy_atom(
                    cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                    cutlass.BFloat16,
                    num_bits_per_copy=128,
                )
                metadata_pred = cute.make_rmem_tensor((8,), cutlass.Boolean)
                metadata_pred.fill(valid_group)
                for field in cutlass.range_constexpr(2):
                    metadata_src_ptr = (
                        meta.iterator
                        + metadata_page * meta.stride[0]
                        + field * meta.stride[1]
                        + self.head * meta.stride[3]
                        + (channel % 16) * meta.stride[2]
                    )
                    metadata_src = cute.make_tensor(
                        cute.make_ptr(
                            cutlass.BFloat16,
                            metadata_src_ptr.toint(),
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        cute.make_layout((8,)),
                    )
                    metadata_dst_ptr = staged_meta.iterator + field * 128 + channel
                    metadata_dst = cute.make_tensor(
                        cute.make_ptr(
                            cutlass.BFloat16,
                            metadata_dst_ptr.toint(),
                            cute.AddressSpace.smem,
                            assumed_align=16,
                        ),
                        cute.make_layout((8,)),
                    )
                    cute.copy(
                        metadata_atom, metadata_src, metadata_dst, pred=metadata_pred
                    )
                cute.arch.cp_async_commit_group()
        elif const_expr(kind == "V"):
            index = n_block * self.tile_n + self.thread
            valid = (n_block >= 0) & (index < self.seqlen) & (slot >= 0)
            pos = index
            if const_expr(self.sparse):
                pos = Int32(-1)
                if valid & (index < selected.shape[1]):
                    pos = selected[req, index]
            staged_scale, staged_minimum = cutlass.BFloat16(0.0), cutlass.BFloat16(0.0)
            if valid & (pos >= 0) & (pos < prefix) & (pos < length):
                physical = Int64(table[req, pos // 16])
                staged_scale = meta[physical, 2, pos % 16, self.head]
                staged_minimum = meta[physical, 3, pos % 16, self.head]
            staged_meta[0, self.thread] = staged_scale
            staged_meta[1, self.thread] = staged_minimum
        cute.arch.cp_async_wait_group(0)
        # Native NamedBarrierFwdSm100 uses IDs 1–10; ID 11 is loader-only.
        ready = pipeline.NamedBarrier(barrier_id=11, num_threads=self.threads)
        ready.arrive_and_wait()
        values = cute.make_rmem_tensor((8,), cutlass.BFloat16)
        scales = cute.make_rmem_tensor((8,), cutlass.BFloat16)
        minima = cute.make_rmem_tensor((8,), cutlass.BFloat16)
        # A complete dense tile shares one precision view and K metadata.
        # Decode columns outside rows to reuse that metadata without tail branches.
        historical = cutlass.Boolean(False)
        if const_expr(not self.sparse):
            tile_end = (n_block + 1) * self.tile_n
            historical = (
                (n_block >= 0)
                & (slot >= 0)
                & (tile_end <= prefix)
                & (tile_end <= length)
                & (tile_end <= self.seqlen)
            )
        if historical:
            for column in cutlass.range_constexpr(cute.size(dest, mode=[2])):
                d = coords[0, 0, column][1]
                if const_expr(kind == "K"):
                    for field in cutlass.range_constexpr(2):
                        smeta_ptr = staged_meta.iterator + field * 128 + d
                        smeta_src = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.BFloat16,
                                smeta_ptr.toint(),
                                cute.AddressSpace.smem,
                                assumed_align=16,
                            ),
                            cute.make_layout((8,)),
                        )
                        smeta_dst = scales if const_expr(field == 0) else minima
                        cute.copy(atom, smeta_src, smeta_dst)
                    key_step = scales.load().to(Float32)
                    key_minimum = minima.load().to(Float32)
                for row in cutlass.range(cute.size(dest, mode=[1]), unroll=1):
                    t = coords[0, row, column][0]
                    if const_expr(kind == "K"):
                        step, minimum = key_step, key_minimum
                    else:
                        scales.fill(staged_meta[0, t])
                        minima.fill(staged_meta[1, t])
                        step = scales.load().to(Float32)
                        minimum = minima.load().to(Float32)
                    coarse = _signed_codes(packed[0, t, d // 8]).to(Float32) + 8.0
                    decoded = coarse * step + minimum
                    if const_expr(self.bits == 8):
                        residual = _signed_codes(packed[1, t, d // 8])
                        decoded = decoded + residual.to(Float32) * (step * (1.0 / 16.0))
                    values.store(decoded.to(cutlass.BFloat16))
                    cute.copy(atom, values, dest[None, row, column])
        else:
            for row in cutlass.range(cute.size(dest, mode=[1]), unroll=1):
                index = n_block * self.tile_n + coords[0, row, 0][0]
                valid = (n_block >= 0) & (index < self.seqlen) & (slot >= 0)
                pos = index
                if const_expr(self.sparse):
                    pos = Int32(-1)
                    if valid & (index < selected.shape[1]):
                        pos = selected[req, index]
                valid = valid & (pos >= 0) & (pos < length)
                old = valid & (pos < prefix)
                for column in cutlass.range_constexpr(cute.size(dest, mode=[2])):
                    d = coords[0, row, column][1]
                    values.fill(0.0)
                    if old:
                        t = coords[0, row, column][0]
                        coarse = _signed_codes(packed[0, t, d // 8]).to(Float32) + 8.0
                        if const_expr(kind == "K"):
                            if const_expr(self.sparse):
                                self.key_metadata(
                                    meta, table, req, pos // 128, d, scales, minima
                                )
                            else:
                                for field in cutlass.range_constexpr(2):
                                    smeta_ptr = staged_meta.iterator + field * 128 + d
                                    smeta_src = cute.make_tensor(
                                        cute.make_ptr(
                                            cutlass.BFloat16,
                                            smeta_ptr.toint(),
                                            cute.AddressSpace.smem,
                                            assumed_align=16,
                                        ),
                                        cute.make_layout((8,)),
                                    )
                                    smeta_dst = (
                                        scales if const_expr(field == 0) else minima
                                    )
                                    cute.copy(atom, smeta_src, smeta_dst)
                        else:
                            scales.fill(staged_meta[0, t])
                            minima.fill(staged_meta[1, t])
                        step = scales.load().to(Float32)
                        decoded = coarse * step + minima.load().to(Float32)
                        if const_expr(self.bits == 8):
                            residual = _signed_codes(packed[1, t, d // 8])
                            decoded = decoded + residual.to(Float32) * (
                                step * (1.0 / 16.0)
                            )
                        values.store(decoded.to(cutlass.BFloat16))
                    elif valid:
                        tail = kt if const_expr(kind == "K") else vt
                        offset = (
                            slot * tail.stride[0]
                            + (pos % 263) * tail.stride[1]
                            + self.head * 128
                            + d
                        )
                        ptr = tail.iterator + offset
                        src = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.BFloat16,
                                ptr.toint(),
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            cute.make_layout((8,)),
                        )
                        cute.copy(atom, src, values)
                    cute.copy(atom, values, dest[None, row, column])
        # Do not let a faster loader warp replace scratch still being decoded.
        ready.arrive_and_wait()


class PackedAttentionSm100(FlashAttentionForwardSm100):
    def __init__(self, *args, bits, sparse, **kwargs):
        if not hasattr(FlashAttentionForwardSm100, "make_paged_kv_manager"):
            raise RuntimeError("FA4 hierarchical loader overlay is not installed")
        super().__init__(*args, **kwargs)
        self.packed_bits, self.packed_sparse = bits, sparse

    def _setup_attributes(self):
        super()._setup_attributes()
        self.kv_stage -= 1  # Reserve room for one packed tile and its metadata.
        assert self.head_dim_padded == self.head_dim_v_padded == 128
        assert self.n_block_size == self.m_block_size == 128 and self.q_stage == 1
        assert not self.use_2cta_instrs and not self.uneven_kv_smem
        assert self.load_warp_ids == (4, 5, 6, 7)

    def extend_shared_storage(self, storage):
        annotations = dict(storage._annotations)
        annotations["packed_tile"] = cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint32, 2 * self.n_block_size * 16], 16
        ]
        annotations["packed_metadata"] = cute.struct.Align[
            cute.struct.MemRange[cutlass.BFloat16, 256], 16
        ]
        extended = cute.struct(
            type("PackedSharedStorage", (), {"__annotations__": annotations})
        )
        assert all(extended._offsets[k] == v for k, v in storage._offsets.items())
        assert extended.__sizeof__() <= 227 * 1024
        return extended

    def prepare_load_aux_data(self, aux_data, storage):
        packed = cute.make_tensor(
            storage.packed_tile.data_ptr(),
            cute.make_layout(
                (2, self.n_block_size, 16), stride=(self.n_block_size * 16, 16, 1)
            ),
        )
        metadata = cute.make_tensor(
            storage.packed_metadata.data_ptr(),
            cute.make_layout((2, 128), stride=(128, 1)),
        )
        return AuxData((*aux_data.tensors, packed, metadata), aux_data.scalars)

    def make_paged_kv_manager(
        self,
        mPageTable,
        mK,
        mV,
        page_size_divmod,
        batch,
        head,
        thread,
        seqlen,
        leftpad,
        tile_n,
        head_dim,
        head_dim_v,
        threads,
        dtype,
        *,
        aux_data,
    ):
        return PackedKVManager(
            aux_data.tensors,
            batch,
            head,
            thread,
            seqlen,
            tile_n,
            threads,
            self.packed_bits,
            self.packed_sparse,
        )

    def load(self, *args, **kwargs):
        # Preserve b33's compute budgets. The four 128-thread warpgroups use
        # (184+64+80+176)*128=64512 registers, within the 65536-register limit.
        cute.arch.setmaxregister_increase(176)
        return super().load(*args, **kwargs)

    @cute.jit
    def load_KV(
        self,
        tma_atom,
        tXgX,
        tXsX,
        paged_kv_manager,
        sX,
        block,
        pipeline_kv,
        producer_state,
        K_or_V: cutlass.Constexpr,
        page_idx=None,
        extra_tx_count=None,
    ):
        pipeline_kv.producer_acquire(producer_state)
        stage = producer_state.index
        paged_kv_manager.load_KV(block, sX[None, None, None, stage], K_or_V)
        cute.arch.fence_proxy("async.shared", space="cta")
        pipeline_kv.producer_commit(producer_state)
