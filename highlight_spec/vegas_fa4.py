"""Opt-in fused Vegas logit collection for the pinned FA4 SM100 backend.

Captures raw QK from FA4's existing accumulator, before scaling/softmax.
No second QK matmul or KV scan. This is an FA4 adaptation, not the paper's
FA3 kernel. The private launcher and compile cache never patch vLLM globals.
GPU numerical validation is required before using its timings as evidence.
"""

from functools import lru_cache
import inspect
from types import FunctionType

import torch

from .selectors import _check_qk


@lru_cache(maxsize=1)
def _backend():
    import cutlass
    import cutlass.cute as cute
    from vllm.vllm_flash_attn.cute import interface
    from vllm.vllm_flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100

    expected = ("self", "tSrS_t2r", "thr_tmem_load", "thr_mma_qk", "batch_idx",
                "head_idx", "m_block", "n_block", "softmax", "seqlen", "aux_data",
                "fastdiv_mods", "head_divmod")
    if tuple(inspect.signature(FlashAttentionForwardSm100.apply_score_mod).parameters) != expected:
        raise RuntimeError("FA4 collector hook changed; review the extension before use")

    @cute.jit
    def collector_marker(scores, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        # Used only to enable FA4's accumulator hook and distinguish its JIT key.
        return scores

    class CollectingForward(FlashAttentionForwardSm100):
        @cute.jit
        def apply_score_mod(self, tSrS_t2r, thr_tmem_load, thr_mma_qk, batch_idx,
                            head_idx, m_block, n_block, softmax, seqlen,
                            aux_data=None, fastdiv_mods=(None, None), head_divmod=None):
            coords = cute.make_identity_tensor((self.m_block_size, self.n_block_size))
            coords = cute.domain_offset((m_block * self.m_block_size,
                                         n_block * self.n_block_size), coords)
            partition = thr_mma_qk.partition_C(coords)
            partition = partition[(None, None), 0, 0]
            indices = thr_tmem_load.partition_D(partition)
            query = indices[0][0]
            head = head_idx
            if cutlass.const_expr(self.pack_gqa):
                query, offset = divmod(query, head_divmod)
                head = head_idx * self.qhead_per_kvhead + offset
            # Unlike score_mod's read-only aux indexing, never wrap padding.
            scores, rows = aux_data.tensors
            if query < seqlen.seqlen_q and head < scores.shape[2]:
                for boundary in cutlass.range_constexpr(2):
                    if query == rows[batch_idx, boundary]:
                        for i in cutlass.range_constexpr(cute.size(tSrS_t2r.shape)):
                            token = indices[i][1]
                            if token < scores.shape[3]:
                                scores[batch_idx, boundary, head, token] = tSrS_t2r[i].to(scores.element_type)
            # score_mod mode applies the scale here; FA4 then uses exp2(log2(e)*S).
            tSrS_t2r.store(tSrS_t2r.load() * softmax.softmax_scale)

    original = interface._flash_attn_fwd
    namespace = dict(original.__globals__)
    namespace["FlashAttentionForwardSm100"] = CollectingForward
    private = FunctionType(original.__code__, namespace, "_flash_attn_fwd",
                           original.__defaults__, original.__closure__)
    private.__kwdefaults__ = original.__kwdefaults__
    private.compile_cache = {}  # Separate from stock FA4, including disk cache.
    namespace["_flash_attn_fwd"] = private
    return private, collector_marker


class FusedVegasAttention:
    """Fixed-shape contiguous KV benchmark adapter for BF16 Qwen3 on B200.

    Construction and allocation stay outside timing. Invoke ``attention``
    then ``reduce``; time reduction separately. A capture has one common
    prefix length and phase. Paged/ragged serving integration is not implied.
    """

    def __init__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *,
                 prefix_tokens: int, phase: str):
        _check_qk(q, k)
        if k.shape != v.shape or k.device != v.device or k.dtype != v.dtype:
            raise ValueError("K/V shape, dtype and device must match")
        if not q.is_cuda or q.dtype != torch.bfloat16 or q.shape[-1] != 128:
            raise ValueError("collector supports CUDA BF16 head dimension 128")
        if torch.cuda.get_device_capability(q.device) != (10, 0):
            raise ValueError("collector requires SM100; other architectures are unvalidated")
        if phase not in {"prefill", "verify"}:
            raise ValueError("phase must be prefill or verify")
        limit = k.shape[1] if phase == "prefill" else k.shape[1] - q.shape[1] + 1
        if not 0 < prefix_tokens <= limit:
            raise ValueError("prefix is outside the selected queries' causal visibility")
        if not all(x.is_contiguous() for x in (q, k, v)):
            raise ValueError("capture tensors must be contiguous")
        self.q, self.k, self.v = q, k, v
        self.launch, self.marker = _backend()
        last = q.shape[1] - 1
        self.boundary_count = 1 if phase == "prefill" or last == 0 else 2
        boundaries = [last, -1] if self.boundary_count == 1 else [0, last]
        self.rows = torch.tensor([boundaries] * q.shape[0], device=q.device, dtype=torch.int32)
        self.scores = torch.empty((q.shape[0], 2, q.shape[2], prefix_tokens),
                                  device=q.device, dtype=torch.float32)
        self.out = torch.empty_like(q)

    def attention(self) -> torch.Tensor:
        self.launch(self.q, self.k, self.v, causal=True, num_splits=1,
                    softmax_scale=self.q.shape[-1] ** -0.5, score_mod=self.marker,
                    aux_tensors=[self.scores, self.rows], out=self.out)
        return self.out

    def reduce(self) -> torch.Tensor:
        # Keep the accumulator precision through head/boundary reduction.
        # BF16 storage introduces avoidable error when signed logits cancel.
        return self.scores[:, :self.boundary_count].mean(dim=(1, 2), dtype=torch.float32)

    @property
    def workspace_bytes(self) -> int:
        return sum(x.numel() * x.element_size() for x in (self.scores, self.rows))
