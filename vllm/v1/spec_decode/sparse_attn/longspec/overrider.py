# SPDX-License-Identifier: Apache-2.0
"""LongSpec drafting: the attention overrider.

Verify pass: every layer's attention hook writes its per-token metric into
one row of a per-layer buffer; the last layer runs one fused selection over
all layers and requests. Draft pass: the first step converts the selected
indices to slots and gathers them, every step appends the newest token, and
masked layers return a zero attention output. All per-round work is static
in shape and free of host syncs, so the verify pass can replay as a CUDA
graph. Design: notes/method.md.
"""

import torch

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.v1.spec_decode.sparse_attn.attn_overrider import BaseAttnOverrider

from .config import LongSpecConfig
from .kernels.mass_select import mass_select
from .kernels.slot_table import index_to_slots
from .layer_skip import install_layer_skip
from .portable.draft_kv import build_draft_kv
from .portable.kernel_support import flash_attn_version
from .portable.score_collection import build_score_collector
from .stats import SelectionStats
from .verify_attention import (
    packed_verify_attention,
    packed_verify_eligible,
)

logger = init_logger(__name__)


class LongSpecAttnOverrider(BaseAttnOverrider):

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self.cfg = cfg = LongSpecConfig.from_vllm_config(vllm_config)
        spec = vllm_config.speculative_config
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        num_query_heads = model_config.get_num_attention_heads(parallel_config)
        layers, batch, max_len = (self.num_layers, self.max_batch_size,
                                  self.max_model_len)

        # Table width: the cap plus the reserved ranges plus the tail the
        # draft appends (verified tokens and its own tokens).
        self._width = (self.max_tokens + cfg.sink + cfg.recent +
                       2 * self.num_spec_tokens + 1)
        self.max_blocks = (self._width + self.block_size - 1) // self.block_size

        fa_version = flash_attn_version()
        num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        # FA3's scheduler packs GQA on its own; FA2 only at one query.
        self._group = num_query_heads // num_kv_heads
        self._packed_verify = (cfg.packed_verify and fa_version == 2
                               and self._group > 1)
        self._scores = build_score_collector(
            spec.sparse_attn_score_source, fa_version, batch, num_query_heads,
            max_len, device)
        self._draft_kv = build_draft_kv(
            spec.sparse_attn_draft_kv, fa_version, layers, batch, self._width,
            self.block_size, device,
            num_kv_heads,
            model_config.get_head_size(), model_config.dtype)
        logger.info(
            "%s on FA%d: theta %.3f, sink %d, recent %d, cap ratio %.3f, "
            "min %d, attn-skip %s, layer-skip %s; scores via %s, draft KV via "
            "%s, packed verify %s", cfg.variant, fa_version, cfg.theta,
            cfg.sink, cfg.recent, cfg.ratio,
            cfg.min_tokens, sorted(cfg.skip_attn_layers),
            sorted(cfg.skip_layers), type(self._scores).__name__,
            type(self._draft_kv).__name__, self._packed_verify)

        i32 = dict(dtype=torch.int32, device=device)
        self._metric = torch.zeros(layers, batch, max_len, dtype=torch.bfloat16,
                                   device=device)
        self._table = torch.zeros(layers, batch, self._width, **i32)
        self._used = torch.zeros(layers, batch, **i32)
        self._valid_lens = torch.zeros(batch, **i32)
        self._k_min = torch.zeros(batch, **i32)
        self._k_max = torch.zeros(batch, **i32)
        self._reduce_entry = torch.zeros(batch, **i32)
        self._valid_all = torch.zeros(layers, batch, **i32)
        self._k_min_all = torch.zeros(layers, batch, **i32)
        self._k_max_all = torch.zeros(layers, batch, **i32)
        self._stats = SelectionStats(layers, device)
        self._metadata_initialized = False
        self.batch_size = 0

        # Compile the selection kernel now, outside any graph capture.
        mass_select(
            torch.zeros(1, 16, dtype=torch.bfloat16, device=device),
            torch.zeros(1, **i32), torch.zeros(1, **i32), torch.zeros(1, **i32),
            torch.zeros(1, 16, **i32), torch.zeros(1, **i32),
            cfg.theta, cfg.sink, cfg.recent)

    # ---- model hooks -----------------------------------------------------

    def bind_model(self, model) -> None:
        if self.cfg.skip_layers:
            install_layer_skip(model, self.cfg.skip_layers, self)

    def stats(self) -> dict:
        return self._stats.snapshot()

    def reset_stats(self) -> None:
        self._stats.reset()

    # ---- verify ----------------------------------------------------------

    def enter_propose(self):
        super().enter_propose()
        self._metadata_initialized = False
        self._draft_kv.begin_propose()

    def _verify_attention(self, *args, **kwargs):
        layer = self.curr_layer
        if layer == 0:
            self._begin_verify(kwargs)
        batch = self.batch_size

        self._scores.verify_kwargs(kwargs, batch)
        kwargs["return_softmax_lse"] = True
        if self._packed_verify and packed_verify_eligible(kwargs, self._group):
            out, lse = packed_verify_attention(
                BaseAttnOverrider._original_attn_func, kwargs)
        else:
            out, lse = BaseAttnOverrider._original_attn_func(*args, **kwargs)
        scale = kwargs.get("softmax_scale")
        if scale is None:
            scale = kwargs["q"].shape[-1] ** -0.5
        self._scores.reduce(
            kwargs=kwargs, lse=lse, softmax_scale=scale,
            valid_lens=self._valid_lens[:batch],
            reduce_entry=self._reduce_entry[:batch],
            output=self._metric[layer, :batch], use_weight=True)

        if layer == self.num_layers - 1:
            self._select()
        return out

    def _begin_verify(self, kwargs: dict) -> None:
        seqlens_k: torch.Tensor = kwargs["seqused_k"]
        cu_seqlens_q: torch.Tensor = kwargs["cu_seqlens_q"]
        seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        batch = self.batch_size = seqlens_k.shape[0]
        cfg = self.cfg

        single_query = seqlens_q == 1
        in_prefill = seqlens_q > self.num_spec_tokens + 1
        padded = seqlens_q == 0  # graph padding: no query, block table -1
        self._reduce_entry.fill_(0)
        self._reduce_entry[:batch].masked_fill_(single_query, 1)
        self._reduce_entry[:batch].masked_fill_(in_prefill, 2)

        # Scored prefix: [0, seqlens_k - seqlens_q + 1), whole cache in prefill.
        valid = seqlens_k + 1 - seqlens_q.masked_fill(in_prefill, 1)
        valid.masked_fill_(padded, 0)
        self._valid_lens.zero_()
        self._valid_lens[:batch] = valid

        k_max = torch.ceil(seqlens_k * cfg.ratio).int()
        k_max.clamp_(min=cfg.min_tokens, max=self.max_tokens)
        k_max.clamp_max_(valid)
        self._k_max.zero_()
        self._k_max[:batch] = k_max
        self._k_min.zero_()
        self._k_min[:batch] = valid.clamp_max(cfg.min_tokens)
        self._metadata_initialized = False

    def _select(self) -> None:
        layers, batch = self.num_layers, self.max_batch_size
        self._valid_all.copy_(self._valid_lens.unsqueeze(0).expand(layers, batch))
        self._k_min_all.copy_(self._k_min.unsqueeze(0).expand(layers, batch))
        self._k_max_all.copy_(self._k_max.unsqueeze(0).expand(layers, batch))
        mass_select(
            self._metric.view(layers * batch, self.max_model_len),
            self._valid_all.view(-1), self._k_min_all.view(-1),
            self._k_max_all.view(-1), self._table.view(layers * batch, -1),
            self._used.view(-1), self.cfg.theta, self.cfg.sink,
            self.cfg.recent)
        self._stats.accumulate(self._used, self._valid_lens)

    # ---- draft -----------------------------------------------------------

    def _draft_attention(self, *args, **kwargs):
        # The verify pass may replay as a graph, so the draft derives its own
        # batch size (sequences, not tokens) from its live metadata.
        self.batch_size = kwargs["cu_seqlens_q"].numel() - 1
        self._begin_draft_layer(kwargs["seqused_k"], kwargs["block_table"])
        layer = self.curr_layer
        batch = self.batch_size
        if layer in self.cfg.skip_attn_layers:
            out = kwargs["out"]
            out.zero_()
            return out
        self._draft_kv.attention_kwargs(
            kwargs, layer, self._table[layer, :batch], self._used[layer, :batch])
        return BaseAttnOverrider._original_attn_func(*args, **kwargs)

    def _begin_draft_layer(self, seqlens_k: torch.Tensor,
                           block_table: torch.Tensor) -> None:
        """Once per draft step, at the first layer the draft reaches."""
        if self.curr_layer != 0:
            return
        batch = self.batch_size
        used = self._used[:, :batch]
        if self._metadata_initialized:
            used += 1
            return
        self._metadata_initialized = True
        valid = self._valid_lens[:batch]
        # Slots for the whole draft: the selection, then [valid, final len).
        index_to_slots(self._table[:, :batch], block_table, used, valid,
                       seqlens_k - 1 + self.num_spec_tokens, self.block_size)
        used += (seqlens_k - valid).unsqueeze(0)
        used.clamp_(min=0)

    def note_skipped_layer(self) -> None:
        """A wrapped decoder layer was bypassed: keep the step boundary and
        the call-order layer counter aligned."""
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, dict):
            metadata = next(iter(metadata.values()))
        self.batch_size = metadata.query_start_loc.numel() - 1
        self._begin_draft_layer(metadata.seq_lens, metadata.block_table)
        self.curr_layer = (self.curr_layer + 1) % self.num_layers
