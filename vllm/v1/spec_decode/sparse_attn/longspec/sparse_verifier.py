# SPDX-License-Identifier: Apache-2.0
"""Opt-in approximate verification: selected history and a complete causal tail."""

import torch

from .kernels.c2q_scores import c2q_lse, c2q_metric
from .kernels.mass_select import mass_select
from .kernels.slot_table import index_to_slots


class SparseVerifier:
    def __init__(self, layers, batch, max_len, heads, queries, spec, device):
        self.spec, self.queries = spec, queries
        self.selected_scores = (
            getattr(spec, "sparse_attn_verify_score_scope", "full") == "selected"
        )
        i32 = dict(device=device, dtype=torch.int32)
        self.known = torch.zeros(batch, **i32)
        self.blocks = torch.full((batch,), -1, **i32)
        # Full width permits an unseen request to use its entire history.
        self.table = torch.zeros(layers, batch, max_len, **i32)
        self.logical = torch.empty_like(self.table) if self.selected_scores else None
        self.used = torch.zeros(layers, batch, **i32)
        self.valid = torch.zeros(layers, batch, **i32)
        self.budget = torch.zeros(layers, batch, **i32)
        self.partial = self.lse = None
        if not self.selected_scores:
            self.partial = torch.empty(
                batch,
                2 * heads,
                (max_len + 127) // 128,
                device=device,
                dtype=torch.float32,
            )
            self.lse = torch.empty(heads, batch * queries, device=device)
        self.counters = torch.zeros(3, device=device, dtype=torch.int64)

    def eligible(self, kwargs):
        batch = kwargs["seqused_k"].shape[0]
        return (
            kwargs.get("max_seqlen_q") == self.queries
            and kwargs["q"].shape[0] == batch * self.queries
            and kwargs.get("causal", False)
            and kwargs.get("window_size", (-1, -1)) in ((-1, -1), [-1, -1])
            and not kwargs.get("softcap", 0)
        )

    def prepare(self, kwargs, metric):
        batch = kwargs["seqused_k"].shape[0]
        seq = kwargs["seqused_k"]
        qlen = kwargs["cu_seqlens_q"].diff()
        prefix = (seq - qlen).clamp_min(0)
        known = torch.minimum(self.known[:batch], prefix)
        known = torch.where(
            self.blocks[:batch] == kwargs["block_table"][:, 0], known, 0
        )
        spec = self.spec
        reserved = known.clamp_max(spec.sparse_attn_sink)
        reserved += (known - reserved).clamp_max(spec.sparse_attn_recent)
        budget = (
            torch.ceil(prefix * spec.sparse_attn_verify_ratio).int()
            - reserved
            - (prefix - known)
        ).clamp_min(0)
        budget = torch.minimum(budget, known - reserved)
        self.valid.zero_()
        self.budget.zero_()
        self.valid[:, :batch] = known
        self.budget[:, :batch] = budget
        layers, capacity, max_len = metric.shape
        mass_select(
            metric.view(layers * capacity, max_len),
            self.valid.view(-1),
            self.budget.view(-1),
            self.budget.view(-1),
            self.table.view(layers * capacity, -1),
            self.used.view(-1),
            1.0,
            spec.sparse_attn_sink,
            spec.sparse_attn_recent,
        )
        index_to_slots(
            self.table[:, :batch],
            kwargs["block_table"],
            self.used[:, :batch],
            known,
            seq,
            kwargs["k"].shape[1],
            self.logical[:, :batch] if self.logical is not None else None,
        )
        self.used[:, :batch] += (seq - known).unsqueeze(0)
        kept = (self.used[:, :batch] - qlen.unsqueeze(0)).clamp_min(0)
        self.counters[0] += (qlen > 0).sum()
        self.counters[1] += kept.sum()
        self.counters[2] += prefix.sum() * layers

    def attention_kwargs(self, kwargs, layer):
        batch = kwargs["seqused_k"].shape[0]
        result = dict(kwargs)
        for name in ("k", "v"):
            cache = kwargs[name]
            result[name] = cache.view(-1, 1, *cache.shape[-2:])
        result["block_table"] = self.table[layer, :batch]
        result["seqused_k"] = self.used[layer, :batch]
        return result

    def full_lse(self, kwargs):
        return c2q_lse(kwargs, self.partial, self.lse)

    def refresh_selected(self, kwargs, layer, lse, valid, entry, output):
        """Score selected keys using sparse LSE; scatter to logical positions.

        Negative sentinels rank below even underflowed zero attention weights.
        Fixed-budget selection is required; these are not attention masses for
        excluded tokens. Full-attention fallback overwrites the valid prefix.
        """
        selected = self.attention_kwargs(kwargs, layer)
        batch = valid.shape[0]
        output.fill_(-1)
        scale = kwargs.get("softmax_scale")
        if scale is None:
            scale = kwargs["q"].shape[-1] ** -0.5
        c2q_metric(
            q=selected["q"],
            k_cache=selected["k"],
            block_table=selected["block_table"],
            cu_seqlens_q=kwargs["cu_seqlens_q"],
            valid_lens=valid,
            reduce_entry=entry,
            lse=lse,
            softmax_scale=scale,
            use_weight=True,
            output=output,
            logical_indices=self.logical[layer, :batch],
            selected_lens=self.used[layer, :batch],
        )

    def remember(self, kwargs, valid):
        batch = kwargs["seqused_k"].shape[0]
        self.known.zero_()
        self.blocks.fill_(-1)
        self.known[:batch] = valid[:batch]
        self.blocks[:batch] = kwargs["block_table"][:, 0]

    def stats(self):
        rounds, kept, full = self.counters.tolist()
        return dict(
            ratio=self.spec.sparse_attn_verify_ratio,
            request_rounds=rounds,
            historical_tokens_kept=kept,
            historical_tokens_available=full,
            mean_historical_fraction=kept / full if full else None,
        )
