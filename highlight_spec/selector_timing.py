"""Standalone native Highlight stages and independent selector checks."""
from dataclasses import replace
from types import MethodType, SimpleNamespace

import torch
import torch.nn.functional as F


class HighlightTiming:
    def __init__(self, capture, tokens):
        from transformers import Qwen3Config
        from spotattention.serve.bench.selector_compare.spotattention import make_config
        from spotattention.serve.indexer import Indexer
        from spotattention.serve.kernels.common import SelectScratch
        from spotattention.serve import kernels
        from spotattention.serve.rope import rope_tables

        self.hidden, self.hidden_all, self.weight, ln_w, ln_b = (
            capture[n].to("cuda").contiguous() for n in
            ("hidden", "hidden_all", "qkw_weight", "ln_weight", "ln_bias"))
        b, length = self.hidden_all.shape[:2]
        if b != 1 or length % 16 or tokens % 16 or tokens > length:
            raise ValueError("Highlight timing requires one complete-block prefix/budget")
        self.tokens, self.length = tokens, length
        cfg = make_config(Qwen3Config(**capture["config"]), length)
        self.cfg = cfg = replace(cfg, selection=replace(cfg.selection,
                                mode="topk_tokens", topk_tokens=tokens, topk_recency=64))
        nb = length // 16
        i32 = dict(device="cuda", dtype=torch.int32)
        table = torch.arange(nb, **i32).view(1, nb)
        indexer = Indexer.__new__(Indexer)
        torch.nn.Module.__init__(indexer)
        indexer.cfg, indexer.layer_idx = cfg, capture["layer"]
        indexer.w_scale = cfg.num_heads ** -0.5 * cfg.head_dim ** -0.5
        indexer.k_norm = torch.nn.LayerNorm(cfg.head_dim, eps=cfg.k_norm_eps,
                                          device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            indexer.k_norm.weight.copy_(ln_w)
            indexer.k_norm.bias.copy_(ln_b)
        indexer._tables = rope_tables(cfg, torch.device("cuda"))
        indexer._unit_scale = None
        indexer._kcfg = kernels.kernel_config(torch.device("cuda"))
        sel = SimpleNamespace(sparse_active=True, reuse_scores=False, num_blocks=nb, fa_table=table,
            scores=torch.empty((1, nb), device="cuda", dtype=torch.bfloat16),
            cmp_table=torch.empty((1, nb), **i32), seqused=torch.empty(1, **i32),
            cnt=torch.empty(1, **i32), ids=torch.empty((1, nb), **i32),
            prompt_len=torch.full((1,), length, **i32), kpred=torch.empty((cfg.num_layers, 1), **i32),
            stats=torch.zeros((cfg.num_layers, 3), device="cuda", dtype=torch.int64),
            select=SelectScratch.alloc(1, torch.device("cuda"), "topk_tokens"))
        sel.scores_of = lambda layer: sel.scores
        indexer.sel = sel
        self.indexer = indexer
        self.positions = torch.tensor([length - 1], device="cuda")
        self.slot = self.positions.clone()
        self.md = SimpleNamespace(block_table=table, seq_lens=torch.full((1,), length, **i32),
            query_start_loc=torch.arange(2, **i32), is_prefilling=torch.zeros(1, device="cuda", dtype=torch.bool))
        self.cache = torch.empty((nb, 16, cfg.kcache_head_size), device="cuda", dtype=torch.bfloat16)
        self.all_positions = torch.arange(length, device="cuda")[None]
        self.offsets = torch.arange(16, device="cuda")
        self.initialize()
        # Split only at the production score/select boundary. Projection,
        # LayerNorm, RoPE, cache write and score kernels are unchanged.
        def score_only(instance, q, w, md, cache):
            self.q, self.w = q, w
            instance._score(q, w, md, cache, instance._kernel_config(q.device), instance.sel.scores)
        indexer._score_and_select = MethodType(score_only, indexer)

    def initialize(self):
        from spotattention.attention.sparse.indexer_rope import apply_qk_rope_fused
        h, d = self.cfg.num_heads, self.cfg.head_dim
        for start in range(0, self.length, 2048):
            hidden = self.hidden_all[:, start:start + 2048]
            q, k, _ = F.linear(hidden, self.weight).split([h*d, h*d, h], -1)
            k = F.layer_norm(k.reshape(1, -1, h, d), (d,),
                            self.indexer.k_norm.weight, self.indexer.k_norm.bias, self.cfg.k_norm_eps)
            _, k = apply_qk_rope_fused(q.reshape(1, -1, h, d), k, *self.indexer._tables,
                                     self.all_positions[:, start:start + hidden.shape[1]], self.cfg.rope_dim)
            self.cache.view(1, self.length, h*d)[:, start:start + hidden.shape[1]].copy_(k.flatten(-2))
        return self.cache

    def score(self):
        self.indexer.run(F.linear(self.hidden, self.weight), self.positions, self.md, self.slot, self.cache)
        return self.indexer.sel.scores

    def rank(self):
        from spotattention.serve.indexer import Indexer
        self.indexer.sel.reuse_scores = True
        Indexer._score_and_select(self.indexer, self.q, self.w, self.md, self.cache)
        self.indexer.sel.reuse_scores = False
        return self.indexer.sel.cmp_table[:, :self.tokens // 16]

    def run(self):
        self.score()
        return self.rank()

    def token_ids(self):
        blocks = self.indexer.sel.cmp_table[:, :self.tokens // 16].long()
        return (blocks[..., None] * 16 + self.offsets).flatten(-2)[:, None]

    def validate(self):
        from spotattention.serve.kernels import reference
        self.score()
        expected = reference.score_blocks(self.q, self.w, self.cache, self.md.block_table,
            self.md.seq_lens, self.md.query_start_loc, self.md.is_prefilling)
        torch.testing.assert_close(self.indexer.sel.scores, expected, rtol=0.015, atol=0.04)
        error = float((self.indexer.sel.scores.float() - expected.float()).abs().max())
        expected = self.indexer.sel.scores.clone()
        self.rank()
        sel = self.indexer.sel
        # Recency is forced inside K. BF16 score ties use lower block IDs.
        expected[:, -4:] = torch.finfo(torch.bfloat16).max
        expected_ids = torch.argsort(expected, descending=True, stable=True)[:, :self.tokens // 16].sort().values
        torch.testing.assert_close(sel.cmp_table[:, :self.tokens // 16].long(), expected_ids, rtol=0, atol=0)
        assert int(sel.seqused[0]) == self.tokens and int(sel.cnt[0]) == self.tokens // 16
        return {"passed": True, "score_max_abs_error": error, "retained_tokens": self.tokens}

    @property
    def cache_bytes(self):
        return self.cache.numel() * self.cache.element_size()

    def close(self):
        from importlib import import_module
        descriptors = import_module("spotattention.serve.kernels.block_scores")._descriptors
        prefix = (self.cache.data_ptr(), tuple(self.cache.shape), self.cache.dtype)
        for key in tuple(descriptors):
            if key[:3] == prefix:
                del descriptors[key]


def validate_triforce(q, k, v, chunks, scores, indices, chunk_size):
    from .selectors import gather_draft_kv
    expected_chunks = k.float().reshape(k.shape[0], -1, chunk_size, k.shape[2], k.shape[3]).mean(2).to(k.dtype)
    torch.testing.assert_close(chunks, expected_chunks, rtol=0, atol=0)
    for head in range(q.shape[2]):
        kh = head // (q.shape[2] // k.shape[2])
        products = q[:, 0, head, None].float() * chunks[:, :, kh].float()
        expected = products.sum(-1)
        bound = products.abs().sum(-1) * (2 * (k.shape[-1] - 1) * torch.finfo(torch.float32).eps)
        bound += expected.abs() * torch.finfo(torch.bfloat16).eps / 2 + 1e-6
        assert torch.all((scores[:, head].float() - expected).abs() <= bound), "TriForce score error"
        ids = indices[:, head]
        assert torch.equal(ids[:, :chunk_size], torch.arange(chunk_size, device=k.device)[None])
        selected = (ids[:, ::chunk_size] // chunk_size)[:, 1:]
        torch.testing.assert_close(scores[:, head, 1:].topk(selected.shape[-1]).values.sort().values,
                                   scores[:, head].gather(-1, selected).sort().values)
        assert torch.unique(ids).numel() == ids.numel(), "duplicate TriForce indices"
    actual_k, actual_v = gather_draft_kv(k, v, indices, query_heads=q.shape[2])
    for head in range(q.shape[2]):
        kh = head // (q.shape[2] // k.shape[2])
        for source, actual in ((k, actual_k), (v, actual_v)):
            torch.testing.assert_close(actual[0, :, head], source[0, :, kh].index_select(0, indices[0, head]), rtol=0, atol=0)
    return {"passed": True, "retained_tokens_per_query_head": indices.shape[-1]}
