"""Paired FFN diagnostic using stock HF Qwen3, FA4 and native Highlight selection.

Eager, batch one. Gather and Python costs make this unsuitable for speed claims.
Only the transient draft tail is replaced; all variants share dense prefix KV.
"""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from .ffn_quant import PackedFFN, TensorStore, relative_l2


class Selector:
    def __init__(self, config, store, layer, prompt_len):
        from spotattention.serve.bench.selector_compare.spotattention import make_config
        from spotattention.serve.indexer import Indexer
        from spotattention.serve.kernels.common import SelectScratch
        from spotattention.serve import kernels
        from spotattention.serve.rope import rope_tables

        cfg = make_config(config, prompt_len + 144, cache_format="bf16")
        cfg = replace(cfg, selection=replace(cfg.selection, mode="topk_fraction",
                      topk_fraction=.07, topk_recency=64, verify_fraction=0.0))
        nb = (prompt_len + 144 + 15) // 16
        i32 = dict(device="cuda", dtype=torch.int32)
        table = torch.arange(nb, **i32).view(1, nb)
        idx = Indexer.__new__(Indexer)
        torch.nn.Module.__init__(idx)
        idx.cfg, idx.layer_idx = cfg, layer
        idx.w_scale = cfg.num_heads ** -.5 * cfg.head_dim ** -.5
        idx.k_norm = torch.nn.LayerNorm(cfg.head_dim, eps=cfg.k_norm_eps,
                                       device="cuda", dtype=torch.bfloat16)
        prefix = f"model.layers.{layer}.self_attn.indexer."
        self.weight = torch.cat([store.get(prefix + n + ".weight") for n in
                                ("wq_b", "wk", "weights_proj")]).cuda()
        idx.k_norm.load_state_dict({n: store.get(prefix + "k_norm." + n) for n in ("weight", "bias")})
        idx._tables = rope_tables(cfg, torch.device("cuda"))
        idx._unit_scale, idx._kcfg = None, kernels.kernel_config(torch.device("cuda"))
        sel = SimpleNamespace(sparse_active=False, reuse_scores=False, num_blocks=nb,
            fa_table=table, scores=torch.empty((1, nb), device="cuda", dtype=torch.bfloat16),
            cmp_table=torch.empty((1, nb), **i32), seqused=torch.empty(1, **i32),
            cnt=torch.empty(1, **i32), ids=torch.empty((1, nb), **i32),
            prompt_len=torch.full((1,), prompt_len, **i32), kpred=torch.empty((cfg.num_layers, 1), **i32),
            stats=torch.zeros((cfg.num_layers, 3), device="cuda", dtype=torch.int64),
            select=SelectScratch.alloc(1, torch.device("cuda"), "topk_fraction"))
        sel.scores_of = lambda layer: sel.scores
        idx.sel = sel
        self.indexer = idx
        self.md = SimpleNamespace(block_table=table, seq_lens=torch.zeros(1, **i32),
            query_start_loc=torch.arange(2, **i32), is_prefilling=torch.zeros(1, device="cuda", dtype=torch.bool))
        self.cache = torch.empty((nb, 16, cfg.kcache_head_size), device="cuda", dtype=cfg.kcache_dtype)
        self.offsets = torch.arange(16, device="cuda")

    def update(self, hidden, positions, sparse, depth):
        idx = self.indexer
        idx.sel.sparse_active, idx.sel.reuse_scores = sparse, depth > 0
        self.md.seq_lens.fill_(int(positions[-1]) + 1)
        self.md.query_start_loc[1] = positions.numel()
        idx.run(F.linear(hidden.flatten(0, 1), self.weight), positions, self.md, positions, self.cache)
        if not sparse:
            return None
        blocks = idx.sel.cmp_table[0, :int(idx.sel.cnt[0])].long()
        tokens = (blocks[:, None] * 16 + self.offsets).flatten()
        tokens = tokens[tokens <= positions[-1]]
        # Native selector must retain the current token and recent window inside K.
        end = int(positions[-1]) + 1
        if not torch.isin(torch.arange(end - 64, end, device="cuda"), tokens).all():
            raise AssertionError("native selector dropped a recent token")
        return tokens


class Probe:
    def __init__(self, checkpoint, quant, prompt_len):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new

        raw = json.loads((Path(checkpoint) / "config.json").read_text())
        theta = raw.get("rope_theta") or raw["rope_parameters"]["rope_theta"]
        raw.update(use_dsa=False, max_position_embeddings=65536,
                   rope_parameters=dict(rope_type="yarn", factor=2.,
                                        original_max_position_embeddings=32768, rope_theta=float(theta)))
        self.model = Qwen3ForCausalLM.from_pretrained(checkpoint, config=Qwen3Config(**raw),
            dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa", local_files_only=True).eval()
        ALL_ATTENTION_FUNCTIONS.register("ffn_probe_fa4", self.attention)
        ALL_MASK_ATTENTION_FUNCTIONS.register("ffn_probe_fa4", lambda **kwargs: None)
        self.model.config._attn_implementation = "ffn_probe_fa4"
        self.original = [layer.mlp for layer in self.model.model.layers]
        workspace = marlin_make_workspace_new(torch.device("cuda"))
        store = TensorStore(quant)
        self.quant = [PackedFFN(store, i, workspace) for i in range(len(self.original))]
        store = TensorStore(checkpoint)
        self.selectors = [Selector(self.model.config, store, i, prompt_len) for i in range(len(self.original))]
        self.sparse, self.depth, self.check_ffn = False, 0, False
        self.ffn_checks = []
        self.hidden = {}
        self.hooks = []
        for i, layer in enumerate(self.model.model.layers):
            def capture(module, args, kwargs, idx=i):
                self.hidden[idx] = kwargs.get("hidden_states", args[0] if args else None)
            self.hooks.append(layer.self_attn.register_forward_pre_hook(capture, with_kwargs=True))
            def ffn_check(module, args, output, idx=i):
                if not self.check_ffn:
                    return
                x = args[0]
                packed = self.quant[idx](x)
                reference = self.quant[idx].reference(x)
                error = relative_l2(packed, reference)
                record = dict(layer=idx, packed_vs_dequant_l2=error,
                    int4_vs_bf16_l2=relative_l2(packed, output),
                    packed_vs_dequant_max_abs=float((packed.float() - reference.float()).abs().max()))
                self.ffn_checks.append(record)
                if error > .01 or not torch.isfinite(packed).all():
                    raise AssertionError(f"packed FFN failed numerical check: {record}")
            self.hooks.append(self.original[i].register_forward_hook(ffn_check))

    def attention(self, module, query, key, value, attention_mask, scaling=None, **kwargs):
        from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd
        q, k, v = (x.transpose(1, 2).contiguous() for x in (query, key, value))
        tokens = self.selectors[module.layer_idx].update(
            self.hidden.pop(module.layer_idx), self.positions, self.sparse, self.depth)
        if self.sparse:
            if q.shape[1] != 1:
                raise ValueError("only single-token sparse drafting is supported")
            k, v = k.index_select(1, tokens), v.index_select(1, tokens)
        return _flash_attn_fwd(q, k, v, causal=not self.sparse,
                               softmax_scale=scaling, num_splits=1)[0], None

    def set_variant(self, variant):
        quant, attention = variant.split("/")
        if quant not in ("bf16", "int4") or attention not in ("full", "highlight"):
            raise ValueError(variant)
        self.sparse = attention == "highlight"
        for layer, mlp in zip(self.model.model.layers, self.quant if quant == "int4" else self.original):
            layer.mlp = mlp

    @torch.inference_mode()
    def forward(self, ids, cache, depth=0, all_logits=False):
        self.depth = depth
        start = cache.get_seq_length()
        self.positions = torch.arange(start, start + len(ids), device="cuda")
        result = self.model(input_ids=torch.tensor([ids], device="cuda"), past_key_values=cache,
                            use_cache=True, logits_to_keep=0 if all_logits else 1)
        if not torch.isfinite(result.logits).all():
            raise AssertionError("non-finite logits")
        return result.logits[0].float()

    @torch.inference_mode()
    def rollout(self, cache, base, anchor, variant, teacher=None):
        cache.crop(base)
        self.set_variant(variant)
        current, predictions = anchor, []
        for depth in range(6):
            token = int(self.forward([current], cache, depth)[-1].argmax())
            predictions.append(token)
            current = teacher[depth] if teacher is not None else token
        cache.crop(base)
        return predictions

    @torch.inference_mode()
    def prefill(self, ids):
        from transformers.cache_utils import DynamicCache
        self.set_variant("bf16/full")
        cache = DynamicCache(config=self.model.config)
        for offset in range(0, len(ids), 2048):
            self.forward(ids[offset:offset + 2048], cache)
        return cache

    @torch.inference_mode()
    def validate_dense_backend(self, ids):
        """Check the FA4 callback against stock HF SDPA on an existing prefix."""
        from transformers.cache_utils import DynamicCache
        self.set_variant("bf16/full")
        fa = self.forward(ids[:64], DynamicCache(config=self.model.config))
        self.model.config._attn_implementation = "sdpa"
        try:
            sdpa = self.model(input_ids=torch.tensor([ids[:64]], device="cuda"),
                              use_cache=False, logits_to_keep=1).logits[0].float()
        finally:
            self.model.config._attn_implementation = "ffn_probe_fa4"
            self.hidden.clear()
        error = relative_l2(fa, sdpa)
        if error > .02 or int(fa.argmax()) != int(sdpa.argmax()):
            raise AssertionError(f"FA4/SDPA dense sanity failed: relative L2 {error}")
        return dict(logit_relative_l2=error, next_token_equal=True, tokens=64)
