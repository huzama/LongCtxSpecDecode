"""Real Qwen3 inputs for selector timing; no quality or throughput measurement.

Chunked dense FA4 prefill followed by six greedy tokens and a seven-query
replay. Capture the same full prompt KV for all selectors. TriForce and
Highlight use the last prefill query; Vegas observes both replay boundaries.
Large tensors live only for this prompt and are never written to disk.
"""
import hashlib
import json
from pathlib import Path

import torch


def tensor_digest(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(memoryview(value.view(torch.uint8).numpy()).cast("B")).hexdigest()


class ModelCapture:
    def __init__(self, checkpoint, context, layers):
        from safetensors import safe_open
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        from benchmarks.decode import rope_parameters, yarn_factor

        self.checkpoint, self.context, self.layers = Path(checkpoint), context, tuple(layers)
        raw = json.loads((self.checkpoint / "config.json").read_text())
        factor = yarn_factor(context + 7)
        if factor is not None:
            raw["rope_parameters"] = rope_parameters(checkpoint, factor)
            raw["max_position_embeddings"] = int(factor * 32768)
        raw["use_dsa"] = False
        self.model = Qwen3ForCausalLM.from_pretrained(
            checkpoint, config=Qwen3Config(**raw), dtype=torch.bfloat16,
            device_map="cuda", attn_implementation="sdpa", local_files_only=True).eval()
        self.config = self.model.config.to_dict()
        if (self.config["model_type"] != "qwen3" or self.config["head_dim"] != 128
                or any(i < 0 or i >= len(self.model.model.layers) for i in layers)):
            raise ValueError("unsupported model/layers")
        self.weights = {}
        names = ("wq_b.weight", "wk.weight", "weights_proj.weight", "k_norm.weight", "k_norm.bias")
        found = {i: {} for i in layers}
        for shard in sorted(self.checkpoint.glob("*.safetensors")):
            with safe_open(shard, framework="pt", device="cpu") as stream:
                keys = set(stream.keys())
                for layer in layers:
                    for name in names:
                        key = f"model.layers.{layer}.self_attn.indexer.{name}"
                        if key in keys:
                            found[layer][name] = stream.get_tensor(key)
        for layer, values in found.items():
            if set(values) != set(names):
                raise ValueError(f"missing indexer weights at layer {layer}")
            self.weights[layer] = (torch.cat([values[n] for n in names[:3]], 0),
                                   values[names[3]], values[names[4]])
        # A new, process-local dispatch name; never replace stock attention.
        ALL_ATTENTION_FUNCTIONS.register("selector_capture_fa4", self.attention)
        ALL_MASK_ATTENTION_FUNCTIONS.register("selector_capture_fa4", lambda **kwargs: None)
        self.model.config._attn_implementation = "selector_capture_fa4"
        self.handles = []
        for layer in layers:
            def hook(module, args, kwargs, idx=layer):
                hidden = kwargs.get("hidden_states", args[0] if args else None)
                if self.phase == "prefill":
                    self.hidden[idx][:, self.offset:self.offset + hidden.shape[1]].copy_(hidden.cpu())
            self.handles.append(self.model.model.layers[layer].self_attn.register_forward_pre_hook(hook, with_kwargs=True))

    def attention(self, module, query, key, value, attention_mask, scaling=None, **kwargs):
        from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd
        if query.shape[0] != 1 or module.sliding_window is not None:
            raise ValueError("capture requires one unpadded full-attention request")
        q, k, v = (x.transpose(1, 2).contiguous() for x in (query, key, value))
        if self.phase == "prefill" and k.shape[1] == self.context and module.layer_idx in self.layers:
            self.retrieval[module.layer_idx] = q[:, -1:].contiguous().cpu()
        if self.phase == "replay" and module.layer_idx in self.layers:
            self.captures[module.layer_idx] = {"q": q.cpu(), "k": k.cpu(), "v": v.cpu()}
        return _flash_attn_fwd(q, k, v, causal=True, softmax_scale=scaling, num_splits=1)[0], None

    @torch.inference_mode()
    def capture(self, ids):
        from transformers.cache_utils import DynamicCache
        if len(ids) != self.context:
            raise ValueError("prompt length differs from approved context")
        self.hidden = {i: torch.empty((1, len(ids), self.config["hidden_size"]),
                        dtype=torch.bfloat16) for i in self.layers}
        self.captures = {}
        self.retrieval = {}
        self.phase = "prefill"
        cache = DynamicCache(config=self.model.config)
        tokens = torch.tensor([ids], device="cuda", dtype=torch.long)
        for start in range(0, len(ids), 2048):
            self.offset = start
            output = self.model(input_ids=tokens[:, start:start + 2048],
                                past_key_values=cache, use_cache=True, logits_to_keep=1)
        generated = [int(output.logits[0, -1].argmax())]
        self.phase = "generate"
        for _ in range(5):
            output = self.model(input_ids=torch.tensor([[generated[-1]]], device="cuda"),
                                past_key_values=cache, use_cache=True, logits_to_keep=1)
            generated.append(int(output.logits[0, -1].argmax()))
        # Fixed six-token input collection, including EOS if present. This is
        # explicitly a timing workload; no answer/acceptance score is reported.
        cache.crop(len(ids) - 1)
        self.phase = "replay"
        replay = torch.tensor([[ids[-1], *generated]], device="cuda")
        self.model(input_ids=replay, past_key_values=cache, use_cache=True, logits_to_keep=1)
        del cache, output
        for layer, capture in self.captures.items():
            weight, ln_w, ln_b = self.weights[layer]
            capture.update(retrieval_q=self.retrieval[layer],
                prefix_tokens=len(ids), phase="verify", model=str(self.checkpoint), layer=layer,
                prompt_sha256=[tensor_digest(tokens)], hidden=self.hidden[layer][:, -1].contiguous(),
                hidden_all=self.hidden[layer], qkw_weight=weight, ln_weight=ln_w, ln_bias=ln_b,
                config=self.config, generated_token_ids=generated,
                trajectory="dense FA4 greedy six-token continuation; seven-query replay",
                checkpoint_config_sha256=hashlib.sha256((self.checkpoint / "config.json").read_bytes()).hexdigest())
        result = self.captures
        self.captures, self.hidden = {}, {}
        torch.cuda.empty_cache()
        return result
