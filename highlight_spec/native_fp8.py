"""Direct Highlight over vLLM FA4's native E4M3 KV cache.

Only configuration and model construction differ from MoLLE. Native vLLM
owns cache writes, query quantization, Q/K/V scales and attention kernels.
The Highlight indexer remains BF16. No hierarchical-cache code is used.
"""
import copy
import functools

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix
from spotattention.serve.buffers import SelectionBuffers
from spotattention.serve.config import MolleConfig
from spotattention.serve.model import MolleQwen3DecoderLayer, MolleQwen3ForCausalLM
from spotattention.serve.stats import SelectionStatsWorker


class NativeFP8Config(MolleConfig):
    def validate_engine(self, config):
        if (not self.enabled or self.spec_decode or config.speculative_config is not None
                or self.kcache_format != "bf16" or self.selection.verify_fraction
                or config.cache_config.cache_dtype != "fp8_e4m3"
                or config.attention_config.flash_attn_version != 4
                or config.model_config.dtype != torch.bfloat16
                or config.quant_config is not None):
            raise ValueError("native FP8 requires direct Highlight, FA4, BF16 weights/indexer and E4M3 KV")
        if torch.cuda.get_device_capability(config.device_config.device)[0] != 10:
            raise ValueError("native FA4 FP8 requires SM100 (B200-class GPUs); no emulated fallback")
        # Upstream's BF16 guard predates its FA4 path. Reuse all other upstream
        # checks on a validation-only copy; never change the engine cache dtype.
        checks = copy.copy(config)
        checks.cache_config = copy.copy(config.cache_config)
        checks.cache_config.cache_dtype = "auto"
        super().validate_engine(checks)


@support_torch_compile(dynamic_arg_dims={
    "input_ids": 0, "positions": -1, "intermediate_tensors": 0, "inputs_embeds": 0})
class NativeFP8Qwen3Model(Qwen2Model):
    hf_to_vllm_mapper = MolleQwen3ForCausalLM.hf_to_vllm_mapper

    def __init__(self, *, vllm_config, prefix=""):
        molle = NativeFP8Config.from_vllm_config(vllm_config)
        from .native_fp8_attention import install
        install()
        sel = SelectionBuffers(vllm_config, molle, vllm_config.device_config.device)
        super().__init__(vllm_config=vllm_config, prefix=prefix,
            decoder_layer_type=functools.partial(MolleQwen3DecoderLayer, molle=molle, sel=sel))
        self.molle, self.sel = molle, sel


class NativeFP8Qwen3ForCausalLM(MolleQwen3ForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        nn.Module.__init__(self)
        self.config = vllm_config.model_config.hf_config
        self.vllm_config, self.quant_config = vllm_config, vllm_config.quant_config
        self.model = NativeFP8Qwen3Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        if not get_pp_group().is_last_rank:
            self.lm_head = PPMissingLayer()
        elif self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(self.config.vocab_size, self.config.hidden_size,
                quant_config=self.quant_config, prefix=maybe_prefix(prefix, "lm_head"))
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors


class NativeFP8StatsWorker(SelectionStatsWorker):
    def native_fp8_eager(self, enabled):
        """Validation only: route fresh requests through the same eager kernels."""
        manager = self.model_runner.cudagraph_manager
        if not hasattr(self, "_fp8_graphs_captured"):
            self._fp8_graphs_captured = manager._graphs_captured
        manager._graphs_captured = False if enabled else self._fp8_graphs_captured

    def native_fp8_info(self, begin=False):
        from spotattention.serve.model import MolleQwen3Attention
        from spotattention.serve.sparse_attn import SparseFAImpl
        runner = self.model_runner
        manager = runner.cudagraph_manager
        if begin:
            if manager is None or not manager.graphs:
                raise RuntimeError("native FP8 decode graphs were not captured")
            if not hasattr(self, "_fp8_graph_original"):
                self._fp8_graph_original = manager.run_fullgraph
                def counted(desc, *args, **kwargs):
                    self._fp8_graph_replays += 1
                    self._fp8_execution.append(dict(kind="graph", requests=desc.num_reqs,
                                                    tokens=desc.num_tokens))
                    return self._fp8_graph_original(desc, *args, **kwargs)
                manager.run_fullgraph = counted
                first = next(m.attn for m in runner.model.modules() if isinstance(m, MolleQwen3Attention))
                original = first.impl.forward
                def eager(layer, query, key, value, kv_cache, attn_metadata, *args, **kwargs):
                    # Graph replay bypasses Python; log only uncaptured decode.
                    if attn_metadata is not None and attn_metadata.max_query_len == 1:
                        self._fp8_execution.append(dict(kind="eager",
                            requests=len(attn_metadata.seq_lens), tokens=attn_metadata.num_actual_tokens))
                    return original(layer, query, key, value, kv_cache, attn_metadata, *args, **kwargs)
                first.impl.forward = eager
            self._fp8_graph_replays = 0
            self._fp8_execution = []
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        rows = []
        for module in runner.model.modules():
            if not isinstance(module, MolleQwen3Attention):
                continue
            attn = module.attn
            cache = attn.kv_cache
            scales = {name: float(getattr(attn, f"_{name}_scale")) for name in ("q", "k", "v")}
            if (cache.element_size() != 1 or attn.kv_cache_dtype != "fp8_e4m3"
                    or not isinstance(attn.impl, SparseFAImpl)
                    or not all(0 < value < float("inf") for value in scales.values())):
                raise RuntimeError("native FP8 cache, backend or scales failed validation")
            rows.append(dict(layer=attn.layer_name, storage_dtype=str(cache.dtype),
                             element_bytes=cache.element_size(), scales=scales))
        if len(rows) != runner.model.config.num_hidden_layers:
            raise RuntimeError("missing native FP8 layers")
        indexer = runner.model.model.molle
        if indexer.kcache_dtype != torch.bfloat16:
            raise RuntimeError("native FP8 validation requires a BF16 indexer")
        return dict(backbone_cache="native_fp8_e4m3", indexer_cache="bf16",
                    attention_bridge="native_fa4_with_qkv_descales",
                    scale_policy="vllm_calculate_kv_scales", layers=rows,
                    graph_replays=getattr(self, "_fp8_graph_replays", 0),
                    target_execution=list(getattr(self, "_fp8_execution", [])),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    kv_cache_groups=len(runner.kv_cache_config.kv_cache_groups))
