"""Reviewed Main Results methods and their shared serving configuration."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Method:
    mode: str
    sparse_draft: bool
    draft_bits: int = 8
    verify_fraction: float = 0.

    @property
    def speculative(self):
        return self.mode == "highlight"


METHODS = {
    "dense8": Method("dense", False),
    "direct8": Method("direct", True),
    "full4-dense8": Method("highlight", False, 4),
    "highlight8-dense8": Method("highlight", True, 8),
    "highlight4-dense8": Method("highlight", True, 4),
    "highlight4-sv50": Method("highlight", True, 4, .5),
    "highlight4-sv40": Method("highlight", True, 4, .4),
    "highlight4-sv25": Method("highlight", True, 4, .25),
}


def engine_options(method, *, batch, context, generation, checkpoint, rope, scheduler,
                   attention_backend='triton'):
    m = METHODS[method]
    if batch < 1 or context < 8 or generation < 2 or context + generation > 131072:
        raise ValueError("invalid batch, context or output reservation")
    if attention_backend not in ('triton', 'fa4'):
        raise ValueError('unsupported hierarchical attention backend')
    hf = dict(architectures=["HierarchicalQwen3ForCausalLM"], use_dsa=m.sparse_draft,
        molle_spec_decode=m.speculative, molle_selection_mode="topk_fraction",
        molle_topk_fraction=.07, molle_topk_recency=64,
        molle_draft_select="round" if m.speculative else "step",
        molle_verify_fraction=m.verify_fraction, molle_kcache_format="bf16",
        molle_hierarchical_draft_bits=m.draft_bits,
        molle_hierarchical_attention_backend=attention_backend)
    if rope is not None:
        hf["rope_parameters"] = rope
    result = dict(model=checkpoint, generation_config="vllm", dtype="bfloat16",
        kv_cache_dtype="auto", block_size=16, max_num_seqs=batch,
        max_model_len=context+generation, enable_prefix_caching=False,
        async_scheduling=False, enable_chunked_prefill=False,
        max_num_batched_tokens=context+generation, gpu_memory_utilization=.9,
        seed=42, hf_overrides=hf, disable_log_stats=False,
        speculative_config=dict(method="custom_class", model="highlight_spec.speculator.HighlightSpeculator",
                                num_speculative_tokens=6) if m.speculative else None,
        compilation_config=dict(cudagraph_mode="FULL_DECODE_ONLY",
            cudagraph_capture_sizes=sorted({1, batch, batch*7} if m.speculative else {1, batch})),
        enforce_eager=False, attention_config=dict(backend="FLASH_ATTN", flash_attn_version=4),
        worker_extension_cls="highlight_spec.hierarchical_backend.HierarchicalStatsWorker")
    if scheduler is not None:
        result["scheduler_cls"] = scheduler
    return result


def build_engine(args, scheduler):
    from benchmarks.decode import rope_parameters, yarn_factor
    from vllm import LLM
    m = METHODS[args.hierarchical_method]
    if (args.mode != m.mode or args.indexer_cache_format != "bf16" or args.spec_tokens != 6
            or args.verify_fraction != m.verify_fraction or args.topk_fraction != .07
            or args.recent != 64 or args.topp is not None
            or getattr(args, "draft_ffn_checkpoint", None) or getattr(args, "draft_attention", None)):
        raise ValueError("hierarchical method flags do not match the frozen Main Results protocol")
    factor = yarn_factor(args.ctx+args.gen)
    rope = rope_parameters(args.checkpoint, factor) if factor else None
    return LLM(**engine_options(args.hierarchical_method, batch=args.batch, context=args.ctx,
        generation=args.gen, checkpoint=args.checkpoint, rope=rope, scheduler=scheduler,
        attention_backend=getattr(args, 'hierarchical_attention_backend', 'triton')))
