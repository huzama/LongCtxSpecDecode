"""How verification cost scales with the number of query tokens.

Speculation pays only if a full-KV pass over gamma+1 query tokens costs about
what a single-token pass costs. That is the memory-bound claim every long-context
speculative method rests on, and it is contested: a production RL framework
measures a large rollout slowdown with speculation enabled, while the roofline
says verification stays memory-bound at long context.

This measures the ratio directly and needs no drafter. r(q) = t(q)/t(1) near 1
means the extra query tokens ride along free; r(q) growing with q means the pass
is compute-bound and no draft design can rescue the round.

The implied round speedup is tau / (gamma * c + r(gamma+1)), where c is the
draft-to-decode cost ratio, so the same table also says which draft cost is
needed to break even in each regime.
"""

import gc
import time
from dataclasses import dataclass, field

import torch

from specdec.config import ModelSpec
from specdec.models import selector
from specdec.models.loading import load_causal_lm
from specdec.runs import append_jsonl, create_run_dir, save_config, save_json


@dataclass
class Config:
    model: str = "Qwen/Qwen3-4B"
    backend: str = "hf"  # "hf", "selector-dense" (Triton dense), "selector-sparse" (top-p draft)
    checkpoint: str = ""  # required for either selector backend
    topp_mass: float = 0.95  # coverage for backend="selector-sparse"
    batches: list[int] = field(default_factory=lambda: [1, 4, 16, 32])
    contexts: list[int] = field(default_factory=lambda: [4096, 8192, 16384, 32768])
    query_lens: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    n_repeat: int = 7
    warmup: int = 3
    prefill_chunk: int = 2048
    kv_budget_gb: float = 32.0  # skip cells whose KV cache would exceed this
    draft_cost_ratios: list[float] = field(default_factory=lambda: [0.05, 0.10, 0.20])
    tau: float = 6.13
    gamma: int = 8
    dtype: str = "bfloat16"
    device: str = "cuda"
    slug: str = "prelim-verify-scaling"


def _kv_gb(model, batch: int, context: int) -> float:
    cfg = getattr(model.config, "text_config", None) or model.config
    per_token = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * getattr(
        cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
    ) * 2
    return batch * context * per_token / 1e9


@torch.no_grad()
def _build_cache(model, batch: int, context: int, chunk: int, device, cache=None):
    """Prefill a cache of exactly `context` tokens, in chunks to bound activation memory."""
    pos = 0
    while pos < context:
        step = min(chunk, context - pos)
        ids = torch.randint(0, 30000, (batch, step), device=device)
        out = model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        pos += step
    return cache


@torch.no_grad()
def _time_forward(model, cache, batch: int, q: int, device, n_repeat: int, warmup: int, keep: int) -> float:
    """Median ms for one forward of q query tokens against the cache, cache restored each time."""
    ids = torch.randint(0, 30000, (batch, q), device=device)
    samples = []
    for i in range(warmup + n_repeat):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
        torch.cuda.synchronize(device)
        dt = (time.perf_counter() - t0) * 1e3
        cache.crop(keep)  # undo this call's append so every repeat sees the same length
        if i >= warmup:
            samples.append(dt)
    samples.sort()
    return samples[len(samples) // 2]


@torch.no_grad()
def run(cfg: Config) -> None:
    run_dir = create_run_dir(cfg.slug)
    save_config(run_dir, cfg)
    if cfg.backend.startswith("selector"):
        from specdec.config import SelectorSpec

        if not cfg.checkpoint:
            raise ValueError("backend=selector-dense requires --checkpoint")
        model, _ = selector.load_selector_model(
            SelectorSpec(checkpoint=cfg.checkpoint, topp_mass=cfg.topp_mass,
                         max_token_length=max(cfg.contexts) + 512,
                         dtype=cfg.dtype, device=cfg.device),
            dense_baseline=(cfg.backend == "selector-dense"),
        )
    else:
        model, _ = load_causal_lm(ModelSpec(cfg.model, cfg.dtype, cfg.device))
    device = torch.device(cfg.device)
    results: list[dict] = []

    query_lens = sorted(set(cfg.query_lens) | {1, cfg.gamma + 1})
    for context in cfg.contexts:
        for batch in cfg.batches:
            kv = _kv_gb(model, batch, context)
            if kv > cfg.kv_budget_gb:
                print(f"skip batch {batch} ctx {context}: KV {kv:.1f} GB over budget")
                continue
            try:
                fresh = selector.make_cache(model) if cfg.backend == "selector-sparse" else None
                cache = _build_cache(model, batch, context, cfg.prefill_chunk, device, fresh)
                times = {
                    q: _time_forward(model, cache, batch, q, device, cfg.n_repeat, cfg.warmup, context)
                    for q in query_lens
                }
            except torch.OutOfMemoryError:
                print(f"skip batch {batch} ctx {context}: out of memory")
                gc.collect(); torch.cuda.empty_cache()
                continue
            base = times[1]
            row = {"model": cfg.model, "batch": batch, "context": context, "kv_gb": round(kv, 2),
                   "backend": cfg.backend,
                   "ms": {str(q): round(t, 3) for q, t in times.items()},
                   "ratio": {str(q): round(t / base, 3) for q, t in times.items()}}
            r_verify = times.get(cfg.gamma + 1)
            if r_verify:
                row["r_at_gamma_plus_1"] = round(r_verify / base, 3)
                row["speedup"] = {
                    str(c): round(cfg.tau / (cfg.gamma * c + r_verify / base), 2)
                    for c in cfg.draft_cost_ratios
                }
            results.append(row)
            append_jsonl(run_dir / "rows.jsonl", row)
            save_json(run_dir / "results.json", results)
            print(f"batch {batch:>3} ctx {context:>6} kv {kv:>5.1f}GB  "
                  f"t(1)={base:.1f}ms  r(9)={row.get('r_at_gamma_plus_1')}")
            del cache
            gc.collect(); torch.cuda.empty_cache()
    print(f"wrote {run_dir}/results.json ({len(results)} cells)")
