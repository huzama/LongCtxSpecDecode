"""Acceptance split by token type: retrieval-hit vs local, on a needle task.

The risk being measured: rejections piling up on exactly the tokens that make
the task long-context (retrieval integration sits in mid/deep layers, and
sparse views can miss needle blocks). Reported as alpha_retrieval vs
alpha_local per configuration.

Two draft backends share the harness:
- exit:   draft = truncated depth of a LayerSkip checkpoint (shared norm+head).
- sparse: draft = the selector model under an aggressive coverage p; target =
  the same model under the gentle p. Teacher-forced sparse forwards run the
  prefill path, whose selection is shared per query tile rather than
  per-query; decode-time alpha can only be better, so this bound is safe.
"""

import gc
from dataclasses import dataclass, field
from typing import Literal

import torch

from specdec.data import needle
from specdec.models import early_exit, selector
from specdec.config import ModelSpec, SelectorSpec
from specdec.models.loading import exit_layers_from_fracs, load_causal_lm, num_layers
from specdec.runs import append_jsonl, create_run_dir, save_config, save_json
from specdec.metrics.acceptance import positionwise_alpha


@dataclass
class Config:
    backend: Literal["exit", "sparse"] = "exit"
    # exit backend
    model: str = "facebook/layerskip-llama3.2-1B"
    exit_frac: float = 0.5
    # sparse backend
    checkpoint: str = ""
    p_target: float = 0.95
    p_draft: list[float] = field(default_factory=lambda: [0.9, 0.7, 0.5])
    # task
    context_tokens: int = 32768
    n_needles: int = 6
    gen_tokens: int = 96
    n_samples: int = 4
    dtype: str = "bfloat16"
    device: str = "cuda"
    slug: str = "prelim-alpha-by-token-type"


def _split(metrics: dict, labels: torch.Tensor) -> dict:
    lab = labels[: metrics["match"].shape[0]]
    out = {}
    for name, sel in (("retrieval", lab), ("local", ~lab)):
        if int(sel.sum()) == 0:
            out[name] = {"n_pos": 0, "alpha_greedy": float("nan"), "alpha_overlap": float("nan")}
        else:
            out[name] = {
                "n_pos": int(sel.sum()),
                "alpha_greedy": float(metrics["match"][sel].float().mean()),
                "alpha_overlap": float(metrics["overlap"][sel].mean()),
            }
    return out


@torch.no_grad()
def _run_exit(cfg: Config, run_dir) -> list[dict]:
    model, tokenizer = load_causal_lm(ModelSpec(cfg.model, cfg.dtype, cfg.device))
    k = exit_layers_from_fracs(num_layers(model), [cfg.exit_frac])[0]
    rows = []
    for s in range(cfg.n_samples):
        sample = needle.build_needle_sample(
            tokenizer, cfg.context_tokens - cfg.gen_tokens, cfg.n_needles, seed=s
        )
        prompt = sample.input_ids.to(cfg.device)
        gen = model.generate(
            prompt, max_new_tokens=cfg.gen_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_ids = gen[0, prompt.shape[1]:]
        labels = needle.label_retrieval_tokens(tokenizer, gen_ids, sample.values)
        hs = early_exit.hidden_states_slice(model, gen, keep_last=gen_ids.shape[0])
        target = early_exit.exit_logits(model, hs[-1], normed=True).squeeze(0)
        draft = early_exit.exit_logits(model, hs[k]).squeeze(0)
        metrics = positionwise_alpha(target, draft)
        row = {"backend": "exit", "model": cfg.model, "exit_layers": k, "sample": s,
               "recall_ok": bool(labels.any()), **_split(metrics, labels)}
        rows.append(row)
        append_jsonl(run_dir / "rows.jsonl", row)
    del model
    gc.collect()
    return rows


@torch.no_grad()
def _sparse_answer_logits(model, full_ids: torch.Tensor, gen_len: int) -> torch.Tensor:
    """Teacher-forced logits on the answer region without materializing the
    full-sequence vocab tensor: decoder forward, slice, then the head."""
    out = model.model(input_ids=full_ids, output_hidden_states=False, use_cache=False)
    hidden = out.last_hidden_state[:, -gen_len - 1 : -1, :]
    return model.get_output_embeddings()(hidden).squeeze(0)


@torch.no_grad()
def _run_sparse(cfg: Config, run_dir) -> list[dict]:
    if not cfg.checkpoint:
        raise ValueError("backend=sparse requires --checkpoint")
    spec = SelectorSpec(
        checkpoint=cfg.checkpoint, topp_mass=cfg.p_target,
        max_token_length=cfg.context_tokens + 512, dtype=cfg.dtype, device=cfg.device,
    )
    model, tokenizer = selector.load_selector_model(spec)
    rows = []
    for s in range(cfg.n_samples):
        sample = needle.build_needle_sample(
            tokenizer, cfg.context_tokens - cfg.gen_tokens, cfg.n_needles, seed=s
        )
        prompt = sample.input_ids.to(cfg.device)
        selector.set_topp(model, spec, p=cfg.p_target)
        gen = model.generate(
            prompt, max_new_tokens=cfg.gen_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_ids = gen[0, prompt.shape[1]:]
        labels = needle.label_retrieval_tokens(tokenizer, gen_ids, sample.values)
        target = _sparse_answer_logits(model, gen, gen_ids.shape[0])
        for p in cfg.p_draft:
            selector.set_topp(model, spec, p=p)
            draft = _sparse_answer_logits(model, gen, gen_ids.shape[0])
            metrics = positionwise_alpha(target, draft)
            row = {"backend": "sparse", "checkpoint": cfg.checkpoint, "p_target": cfg.p_target,
                   "p_draft": p, "sample": s, "recall_ok": bool(labels.any()),
                   **_split(metrics, labels)}
            rows.append(row)
            append_jsonl(run_dir / "rows.jsonl", row)
        selector.set_topp(model, spec, p=cfg.p_target)
    del model
    gc.collect()
    return rows


def run(cfg: Config) -> None:
    run_dir = create_run_dir(cfg.slug)
    save_config(run_dir, cfg)
    rows = _run_exit(cfg, run_dir) if cfg.backend == "exit" else _run_sparse(cfg, run_dir)
    save_json(run_dir / "results.json", rows)
    print(f"wrote {run_dir}/results.json ({len(rows)} rows)")
