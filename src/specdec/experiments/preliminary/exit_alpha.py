"""Exit acceptance vs depth on LayerSkip checkpoints, zero training.

Measures alpha(k) at 8-32K contexts on Meta's released LayerSkip models: the
free estimate of the exit term before any LoRA spend. Position-wise alpha on
the model's own greedy continuation (see core.spec for the two metrics).

Decision this feeds: whether the exit axis of the design survives at long
context (design doc, first experiments).
"""

import gc
from dataclasses import dataclass, field

import torch

from specdec.data import text
from specdec.models import early_exit
from specdec.config import ModelSpec
from specdec.models.loading import exit_layers_from_fracs, load_causal_lm, num_layers
from specdec.runs import append_jsonl, create_run_dir, save_config, save_json


@dataclass
class Config:
    models: list[str] = field(
        default_factory=lambda: [
            "facebook/layerskip-llama3.2-1B",
            "facebook/layerskip-llama3-8B",
        ]
    )
    context_lengths: list[int] = field(default_factory=lambda: [8192, 16384, 32768])
    exit_fracs: list[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])
    gen_tokens: int = 128
    n_samples: int = 4
    dataset: str = "pg19"
    seed: int = 0
    dtype: str = "bfloat16"
    device: str = "cuda"
    slug: str = "prelim-exit-alpha"


@torch.no_grad()
def run(cfg: Config) -> None:
    run_dir = create_run_dir(cfg.slug)
    save_config(run_dir, cfg)
    rows_path = run_dir / "rows.jsonl"
    aggregate: list[dict] = []

    for name in cfg.models:
        model, tokenizer = load_causal_lm(ModelSpec(name, cfg.dtype, cfg.device))
        n_layers = num_layers(model)
        exit_layers = exit_layers_from_fracs(n_layers, cfg.exit_fracs)
        for ctx in cfg.context_lengths:
            per_k: dict[int, list[dict]] = {k: [] for k in exit_layers}
            for s in range(cfg.n_samples):
                prompt = text.sample_long_ids(
                    tokenizer, ctx - cfg.gen_tokens, seed=cfg.seed + s, dataset=cfg.dataset
                ).to(cfg.device)
                gen = model.generate(
                    prompt, max_new_tokens=cfg.gen_tokens, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                results = early_exit.sweep_exit_alpha(model, gen, cfg.gen_tokens, exit_layers)
                for k, metrics in results.items():
                    row = {"model": name, "context": ctx, "sample": s,
                           "exit_layers": k, "n_layers": n_layers, **metrics}
                    append_jsonl(rows_path, row)
                    per_k[k].append(metrics)
            for k, ms in per_k.items():
                aggregate.append(
                    {
                        "model": name,
                        "context": ctx,
                        "exit_layers": k,
                        "exit_frac": k / n_layers,
                        "alpha_greedy": sum(m["alpha_greedy"] for m in ms) / len(ms),
                        "alpha_overlap": sum(m["alpha_overlap"] for m in ms) / len(ms),
                        "n_samples": len(ms),
                    }
                )
            save_json(run_dir / "results.json", aggregate)
        del model
        gc.collect()
        if cfg.device.startswith("cuda"):
            torch.cuda.empty_cache()
    print(f"wrote {run_dir}/results.json ({len(aggregate)} cells)")
