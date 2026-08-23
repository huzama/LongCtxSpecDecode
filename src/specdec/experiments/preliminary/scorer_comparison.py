"""Trained versus training-free block scoring at matched budget.

The question: how much selection quality do we give up by replacing the trained
selector with a scorer that needs no training and no preparation? This is the
number that prices the training-free direction, and the survey found no
published measurement of it at matched density for any scorer pair.

Method: during a real sparse decode, at each recorded step and layer, rank the
same residual block set four ways from identical inputs (oracle true attention
mass, trained selector, Quest-style min/max bound, mean-plus-spread), then
compare the top-k sets at matched k. Reported per budget fraction, so the
generous regime where published equivalence holds and the aggressive regime
nobody has measured both appear in one table.

Metrics: recall of the oracle's top-k, captured true attention mass, and mass
efficiency (captured divided by the best achievable at that k).
"""

from dataclasses import dataclass, field

import torch

from specdec.config import SelectorSpec
from specdec.data import text
from specdec.models import selector
from specdec.runs import append_jsonl, create_run_dir, save_config, save_json


@dataclass
class Config:
    checkpoint: str
    topp_mass: float = 0.95
    prompt_tokens: int = 32768
    gen_tokens: int = 64
    step_stride: int = 16
    n_samples: int = 2
    datasets: list[str] = field(default_factory=lambda: ["pg19", "fineweb"])
    budget_fracs: list[float] = field(
        default_factory=lambda: [0.50, 0.25, 0.10, 0.05, 0.02, 0.01]
    )
    spread: float = 1.0
    head_agg: str = "mean"  # oracle aggregation over query heads: mean or max
    dtype: str = "bfloat16"
    device: str = "cuda"
    slug: str = "prelim-scorer-comparison"

    def spec(self) -> SelectorSpec:
        return SelectorSpec(
            checkpoint=self.checkpoint,
            topp_mass=self.topp_mass,
            max_token_length=self.prompt_tokens + self.gen_tokens + 512,
            dtype=self.dtype,
            device=self.device,
        )


SCORERS = ("trained", "quest", "mean_std")


def _summarize(rows: list[dict], n_layers: int) -> list[dict]:
    """Aggregate by (budget fraction, depth band); bands are thirds of the stack."""
    third = max(1, n_layers // 3)
    bands = {"shallow": range(0, third), "middle": range(third, 2 * third), "deep": range(2 * third, n_layers)}
    out = []
    for frac in sorted({r["budget_frac"] for r in rows}, reverse=True):
        for band, layers in bands.items():
            sel = [r for r in rows if r["budget_frac"] == frac and r["layer"] in layers]
            if not sel:
                continue
            row = {"budget_frac": frac, "band": band, "n": len(sel),
                   "mean_k": sum(r["k"] for r in sel) / len(sel),
                   "oracle_captured": sum(r["oracle_captured"] for r in sel) / len(sel)}
            for name in ("trained_shift-1", "trained_shift+1"):
                key = name.replace("trained_shift", "trained_efficiency_shift")
                vals = [r[key] for r in sel if key in r and r[key] == r[key]]
                if vals:
                    row[f"{name}_efficiency"] = sum(vals) / len(vals)
            for name in SCORERS:
                for metric in ("recall", "captured", "efficiency"):
                    vals = [r[f"{name}_{metric}"] for r in sel if r[f"{name}_{metric}"] == r[f"{name}_{metric}"]]
                    row[f"{name}_{metric}"] = sum(vals) / len(vals) if vals else float("nan")
            out.append(row)
    return out


@torch.no_grad()
def run(cfg: Config) -> None:
    run_dir = create_run_dir(cfg.slug)
    save_config(run_dir, cfg)
    spec = cfg.spec()
    model, tokenizer = selector.load_selector_model(spec)
    n_layers = len(selector.dsa_modules(model))

    rows: list[dict] = []
    for dataset in cfg.datasets:
        for s in range(cfg.n_samples):
            prompt = text.sample_long_ids(
                tokenizer, cfg.prompt_tokens, seed=s, dataset=dataset
            ).to(cfg.device)
            with selector.ScorerComparisonRecorder(
                model, budget_fracs=cfg.budget_fracs, step_stride=cfg.step_stride,
                spread=cfg.spread, head_agg=cfg.head_agg,
            ) as rec:
                model.generate(
                    prompt, max_new_tokens=cfg.gen_tokens, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            for row in rec.records:
                tagged = {"dataset": dataset, "sample": s, **row}
                rows.append(tagged)
                append_jsonl(run_dir / "rows.jsonl", tagged)
            save_json(run_dir / "results.json", _summarize(rows, n_layers))
    print(f"wrote {run_dir}/results.json ({len(rows)} records, {n_layers} layers)")
