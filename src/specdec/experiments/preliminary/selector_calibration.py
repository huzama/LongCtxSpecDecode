"""Selector calibration on draft-side states, inside the decode loop.

The published calibration is measured on full-model runs; here the model
decodes with the dual top-p rule active, so every recorded query was computed
through sparse attention. Per decode step and layer we compare the selector's
block-score ranking against the true attention block mass from the same query
and cache (core.selector.CalibrationRecorder).

Decision this feeds: whether top-p budgets keep meaning inside the speculation
loop (design doc, kill-test condition 3). Read the per-layer summary with the
depth asymmetry in mind: the shallow-layer rows are the draft-relevant slice.
"""

from dataclasses import dataclass, field

import torch

from specdec.data import text
from specdec.models import selector
from specdec.config import SelectorSpec
from specdec.runs import append_jsonl, create_run_dir, save_config, save_json


@dataclass
class Config:
    checkpoint: str
    topp_mass: float = 0.95
    prompt_tokens: int = 32768
    gen_tokens: int = 64
    n_samples: int = 2
    datasets: list[str] = field(default_factory=lambda: ["pg19", "fineweb"])
    dtype: str = "bfloat16"
    device: str = "cuda"
    slug: str = "prelim-selector-calibration"

    def spec(self) -> SelectorSpec:
        return SelectorSpec(
            checkpoint=self.checkpoint,
            topp_mass=self.topp_mass,
            max_token_length=self.prompt_tokens + self.gen_tokens + 512,
            dtype=self.dtype,
            device=self.device,
        )


@torch.no_grad()
def run(cfg: Config) -> None:
    run_dir = create_run_dir(cfg.slug)
    save_config(run_dir, cfg)
    model, tokenizer = selector.load_selector_model(cfg.spec())

    summaries: list[dict] = []
    for dataset in cfg.datasets:
        for s in range(cfg.n_samples):
            prompt = text.sample_long_ids(
                tokenizer, cfg.prompt_tokens, seed=s, dataset=dataset
            ).to(cfg.device)
            with selector.CalibrationRecorder(model) as rec:
                model.generate(
                    prompt, max_new_tokens=cfg.gen_tokens, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            for row in rec.records:
                append_jsonl(run_dir / "rows.jsonl", {"dataset": dataset, "sample": s, **row})
            for layer_row in rec.summary_by_layer():
                summaries.append({"dataset": dataset, "sample": s, **layer_row})
            save_json(run_dir / "results.json", summaries)
    print(f"wrote {run_dir}/results.json ({len(summaries)} layer summaries)")
