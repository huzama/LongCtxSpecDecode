"""Autoregressive decode throughput: the baseline under every speedup claim.

Dense AR is measured here with the plain HF path (core.measure). Sparse AR
must come from the selector repo's own tuned kernels, so this experiment
emits (and optionally runs) its `latency` subcommand with the matching
lengths instead of reimplementing that benchmark; raw output is archived in
the run dir either way.
"""

import gc
import shlex
import subprocess
from dataclasses import dataclass, field

import torch

from specdec.data import text
from specdec.models import selector
from specdec.config import ModelSpec
from specdec.metrics.timing import decode_ms_per_token
from specdec.models.loading import load_causal_lm
from specdec.runs import create_run_dir, save_config, save_json


@dataclass
class Config:
    models: list[str] = field(
        default_factory=lambda: ["meta-llama/Llama-3.1-8B", "Qwen/Qwen3-8B"]
    )
    prompt_lengths: list[int] = field(default_factory=lambda: [32768, 65536])
    new_tokens: int = 64
    warmup_tokens: int = 8
    dataset: str = "pg19"
    # Selector side: when set, the latency command for these checkpoints is
    # written to the run dir; with run_selector_latency=True it is also executed.
    selector_checkpoints: list[str] = field(default_factory=list)
    selector_topp: float = 0.95
    run_selector_latency: bool = False
    dtype: str = "bfloat16"
    device: str = "cuda"
    slug: str = "prelim-ar-throughput"


def _selector_latency_command(cfg: Config, checkpoint: str) -> str:
    # Fixed-K mode (the checkpoint's saved index_topk): the selector repo's
    # latency CLI rejects top-p under CUDA-graph capture (data-dependent K),
    # and its non-graph top-p decode is not representative of tuned speed.
    root = selector.selector_root()
    lengths = " ".join(str(n) for n in cfg.prompt_lengths)
    # Distinct run name per checkpoint: the latency runner resumes cells from
    # its run dir keyed by (mode, length) only, so a shared default name would
    # silently mix results across checkpoints.
    tag = checkpoint.rstrip("/").split("/")[-3]
    return (
        f"cd {root} && {root}/.venv/bin/python molle.py latency "
        f"--checkpoint {shlex.quote(checkpoint)} "
        f"--input-seq-length {lengths} "
        f"--run-name lat-{shlex.quote(tag)} "
        f"--modes vanilla molle_sparse_paged"
    )


@torch.no_grad()
def run(cfg: Config) -> None:
    run_dir = create_run_dir(cfg.slug)
    save_config(run_dir, cfg)
    results: list[dict] = []

    for name in cfg.models:
        model, tokenizer = load_causal_lm(ModelSpec(name, cfg.dtype, cfg.device))
        for length in cfg.prompt_lengths:
            prompt = text.sample_long_ids(tokenizer, length, seed=0, dataset=cfg.dataset)
            timing = decode_ms_per_token(
                model, prompt.to(cfg.device), cfg.new_tokens, cfg.warmup_tokens
            )
            results.append({"model": name, "path": "dense-hf", **timing})
            save_json(run_dir / "results.json", results)
        del model
        gc.collect()
        if cfg.device.startswith("cuda"):
            torch.cuda.empty_cache()

    commands = [_selector_latency_command(cfg, ckpt) for ckpt in cfg.selector_checkpoints]
    if commands:
        (run_dir / "selector-latency-commands.sh").write_text(
            "\n".join(commands) + "\n", encoding="utf-8"
        )
    if cfg.run_selector_latency:
        for ckpt, cmd in zip(cfg.selector_checkpoints, commands):
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            log = run_dir / f"selector-latency-{ckpt.replace('/', '_')}.log"
            log.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
            if proc.returncode != 0:
                raise RuntimeError(f"selector latency failed for {ckpt}; see {log}")
    print(f"wrote {run_dir}/results.json ({len(results)} dense cells, "
          f"{len(commands)} selector commands)")
