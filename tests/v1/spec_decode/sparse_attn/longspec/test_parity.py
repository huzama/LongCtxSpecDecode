# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Seeded synthetic-token regressions for target-weight and W4 draft copies."""

import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = [
    pytest.mark.slow_test,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU"),
    pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc"),
]
REPO = Path(__file__).resolve().parents[5]
GRID = REPO / "benchmarks/longspec/grid.py"


@pytest.mark.parametrize(
    "model,checkpoint,ctx,theta,cap,min_alpha",
    [
        ("Qwen/Qwen3-0.6B", "target", 4096, "1", "1", 0.98),
        ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-0.6B", 4096, "1", "1", 0.98),
        (
            "Qwen/Qwen3-4B",
            "RedHatAI/Qwen3-4B-quantized.w4a16",
            8192,
            "0.85",
            "0.15",
            0.5,
        ),
    ],
)
def test_dense_output(tmp_path, model, checkpoint, ctx, theta, cap, min_alpha):
    from transformers import AutoTokenizer

    # Preserve the original controlled inputs. Synthetic prompts are test
    # fixtures only; the benchmark itself supports LongBench v2 exclusively.
    prompts_dir = tmp_path / "prompts"
    cache = prompts_dir / model.replace("/", "--") / "42"
    cache.mkdir(parents=True)
    vocab = AutoTokenizer.from_pretrained(model).vocab_size
    for slot in range(2):
        rng = random.Random(42 + slot)
        tokens = [rng.randrange(1000, vocab - 1000) for _ in range(ctx)]
        (cache / f"longbench2-{ctx}-{slot}.json").write_text(json.dumps(tokens))
    modes = (
        ("dense", "vegas", "coverage")
        if checkpoint == "target"
        else ("dense", "coverage")
    )
    cmd = [
        sys.executable,
        str(GRID),
        "--cells",
        ",".join(f"{ctx}:2:{mode}" for mode in modes),
        "--prompts-dir",
        str(prompts_dir),
        "--samples",
        "1",
        "--model",
        model,
        "--draft-weights",
        checkpoint,
        "--gen",
        "64",
        "--theta",
        theta,
        "--ratio",
        cap,
        "--out",
        str(tmp_path),
        "--gpu-mem-util",
        "0.7",
    ]
    log_path = tmp_path / "engine.log"
    with log_path.open("w") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    assert result.returncode == 0, log_path.read_text()[-12000:]
    records = [
        json.loads(line)
        for line in (tmp_path / "results.jsonl").read_text().splitlines()
    ]
    assert all(r["decode_seconds"] > 0 and r["decode_tok_s"] > 0 for r in records)
    spec = next(r for r in records if r["mode"] == "coverage")
    assert spec["alpha"] >= min_alpha, spec
    comparison = json.loads((tmp_path / "output_comparison.json").read_text())
    assert comparison[f"{ctx}:2:coverage"]["identical"], comparison
