# SPDX-License-Identifier: Apache-2.0
"""End to end through the grid runner: drafting on a separately loaded
weight copy of the same checkpoint reproduces dense greedy output token
for token, proving the grafted-attention plumbing changes nothing.
Slow: two engines."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU"),
    pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc"),
]

REPO = Path(subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], text=True).strip())
GRID = REPO / "benchmarks" / "longspec" / "grid.py"


@pytest.mark.slow
def test_copied_weights_match_dense(tmp_path):
    cmd = [sys.executable, str(GRID), "--cells",
           "4096:2:dense,4096:2:longspec", "--parity",
           "--model", "Qwen/Qwen3-0.6B", "--draft-weights", "Qwen/Qwen3-0.6B",
           "--gen", "64", "--theta", "1.0",
           "--ratio", "1.0", "--prompt-source", "synthetic",
           "--prompts-dir", str(tmp_path / "prompts"), "--out", str(tmp_path),
           "--drain", "5", "--gpu-mem-util", "0.4"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    records = [json.loads(line)
               for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    longspec = next(r for r in records if r["mode"] == "longspec")
    assert longspec["alpha"] >= 0.98, longspec
    assert json.loads((tmp_path / "parity.json").read_text())["ok"]


@pytest.mark.slow
def test_quantized_draft_is_lossless(tmp_path):
    """The verify emits its own argmax regardless of the draft, so a W4
    drafter must reproduce dense output token for token; only acceptance
    may move. Heavy: the 4B target and its W4 checkpoint."""
    cmd = [sys.executable, str(GRID), "--cells",
           "8192:2:dense,8192:2:coverage", "--parity",
           "--model", "Qwen/Qwen3-4B",
           "--draft-weights", "RedHatAI/Qwen3-4B-quantized.w4a16",
           "--gen", "64", "--theta", "0.98", "--ratio", "0.15",
           "--min-tokens", "0", "--prompt-source", "synthetic",
           "--prompts-dir", str(tmp_path / "prompts"), "--out", str(tmp_path),
           "--drain", "5", "--gpu-mem-util", "0.7"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    records = [json.loads(line)
               for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    coverage = next(r for r in records if r["mode"] == "coverage")
    # Quantization costs acceptance, never output.
    assert coverage["alpha"] >= 0.5, coverage
    assert json.loads((tmp_path / "parity.json").read_text())["ok"]
