# SPDX-License-Identifier: Apache-2.0
"""End to end through the grid runner on a small model: coverage at theta 1
and an uncapped budget reproduces dense greedy output token for token, and
the drafter agrees with the target almost always. Slow: two engines."""
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
def test_full_budget_matches_dense(tmp_path):
    cmd = [sys.executable, str(GRID), "--cells",
           "4096:2:dense,4096:2:coverage", "--parity",
           "--model", "Qwen/Qwen3-0.6B", "--gen", "64", "--theta", "1.0",
           "--ratio", "1.0", "--prompt-source", "synthetic",
           "--prompts-dir", str(tmp_path / "prompts"), "--out", str(tmp_path),
           "--drain", "5", "--gpu-mem-util", "0.4"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    records = [json.loads(line)
               for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    coverage = next(r for r in records if r["mode"] == "coverage")
    assert coverage["alpha"] >= 0.98, coverage
    assert json.loads((tmp_path / "parity.json").read_text())["ok"]
