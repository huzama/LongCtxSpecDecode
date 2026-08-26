# SPDX-License-Identifier: Apache-2.0
"""Statistics accumulate over all rows and ignore padded ones."""
import pytest
import torch

from vllm.v1.spec_decode.sparse_attn.longspec.stats import SelectionStats

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")


def test_accumulate_snapshot_reset():
    device = "cuda"
    stats = SelectionStats(2, device)
    used = torch.tensor([[10, 20, 0], [5, 5, 0]], dtype=torch.int32,
                        device=device)
    valid = torch.tensor([100, 200, 0], dtype=torch.int32, device=device)
    stats.accumulate(used, valid)
    stats.accumulate(used, valid)
    snap = stats.snapshot()
    assert snap["request_rounds"] == 4
    assert snap["mean_valid"] == 150.0
    assert snap["mean_used_per_layer"] == [15.0, 5.0]
    assert snap["mean_ratio_per_layer"] == [0.1, 5 / 150]
    stats.reset()
    snap = stats.snapshot()
    assert snap["request_rounds"] == 0
    assert snap["mean_used_per_layer"] == [0.0, 0.0]
