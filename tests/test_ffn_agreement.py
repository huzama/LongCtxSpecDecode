"""CPU checks for the diagnostic's quantization and acceptance accounting."""
import json

import pytest
import torch

from highlight_spec.ffn_quant import accepted_prefix, dequantize, unpack_unsigned
from benchmarks.ffn_agreement import report


def test_unpack_signed_int32_and_output_packed_zeros():
    # Nibbles 0..7 and 8..15 in least-to-most-significant order.
    packed = torch.tensor([[0x76543210, -0x01234568]], dtype=torch.int32)
    assert unpack_unsigned(packed, 1).tolist() == [list(range(16))]
    assert unpack_unsigned(packed.t(), 0).flatten().tolist() == list(range(16))


def test_asymmetric_groups_use_independent_zero_points():
    # Eight output channels, two groups of eight input channels; each nibble is 9.
    weights = torch.full((8, 2), -1717986919, dtype=torch.int32)  # 0x99999999
    zeros = torch.tensor([[0x76543210, -0x01234568]], dtype=torch.int32)
    scales = torch.tensor([[.5, 2.]] * 8, dtype=torch.bfloat16)
    result = dequantize(weights, scales, zeros, group_size=8)
    expected = torch.tensor([[(9 - i) * .5] * 8 + [(9 - (i + 8)) * 2.] * 8
                             for i in range(8)], dtype=torch.bfloat16)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("verified,expected", [([8, 2, 3, 4], 0), ([1, 8, 3, 4], 1), ([1, 2, 3, 4], 4)])
def test_acceptance_stops_at_first_rejection(verified, expected):
    assert accepted_prefix([1, 2, 3, 4], verified) == expected


def test_acceptance_rejects_truncated_verification():
    with pytest.raises(ValueError):
        accepted_prefix([1, 2], [1])


def test_report_refuses_partial_evidence(tmp_path):
    from argparse import Namespace
    with pytest.raises(ValueError, match="six distinct"):
        report(Namespace(plan=tmp_path / "plan.json"))


def test_report_pools_raw_counts(tmp_path):
    from argparse import Namespace
    from benchmarks.ffn_agreement import VARIANTS
    out = tmp_path / "gpu-1"
    out.mkdir()
    for slot in range(6):
        rows = [dict(slot=slot, offset=offset, variant=variant, matches=slot, accepted=slot)
                for offset in range(0, 128, 16) for variant in VARIANTS]
        (out / f"prompt-{slot}.json").write_text(json.dumps(dict(slot=slot, rows=rows)))
    report(Namespace(plan=tmp_path / "plan.json"))
    summary = json.loads((tmp_path / "summary.json").read_text())
    for value in summary.values():
        assert value["matches"] == 120 and value["predictions"] == 288
        assert value["mean_accepted"] == 2.5
