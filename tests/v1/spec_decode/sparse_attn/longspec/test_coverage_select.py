# SPDX-License-Identifier: Apache-2.0
"""The fused selection must match a float64 reference: reserved ranges, the
mass crossing, tie resolution, clamps, and empty rows."""
import pytest
import torch

from vllm.v1.spec_decode.sparse_attn.longspec.kernels.coverage_select import (
    coverage_select,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")

EPS = 1e-3  # relative slack on theta for near-exact crossings


def reference_k(row: torch.Tensor, S: int, R: int, theta: float) -> int:
    """Smallest k whose mass with the reserved ranges reaches theta * total.
    ``row`` is the scored prefix in float64."""
    P = row.numel()
    s_eff = min(S, P)
    r_eff = min(R, P - s_eff)
    cand = row[s_eff:P - r_eff]
    if cand.numel() == 0 or theta >= 1:
        return cand.numel()
    total = row.sum()
    reserved = total - cand.sum()
    target = theta * total - reserved
    if total <= 0 or target <= 0:
        return 0
    cum = cand.sort(descending=True).values.cumsum(0)
    hit = (cum >= target).nonzero()
    return int(hit[0]) + 1 if hit.numel() else cand.numel()


def clamp(k: int, k_min: int, k_max: int, n_cand: int) -> int:
    return min(max(k, k_min), k_max, n_cand)


def check_row(metric_row, table_row, used, P, S, R, theta, k_min, k_max):
    row = metric_row[:P].double()
    s_eff = min(S, P)
    r_eff = min(R, P - s_eff)
    n_cand = P - s_eff - r_eff
    k_lo = clamp(reference_k(row, S, R, theta * (1 - EPS)), k_min, k_max, n_cand)
    k_hi = clamp(reference_k(row, S, R, theta * (1 + EPS)), k_min, k_max, n_cand)
    k_ref = clamp(reference_k(row, S, R, theta), k_min, k_max, n_cand)
    k = int(used) - s_eff - r_eff
    assert k_lo <= k <= k_hi, (k, k_lo, k_ref, k_hi)
    idx = table_row[:int(used)].long()
    assert idx.unique().numel() == idx.numel()
    assert idx.min() >= 0 and idx.max() < P
    reserved = torch.cat([torch.arange(s_eff), torch.arange(P - r_eff, P)])
    assert torch.equal(idx[k:].sort().values.cpu(), reserved)
    chosen = idx[:k]
    assert torch.all((chosen >= s_eff) & (chosen < P - r_eff))
    cand = row[s_eff:P - r_eff]
    top = cand.sort(descending=True).values[:k]
    assert torch.equal(row[chosen].sort(descending=True).values, top)


def softmax_rows(rows, max_len, P_list, device, sparse=False):
    torch.manual_seed(1)
    m = torch.zeros(rows, max_len, device=device, dtype=torch.float32)
    for r, P in enumerate(P_list):
        logits = torch.randn(P, device=device) * (6.0 if sparse else 1.0)
        m[r, :P] = torch.softmax(logits, 0)
    return m.to(torch.bfloat16)


def run(metric, P_list, S, R, theta, k_min_list, k_max_list, width=None):
    device = metric.device
    rows = metric.shape[0]
    valid = torch.tensor(P_list, dtype=torch.int32, device=device)
    k_min = torch.tensor(k_min_list, dtype=torch.int32, device=device)
    k_max = torch.tensor(k_max_list, dtype=torch.int32, device=device)
    width = width or (max(P_list) + S + R)
    table = torch.full((rows, width), -1, dtype=torch.int32, device=device)
    used = torch.full((rows,), -1, dtype=torch.int32, device=device)
    coverage_select(metric, valid, k_min, k_max, table, used, theta, S, R)
    return table, used


@pytest.mark.parametrize("theta", [0.5, 0.9, 1.0])
@pytest.mark.parametrize("sparse", [False, True])
def test_matches_reference(theta, sparse):
    device, S, R = "cuda", 4, 8
    P_list = [0, 1, S + R - 1, S + R, 100, 4096, 131072]
    max_len = max(P_list)
    metric = softmax_rows(len(P_list), max_len, P_list, device, sparse)
    k_min = [0] * len(P_list)
    k_max = [max_len] * len(P_list)
    table, used = run(metric, P_list, S, R, theta, k_min, k_max)
    for r, P in enumerate(P_list):
        if P == 0:
            assert int(used[r]) == 0 and torch.all(table[r] == -1)
            continue
        check_row(metric[r], table[r], used[r], P, S, R, theta, k_min[r],
                  k_max[r])


def test_theta_one_selects_every_candidate():
    device, S, R, P = "cuda", 4, 8, 3000
    metric = softmax_rows(1, P, [P], device)
    metric[0, 1000:2000] = 0  # zero-mass candidates count too
    table, used = run(metric, [P], S, R, 1.0, [0], [P])
    assert int(used[0]) == P
    assert torch.equal(table[0, :P].sort().values.cpu(),
                       torch.arange(P, dtype=torch.int32))


def test_clamps():
    device, S, R, P = "cuda", 4, 8, 2048
    metric = softmax_rows(4, P, [P] * 4, device)
    row = metric[0, :P].double()
    k_star = reference_k(row, S, R, 0.9)
    n_cand = P - S - R
    assert 100 < k_star < n_cand - 100
    k_min = [0, k_star + 50, 0, k_star + 50]
    k_max = [P, P, k_star - 20, k_star - 20]
    table, used = run(metric, [P] * 4, S, R, 0.9, k_min, k_max)
    for r in range(4):
        check_row(metric[r], table[r], used[r], P, S, R, 0.9, k_min[r],
                  k_max[r])
    assert int(used[1]) - S - R == k_star + 50
    assert int(used[2]) - S - R == k_star - 20
    assert int(used[3]) - S - R == k_star - 20  # cap wins over the floor


def test_zero_and_tied_rows():
    device, S, R, P = "cuda", 2, 2, 512
    metric = torch.zeros(3, P, device=device, dtype=torch.bfloat16)
    metric[1, :P] = 1.0 / P            # all tied
    metric[2, 10:20] = 0.1             # ten equal spikes, rest zero
    table, used = run(metric, [P] * 3, S, R, 0.9, [0] * 3, [P] * 3)
    assert int(used[0]) == S + R       # no mass: reserved only
    for r in (1, 2):
        check_row(metric[r], table[r], used[r], P, S, R, 0.9, 0, P)
    assert int(used[2]) - S - R == 9    # 0.9 of ten equal spikes


def test_reserved_covers_short_rows():
    device, S, R = "cuda", 4, 8
    P_list = [1, 5, 11, 12]
    metric = softmax_rows(4, 12, P_list, device)
    table, used = run(metric, P_list, S, R, 0.9, [0] * 4, [12] * 4)
    for r, P in enumerate(P_list):
        assert int(used[r]) == P
        assert torch.equal(table[r, :P].sort().values.cpu(),
                           torch.arange(P, dtype=torch.int32))


def test_min_tokens_fills_from_the_top():
    device, S, R, P = "cuda", 0, 0, 1024
    metric = softmax_rows(1, P, [P], device, sparse=True)
    table, used = run(metric, [P], S, R, 0.5, [300], [P])
    assert int(used[0]) == 300
    check_row(metric[0], table[0], used[0], P, S, R, 0.5, 300, P)
