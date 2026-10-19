"""Draft inputs: the first step after verification and the in-graph advance."""

import pytest
import torch

from vllm.v1.worker.gpu.input_batch import InputBuffers

from highlight_spec.speculator import advance, prepare_first_step

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def i32(values):
    return torch.tensor(values, dtype=torch.int32, device="cuda")


def i64(values):
    return torch.tensor(values, dtype=torch.int64, device="cuda")


def test_first_step_follows_accepted_prefix():
    max_reqs, max_len = 5, 108
    buffers = InputBuffers(max_num_reqs=max_reqs, max_num_tokens=16, device=torch.device("cuda"))
    buffers.seq_lens.fill_(-1)
    last_sampled = i64(range(100, 100 + max_reqs)).view(max_reqs, 1)
    next_prefill = i32(range(200, 200 + max_reqs)).view(1, max_reqs)
    # Row 0: 4 of 6 drafts rejected. Row 1: all accepted, at the length limit.
    # Row 2: mid chunked prefill, nothing sampled.
    prepare_first_step(
        buffers, i32([107, 108, 9]), i32([2, 0, 4]), i32([3, 7, 0]), i32([4, 0, 0]),
        last_sampled, next_prefill, max_len, max_reqs,
    )
    assert buffers.input_ids[:3].tolist() == [102, 100, 204]
    assert buffers.positions[:3].tolist() == [103, 107, 9]
    assert buffers.seq_lens.tolist() == [104, 108, 10, 0, 0]
    assert buffers.query_start_loc.tolist() == [0, 1, 2, 3, 3, 3]


def test_advance_feeds_next_step_and_skips_padding():
    buffers = InputBuffers(max_num_reqs=4, max_num_tokens=8, device=torch.device("cuda"))
    buffers.positions[:3] = i64([5, 0, 99])
    buffers.seq_lens[:3] = i32([6, 0, 100])
    draft_tokens = torch.zeros((3, 2), dtype=torch.int64, device="cuda")
    step = i64(1)
    advance(draft_tokens, step, i64([11, 12, 13]), buffers, 100)
    assert draft_tokens[:, 1].tolist() == [11, 12, 13]
    assert buffers.input_ids[:3].tolist() == [11, 12, 13]
    assert buffers.positions[:3].tolist() == [6, 1, 99]
    assert buffers.seq_lens[:3].tolist() == [7, 0, 100]
    assert step.item() == 0


@pytest.mark.parametrize("capacity,active", [(5, 3), (42, 25), (64, 64)])
def test_captured_proposal_updates_live_state_across_rounds(capacity, active):
    """A captured six-step loop must read new rejection/slot data each round."""
    buffers = InputBuffers(max_num_reqs=capacity, max_num_tokens=capacity*7,
                           device=torch.device("cuda"))
    seq = torch.full((active,), 127, dtype=torch.int32, device="cuda")
    idx = torch.arange(active-1, -1, -1, device="cuda", dtype=torch.int32)
    sampled = torch.ones(active, dtype=torch.int32, device="cuda")
    rejected = torch.zeros_like(sampled)
    # Non-contiguous rows exercise the native last_sampled layout.
    last = torch.arange(capacity*7, device="cuda", dtype=torch.int64).reshape(capacity, 7)
    next_tokens = torch.arange(capacity*2, device="cuda", dtype=torch.int32)[::2][None]
    tokens = torch.arange(100, 100+capacity, device="cuda", dtype=torch.int64)
    drafts = torch.full((capacity, 6), -1, dtype=torch.int64, device="cuda")
    step = i64(0)

    def call():
        prepare_first_step(buffers, seq, idx, sampled, rejected, last, next_tokens, 132, capacity)
        step.zero_()
        for _ in range(6):
            advance(drafts, step, tokens, buffers, 132)

    for _ in range(3): call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): call()
    for reject in (0, 6, 2):
        rejected.fill_(reject)
        graph.replay()
        expected_len = min(132, 128-reject+6)
        assert buffers.seq_lens[:active].tolist() == [expected_len]*active
        assert buffers.positions[:active].tolist() == [min(131,127-reject+6)]*active
        assert buffers.seq_lens[active:].tolist() == [0]*(capacity-active)
        assert buffers.query_start_loc.tolist() == list(range(active))+[active]*(capacity+1-active)
        assert step.item() == 0
        torch.testing.assert_close(drafts, tokens[:, None].expand(-1, 6), rtol=0, atol=0)
