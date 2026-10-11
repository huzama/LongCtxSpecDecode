"""Semantic tests, independent of vLLM and GPU availability."""
import pytest
import torch

from highlight_spec.selectors import (
    SelectorBudget, gather_draft_kv, triforce_chunk_keys, triforce_scores,
    triforce_select, triforce_update_tail, vegas_reference_scores, vegas_select,
)


def test_budget_protocols_never_silently_round_matched_tokens():
    matched = SelectorBudget.matched(32768, 2304, highlight_block=16, triforce_chunk=8)
    assert matched.highlight_tokens == matched.vegas_tokens == matched.triforce_tokens == 2304
    paper = SelectorBudget.paper(32768, highlight_tokens=2304)
    assert (paper.vegas_tokens, paper.triforce_tokens, paper.triforce_chunk) == (2294, 4096, 8)
    with pytest.raises(ValueError, match="granularities"):
        SelectorBudget.matched(32768, 2294, highlight_block=16, triforce_chunk=8)
    with pytest.raises(ValueError, match="4096"):
        SelectorBudget.paper(1024, highlight_tokens=64)


@pytest.mark.parametrize("phase,qlen", [("verify", 1), ("verify", 7), ("prefill", 9)])
def test_vegas_boundary_logits_and_gqa(phase, qlen):
    rng = torch.Generator().manual_seed(17)
    q = torch.randn(2, qlen, 6, 4, generator=rng)
    k = torch.randn(2, 13, 2, 4, generator=rng)
    prefix = 13 if phase == "prefill" else 13 - qlen + 1
    rows = [qlen - 1] if phase == "prefill" else [0, qlen - 1]
    expected = torch.stack([
        torch.stack([sum(torch.dot(q[b, r, h], k[b, t, h // 3])
                         for r in rows for h in range(6)) / (len(rows) * 6)
                     for t in range(prefix)]) for b in range(2)])
    torch.testing.assert_close(vegas_reference_scores(q, k, prefix_tokens=prefix, phase=phase), expected)


def test_vegas_ignores_middle_queries_but_keeps_last_even_if_rejected():
    q = torch.tensor([1., 10000., 3.]).view(1, 3, 1, 1)
    k = torch.arange(1., 6.).view(1, 5, 1, 1)
    scores = vegas_reference_scores(q, k, prefix_tokens=3, phase="verify")
    assert scores.tolist() == [[2., 4., 6.]]
    assert vegas_select(scores, tokens=2).tolist() == [[[2, 1]]]
    with pytest.raises(ValueError, match="causal"):
        vegas_reference_scores(q, k, prefix_tokens=4, phase="verify")


def test_triforce_preserves_distinct_query_heads_sharing_one_kv_head():
    # Head 0 ranks the positive chunk, head 1 the negative chunk. Chunk 0 forced.
    k = torch.tensor([0., 0., 3., 5., -5., -3., 1., 1.]).view(1, 8, 1, 1)
    q = torch.tensor([1., -1.]).view(1, 1, 2, 1)
    means = triforce_chunk_keys(k, chunk_size=2)
    scores = triforce_scores(q, means)
    indices = triforce_select(scores, tokens=4, chunk_size=2)
    assert indices.tolist() == [[[0, 1, 2, 3], [0, 1, 4, 5]]]
    selected_k, selected_v = gather_draft_kv(k, k + 10, indices, query_heads=2)
    assert selected_k.shape == (1, 4, 2, 1)
    assert selected_k[0, :, 0, 0].tolist() == [0., 0., 3., 5.]
    assert selected_k[0, :, 1, 0].tolist() == [0., 0., -5., -3.]
    torch.testing.assert_close(selected_v, selected_k + 10)


def test_chunk_zero_forced_even_when_it_has_the_lowest_score():
    scores = torch.tensor([[[-100., 9., 3.], [-100., 1., 8.]]])
    assert triforce_select(scores, tokens=2, chunk_size=2).tolist() == [[[0, 1], [0, 1]]]
    with pytest.raises(ValueError, match="whole-chunk"):
        triforce_select(scores, tokens=3, chunk_size=2)
    with pytest.raises(ValueError, match="complete"):
        triforce_chunk_keys(torch.zeros(1, 5, 1, 2), chunk_size=2)


def test_shared_selection_keeps_native_kv_heads_and_batch_mapping():
    k = torch.arange(2 * 7 * 2 * 3).reshape(2, 7, 2, 3).float()
    indices = torch.tensor([[[6, 0]], [[1, 5]]])
    selected, _ = gather_draft_kv(k, k, indices, query_heads=6)
    assert selected.shape == (2, 2, 2, 3)
    for b in range(2):
        torch.testing.assert_close(selected[b], k[b, indices[b, 0]])


def test_tail_update_reuses_cache_and_expands_gqa_without_rescoring():
    draft = torch.full((1, 4, 4, 1), -1.)
    value = draft.clone()
    accepted = torch.tensor([2., 3., 4., 5.]).view(1, 2, 2, 1)
    triforce_update_tail(draft, value, accepted, accepted + 10, query_heads=4)
    assert draft[0, :, :, 0].tolist() == [[-1.] * 4, [-1.] * 4, [2., 2., 3., 3.], [4., 4., 5., 5.]]
    torch.testing.assert_close(value[:, -2:], draft[:, -2:] + 10)
    before = draft.clone()
    triforce_update_tail(draft, value, accepted[:, :0], accepted[:, :0], query_heads=4)
    assert torch.equal(before, draft)
    with pytest.raises(ValueError, match="rebuild"):
        triforce_update_tail(draft, value, accepted.repeat(1, 3, 1, 1),
                             accepted.repeat(1, 3, 1, 1), query_heads=4)


def test_capture_requires_provenance_and_explicit_retrieval_query(tmp_path):
    from benchmarks.selectors import load_capture
    capture = {
        "q": torch.ones(1, 3, 4, 8), "k": torch.ones(1, 8, 2, 8),
        "v": torch.ones(1, 8, 2, 8), "retrieval_q": torch.ones(1, 1, 4, 8),
        "prefix_tokens": 6, "phase": "verify", "model": "fixture",
        "layer": 0, "prompt_sha256": ["a" * 64],
    }
    path = tmp_path / "capture.pt"
    torch.save(capture, path)
    assert load_capture(path)["prefix_tokens"] == 6
    capture["prefix_tokens"] = 7
    torch.save(capture, path)
    with pytest.raises(ValueError, match="causal"):
        load_capture(path)
    capture["prefix_tokens"] = 6
    del capture["retrieval_q"]
    torch.save(capture, path)
    with pytest.raises(ValueError, match="requires"):
        load_capture(path)


def test_reference_cache_initializes_once_and_accepts_only_growing_committed_suffix(monkeypatch):
    import highlight_spec.selectors as selectors
    k = torch.arange(8.).view(1, 8, 1, 1)
    q = torch.tensor([1., -1.]).view(1, 1, 2, 1)
    cache = selectors.TriForceRetrievalCache(q, k, k + 20, tokens=4, chunk_size=2)
    initial = cache.k.clone()

    def forbidden(*args, **kwargs):
        raise AssertionError("tail update must not rescore or reselect")

    monkeypatch.setattr(selectors, "triforce_scores", forbidden)
    monkeypatch.setattr(selectors, "triforce_select", forbidden)
    accepted = torch.tensor([8., 9.]).view(1, 2, 1, 1)
    cache.update(accepted[:, :1], accepted[:, :1] + 20)
    cache.update(accepted, accepted + 20)
    torch.testing.assert_close(cache.k[:, :2], initial[:, :2])
    assert cache.indices[0, :, -2:].tolist() == [[8, 9], [8, 9]]
    assert cache.accepted_tokens == 2
    assert cache.kv_bytes == 64
    with pytest.raises(ValueError, match="shrink"):
        cache.update(accepted[:, :1], accepted[:, :1])
    assert cache.accepted_tokens == 2
