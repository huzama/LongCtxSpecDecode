# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Method defaults and rejected configuration values."""

import pytest

from vllm.config.speculative import SpeculativeConfig


def config(algorithm="coverage", **kwargs):
    return SpeculativeConfig(
        method="sparse_attn",
        num_speculative_tokens=6,
        sparse_attn_algorithm=algorithm,
        **kwargs,
    )


@pytest.mark.parametrize(
    "algorithm,ratio,floor",
    [("coverage", 0.15, 0), ("vegas", 0.05, 256), ("streamingllm", 0.05, 256)],
)
def test_defaults(algorithm, ratio, floor):
    spec = config(algorithm)
    assert (spec.sparse_attn_ratio, spec.sparse_attn_min_tokens) == (ratio, floor)
    assert (spec.sparse_attn_theta, spec.sparse_attn_sink, spec.sparse_attn_recent) == (
        0.85,
        4,
        64,
    )
    assert spec.sparse_attn_collect_stats is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("sparse_attn_theta", 0),
        ("sparse_attn_theta", 1.5),
        ("sparse_attn_ratio", 0),
        ("sparse_attn_ratio", 1.5),
        ("sparse_attn_min_tokens", -1),
        ("sparse_attn_algorithm", "longspec"),
    ],
)
def test_invalid(field, value):
    with pytest.raises(ValueError):
        SpeculativeConfig(
            method="sparse_attn", num_speculative_tokens=6, **{field: value}
        )


def test_explicit_budget():
    spec = config(sparse_attn_ratio=1, sparse_attn_min_tokens=17)
    assert spec.sparse_attn_ratio == 1 and spec.sparse_attn_min_tokens == 17


def test_fp8_cache_rejected_before_initialization():
    from types import SimpleNamespace

    from vllm.v1.spec_decode.sparse_attn.attn_overrider import build_attention_overrider

    cfg = SimpleNamespace(
        speculative_config=config(), cache_config=SimpleNamespace(cache_dtype="fp8")
    )
    with pytest.raises(ValueError, match="non-FP8"):
        build_attention_overrider(cfg, None)


def greedy_metadata():
    from types import SimpleNamespace

    return SimpleNamespace(
        all_greedy=True,
        no_penalties=True,
        max_num_logprobs=None,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=SimpleNamespace(non_argmax_invariant=[]),
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("all_greedy", False),
        ("no_penalties", False),
        ("max_num_logprobs", 0),
        ("allowed_token_ids_mask", object()),
        ("bad_words_token_ids", {0: [[1]]}),
    ],
)
def test_round_graph_rejects_request_dependent_sampling(field, value):
    from vllm.v1.spec_decode.sparse_attn.proposer import plain_greedy

    metadata = greedy_metadata()
    assert plain_greedy(metadata)
    setattr(metadata, field, value)
    assert not plain_greedy(metadata)


def test_round_graph_checks_active_logits_processors():
    from vllm.v1.sample.logits_processor.builtin import (
        LogitBiasLogitsProcessor,
        MinTokensLogitsProcessor,
    )
    from vllm.v1.spec_decode.sparse_attn.proposer import plain_greedy

    metadata = greedy_metadata()
    bias = LogitBiasLogitsProcessor.__new__(LogitBiasLogitsProcessor)
    minimum = MinTokensLogitsProcessor.__new__(MinTokensLogitsProcessor)
    bias.biases, minimum.min_toks = {}, {}
    metadata.logitsprocs.non_argmax_invariant = [bias, minimum]
    assert plain_greedy(metadata)
    bias.biases = {0: {5: 1.0}}
    assert not plain_greedy(metadata)
    bias.biases = {}
    minimum.min_toks = {0: (5, [], {2})}
    assert not plain_greedy(metadata)
    metadata.logitsprocs.non_argmax_invariant = [object()]
    assert not plain_greedy(metadata)


def test_round_graph_refreshes_inputs_and_consumes_handoff_once():
    from types import SimpleNamespace as NS
    from unittest.mock import Mock

    import torch

    from vllm.v1.spec_decode.sparse_attn.proposer import (
        SparseAttnProposer,
        SparseAttnRoundGraphs,
    )

    proposer = SparseAttnProposer.__new__(SparseAttnProposer)
    proposer.num_speculative_tokens = 2
    proposer._sampled_token_ids = torch.zeros(2, 2, dtype=torch.int32)
    graphs = SparseAttnRoundGraphs.__new__(SparseAttnRoundGraphs)
    graphs.proposer = proposer
    graphs.ids = torch.zeros(6, dtype=torch.int64)
    graphs.seq_lens = torch.zeros(2, dtype=torch.int32)
    graphs.block_table = torch.zeros(2, 3, dtype=torch.int32)
    graphs.hidden = torch.zeros(6, 1)
    graphs.sampled = torch.ones(2, 3, dtype=torch.int32)
    graphs.verify_replays = graphs.draft_replays = 0
    verify, draft = NS(replay=Mock()), NS(replay=Mock())
    graphs.graphs = {2: (verify, draft, ())}
    common = NS(
        num_actual_tokens=6,
        seq_lens=torch.tensor([10, 20]),
        block_table_tensor=torch.tensor([[1, 2, 3], [4, 5, 6]]),
    )
    spec = NS(num_draft_tokens=[2, 2])
    ids = torch.arange(6)
    first_output = graphs.verify(ids, common, spec, greedy_metadata())
    assert first_output is not None
    torch.testing.assert_close(graphs.ids, ids)
    assert graphs.seq_lens.tolist() == [10, 20]
    assert graphs.block_table.tolist() == [[1, 2, 3], [4, 5, 6]]
    assert graphs.draft_after_verify(2) is not None
    assert graphs.draft_after_verify(2) is None
    assert verify.replay.call_count == draft.replay.call_count == 1

    # Same graph, new tokens, contexts and physical blocks.
    common.seq_lens += 5
    common.block_table_tensor += 10
    graphs.verify(ids + 7, common, spec, greedy_metadata())
    assert graphs.ids.tolist() == list(range(7, 13))
    assert graphs.seq_lens.tolist() == [15, 25]
    assert graphs.block_table[0].tolist() == [11, 12, 13]
    spec.num_draft_tokens = [1, 2]
    assert graphs.verify(ids, common, spec, greedy_metadata()) is None
    assert graphs.draft_after_verify(2) is None
    assert verify.replay.call_count == 2
