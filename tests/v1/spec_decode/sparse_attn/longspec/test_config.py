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
    [("coverage", 0.07, 0), ("vegas", 0.05, 256), ("streamingllm", 0.05, 256)],
)
def test_defaults(algorithm, ratio, floor):
    spec = config(algorithm)
    assert (spec.sparse_attn_ratio, spec.sparse_attn_min_tokens) == (ratio, floor)
    assert (spec.sparse_attn_theta, spec.sparse_attn_sink, spec.sparse_attn_recent) == (
        0.85,
        4,
        64,
    )
    assert spec.sparse_attn_fixed_budget is (algorithm == "coverage")
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


def test_ffn_draft_requires_checkpoint_and_changes_graph_hash():
    hashes = set()
    for scope in ("all", "ffn", "gate_up", "down"):
        if scope != "all":
            with pytest.raises(ValueError, match="draft checkpoint"):
                config(sparse_attn_draft_weights_scope=scope)
        hashes.add(config(sparse_attn_draft_weights="checkpoint",
                          sparse_attn_draft_weights_scope=scope).compute_hash())
    assert len(hashes) == 4


@pytest.mark.parametrize("scope", ["all", "ffn", "gate_up", "down"])
def test_draft_module_sharing_preserves_target(scope):
    ffn_only = scope != "all"
    import copy

    import torch
    from torch import nn

    from vllm.v1.spec_decode.sparse_attn.draft_weights import (
        _share_target_modules,
        _validate_int4_ffns,
    )

    target = nn.Module()
    target.model = nn.Module()
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.attn = nn.Identity()
    layer.self_attn.qkv_proj = nn.Linear(4, 4, dtype=torch.bfloat16)
    layer.mlp = nn.Module()
    layer.mlp.gate_up_proj = nn.Linear(4, 4, dtype=torch.bfloat16)
    layer.mlp.down_proj = nn.Linear(4, 4, dtype=torch.bfloat16)
    layer.input_layernorm = nn.LayerNorm(4, dtype=torch.bfloat16)
    layer.post_attention_layernorm = nn.LayerNorm(4, dtype=torch.bfloat16)
    target.model.layers = nn.ModuleList([layer])
    target.model.norm = nn.LayerNorm(4, dtype=torch.bfloat16)
    target.model.embed_tokens = nn.Embedding(8, 4, dtype=torch.bfloat16)
    target.lm_head = nn.Linear(4, 8, dtype=torch.bfloat16)
    draft = copy.deepcopy(target)
    original = {name: param.clone() for name, param in target.named_parameters()}
    draft_ffn = draft.model.layers[0].mlp
    _share_target_modules(target, draft, ffn_only=ffn_only, scope=scope)
    assert draft.model.layers[0].mlp is draft_ffn
    assert draft_ffn is not layer.mlp
    assert (draft_ffn.gate_up_proj is layer.mlp.gate_up_proj) == (scope == "down")
    assert (draft_ffn.down_proj is layer.mlp.down_proj) == (scope == "gate_up")
    assert (draft.model.layers[0].self_attn is layer.self_attn) == ffn_only
    assert draft.model.layers[0].self_attn.attn is layer.self_attn.attn
    assert (draft.model.norm is target.model.norm) == ffn_only
    assert draft.model.embed_tokens is target.model.embed_tokens
    assert draft.lm_head is target.lm_head
    for name, param in target.named_parameters():
        torch.testing.assert_close(param, original[name], rtol=0, atol=0)
    # Reject a BF16 checkpoint instead of silently calling it INT4.
    with pytest.raises(ValueError, match="INT4 W4A16"):
        _validate_int4_ffns(draft)

    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        CompressedTensorsWNA16,
    )

    for projection in (draft_ffn.gate_up_proj, draft_ffn.down_proj):
        projection.scheme = CompressedTensorsWNA16("group", 4, group_size=128)
    _validate_int4_ffns(draft)
    draft_ffn.down_proj.scheme = CompressedTensorsWNA16("group", 8, group_size=128)
    with pytest.raises(ValueError, match="layer 0 down_proj"):
        _validate_int4_ffns(draft)


def test_draft_loader_resets_revision_and_cleans_registry_on_failure(monkeypatch):
    from dataclasses import dataclass
    from types import SimpleNamespace as NS

    from vllm.model_executor import model_loader
    from vllm.v1.spec_decode.sparse_attn.draft_weights import load_draft_model

    @dataclass
    class Model:
        model: str = "target"
        revision: str | None = "target-commit"
        code_revision: str | None = "target-code-commit"
        quantization: str | None = None

    @dataclass
    class Config:
        model_config: Model
        compilation_config: object
        quant_config: object = None

    target_attention = object()
    registry = {"target": target_attention}
    cfg = Config(Model(), NS(static_forward_context=registry))

    def failing_loader(*, vllm_config, prefix):
        model = vllm_config.model_config
        assert model.model == "/pinned/draft"
        assert model.revision is model.code_revision is None
        registry[prefix] = object()
        raise RuntimeError("incomplete checkpoint")

    monkeypatch.setattr(model_loader, "get_model", failing_loader)
    with pytest.raises(RuntimeError, match="incomplete checkpoint"):
        load_draft_model(cfg, None, "/pinned/draft")
    assert cfg.model_config.revision == "target-commit"
    assert registry == {"target": target_attention}


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


def test_dense_graph_refreshes_inputs_and_rejects_prefill():
    from types import SimpleNamespace as NS
    from unittest.mock import Mock

    import torch

    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
    from vllm.v1.spec_decode.sparse_attn.proposer import DenseDecodeGraphs

    graphs = DenseDecodeGraphs.__new__(DenseDecodeGraphs)
    graphs.ids = torch.zeros(1, dtype=torch.int32)
    graphs.positions = torch.zeros(1, dtype=torch.int64)
    graphs.seq_lens = torch.zeros(1, dtype=torch.int32)
    graphs.block_table = torch.zeros(1, 3, dtype=torch.int32)
    graphs.slots = torch.zeros(1, dtype=torch.int64)
    graphs.hidden = torch.zeros(1, 2)
    graphs.sampled = torch.tensor([[7]], dtype=torch.int32)
    graph = NS(replay=Mock())
    graphs.graphs = {1: (graph,)}
    graphs.decode_replays = 0
    attention = FlashAttentionMetadata(
        num_actual_tokens=1,
        max_query_len=1,
        query_start_loc=torch.tensor([0, 1]),
        max_seq_len=33,
        seq_lens=torch.tensor([33]),
        block_table=torch.tensor([[4, 5, 6]]),
        slot_mapping=torch.tensor([96]),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
    )
    metadata = {"layer": attention}
    result = graphs.replay(
        torch.tensor([9]), torch.tensor([32]), metadata, greedy_metadata()
    )
    assert result[1].sampled_token_ids.tolist() == [[7]]
    assert graphs.ids.tolist() == [9]
    assert graphs.positions.tolist() == [32]
    assert graphs.seq_lens.tolist() == [33]
    assert graphs.block_table.tolist() == [[4, 5, 6]]
    assert graphs.slots.tolist() == [96]
    assert graphs.decode_replays == 1
    attention.max_query_len = 2
    assert (
        graphs.replay(
            torch.tensor([8]), torch.tensor([33]), metadata, greedy_metadata()
        )
        is None
    )
    graph.replay.assert_called_once()


def test_fixed_budget_includes_reserved_tokens_and_handles_padding():
    from types import SimpleNamespace as NS

    import torch

    from vllm.v1.spec_decode.sparse_attn.longspec.overrider import LongSpecAttnOverrider

    # Exercise real metadata/budget preparation without constructing GPU kernels.
    state = NS(
        spec=config(sparse_attn_ratio=0.07, sparse_attn_fixed_budget=True),
        num_spec_tokens=6,
        max_tokens=10000,
        _reduce_entry=torch.zeros(4, dtype=torch.int32),
        _valid_lens=torch.zeros(4, dtype=torch.int32),
        _k_max=torch.zeros(4, dtype=torch.int32),
        _k_min=torch.zeros(4, dtype=torch.int32),
    )
    kwargs = dict(
        seqused_k=torch.tensor([1006, 56, 3, 0]),
        cu_seqlens_q=torch.tensor([0, 7, 14, 15, 15]),
    )
    LongSpecAttnOverrider._begin_verify(state, kwargs)
    assert state._valid_lens.tolist() == [1000, 50, 3, 0]
    # 70 total tokens: 4 sink + 64 recent + 2 selected. Short prefixes retain all.
    assert state._k_min.tolist() == state._k_max.tolist() == [2, 0, 0, 0]
    state.spec.sparse_attn_fixed_budget = False
    LongSpecAttnOverrider._begin_verify(state, kwargs)
    assert state._k_min.tolist() == [0, 0, 0, 0]
    assert state._k_max.tolist() == [71, 4, 1, 0]


def test_sparse_verifier_is_opt_in_and_changes_graph_hash():
    full = config()
    sparse = config(sparse_attn_verify_ratio=0.5)
    assert full.sparse_attn_verify_ratio == 1
    assert full.compute_hash() != sparse.compute_hash()
    for ratio in (0, 1.1):
        with pytest.raises(ValueError):
            config(sparse_attn_verify_ratio=ratio)
    with pytest.raises(ValueError, match="coverage"):
        config("vegas", sparse_attn_verify_ratio=0.5)


def test_selected_scoring_requires_sparse_fixed_budget_and_changes_hash():
    sparse = config(sparse_attn_verify_ratio=0.5)
    selected = config(
        sparse_attn_verify_ratio=0.5, sparse_attn_verify_score_scope="selected"
    )
    assert sparse.sparse_attn_verify_score_scope == "full"
    assert selected.compute_hash() != sparse.compute_hash()
    for kwargs in (
        {},
        {"sparse_attn_verify_ratio": 0.5, "sparse_attn_fixed_budget": False},
    ):
        with pytest.raises(ValueError, match="selected scoring requires"):
            config(sparse_attn_verify_score_scope="selected", **kwargs)


@pytest.mark.parametrize(
    "change",
    [
        {"sparse_attn_fixed_budget": False},
        {"sparse_attn_ratio": 0.08},
        {"sparse_attn_sink": 8},
        {"sparse_attn_recent": 32},
        {"sparse_attn_min_tokens": 128},
        {"sparse_attn_collect_stats": True},
    ],
)
def test_selection_constants_change_compilation_hash(change):
    assert config().compute_hash() != config(**change).compute_hash()


@pytest.mark.parametrize("confirmed_round", [False, True])
def test_seven_token_prefill_is_not_sparse_verification(monkeypatch, confirmed_round):
    from types import SimpleNamespace as NS
    from unittest.mock import Mock

    import torch

    from vllm.v1.spec_decode.sparse_attn.attn_overrider import BaseAttnOverrider
    from vllm.v1.spec_decode.sparse_attn.longspec.overrider import LongSpecAttnOverrider

    verifier = NS(
        eligible=Mock(return_value=True),
        prepare=Mock(),
        attention_kwargs=lambda kwargs, layer: kwargs,
        selected_scores=False,
        full_lse=Mock(return_value=torch.zeros(1)),
        remember=Mock(),
    )
    state = NS(
        curr_layer=0,
        num_layers=1,
        batch_size=1,
        in_sparse_verify=confirmed_round,
        _sparse_verifier=verifier,
        _begin_verify=Mock(),
        _packed_verify=False,
        _scores=NS(verify_kwargs=Mock(), reduce=Mock()),
        _select=Mock(),
        _metric=torch.zeros(1, 1, 16),
        _valid_lens=torch.tensor([10]),
        _reduce_entry=torch.zeros(1, dtype=torch.int32),
    )
    monkeypatch.setattr(
        BaseAttnOverrider,
        "_original_attn_func",
        lambda **kw: (torch.zeros(7, 1, 4), torch.zeros(1, 7)),
        raising=False,
    )
    LongSpecAttnOverrider._verify_attention(state, q=torch.zeros(7, 1, 4))
    assert verifier.prepare.called == confirmed_round
    assert verifier.full_lse.called == confirmed_round
