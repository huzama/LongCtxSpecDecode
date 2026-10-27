# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for Vegas draft replay inputs and per-forward dispatch."""

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.spec_decode.sparse_attn import proposer as module
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


@pytest.fixture
def draft(monkeypatch):
    p = object.__new__(module.SparseAttnProposer)
    p._hierarchical = True
    p._is_vegas = True
    p.max_model_len, p.block_size, p.num_speculative_tokens = 64, 16, 6
    p.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        compilation_config=SimpleNamespace(cudagraph_num_of_warmups=1),
    )
    p.speculative_config = SimpleNamespace(enforce_eager=False)
    p.runner = SimpleNamespace(hierarchical_context=SimpleNamespace(
        drafting=False, draft_committed=torch.zeros(4, dtype=torch.int32)))
    p.input_ids = torch.zeros(4, dtype=torch.int32)
    p.positions = torch.zeros(4, dtype=torch.int64)
    p._sampled_token_ids = torch.zeros(4, 6, dtype=torch.int32)
    p._draft_seq_lens = torch.zeros(4, dtype=torch.int32)
    p._draft_slots = torch.full((4,), -1, dtype=torch.int64)
    p._draft_blocks = torch.zeros(4, 4, dtype=torch.int32)
    p._draft_query_start = torch.arange(5, dtype=torch.int32)
    p._draft_query_start_cpu = torch.arange(5, dtype=torch.int32)
    p.arange = p._draft_query_start
    p.token_arange_np = np.arange(5)
    p.attn_layer_names = ["layer"]
    p.allowed_attn_types = (SimpleNamespace,)
    p.attn_metadata_builder = SimpleNamespace(build_for_drafting=lambda
        common_attn_metadata, draft_index: SimpleNamespace(
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            max_seq_len=common_attn_metadata.max_seq_len))
    p._get_attention_metadata_builder = lambda: p.attn_metadata_builder
    p.attn_overrider = SimpleNamespace(
        enter_propose=lambda: None, exit_propose=lambda: None,
        _page_table=torch.zeros(1, 4, 8, dtype=torch.int32),
        _topk_budget=torch.zeros(4, dtype=torch.int32),
        _topk_valid_lens=torch.zeros(4, dtype=torch.int32),
    )
    p._draft_batch_descriptor = BatchDescriptor(4, 4, True)
    p._determine_batch_execution_and_padding = lambda n: (
        CUDAGraphMode.FULL, 4, None)
    p.cudagraph_dispatcher = SimpleNamespace(get_capture_descs=lambda: [
        (CUDAGraphMode.FULL, [p._draft_batch_descriptor])])
    calls, contexts = [], []

    @contextmanager
    def forward_context(metadata, config, **kwargs):
        contexts.append(kwargs)
        yield

    monkeypatch.setattr(module, "set_forward_context", forward_context)

    def forward(variant, **kwargs):
        assert p.runner.hierarchical_context.drafting == p._hierarchical
        calls.append(dict(
            variant=variant, positions=kwargs["positions"].clone(),
            lengths=p._draft_seq_lens.clone(), slots=p._draft_slots.clone(),
            committed=p.runner.hierarchical_context.draft_committed.clone(),
            budgets=p.attn_overrider._topk_budget.clone(),
            valid_lengths=p.attn_overrider._topk_valid_lens.clone(),
            addresses=(p._draft_seq_lens.data_ptr(), p._draft_slots.data_ptr(),
                       p._draft_blocks.data_ptr()),
        ))
        return torch.zeros(len(kwargs["input_ids"]), 8)

    p._draft_forward_models = (
        lambda **kwargs: forward("first", **kwargs),
        lambda **kwargs: forward("continue", **kwargs),
    )

    def save(draft_depth, hidden_states, sampling_metadata):
        p._sampled_token_ids[:p.batch_size, draft_depth].fill_(draft_depth + 1)

    p._save_hidden_states_and_sample = save
    return p, calls, contexts


def metadata():
    return CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 7, 14], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 7, 14], dtype=torch.int32),
        seq_lens=torch.tensor([20, 64], dtype=torch.int32),
        num_reqs=2, num_actual_tokens=14, max_query_len=7, max_seq_len=64,
        block_table_tensor=torch.tensor([[4, 5, 6, 7], [8, 9, 10, 11]],
                                        dtype=torch.int32),
        slot_mapping=torch.zeros(14, dtype=torch.int64),
    )


@pytest.mark.parametrize("hierarchical", [False, True])
def test_one_forward_per_draft_and_stable_padded_metadata(draft, hierarchical):
    p, calls, contexts = draft
    p._hierarchical = hierarchical
    result = p.propose(torch.tensor([10, 20]), None, metadata(), None,
                       num_rejected_tokens_gpu=torch.tensor([3, 1]))
    assert result.shape == (2, 6)
    assert [c["variant"] for c in calls] == ["first"] + ["continue"] * 5
    assert len({c["addresses"] for c in calls}) == 1
    if hierarchical:
        assert all(c["committed"].tolist() == [17, 63, 0, 0] for c in calls)
    assert calls[0]["positions"].tolist() == [17, 63, 0, 0]
    assert calls[0]["slots"].tolist() == (
        [0, 0, -1, -1] if hierarchical else [81, 191, -1, -1])
    assert all(c["slots"][2:].tolist() == [-1, -1] for c in calls)
    assert all(c["lengths"][2:].tolist() == [0, 0] for c in calls)
    assert all(c["slots"][1].item() == -1 for c in calls[1:])
    assert all(c["batch_descriptor"] is p._draft_batch_descriptor for c in contexts)
    assert not p.runner.hierarchical_context.drafting


def test_packed_slot_markers_skip_page_lookup_and_reset_each_proposal(
    draft, monkeypatch
):
    p, calls, _ = draft

    def unexpected_page_lookup(*args, **kwargs):
        raise AssertionError("packed draft must not compute physical slot IDs")

    monkeypatch.setattr(torch.Tensor, "gather", unexpected_page_lookup)
    p.propose(torch.tensor([10, 20]), None, metadata(), None,
              num_rejected_tokens_gpu=torch.tensor([3, 1]))
    assert calls[-1]["slots"].tolist() == [0, -1, -1, -1]

    calls.clear()
    p.propose(torch.tensor([30, 40]), None, metadata(), None,
              num_rejected_tokens_gpu=torch.tensor([3, 2]))
    assert [c["slots"][1].item() for c in calls] == [0, 0, -1, -1, -1, -1]
    assert all(c["slots"][2:].tolist() == [-1, -1] for c in calls)


def test_bf16_drafts_keep_physical_slots_and_sticky_overflow(draft):
    p, calls, _ = draft
    p._hierarchical = False
    p.attn_overrider._topk_budget[2:].fill_(9)
    p.attn_overrider._topk_valid_lens[2:].fill_(32)
    p.propose(torch.tensor([10, 20]), None, metadata(), None,
              num_rejected_tokens_gpu=torch.tensor([3, 1]))
    assert [c["slots"][:2].tolist() for c in calls] == [
        [81 + step, 191 if step == 0 else -1] for step in range(6)]
    assert all(c["budgets"][2:].tolist() == [0, 0] for c in calls)
    assert all(c["valid_lengths"][2:].tolist() == [0, 0] for c in calls)


@pytest.mark.parametrize("hierarchical", [False, True])
def test_capture_separates_first_and_continuation_forward(draft, hierarchical):
    p, calls, contexts = draft
    p._hierarchical = hierarchical
    p.capture_draft_graphs()
    assert [c["variant"] for c in calls] == ["first", "continue"] * 2
    assert [c["cudagraph_runtime_mode"] for c in contexts] == (
        [CUDAGraphMode.NONE] * 2 + [CUDAGraphMode.FULL] * 2)
    assert not p.runner.hierarchical_context.drafting


def test_bf16_capture_resets_full_gather_then_appends(draft, monkeypatch):
    from vllm.v1.spec_decode.sparse_attn.attn_overrider.utils import vegas_fallback

    p, _, _ = draft
    p._hierarchical = False
    cache = vegas_fallback.GatheredDraftKV(
        1, 4, 16, 16, 1, 8, torch.float32, "cpu")
    starts = []
    monkeypatch.setattr(vegas_fallback, "gather_tokens",
                        lambda k, v, slots, begin, end, *dst:
                        starts.append(begin.clone()))
    p.attn_overrider.enter_propose = cache.begin_propose

    def forward(**kwargs):
        cache.attention_kwargs(
            dict(k=cache.k, v=cache.v, max_seqlen_k=64), 0,
            p.attn_overrider._page_table[0], p._draft_seq_lens)
        return torch.zeros(4, 8)

    p._draft_forward_models = (forward, forward)
    p.capture_draft_graphs()
    assert [x.tolist() for x in starts] == [[0] * 4, [1] * 4] * 2


def test_failed_forward_restores_verifier_mode(draft):
    p, _, _ = draft

    def fail(**kwargs):
        raise RuntimeError("forward failed")

    p._draft_forward_models = (fail, fail)
    with pytest.raises(RuntimeError, match="forward failed"):
        p.propose(torch.tensor([10, 20]), None, metadata(), None)
    assert not p.runner.hierarchical_context.drafting


@pytest.mark.parametrize("hierarchical", [False, True])
def test_warmup_bypasses_graph_dispatch(draft, hierarchical):
    p, _, _ = draft
    p._hierarchical = hierarchical
    mode, tokens, dp_tokens = (
        module.SparseAttnProposer._determine_batch_execution_and_padding(
            p, 3, use_cudagraphs=False
        )
    )
    assert mode == CUDAGraphMode.NONE
    assert tokens == 3 and dp_tokens is None
    assert p._draft_batch_descriptor == BatchDescriptor(3)


@pytest.mark.parametrize("hierarchical", [False, True])
@pytest.mark.parametrize("config_name", ["model_config", "speculative_config"])
def test_explicit_eager_mode_bypasses_graph_dispatch(
    draft, hierarchical, config_name
):
    p, _, _ = draft
    p._hierarchical = hierarchical
    config = (p.vllm_config.model_config if config_name == "model_config"
              else p.speculative_config)
    config.enforce_eager = True
    mode, tokens, dp_tokens = (
        module.SparseAttnProposer._determine_batch_execution_and_padding(p, 3))
    assert (mode, tokens, dp_tokens) == (CUDAGraphMode.NONE, 3, None)


@pytest.mark.parametrize("hierarchical", [False, True])
def test_draft_capture_sizes_survive_verifier_rounding(draft, hierarchical):
    p, _, _ = draft
    p._hierarchical = hierarchical
    config = p.vllm_config.compilation_config
    config.max_cudagraph_capture_size = 32
    config.cudagraph_capture_sizes = [1, 2, 4, 8, 16, 32]
    modes = []
    p.cudagraph_dispatcher.initialize_cudagraph_keys = modes.append
    p.initialize_cudagraph_keys(CUDAGraphMode.FULL_AND_PIECEWISE)
    config.max_cudagraph_capture_size = 28
    config.cudagraph_capture_sizes = [7, 14, 21, 28]
    saved = p.cudagraph_dispatcher.compilation_config
    assert saved.max_cudagraph_capture_size == 32
    assert saved.cudagraph_capture_sizes == [1, 2, 4, 8, 16, 32]
    assert modes == [CUDAGraphMode.FULL_DECODE_ONLY]


@pytest.mark.parametrize(
    "cache_dtype,computed,queries,forced,disable_full,capacity,uniform",
    [
        ("int4_4", [0], [7], None, True, 7, True),
        ("int4_4", [20, 0], [7, 7], None, True, 14, True),
        ("int4_4", [20, 63], [7, 1], None, False, 14, True),
        ("int4_4", [63, 63], [1, 1], None, False, 14, True),
        ("int4_4", [20, 20], [7, 7], None, False, 14, True),
        # Startup capture explicitly declares its batch type and has no live
        # request state. It must still capture both verifier and mixed graphs.
        ("int4_4", [0], [7], True, False, 7, True),
        ("int4_4", [0], [7], False, False, 7, False),
        ("auto", [0], [7], None, False, 7, True),
        ("auto", [20, 63], [7, 1], None, False, 14, True),
    ],
)
def test_verifier_dispatch_preserves_prefills_and_short_decode_rows(
    cache_dtype, computed, queries, forced, disable_full, capacity, uniform
):
    calls = []

    def dispatch(**kwargs):
        calls.append(kwargs)
        mode = (CUDAGraphMode.PIECEWISE if kwargs["disable_full"]
                else CUDAGraphMode.FULL)
        return mode, BatchDescriptor(kwargs["num_tokens"])

    runner = verifier_runner(cache_dtype, computed, dispatch)
    mode, descriptor, *_ = (
        GPUModelRunner._determine_batch_execution_and_padding(
            runner, num_tokens=sum(queries), num_reqs=len(queries),
            num_scheduled_tokens_np=np.array(queries),
            max_num_scheduled_tokens=max(queries), use_cascade_attn=False,
            force_uniform_decode=forced))
    assert len(calls) == 1
    assert calls[0]["disable_full"] is disable_full
    assert calls[0]["uniform_decode"] is uniform
    assert descriptor.num_tokens == capacity
    assert mode == (CUDAGraphMode.PIECEWISE if disable_full
                    else CUDAGraphMode.FULL)


def verifier_runner(cache_dtype, computed, dispatch,
                    mode=CUDAGraphMode.FULL_AND_PIECEWISE):
    return SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype=cache_dtype),
        speculative_config=SimpleNamespace(
            method="sparse_attn", sparse_attn_algorithm="vegas"),
        compilation_config=SimpleNamespace(
            cudagraph_mode=mode,
            pass_config=SimpleNamespace(enable_sp=False)),
        model_config=SimpleNamespace(is_encoder_decoder=False),
        input_batch=SimpleNamespace(
            num_computed_tokens_cpu=np.array(computed),
            num_prompt_tokens=np.ones(len(computed), dtype=np.int32),
            lora_id_to_lora_request={}),
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=1),
            observability_config=SimpleNamespace(cudagraph_metrics=False)),
        uniform_decode_query_len=7,
        _is_uniform_decode=GPUModelRunner._is_uniform_decode,
        _pad_for_sequence_parallelism=lambda n: n,
        cudagraph_dispatcher=SimpleNamespace(dispatch=dispatch),
    )


def native_dispatcher(capacity):
    dispatcher = object.__new__(CudagraphDispatcher)
    dispatcher.keys_initialized = True
    dispatcher.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    dispatcher.compilation_config = SimpleNamespace(
        max_cudagraph_capture_size=capacity)
    dispatcher.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=4))
    dispatcher.uniform_decode_query_len = 7
    dispatcher._bs_to_padded_graph_size = [0] + [
        ((n + 6) // 7) * 7 for n in range(1, capacity + 1)]
    dispatcher.cudagraph_keys = {
        CUDAGraphMode.FULL: {
            BatchDescriptor(n, n // 7, True)
            for n in range(7, capacity + 1, 7)},
        CUDAGraphMode.PIECEWISE: set(),
    }
    return dispatcher


@pytest.mark.parametrize("queries", [[7, 1], [1, 1], [6, 2, 1]])
@pytest.mark.parametrize("cache_dtype", ["auto", "int4_4"])
def test_short_decode_rows_reuse_native_full_capture(queries, cache_dtype):
    runner = verifier_runner(
        cache_dtype, [20] * len(queries), native_dispatcher(28).dispatch)
    mode, descriptor, *_ = (
        GPUModelRunner._determine_batch_execution_and_padding(
            runner, num_tokens=sum(queries), num_reqs=len(queries),
            num_scheduled_tokens_np=np.array(queries),
            max_num_scheduled_tokens=max(queries), use_cascade_attn=False))
    assert mode == CUDAGraphMode.FULL
    assert descriptor == BatchDescriptor(len(queries) * 7, len(queries), True)


@pytest.mark.parametrize("reason", ["other_speculator", "unfinished_prefill"])
def test_other_batches_keep_native_nonuniform_dispatch(reason):
    runner = verifier_runner("auto", [20, 63], native_dispatcher(28).dispatch)
    if reason == "other_speculator":
        runner.speculative_config = None
    else:
        runner.input_batch.num_prompt_tokens[0] = 32
    mode, descriptor, *_ = (
        GPUModelRunner._determine_batch_execution_and_padding(
            runner, num_tokens=8, num_reqs=2,
            num_scheduled_tokens_np=np.array([7, 1]),
            max_num_scheduled_tokens=7, use_cascade_attn=False))
    assert mode == CUDAGraphMode.NONE
    assert descriptor.num_tokens == 8


def test_missing_verifier_capture_reports_required_capacity():
    runner = verifier_runner("int4_4", [20, 63], native_dispatcher(7).dispatch)
    with pytest.raises(RuntimeError, match=r"verifier.*2 requests \(14 tokens"):
        GPUModelRunner._determine_batch_execution_and_padding(
            runner, num_tokens=8, num_reqs=2,
            num_scheduled_tokens_np=np.array([7, 1]),
            max_num_scheduled_tokens=7, use_cascade_attn=False)


def test_missing_draft_capture_reports_required_capacity(draft):
    p, _, _ = draft
    p.cudagraph_dispatcher = SimpleNamespace(
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        dispatch=lambda **kw: (CUDAGraphMode.NONE, BatchDescriptor(3)))
    with pytest.raises(RuntimeError, match=r"draft.*3 requests \(3 tokens"):
        module.SparseAttnProposer._determine_batch_execution_and_padding(p, 3)


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE])
def test_explicit_non_full_verifier_mode_keeps_actual_token_count(mode):
    runner = verifier_runner(
        "int4_4", [20, 63],
        lambda **kw: (mode, BatchDescriptor(kw["num_tokens"])), mode=mode)
    result_mode, descriptor, *_ = (
        GPUModelRunner._determine_batch_execution_and_padding(
            runner, num_tokens=8, num_reqs=2,
            num_scheduled_tokens_np=np.array([7, 1]),
            max_num_scheduled_tokens=7, use_cascade_attn=False))
    assert result_mode == mode and descriptor.num_tokens == 8
