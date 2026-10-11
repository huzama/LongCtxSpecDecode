import json

import pytest

from benchmarks import accuracy, decode


@pytest.mark.parametrize(
    "total, factor",
    [(32640 + 128, None), (32769, 2.0), (65536, 2.0), (65537, 3.0), (131072, 4.0)],
)
def test_yarn_covers_the_total_with_native_multiples(total, factor):
    assert decode.yarn_factor(total) == factor


def test_yarn_refuses_past_the_validated_window():
    with pytest.raises(ValueError, match="131072"):
        decode.yarn_factor(131073)


def test_rope_parameters_keep_the_checkpoint_theta(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"rope_parameters": {"rope_theta": 1000000, "rope_type": "default"}}))
    assert decode.rope_parameters(str(tmp_path), 2.0) == {
        "rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 32768, "rope_theta": 1e6,
    }


def test_sampling_is_explicitly_greedy():
    p = decode.sampling_params(128, ignore_eos=False)
    assert (p.temperature, p.top_p, p.top_k, p.min_p) == (0.0, 1.0, 0, 0.0)
    assert (p.presence_penalty, p.frequency_penalty, p.repetition_penalty) == (0.0, 0.0, 1.0)
    assert (p.max_tokens, p.min_tokens, p.ignore_eos, p.skip_special_tokens) == (128, 0, False, False)


@pytest.mark.parametrize(
    "text, choice",
    [
        ("The correct answer is (C).<|im_end|>", "C"),
        ("The correct answer is **(B)**", "B"),
        ("The correct answer is D because", "D"),
        ("<think>still thinking", None),
        ("<think>The correct answer is (A)</think>The correct answer is (B)", "B"),
        ("I cannot tell.", None),
    ],
)
def test_extract_answer(text, choice):
    assert accuracy.extract_answer(text) == choice


def _args(**kw):
    import argparse

    p = argparse.ArgumentParser()
    decode.add_engine_args(p)
    args = p.parse_args(["--checkpoint", "x", "--mode", "highlight", *kw.pop("argv", [])])
    args.ctx = kw.pop("ctx", 32640)
    return args


def test_topk_selection_by_default():
    assert decode.selection_overrides(_args()) == {
        "molle_selection_mode": "topk_fraction", "molle_topk_fraction": 0.07, "molle_topk_recency": 64,
        "molle_draft_select": "step", "molle_verify_fraction": 0.0,
    }


@pytest.mark.parametrize("mode", ["direct", "highlight"])
def test_mxfp4_changes_only_the_indexer_cache(monkeypatch, mode):
    import vllm
    args = _args(argv=["--indexer-cache-format", "mxfp4"])
    args.mode, args.batch, args.gen = mode, 6, 128
    monkeypatch.setattr(vllm, "LLM", lambda **kwargs: kwargs)
    config = decode.build_engine(args)
    assert config["hf_overrides"]["molle_kcache_format"] == "mxfp4"
    assert config["dtype"] == "bfloat16" and config["kv_cache_dtype"] == "auto"
    assert config["hf_overrides"]["molle_topk_fraction"] == .07
    args.mode = "dense"
    with pytest.raises(ValueError, match="no Highlight indexer"):
        decode.build_engine(args)


@pytest.mark.parametrize("mode", ["dense", "direct", "highlight"])
def test_engine_selects_direct_or_speculative_execution(monkeypatch, mode):
    import vllm

    args = _args()
    args.mode, args.batch, args.gen = mode, 6, 128
    monkeypatch.setattr(vllm, "LLM", lambda **kwargs: kwargs)
    config = decode.build_engine(args)
    overrides = config["hf_overrides"]
    if mode == "highlight":
        assert config["speculative_config"]["num_speculative_tokens"] == 6
        assert overrides["molle_spec_decode"] is True
        assert config["compilation_config"]["cudagraph_capture_sizes"] == [1, 6, 42]
    else:
        assert config["speculative_config"] is None
        assert config["compilation_config"]["cudagraph_capture_sizes"] == [1, 6]
        assert overrides["use_dsa"] is (mode == "direct")
    if mode == "direct":
        assert overrides["molle_spec_decode"] is False
        assert overrides["molle_topk_fraction"] == .07
        assert overrides["molle_draft_select"] == "step"


@pytest.mark.parametrize("flags", [["--draft-select", "round"], ["--verify-fraction", ".5"]])
def test_direct_rejects_speculative_settings(flags):
    args = _args(argv=flags)
    args.mode = "direct"
    with pytest.raises(ValueError, match="no verifier"):
        decode.selection_overrides(args)


def test_topp_selection_floor_is_a_fraction_of_the_prompt():
    o = decode.selection_overrides(_args(argv=["--topp", "0.85", "--draft-select", "round"]))
    assert o == {
        "molle_selection_mode": "topp", "molle_topp": 0.85, "molle_recency_window": 64,
        "molle_sink_boundary": 16, "molle_topp_kmin": 327, "molle_draft_select": "round",
        "molle_verify_fraction": 0.0,
    }
    assert decode.selection_overrides(_args(argv=["--topp", "0.85"], ctx=32700))["molle_topp_kmin"] == 327


def test_topp_cap_matches_the_topk_budget():
    o = decode.selection_overrides(_args(argv=["--topp", "0.85", "--sink", "0", "--topp-max-fraction", "0.07"]))
    assert o["molle_sink_boundary"] == 0 and o["molle_topp_kmax"] == 2284  # 142 blocks, as top-K 7%


def test_prompt_groups_wrap_and_weight_every_prompt_equally():
    import argparse

    groups = decode.prompt_groups(argparse.Namespace(samples=5, batch=3), list("abcde"))
    assert [slots for _, slots in groups] == [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 0], [4, 0, 1]]
    assert groups[3][0] == ["d", "e", "a"]
    single = decode.prompt_groups(argparse.Namespace(samples=1, batch=2), ["a", "b"])
    assert single == [(["a", "b"], [0, 1])]


def test_summary_pools_tokens_over_seconds():
    records = [
        {"decode_tokens": 100, "decode_seconds": 1.0, "kv_cache_blocks": 7, "drafts": 10, "draft_tokens": 60, "accepted": 54},
        {"decode_tokens": 300, "decode_seconds": 2.0, "kv_cache_blocks": 7, "drafts": 30, "draft_tokens": 180, "accepted": 150},
    ]
    s = decode.summarize_run(records)
    assert s["decode_tok_s"] == 400 / 3 and s["alpha"] == 204 / 240 and s["tau"] == 1 + 204 / 40


def test_source_digest_is_stable():
    assert decode.source_sha256() == decode.source_sha256() and len(decode.source_sha256()) == 64


def test_blocks_cover_prompt_outputs_and_lookahead():
    # 130,560 + 512 + 6 tokens -> 8,193 blocks each: six fit in 52,167, seven do not.
    assert decode.blocks_needed([130560] * 6, 512, 6, 16) == 49158
    assert decode.blocks_needed([130560] * 7, 512, 6, 16) == 57351
    assert decode.blocks_needed([10, 20], 6, 0, 16) == 1 + 2


def test_render_fills_fields_before_inserted_text():
    q = {"question": " Who? ", "choices": [" a ", "b", "c", "d"]}
    text = accuracy.render(accuracy.COT_PROMPT, q, document=" doc keeps $Q$ and $C_A$ ")
    assert "<text>\ndoc keeps $Q$ and $C_A$\n</text>" in text
    assert "question: Who?\nChoices:\n(A) a\n(B) b" in text
    assert text.endswith("Let’s think step by step:")
    answer = accuracy.render(accuracy.COT_ANSWER_PROMPT, q, cot="first $DOC$")
    assert "omitted here" in answer and "step by step: first $DOC$\n\nBased on the above" in answer


def test_ruler_pairs_scores_per_row():
    from benchmarks import ruler

    report = ruler.paired([1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 1.0], resamples=200)
    assert (report["n"], report["dense"], report["candidate"], report["delta_pp"]) == (4, 75.0, 75.0, 0.0)
    assert (report["lost"], report["recovered"]) == (1, 1)
    low, high = report["paired_bootstrap_95pct_delta_pp"]
    assert low <= 0.0 <= high
