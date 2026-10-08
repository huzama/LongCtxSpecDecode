# SPDX-License-Identifier: Apache-2.0
"""Answer parsing and paired accounting, including losses hidden by recoveries."""

import json
from types import SimpleNamespace as NS

import pytest

from benchmarks.longspec import accuracy, grid
from benchmarks.longspec.accuracy import extract_answer, paired_stats


@pytest.mark.parametrize(
    "text,expected",
    [
        ("The correct answer is (B)", "B"),
        ("**The correct answer is C**", "C"),
        ("<think>The correct answer is (A)", None),
        ("<think>The correct answer is (A)</think>The correct answer is (D)", "D"),
        ("The correct answer is Banana", None),
        ("I choose A", None),
    ],
)
def test_final_answer_parser(text, expected):
    assert extract_answer(text) == expected


def test_equal_accuracy_retains_separate_losses_and_recoveries():
    dense = [
        dict(
            id=str(i),
            answer="A",
            prompt_sha256=str(i),
            correct=i == 0,
            complete=True,
            parsed=True,
        )
        for i in range(3)
    ]
    sparse = [
        dict(r, correct=i == 1, complete=i != 2, parsed=i != 2)
        for i, r in enumerate(dense)
    ]
    result = paired_stats(dense, sparse)
    assert result["delta_pp"] == 0
    assert result["lost"] == result["recovered"] == 1
    assert result["n"] == 3 and result["candidate_capped"] == 1
    sparse[0]["prompt_sha256"] = "different"
    with pytest.raises(ValueError, match="differ"):
        paired_stats(dense, sparse)


def test_prepare_freezes_untruncated_nonthinking_prompts_and_exclusions(
    tmp_path, monkeypatch
):
    import transformers

    development = json.loads(
        accuracy.Path(accuracy.__file__).with_name("development_ids.json").read_text()
    )
    template_calls = []

    def tokenize(messages, **kwargs):
        template_calls.append((messages, kwargs))
        return [ord(c) for c in messages[0]["content"]]

    tokenizer = NS(
        chat_template="test", get_vocab=lambda: {}, apply_chat_template=tokenize
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer
    )
    monkeypatch.setattr(
        grid, "resolve_revision", lambda args: setattr(args, "revision", "commit")
    )
    document = "complete document " * 20
    item = dict(
        context=document,
        question="Which?",
        answer="A",
        domain="test",
        difficulty="easy",
    )
    item.update({f"choice_{c}": c for c in "ABCD"})
    monkeypatch.setattr(
        grid,
        "load_dataset",
        lambda: [dict(item, _id=i) for i in (development[0], "extra", "fresh")],
    )
    exclusions = tmp_path / "exclude.json"
    exclusions.write_text('["extra"]')
    out = tmp_path / "questions.json"
    argv = [
        "--ctx",
        "4096",
        "--samples",
        "1",
        "--exclude",
        str(exclusions),
        "--out",
        str(out),
    ]
    accuracy.prepare(argv)
    manifest = json.loads(out.read_text())
    assert [q["id"] for q in manifest["questions"]] == ["fresh"]
    assert manifest["protocol"]["excluded_ids"] == sorted([*development, "extra"])
    assert manifest["protocol"]["revision"] == "commit"
    text = "".join(chr(t) for t in manifest["questions"][0]["prompt_token_ids"])
    assert document.strip() in text
    assert all(
        k["enable_thinking"] is False and k["tokenize"] for _, k in template_calls
    )
    out.unlink()
    argv[argv.index("--samples") + 1] = "2"
    with pytest.raises(SystemExit):
        accuracy.prepare(argv)
    assert not out.exists()


def test_completed_answers_rescore_caps_and_reject_tampered_status(tmp_path):
    accuracy.write_json(tmp_path / "config.json", {"questions": ["a"]})
    accuracy.write_json(tmp_path / "finished.json", {"questions": 1})
    row = dict(
        id="a",
        answer="A",
        output_text="The correct answer is (A)",
        complete=False,
        finish_reason="length",
        parsed=True,
        correct=False,
    )
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps(row) + "\n")
    _, rows = accuracy.load_completed(tmp_path)
    assert rows[0]["correct"] is False
    row["complete"] = True
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="termination"):
        accuracy.load_completed(tmp_path)


def test_run_uses_frozen_tokens_eos_and_accuracy_budget(tmp_path, monkeypatch):
    tokens = [10, 20, 30]
    manifest = dict(
        protocol=dict(
            model="Qwen/Qwen3-8B", revision="frozen", seed=42, thinking=False
        ),
        ctx=4096,
        questions=[
            dict(
                id="a",
                answer="B",
                prompt_token_ids=tokens,
                prompt_sha256=grid.digest(tokens),
                prompt_tokens=3,
            )
        ],
    )
    questions = tmp_path / "questions.json"
    accuracy.write_json(questions, manifest)
    config = NS(
        model_config=NS(
            dtype="bfloat16",
            generation_config="vllm",
            try_get_generation_config=lambda: {"eos_token_id": [151645, 151643]},
        ),
        cache_config=NS(cache_dtype="auto"),
        attention_config=NS(flash_attn_version=4),
    )
    llm = NS(llm_engine=NS(vllm_config=config), collective_rpc=lambda fn: [None])

    def engine(args):
        assert args.revision == "frozen" and args.gen == 16384
        return llm, None

    def generate(actual_llm, prompts, max_tokens):
        assert actual_llm is llm and prompts == [tokens] and max_tokens == 16384
        return 0.1, [
            NS(
                prompt_token_ids=tokens,
                outputs=[
                    NS(
                        text="The correct answer is (B)",
                        finish_reason="stop",
                        stop_reason=151643,
                        token_ids=[40, 151643],
                    )
                ],
            )
        ]

    monkeypatch.setattr(grid, "build_engine", engine)
    monkeypatch.setattr(grid, "generate", generate)
    monkeypatch.setattr(grid, "provenance", lambda: {})
    out = tmp_path / "run"
    accuracy.run(
        [
            "--questions",
            str(questions),
            "--out",
            str(out),
            "--model",
            "Qwen/Qwen3-8B",
            "--ctx",
            "4096",
            "--mode",
            "dense",
        ]
    )
    saved, rows = accuracy.load_completed(out)
    assert saved["eos_token_ids"] == [151645, 151643]
    assert saved["gen"] == 16384 and saved["ignore_eos"] is False
    assert rows[0]["correct"] and rows[0]["output_token_ids"] == [40, 151643]
