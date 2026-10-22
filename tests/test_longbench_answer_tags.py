"""LongBench prompt parity, cohort preservation and answer-tag scoring."""
import json
from pathlib import Path

import pytest

from benchmarks import accuracy, main_quality
from benchmarks.main_results import save
from spotattention.data._prompts import render_prompt


@pytest.mark.parametrize("text,prediction,correct", [
    ("B</answer>", "B", True),
    ("B</answer> trailing text without EOS", "B", True),
    ("C</answer>", "C", False),
    ("B", None, False),
    ("</answer>", None, False),
    ("A</answer><answer>B</answer>", "B", True),
])
def test_scoring_uses_complete_tags_and_not_eos(text, prediction, correct):
    q = dict(question="Which?", choices=["a", "b", "c", "d"], answer="B")
    assert accuracy.score_answer_tags(text, q) == (prediction, correct)


def test_preparation_preserves_cohort_and_uses_molle_prompt(tmp_path, monkeypatch):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
            return json.dumps(messages) + "ASSISTANT:"
        def encode(self, text, **kwargs):
            assert kwargs == dict(add_special_tokens=False)
            return list(text.encode())
    tokenizer = Tokenizer()
    rows = [dict(id=str(i), question=f"Q{i}", choices=["a", "b", "c", "d"], answer="B",
                 prompt_token_ids=[i], prompt_tokens=1, prompt_sha256=accuracy.decode.digest([i])) for i in range(2)]
    manifest = dict(ctx=114688, protocol=dict(thinking=False), questions=rows)
    source = tmp_path/"original.json"
    save(source, manifest)
    documents = {q["id"]: "Full original document " + q["id"] for q in rows}
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda _: tokenizer)
    monkeypatch.setattr(accuracy, "load_documents", lambda *args: documents)
    out = tmp_path/"answer-tags.json"
    new = accuracy.prepare_answer_tags(source, "checkpoint", out)
    assert json.loads(source.read_text()) == manifest
    assert new["source_manifest_sha256"] == accuracy.decode.digest(manifest)
    assert new["protocol"]["generation_tokens"] == 8192
    for old, row in zip(rows, new["questions"], strict=True):
        assert {k: row[k] for k in ("id", "question", "choices", "answer")} == {
            k: old[k] for k in ("id", "question", "choices", "answer")}
        prompt = render_prompt(accuracy.answer_record(old, documents[old["id"]]), tokenizer)
        assert prompt.endswith("<answer>") and documents[old["id"]] in prompt
        assert row["prompt_token_ids"] == tokenizer.encode(prompt, add_special_tokens=False)
        assert row["prompt_sha256"] == accuracy.decode.digest(row["prompt_token_ids"])
    accuracy.load_questions(out)


def test_new_quality_scope_and_worker_preserve_ruler(tmp_path, monkeypatch):
    methods = ["dense8", "highlight4-sv25"]
    jobs = main_quality.make_jobs(Path("/runtime"), tmp_path, "4b", methods, "answer-tags")
    assert len(jobs) == 6
    assert all("/answer-tags/" in j["name"] for j in jobs if j["stage"] == "longbench")
    plan = dict(schema="main-quality-v2", host=main_quality.socket.gethostname(), model="4b",
                runtime_root="/runtime", allowed_gpus=[0,1,2,3],
                gpu_authorization=dict(allowed_gpus=[0,1,2,3], user_instruction="Use all four"),
                protocol=dict(methods=methods, longbench_format="answer-tags", longbench_tokens=8192,
                              attention_backend="fa4"), jobs=jobs,
                inputs=dict(checkpoint="/model", questions="/questions", ruler={"65536":"/ruler"}))
    save(tmp_path/"plan.json", plan)
    monkeypatch.setattr(main_quality, "_verify_payload", lambda _: None)
    main_quality.verify_plan(tmp_path)
    calls = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(accuracy, "main", lambda argv: calls.append(argv))
    monkeypatch.setattr(main_quality, "validate", lambda *args: {})
    main_quality.run_job(tmp_path, next(j["name"] for j in jobs if j["stage"] == "longbench"))
    assert "--answer-tags" in calls[0] and "--cot" not in calls[0]
    assert calls[0][calls[0].index("--gen")+1] == "8192"
    plan["protocol"]["longbench_tokens"] = 1024
    save(tmp_path/"plan.json", plan)
    with pytest.raises(ValueError, match="answer-tag protocol"):
        main_quality.verify_plan(tmp_path)
