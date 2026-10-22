"""Answer accuracy on Spot's runtime: dense or Highlight, one frozen question at a time.

Branch thesis freezes the questions (``python -m benchmarks.longspec.accuracy
prepare``) and compares two completed runs (``accuracy compare``); this module
runs them on the same checkpoint and writes the files that ``compare`` reads.
Batch 1, greedy, thinking off (the frozen chat template), EOS stopping.

``--cot`` runs LongBench v2's official chain of thought instead of the frozen
prompts: reason over the document for up to ``--gen`` tokens (officially
1024), then answer from the question and that reasoning alone in up to 128.
The documents come from the frozen dataset revision and must reproduce every
frozen prompt token for token.

  .venv/bin/python -m benchmarks.accuracy --checkpoint <spot ckpt> \
      --mode highlight --questions <questions.json> --out <dir>
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import sys
from pathlib import Path

from benchmarks import decode

# THUDM/LongBench prompts/0shot_cot.txt and prompts/0shot_cot_ans.txt, verbatim.
COT_PROMPT = (
    "Please read the following text and answer the questions below.\n\n<text>\n$DOC$\n</text>\n\n"
    "What is the correct answer to this question: $Q$\nChoices:\n(A) $C_A$\n(B) $C_B$\n(C) $C_C$\n(D) $C_D$\n\n"
    "Let\u2019s think step by step:"
)
COT_ANSWER_PROMPT = (
    "Please read the following text and answer the questions below.\n\nThe text is too long and omitted here.\n\n"
    "What is the correct answer to this question: $Q$\nChoices:\n(A) $C_A$\n(B) $C_B$\n(C) $C_C$\n(D) $C_D$\n\n"
    "Let\u2019s think step by step: $COT$\n\n"
    "Based on the above, what is the single, most likely answer choice? "
    'Format your response as follows: "The correct answer is (insert answer here)".'
)
COT_ANSWER_TOKENS = 128  # the official answer limit; the reasoning limit is --gen
ANSWER_TAG_TOKENS = 8192
ANSWER_TAG_FORMAT = "molle-answer-tags"


def answer_record(q: dict, document: str = "") -> dict:
    """Use the same multiple-choice prompt and extraction contract as MoLLE."""
    return dict(context=document, question=q["question"], choices=q["choices"],
                prompt_kind="mc", enable_thinking=False, prefill_answer_tag=True,
                strict_tag=True, scorer_kind="mc_letter", answer_letter=q["answer"])


def score_answer_tags(text: str, q: dict) -> tuple[str | None, bool]:
    from spotattention.data._scorers import extract_answer, score_row

    record = answer_record(q)
    extracted, tagged = extract_answer(text, record)
    match = re.search(r"\b([A-D])\b", extracted) if tagged else None
    return (match.group(1) if match else None), bool(score_row(extracted, tagged, record))


def prepare_answer_tags(questions: Path, checkpoint: str, out: Path) -> dict:
    """Freeze new prompts for exactly the existing question IDs and gold answers."""
    from transformers import AutoTokenizer
    from spotattention.data._prompts import MC_TEMPLATE, SYSTEM_PROMPT, render_prompt

    if out.exists():
        raise ValueError("use a new question manifest")
    manifest, rows = load_questions(questions)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    documents = load_documents(manifest, rows, tokenizer)
    prepared = []
    for q in rows:
        prompt = render_prompt(answer_record(q, documents[q["id"]]), tokenizer, enable_thinking=False)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > manifest["ctx"]:
            raise ValueError(f"question {q['id']}: answer-tag prompt exceeds context budget")
        prepared.append({**q, "prompt_token_ids": ids, "prompt_sha256": decode.digest(ids),
                         "prompt_tokens": len(ids)})
    if manifest["ctx"] + ANSWER_TAG_TOKENS > 131072:
        raise ValueError("question and output budgets exceed the agreed context limit")
    protocol = {**manifest["protocol"], "prompt_format": ANSWER_TAG_FORMAT,
                "template": MC_TEMPLATE, "system_prompt": SYSTEM_PROMPT,
                "prefill_answer_tag": True, "strict_tag": True,
                "generation_tokens": ANSWER_TAG_TOKENS,
                "scoring": "MoLLE mc_letter; complete answer tags required; EOS not required for a valid answer"}
    result = {**manifest, "protocol": protocol, "questions": prepared,
              "source_manifest": str(questions), "source_manifest_sha256": decode.digest(manifest)}
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, result)
    return result


def render(template: str, q: dict, *, document: str | None = None, cot: str | None = None) -> str:
    """Fill a LongBench v2 template. Fields go in first and the document last, so inserted text is never rewritten."""
    text = template.replace("$Q$", q["question"].strip())
    for letter, choice in zip("ABCD", q["choices"]):
        text = text.replace(f"$C_{letter}$", choice.strip())
    if cot is not None:
        text = text.replace("$COT$", cot)
    if document is not None:
        text = text.replace("$DOC$", document.strip())
    return text


def chat_ids(tokenizer, text: str) -> list[int]:
    """One user turn with thinking off, as branch thesis froze the prompts."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return tokenizer.encode(rendered, add_special_tokens=False)


def load_documents(manifest: dict, rows: list[dict], tokenizer) -> dict[str, str]:
    """Documents of the frozen questions; each must rebuild its frozen prompt exactly."""
    from huggingface_hub import hf_hub_download

    protocol = manifest["protocol"]
    if protocol["chat_template"] != tokenizer.chat_template or protocol["tokenizer_sha256"] != decode.digest(
        tokenizer.get_vocab()
    ):
        raise ValueError("the checkpoint tokenizer differs from the frozen one")
    path = hf_hub_download(protocol["dataset"], "data.json", repo_type="dataset", revision=protocol["dataset_revision"],
                           cache_dir=os.environ.get("HF_HUB_CACHE", decode.ROOT / "outputs/hf/hub"))
    items = {item["_id"]: item for item in json.loads(Path(path).read_text())}
    documents = {}
    for q in rows:
        document = items[q["id"]]["context"]
        if decode.digest(chat_ids(tokenizer, render(protocol["template"], q, document=document))) != q["prompt_sha256"]:
            raise ValueError(f"question {q['id']}: the dataset no longer rebuilds its frozen prompt")
        documents[q["id"]] = document
    return documents


def extract_answer(text: str) -> str | None:
    """The final A-D choice; branch thesis's rule, so ``compare`` rescoring agrees."""
    if "<think>" in text and "</think>" not in text:
        return None
    final = text.rsplit("</think>", 1)[-1].replace("*", "")
    for pattern in (r"The correct answer is \(([A-D])\)", r"The correct answer is ([A-D])\b"):
        match = re.search(pattern, final)
        if match:
            return match.group(1)
    return None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    decode.add_engine_args(p)
    p.add_argument("--questions", required=True, help="manifest from thesis's accuracy prepare")
    p.add_argument("--gen", type=int, default=128, help="output token limit; with --cot, the reasoning limit (official 1024)")
    format_args = p.add_mutually_exclusive_group()
    format_args.add_argument("--cot", action="store_true", help="LongBench v2 official chain of thought")
    format_args.add_argument("--answer-tags", action="store_true", help="single-stage MoLLE answer-tag prompts")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    args.batch = 1
    return args


def load_questions(path: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads(path.read_text())
    rows = manifest["questions"]
    if manifest["protocol"]["thinking"] is not False:
        raise ValueError("frozen questions must have thinking off")
    if not rows or len({q["id"] for q in rows}) != len(rows):
        raise ValueError("empty or duplicate questions")
    for q in rows:
        if q["prompt_sha256"] != decode.digest(q["prompt_token_ids"]) or q["prompt_tokens"] > manifest["ctx"]:
            raise ValueError(f"question {q['id']} differs from the frozen manifest")
    return manifest, rows


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main(argv=None) -> None:
    import spotattention
    from vllm.inputs import TokensPrompt

    args = parse_args(argv)
    manifest, rows = load_questions(Path(args.questions))
    if args.answer_tags:
        if (manifest["protocol"].get("prompt_format") != ANSWER_TAG_FORMAT
                or manifest["protocol"].get("generation_tokens") != ANSWER_TAG_TOKENS
                or args.gen != ANSWER_TAG_TOKENS):
            raise ValueError("answer-tag evaluation requires its frozen prompts and 8192-token cap")
    elif manifest["protocol"].get("prompt_format") == ANSWER_TAG_FORMAT:
        raise ValueError("answer-tag prompts require --answer-tags")
    args.ctx = manifest["ctx"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "config.json").exists() or (out / "results.jsonl").exists():
        raise ValueError("use a fresh output directory")
    tokenizer = documents = None
    if args.cot:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
        documents = load_documents(manifest, rows, tokenizer)
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(decode.ROOT), os.environ.get("PYTHONPATH")]))
    llm = decode.build_engine(args, scheduler_cls=None)
    engine = llm.llm_engine.vllm_config
    eos_ids = decode.configured_eos_ids(args.checkpoint) if args.hierarchical_method or args.native_fp8_highlight else None
    params = decode.sampling_params(args.gen, ignore_eos=False, eos_token_ids=eos_ids)
    answer_params = decode.sampling_params(COT_ANSWER_TOKENS, ignore_eos=False, eos_token_ids=eos_ids)

    def generate(ids, sampling):
        [result] = llm.generate([TokensPrompt(prompt_token_ids=ids)], sampling, use_tqdm=False)
        if list(result.prompt_token_ids) != ids:
            raise RuntimeError("engine changed the input tokens")
        if args.hierarchical_method or args.native_fp8_highlight:
            from benchmarks.multilexsum import output_audit
            output_audit(list(result.outputs[0].token_ids), eos_ids, sampling.max_tokens,
                         result.outputs[0].finish_reason)
            if args.hierarchical_method:
                llm.collective_rpc("hierarchical_cache_info")
        return result.outputs[0]
    highlight = args.mode == "highlight"
    # The keys thesis's compare requires equal between the dense and candidate runs.
    config = {
        "manifest_sha256": decode.digest(manifest),
        "questions": [q["id"] for q in rows],
        "model": args.checkpoint,
        "revision": manifest["protocol"]["revision"],
        "ctx": args.ctx,
        "gen": args.gen,
        "thinking": False,
        "batch": args.batch,
        "async_scheduling": engine.scheduler_config.async_scheduling,
        "prefix_caching": engine.cache_config.enable_prefix_caching,
        "temperature": params.temperature,
        "sampling": {k: getattr(params, k) for k in (
            "temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty",
            "repetition_penalty", "min_tokens", "skip_special_tokens")},
        "ignore_eos": False,
        "generation_config": engine.model_config.generation_config,
        "eos_token_ids": engine.model_config.try_get_generation_config().get("eos_token_id"),
        "dtype": str(engine.model_config.dtype),
        "kv_cache_dtype": engine.cache_config.cache_dtype,
        "indexer_cache_format": getattr(engine.model_config.hf_config, "molle_kcache_format", "bf16"),
        "flash_attn_version": engine.attention_config.flash_attn_version,
        "seed": args.seed,
        "yarn_factor": decode.yarn_factor(args.ctx + args.gen),
        "yarn_original_window": decode.NATIVE_WINDOW,
        "rope_parameters": getattr(engine.model_config.hf_config, "rope_parameters", None),
        "mode": args.mode,
        "hierarchical_method": args.hierarchical_method,
        "backbone_cache": ("hierarchical_4plus4" if args.hierarchical_method else
                           "native_fp8_e4m3" if args.native_fp8_highlight else "bf16"),
        "calculate_kv_scales": engine.cache_config.calculate_kv_scales,
        "eos_audit_ids": eos_ids,
        "spec_tokens": args.spec_tokens if highlight else 0,
        "selection": decode.selection_overrides(args) if args.mode != "dense" else None,
        "cot": {
            "source": "THUDM/LongBench prompts/0shot_cot.txt, prompts/0shot_cot_ans.txt",
            "prompt": COT_PROMPT,
            "answer_prompt": COT_ANSWER_PROMPT,
            "reasoning_tokens": args.gen,
            "answer_tokens": COT_ANSWER_TOKENS,
        } if args.cot else None,
        "answer_format": ANSWER_TAG_FORMAT if args.answer_tags else None,
        "provenance": {
            "versions": {n: importlib.metadata.version(n) for n in ("torch", "vllm", "triton")},
            "spotattention_commit": decode.git_sha(Path(spotattention.__file__).parent),
            "git_sha": decode.git_sha(decode.ROOT),
            "source_sha256": decode.source_sha256(),
        },
    }
    write_json(out / "config.json", config)
    results = []
    totals = {"drafts": 0, "draft_tokens": 0, "accepted": 0}
    with (out / "results.jsonl").open("x") as stream:
        for q in rows:
            before = decode.spec_counters(llm, args.spec_tokens) if highlight else None
            row = {k: v for k, v in q.items() if k != "prompt_token_ids"}
            if args.cot:
                prompt = chat_ids(tokenizer, render(COT_PROMPT, q, document=documents[q["id"]]))
                if len(prompt) > args.ctx:
                    raise ValueError(f"question {q['id']}: reasoning prompt exceeds --ctx")
                reasoning = generate(prompt, params)
                answer = generate(chat_ids(tokenizer, render(COT_ANSWER_PROMPT, q, cot=reasoning.text.strip())),
                                  answer_params)
                row.update(
                    reasoning_prompt_tokens=len(prompt),
                    reasoning_text=reasoning.text,
                    reasoning_token_ids=list(reasoning.token_ids),
                    reasoning_tokens=len(reasoning.token_ids),
                    reasoning_finish_reason=reasoning.finish_reason,
                )
            else:
                answer = generate(q["prompt_token_ids"], params)
            prediction, correct = (score_answer_tags(answer.text, q) if args.answer_tags
                                   else (extract_answer(answer.text), None))
            complete = answer.finish_reason == "stop"
            row.update(
                prediction=prediction,
                correct=correct if args.answer_tags else complete and prediction == q["answer"],
                complete=complete,
                parsed=prediction is not None,
                output_text=answer.text,
                output_token_ids=list(answer.token_ids),
                output_tokens=len(answer.token_ids),
                finish_reason=answer.finish_reason,
                stop_reason=answer.stop_reason,
            )
            if highlight:
                spec = decode.spec_record(before, decode.spec_counters(llm, args.spec_tokens))
                row["spec"] = {k: spec[k] for k in ("drafts", "draft_tokens", "accepted", "alpha")}
                for k in totals:
                    totals[k] += spec[k]
            results.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"{len(results)}/{len(rows)} answers saved", flush=True)
    finished = {
        "questions": len(results),
        "correct": sum(r["correct"] for r in results),
        "capped": sum(not r["complete"] for r in results),
        "unparsed": sum(not r["parsed"] for r in results),
    }
    if args.cot:
        finished["reasoning_capped"] = sum(r["reasoning_finish_reason"] != "stop" for r in results)
    if highlight:
        finished["spec"] = {**totals, "alpha": totals["accepted"] / totals["draft_tokens"] if totals["draft_tokens"] else None}
    write_json(out / "finished.json", finished)
    if args.hierarchical_method or args.native_fp8_highlight:
        rpc = "hierarchical_cache_info" if args.hierarchical_method else "native_fp8_info"
        write_json(out / "cache.json", llm.collective_rpc(rpc)[0])
    print(json.dumps(finished), flush=True)


if __name__ == "__main__":
    sys.exit(main())
