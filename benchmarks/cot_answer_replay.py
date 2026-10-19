# SPDX-License-Identifier: Apache-2.0
"""Dense final-answer replay from saved CoT reasoning; no document regeneration."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket

from selector_diagnosis import wait_idle


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def run(args):
    from transformers import AutoTokenizer
    from vllm.inputs import TokensPrompt
    from benchmarks import decode
    from benchmarks.accuracy import chat_ids, render, extract_answer, COT_ANSWER_PROMPT
    from benchmarks.selector_campaign import source_hashes, atomic_json

    source_config = args.source.parent / "config.json"
    original = json.loads(source_config.read_text())
    rows, controls = read_rows(args.source), read_rows(args.control)
    if (len(rows) != 60 or [r["id"] for r in rows] != original["questions"]
            or [r["id"] for r in controls] != original["questions"]
            or original["mode"] != "direct" or original["indexer_cache_format"] != "mxfp4"
            or original["cot"]["answer_prompt"] != COT_ANSWER_PROMPT):
        raise RuntimeError("Unexpected source protocol or row order")
    tokenizer = AutoTokenizer.from_pretrained(original["model"], local_files_only=True)
    prompts = [chat_ids(tokenizer, render(COT_ANSWER_PROMPT, row, cot=row["reasoning_text"].strip())) for row in rows]
    parser = argparse.ArgumentParser()
    decode.add_engine_args(parser)
    runtime = parser.parse_args(["--checkpoint", original["model"], "--mode", "dense",
                                "--seed", str(original["seed"]), "--flash-attn-version", str(original["flash_attn_version"])])
    runtime.ctx, runtime.gen, runtime.batch = original["ctx"], original["gen"], 1
    params = decode.sampling_params(original["cot"]["answer_tokens"], ignore_eos=False)
    for key, expected in original["sampling"].items():
        if getattr(params, key) != expected:
            raise RuntimeError(f"Sampling differs: {key}")
    own_files = [Path(__file__).resolve(), Path(__file__).with_name("selector_diagnosis.py").resolve()]
    sources = {**source_hashes(), **{str(p): sha(p) for p in own_files}}
    inputs = {str(p): sha(p) for p in (args.source, args.control, source_config)}
    atomic_json(args.out / "prompts.json", {"rows": [
        {"id": row["id"], "prompt_token_ids": ids, "prompt_sha256": decode.digest(ids)}
        for row, ids in zip(rows, prompts)]})
    atomic_json(args.out / "config.json", {"source_sha256": sources, "input_sha256": inputs,
        "original": original, "change": "dense final-answer attention only; reuse saved MXFP4 Direct Highlight reasoning",
        "physical_gpu": args.gpu, "host": socket.gethostname(), "batch": 1,
        "answer_cap": params.max_tokens, "thinking": False,
        "long_context_generation": False, "quality_only": True})
    llm = decode.build_engine(runtime, scheduler_cls=None)
    engine = llm.llm_engine.vllm_config
    if (engine.model_config.hf_config.use_dsa
            or engine.model_config.hf_config.rope_parameters != original["rope_parameters"]):
        raise RuntimeError("Dense mode or YaRN differs")

    def generate(ids):
        [output] = llm.generate([TokensPrompt(prompt_token_ids=ids)], params, use_tqdm=False)
        if list(output.prompt_token_ids) != ids:
            raise RuntimeError("Engine changed prompt tokens")
        return output.outputs[0]

    checks = []
    for row in controls[:3]:
        ids = chat_ids(tokenizer, render(COT_ANSWER_PROMPT, row, cot=row["reasoning_text"].strip()))
        actual = generate(ids)
        checks.append({"id": row["id"], "matches": list(actual.token_ids) == row["output_token_ids"]
                       and actual.finish_reason == row["finish_reason"],
                       "reference": row["output_token_ids"], "actual": list(actual.token_ids)})
    atomic_json(args.out / "validation.json", {"checks": checks, "passed": all(r["matches"] for r in checks)})
    if not all(r["matches"] for r in checks):
        raise RuntimeError("Replay differs from the saved dense controls")
    results = []
    with (args.out / "results.jsonl").open("x") as stream:
        for row, ids in zip(rows, prompts):
            answer = generate(ids)
            prediction = extract_answer(answer.text)
            result = {"id": row["id"], "prompt_sha256": decode.digest(ids), "prompt_tokens": len(ids),
                      "answer": row["answer"], "prediction": prediction, "parsed": prediction is not None,
                      "complete": answer.finish_reason == "stop",
                      "correct": answer.finish_reason == "stop" and prediction == row["answer"],
                      "output_text": answer.text, "output_token_ids": list(answer.token_ids),
                      "finish_reason": answer.finish_reason, "stop_reason": answer.stop_reason,
                      "previous_correct": row["correct"], "previous_parsed": row["parsed"]}
            results.append(result)
            stream.write(json.dumps(result) + "\n")
            stream.flush()
            print(f"dense final answer {len(results)}/60", flush=True)
    if sources != {**source_hashes(), **{str(p): sha(p) for p in own_files}} or inputs != {
            str(p): sha(p) for p in (args.source, args.control, source_config)}:
        raise RuntimeError("Sources or inputs changed during replay")
    atomic_json(args.out / "finished.json", {"state": "complete", "rows": len(results),
        "correct": sum(r["correct"] for r in results), "parsed": sum(r["parsed"] for r in results),
        "complete": sum(r["complete"] for r in results),
        "previous_correct": sum(r["previous_correct"] for r in results),
        "previous_parsed": sum(r["previous_parsed"] for r in results),
        "recovered": sum(r["correct"] and not r["previous_correct"] for r in results),
        "lost": sum(not r["correct"] and r["previous_correct"] for r in results)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    parser.add_argument("--lock", required=True)
    args = parser.parse_args()
    if socket.gethostname() != "DCTN-1008231220" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        parser.error("Only the assigned b200-1 GPU is allowed")
    args.out.mkdir(parents=True, exist_ok=False)
    with Path(args.lock).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        wait_idle(args.gpu)
        run(args)


if __name__ == "__main__":
    main()
