# SPDX-License-Identifier: Apache-2.0
"""Freeze unseen complete-document questions, run greedy answers, compare pairs.

Use ``python -m benchmarks.longspec.accuracy {prepare,run,compare} --help``.
Preparation is CPU-only. Inference runs only through the explicit run command.
"""

import argparse
import json
import os
import random
import re
from pathlib import Path

from benchmarks.longspec import grid


def extract_answer(text):
    if "<think>" in text and "</think>" not in text:
        return None
    final = text.rsplit("</think>", 1)[-1].replace("*", "")
    for pattern in (
        r"The correct answer is \(([A-D])\)",
        r"The correct answer is ([A-D])\b",
    ):
        match = re.search(pattern, final)
        if match:
            return match.group(1)
    return None


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def prepare(argv):
    from transformers import AutoTokenizer

    p = argparse.ArgumentParser(description="Freeze unseen, untruncated questions")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--revision")
    p.add_argument("--ctx", type=int, required=True, help="maximum prompt tokens")
    p.add_argument("--min-ctx", type=int, default=0)
    p.add_argument("--samples", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--exclude",
        type=Path,
        action="append",
        default=[],
        help="additional JSON lists of used question IDs",
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    if args.out.exists() or not 0 <= args.min_ctx < args.ctx <= grid.MAX_WINDOW:
        p.error("use a new output file and valid context bounds")
    if args.samples < 1:
        p.error("samples must be positive")
    grid.resolve_revision(args)
    if not args.revision:
        p.error("a resolved model commit is required for frozen evaluation")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    exclusions = [Path(__file__).with_name("development_ids.json"), *args.exclude]
    excluded = set()
    for path in exclusions:
        excluded.update(grid.question_ids(path))
    items = grid.load_dataset()
    random.Random(args.seed).shuffle(items)
    rows = []
    for item in items:
        if item["_id"] in excluded:
            continue
        text = grid.question_text(item).replace("$DOC$", item["context"].strip())
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not args.min_ctx < len(ids) <= args.ctx:
            continue
        rows.append(
            dict(
                id=item["_id"],
                question=item["question"],
                answer=item["answer"],
                choices=[item[f"choice_{c}"] for c in "ABCD"],
                domain=item["domain"],
                difficulty=item["difficulty"],
                prompt_token_ids=ids,
                prompt_sha256=grid.digest(ids),
                prompt_tokens=len(ids),
            )
        )
        if len(rows) == args.samples:
            break
    if len(rows) != args.samples:
        p.error(f"only {len(rows)} eligible unseen questions; requested {args.samples}")
    protocol = grid.prompt_protocol(
        argparse.Namespace(**(vars(args) | {"exclude": None})), tokenizer
    )
    protocol["excluded_ids"] = sorted(excluded)
    manifest = dict(
        protocol=protocol,
        ctx=args.ctx,
        min_ctx=args.min_ctx,
        excluded_ids=sorted(excluded),
        questions=rows,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, manifest)
    print(f"Frozen {len(rows)} questions; sha256={grid.digest(manifest)}")


def run(argv):
    p = argparse.ArgumentParser(description="Generate answers; accepts grid options")
    p.add_argument("--questions", type=Path, required=True)
    own, rest = p.parse_known_args(argv)
    args = grid.parse_args(["--gen", "16384", *rest])
    if args.cells or args.batch != 1 or args.measurement != "latency":
        p.error("answer accuracy uses one engine and batch 1 with EOS stopping")
    manifest = json.loads(own.questions.read_text())
    protocol = manifest["protocol"]
    if (args.model, args.ctx, args.seed) != (
        protocol["model"],
        manifest["ctx"],
        protocol["seed"],
    ) or protocol["thinking"] is not False:
        p.error("model, context, seed or thinking differs from the frozen questions")
    if args.revision and args.revision != protocol["revision"]:
        p.error("revision differs from the frozen questions")
    args.revision = protocol["revision"]
    rows = manifest["questions"]
    if not rows or len({q["id"] for q in rows}) != len(rows):
        p.error("empty or duplicate questions")
    for q in rows:
        if (
            q["prompt_sha256"] != grid.digest(q["prompt_token_ids"])
            or len(q["prompt_token_ids"]) > args.ctx
        ):
            p.error("question tokens differ from the frozen manifest")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "config.json").exists() or (out / "results.jsonl").exists():
        p.error("use a fresh output directory")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    llm, factor = grid.build_engine(args)
    engine = llm.llm_engine.vllm_config
    config = dict(
        manifest_sha256=grid.digest(manifest),
        questions=[q["id"] for q in rows],
        model=args.model,
        revision=args.revision,
        ctx=args.ctx,
        gen=args.gen,
        thinking=False,
        temperature=0,
        ignore_eos=False,
        generation_config=engine.model_config.generation_config,
        eos_token_ids=engine.model_config.try_get_generation_config().get(
            "eos_token_id"
        ),
        dtype=str(engine.model_config.dtype),
        kv_cache_dtype=engine.cache_config.cache_dtype,
        flash_attn_version=engine.attention_config.flash_attn_version,
        seed=args.seed,
        yarn_factor=factor,
        yarn_original_window=grid.NATIVE_WINDOW,
        mode=args.mode,
        speculative_config=grid.speculative_config(args),
        provenance=grid.provenance(),
    )
    write_json(out / "config.json", config)
    results = []
    with (out / "results.jsonl").open("x") as stream:
        for q in rows:
            _, outputs = grid.generate(llm, [q["prompt_token_ids"]], args.gen)
            result = outputs[0]
            if result.prompt_token_ids != q["prompt_token_ids"]:
                raise RuntimeError("engine changed frozen input tokens")
            answer = result.outputs[0]
            prediction = extract_answer(answer.text)
            complete = answer.finish_reason == "stop"
            row = {k: v for k, v in q.items() if k != "prompt_token_ids"}
            row.update(
                prediction=prediction,
                correct=complete and prediction == q["answer"],
                complete=complete,
                parsed=prediction is not None,
                output_text=answer.text,
                output_token_ids=list(answer.token_ids),
                output_tokens=len(answer.token_ids),
                finish_reason=answer.finish_reason,
                stop_reason=answer.stop_reason,
            )
            results.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"{len(results)}/{len(rows)} answers saved", flush=True)
    write_json(
        out / "finished.json",
        dict(
            questions=len(results),
            correct=sum(r["correct"] for r in results),
            capped=sum(not r["complete"] for r in results),
            unparsed=sum(not r["parsed"] for r in results),
            graph_replays=llm.collective_rpc(grid._round_graph_stats)[0],
        ),
    )


def load_completed(directory):
    config = json.loads((directory / "config.json").read_text())
    finished = json.loads((directory / "finished.json").read_text())
    rows = [
        json.loads(s) for s in (directory / "results.jsonl").read_text().splitlines()
    ]
    expected = config["questions"]
    if (
        len(rows) != len(expected)
        or finished["questions"] != len(expected)
        or [r["id"] for r in rows] != expected
        or len(set(expected)) != len(expected)
    ):
        raise ValueError("missing, duplicate or reordered answers")
    for r in rows:
        if r["complete"] != (r["finish_reason"] == "stop"):
            raise ValueError("stored completion differs from termination reason")
        if r["parsed"] != (extract_answer(r["output_text"]) is not None):
            raise ValueError("stored parse status differs from saved answer")
        correct = r["complete"] and extract_answer(r["output_text"]) == r["answer"]
        if r["correct"] != correct:
            raise ValueError("stored correctness differs from rescored answer")
    return config, rows


def paired_stats(dense, candidate):
    if not dense or len(dense) != len(candidate):
        raise ValueError("empty or unequal question sets")
    for a, b in zip(dense, candidate):
        if any(a[k] != b[k] for k in ("id", "answer", "prompt_sha256")):
            raise ValueError("paired questions differ")
    d, s = sum(r["correct"] for r in dense), sum(r["correct"] for r in candidate)
    delta = [int(b["correct"]) - int(a["correct"]) for a, b in zip(dense, candidate)]
    rng = random.Random(42)
    boot = sorted(
        100 * sum(rng.choices(delta, k=len(delta))) / len(delta) for _ in range(10000)
    )
    return dict(
        n=len(dense),
        dense_correct=d,
        candidate_correct=s,
        delta_pp=100 * (s - d) / len(dense),
        lost=sum(a["correct"] and not b["correct"] for a, b in zip(dense, candidate)),
        recovered=sum(
            b["correct"] and not a["correct"] for a, b in zip(dense, candidate)
        ),
        dense_capped=sum(not r["complete"] for r in dense),
        candidate_capped=sum(not r["complete"] for r in candidate),
        dense_unparsed=sum(not r["parsed"] for r in dense),
        candidate_unparsed=sum(not r["parsed"] for r in candidate),
        paired_bootstrap_95pct_delta_pp=[boot[249], boot[9749]],
    )


def compare(argv):
    p = argparse.ArgumentParser(description="Compare completed, matched answer runs")
    p.add_argument("--dense", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    a, dense = load_completed(args.dense)
    b, candidate = load_completed(args.candidate)
    if a["mode"] != "dense":
        p.error("reference must be dense")
    keys = (
        "manifest_sha256",
        "model",
        "revision",
        "ctx",
        "gen",
        "thinking",
        "temperature",
        "ignore_eos",
        "generation_config",
        "eos_token_ids",
        "dtype",
        "kv_cache_dtype",
        "flash_attn_version",
        "seed",
        "yarn_factor",
        "yarn_original_window",
    )
    if any(a[k] != b[k] for k in keys):
        p.error("generation protocols differ")
    report = dict(overall=paired_stats(dense, candidate))
    for field in ("domain", "difficulty"):
        report[field] = {
            value: paired_stats(
                [r for r in dense if r[field] == value],
                [r for r in candidate if r[field] == value],
            )
            for value in sorted({r[field] for r in dense})
        }
    write_json(args.out, report)
    print(json.dumps(report["overall"]))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2 or sys.argv[1] not in ("prepare", "run", "compare"):
        raise SystemExit("expected prepare, run, or compare")
    globals()[sys.argv[1]](sys.argv[2:])
