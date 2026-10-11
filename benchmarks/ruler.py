"""RULER retrieval: dense, direct Highlight or speculation on frozen prompts.

Spot's evaluation (``spotattention.data.eval.ruler``) builds the six NIAH
tasks and scores them. ``prepare`` freezes its prompts for one length and
tokenizer, so every compared run reads identical tokens; ``run`` answers them
at batch 1 (greedy, thinking off, EOS stopping) with Spot's scorer;
``compare`` pairs a candidate run with the dense run.

  .venv/bin/python -m benchmarks.ruler prepare --checkpoint <ckpt> --length 65536 --out <rows.json>
  .venv/bin/python -m benchmarks.ruler run --checkpoint <ckpt> --mode highlight --rows <rows.json> --out <dir>
  .venv/bin/python -m benchmarks.ruler compare --dense <dir> --candidate <dir> --out <json>
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import random
import sys
from pathlib import Path

from benchmarks import decode

ROW_FIELDS = ("outputs", "task", "length_bin", "actual_length", "prompt_kind", "scorer_kind",
              "prefill_answer_tag", "strict_tag", "enable_thinking", "choices", "answer_letter")
# Run settings that must match between the dense run and a candidate.
PAIRED_KEYS = ("rows_sha256", "model", "ctx", "gen", "sampling", "ignore_eos", "dtype", "kv_cache_dtype",
               "flash_attn_version", "seed", "yarn_factor", "rope_parameters")


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def prepare(argv=None) -> None:
    from transformers import AutoTokenizer

    import spotattention
    from spotattention.data.eval import get_eval_dataset

    p = argparse.ArgumentParser(description="Freeze Spot's RULER prompts for one length")
    p.add_argument("--checkpoint", required=True, help="tokenizer source")
    p.add_argument("--length", type=int, required=True, help="haystack tokens")
    p.add_argument("--samples-per-task", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        p.error("use a new output file")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    data = get_eval_dataset(
        "ruler", tokenizer=tokenizer, lengths=[args.length], samples_per_task=args.samples_per_task,
        eval_samples=6 * args.samples_per_task, seed=args.seed, enable_thinking_override=False,
    )
    rows = []
    for i in range(len(data)):
        row = data[i]
        ids = [int(t) for t in row["input_ids"]]
        rows.append({"id": f"{args.length}-{i}", "prompt_token_ids": ids, "prompt_sha256": decode.digest(ids),
                     **{k: row.get(k) for k in ROW_FIELDS}})
    if len(rows) != 6 * args.samples_per_task:
        p.error(f"Spot's loader returned {len(rows)} rows, expected {6 * args.samples_per_task}")
    manifest = {
        "protocol": {
            "source": "spotattention.data.eval.ruler",
            "spotattention_commit": decode.git_sha(Path(spotattention.__file__).parent),
            "tokenizer": args.checkpoint,
            "tokenizer_sha256": decode.digest(tokenizer.get_vocab()),
            "chat_template": tokenizer.chat_template,
            "length": args.length,
            "samples_per_task": args.samples_per_task,
            "seed": args.seed,
            "thinking": False,
        },
        "rows": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, manifest)
    print(f"Frozen {len(rows)} rows, longest {max(r['actual_length'] for r in rows)} tokens; "
          f"sha256={decode.digest(manifest)}")


def load_rows(path: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads(path.read_text())
    rows = manifest["rows"]
    if manifest["protocol"]["thinking"] is not False:
        raise ValueError("frozen rows must have thinking off")
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("empty or duplicate rows")
    for r in rows:
        if r["prompt_sha256"] != decode.digest(r["prompt_token_ids"]):
            raise ValueError(f"row {r['id']} differs from the frozen manifest")
    return manifest, rows


def run(argv=None) -> None:
    import spotattention
    from spotattention.data._scorers import extract_answer, score_row
    from vllm.inputs import TokensPrompt

    p = argparse.ArgumentParser(description="Answer frozen RULER rows at batch 1")
    decode.add_engine_args(p)
    p.add_argument("--rows", required=True, help="manifest from prepare")
    p.add_argument("--gen", type=int, default=1024, help="output token limit")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    args.batch = 1
    manifest, rows = load_rows(Path(args.rows))
    args.ctx = max(r["actual_length"] for r in rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "config.json").exists() or (out / "results.jsonl").exists():
        raise ValueError("use a fresh output directory")
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(decode.ROOT), os.environ.get("PYTHONPATH")]))
    llm = decode.build_engine(args, scheduler_cls=None)
    engine = llm.llm_engine.vllm_config
    params = decode.sampling_params(args.gen, ignore_eos=False)
    highlight = args.mode == "highlight"
    config = {
        "rows_sha256": decode.digest(manifest),
        "rows": [r["id"] for r in rows],
        "model": args.checkpoint,
        "ctx": args.ctx,
        "gen": args.gen,
        "thinking": False,
        "sampling": {k: getattr(params, k) for k in (
            "temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty",
            "repetition_penalty", "min_tokens", "skip_special_tokens")},
        "ignore_eos": False,
        "dtype": str(engine.model_config.dtype),
        "kv_cache_dtype": engine.cache_config.cache_dtype,
        "flash_attn_version": engine.attention_config.flash_attn_version,
        "seed": args.seed,
        "yarn_factor": decode.yarn_factor(args.ctx + args.gen),
        "rope_parameters": getattr(engine.model_config.hf_config, "rope_parameters", None),
        "mode": args.mode,
        "spec_tokens": args.spec_tokens if highlight else 0,
        "selection": decode.selection_overrides(args) if args.mode != "dense" else None,
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
    selection_before = decode.selection_snapshot(llm) if args.mode == "direct" else None
    with (out / "results.jsonl").open("x") as stream:
        for r in rows:
            before = decode.spec_counters(llm, args.spec_tokens) if highlight else None
            [result] = llm.generate([TokensPrompt(prompt_token_ids=r["prompt_token_ids"])], params, use_tqdm=False)
            if list(result.prompt_token_ids) != r["prompt_token_ids"]:
                raise RuntimeError("engine changed frozen input tokens")
            answer = result.outputs[0]
            extracted, had_tag = extract_answer(answer.text, r)
            row = {k: r[k] for k in ("id", "task", "length_bin", "actual_length", "outputs")}
            row.update(
                score=score_row(extracted, had_tag, r),
                extracted=extracted,
                had_tag=had_tag,
                output_text=answer.text,
                output_tokens=len(answer.token_ids),
                finish_reason=answer.finish_reason,
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
    tasks = sorted({r["task"] for r in results})
    finished = {
        "rows": len(results),
        "score": sum(r["score"] for r in results) / len(results),
        "per_task": {t: sum(r["score"] for r in results if r["task"] == t) / sum(r["task"] == t for r in results)
                     for t in tasks},
        "capped": sum(r["finish_reason"] != "stop" for r in results),
        "untagged": sum(not r["had_tag"] for r in results),
    }
    if highlight:
        finished["spec"] = {**totals, "alpha": totals["accepted"] / totals["draft_tokens"] if totals["draft_tokens"] else None}
    if selection_before is not None:
        finished["decode_kv"] = decode.selection_record(selection_before, decode.selection_snapshot(llm))
    write_json(out / "finished.json", finished)
    print(json.dumps(finished), flush=True)


def paired(dense: list[float], candidate: list[float], *, resamples: int = 10000) -> dict:
    """Paired score difference in percentage points with a 95% bootstrap interval."""
    if not dense or len(dense) != len(candidate):
        raise ValueError("empty or unequal row sets")
    delta = [b - a for a, b in zip(dense, candidate)]
    rng = random.Random(42)
    boot = sorted(100 * sum(rng.choices(delta, k=len(delta))) / len(delta) for _ in range(resamples))
    return {
        "n": len(delta),
        "dense": 100 * sum(dense) / len(dense),
        "candidate": 100 * sum(candidate) / len(candidate),
        "delta_pp": 100 * sum(delta) / len(delta),
        "lost": sum(d < 0 for d in delta),
        "recovered": sum(d > 0 for d in delta),
        "paired_bootstrap_95pct_delta_pp": [boot[int(0.025 * resamples) - 1], boot[int(0.975 * resamples) - 1]],
    }


def compare(argv=None) -> None:
    p = argparse.ArgumentParser(description="Pair a candidate RULER run with the dense run")
    p.add_argument("--dense", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    configs = [json.loads((d / "config.json").read_text()) for d in (args.dense, args.candidate)]
    if configs[0]["mode"] != "dense":
        p.error("reference must be dense")
    if any(configs[0][k] != configs[1][k] for k in PAIRED_KEYS):
        p.error("run settings differ")
    a, b = ([json.loads(line) for line in (d / "results.jsonl").open()] for d in (args.dense, args.candidate))
    if [r["id"] for r in a] != configs[0]["rows"] or [r["id"] for r in b] != configs[1]["rows"]:
        p.error("incomplete run")
    report = {"overall": paired([r["score"] for r in a], [r["score"] for r in b])}
    report["per_task"] = {
        t: paired([r["score"] for r in a if r["task"] == t], [r["score"] for r in b if r["task"] == t])
        for t in sorted({r["task"] for r in a})
    }
    write_json(args.out, report)
    print(json.dumps(report["overall"]))


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("prepare", "run", "compare"):
        raise SystemExit("expected prepare, run, or compare")
    globals()[sys.argv[1]](sys.argv[2:])
