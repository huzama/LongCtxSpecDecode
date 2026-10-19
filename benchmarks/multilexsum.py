"""Freeze, run and audit an EOS-stopped dense Multi-LexSum workload suite.

prepare-suite --checkpoint PATH --data PARQUET --out DIRECTORY
run --plan DIRECTORY/ctx-32768/plan.json --runtime-root PATH --gpu 1
report --suite DIRECTORY/suite.json

The suite uses the same ten cases at 32K, 64K and 128K, thinking off,
with a detailed-summary instruction and a 512-token output cap.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import socket
import statistics
import time

from benchmarks.ffn_agreement import gpu_lock, save, sha

REVISION = "80d2f6628cd245985675aad1c9aaa11401142c6a"
CONTEXTS = (32768, 65536, 130560)
CONCISE_INSTRUCTION = (
    "You are given legal documents from a civil rights lawsuit. Write a concise "
    "summary of one paragraph (200 to 250 words), covering the background, "
    "parties involved, and outcomes supported by the supplied documents. "
    "Do not invent an outcome if it is absent from the supplied documents.\n\n"
    "Legal documents:\n"
)
DETAILED_INSTRUCTION = (
    "You are given legal documents from a civil rights lawsuit. Write a detailed "
    "summary (500 to 700 words), covering the background, parties involved, "
    "claims, proceedings, and outcomes supported by the supplied documents. "
    "Do not invent an outcome if it is absent from the supplied documents.\n\n"
    "Legal documents:\n"
)
SUFFIX = "\n\nNow summarize the case using only the documents above. Summary:"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def output_audit(ids, eos_ids, cap, finish_reason):
    """Reject post-EOS output and unexplained early completion."""
    if not ids or len(ids) > cap:
        raise ValueError("empty output or output beyond the cap")
    positions = [i + 1 for i, token in enumerate(ids) if token in eos_ids]
    if positions and positions != [len(ids)]:
        raise ValueError("tokens were generated after EOS")
    if positions and finish_reason != "stop":
        raise ValueError("EOS did not terminate the request")
    if not positions and (len(ids) != cap or finish_reason != "length"):
        raise ValueError("request stopped without EOS or the token cap")
    return dict(output_tokens=len(ids), eos_position=positions[0] if positions else None,
                cap_hit=not positions, post_eos_tokens=0)


def prepare(args):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    if args.ctx not in CONTEXTS:
        raise ValueError("context must be 32768, 65536 or 130560")
    parent_rows = json.loads(Path(args.cases_from).read_text()) if args.cases_from else None
    parent_ids = [r["id"] for r in parent_rows] if parent_rows else None
    count = getattr(args, "count", 10)
    if count < 1:
        raise ValueError("case count must be positive")
    if parent_ids is not None and (len(parent_ids) != count or len(set(parent_ids)) != count):
        raise ValueError("original case count or identity differs")
    instruction = DETAILED_INSTRUCTION if args.style == "detailed" else CONCISE_INSTRUCTION
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(args.checkpoint).resolve()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    marker = "__MULTILEXSUM_DOCUMENTS__"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction + marker + SUFFIX}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prefix, suffix = rendered.split(marker)
    before = tokenizer.encode(prefix, add_special_tokens=False)
    after = tokenizer.encode(suffix, add_special_tokens=False)
    budget = args.ctx - len(before) - len(after)
    rows = []
    for ordinal, item in enumerate(pq.read_table(args.data).to_pylist()):
        if parent_ids is not None and item["id"] not in parent_ids:
            continue
        if not item["summary/short"]:
            continue
        source = "\n\n".join(item["sources"])
        source_ids = tokenizer.encode(source, add_special_tokens=False)
        if len(source_ids) < budget:
            continue
        ids = before + source_ids[:budget] + after
        assert len(ids) == args.ctx
        rows.append(dict(id=item["id"], dataset_ordinal=ordinal,
                         case_name=item["case_metadata"]["case_name"],
                         prompt_token_ids=ids, prompt_sha256=digest(ids),
                         input_tokens=len(ids), source_tokens=len(source_ids),
                         source_tokens_used=budget, source_sha256=digest(source),
                         source_documents=len(item["sources"]),
                         source_cropped=len(source_ids) > budget,
                         reference_short=item["summary/short"]))
        if len(rows) == count:
            break
    if len(rows) != count or len({r["id"] for r in rows}) != count:
        raise ValueError(f"validation split does not supply {count} distinct eligible cases")
    if len({r["prompt_sha256"] for r in rows}) != count:
        raise ValueError("selected cases have duplicate tokenized prompts")
    if parent_ids is not None:
        if [r["id"] for r in rows] != parent_ids:
            raise ValueError("case identity/order differs from original pilot")
        for row, parent in zip(rows, parent_rows):
            for key in ("source_sha256", "source_tokens", "dataset_ordinal", "reference_short"):
                if row[key] != parent[key]:
                    raise ValueError(f"original case content changed: {row['id']} {key}")
    save(out / "prompts.json", rows)
    root = Path(__file__).resolve().parents[1]
    snapshot = out / "source"
    for name in ("benchmarks/multilexsum.py", "benchmarks/decode.py", "benchmarks/ffn_agreement.py"):
        dest = snapshot / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, dest)
    (snapshot / "benchmarks/__init__.py").touch()
    shutil.copytree(root / "highlight_spec", snapshot / "highlight_spec",
                    ignore=shutil.ignore_patterns("__pycache__"))
    config = json.loads((checkpoint / "config.json").read_text())
    generation = json.loads((checkpoint / "generation_config.json").read_text())
    eos_ids = set()
    for value in (config["eos_token_id"], generation["eos_token_id"], tokenizer.eos_token_id):
        eos_ids.update(value if isinstance(value, list) else [value])
    plan = dict(checkpoint=str(checkpoint), host=socket.gethostname(),
        dataset="allenai/multi_lexsum", dataset_version="v20230518",
        dataset_revision=REVISION, split="validation", data_sha256=sha(args.data),
        selection=("reuse original pilot cases in original order" if parent_ids else
                   f"first {count} cases in dataset order with a short reference and sufficient source tokens"),
        cases=count,
        parent_prompts_sha256=sha(args.cases_from) if args.cases_from else None,
        prompt_style=args.style, instruction=instruction, suffix=SUFFIX,
        instruction_template_tokens=len(before) + len(after), source_prefix_tokens=budget,
        source_start=len(before),
        input_policy="source token prefix; preserve instruction and Qwen3 chat template; no padding",
        context=args.ctx, max_output_tokens=512, batch=1, mode="dense", thinking=False,
        ignore_eos=False, min_tokens=0, seed=42, eos_token_ids=sorted(eos_ids),
        yarn_factor=-(-(args.ctx + 512) // 32768), flash_attn_version=4, allowed_gpus=[1, 2],
        tokenizer_sha256=digest(tokenizer.get_vocab()), chat_template=tokenizer.chat_template,
        prompts_sha256=sha(out / "prompts.json"),
        checkpoint_config_sha256=sha(checkpoint / "config.json"),
        source_sha256={str(p.relative_to(snapshot)): sha(p) for p in sorted(snapshot.rglob("*.py"))},
        runtime={n: importlib.metadata.version(n) for n in ("torch", "vllm", "transformers", "pyarrow")},
        purpose="workload pilot; no speedup or saturated-throughput claim")
    save(out / "plan.json", plan)
    print(json.dumps(dict(status="prepared", cases=[r["id"] for r in rows],
                          eos_token_ids=sorted(eos_ids), plan=str(out / "plan.json"))), flush=True)


def prepare_suite(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    largest = out / f"ctx-{max(CONTEXTS)}"
    count = getattr(args, "count", 10)
    for ctx in (max(CONTEXTS), *CONTEXTS[:-1]):
        prepare(argparse.Namespace(checkpoint=args.checkpoint, data=args.data,
            out=out / f"ctx-{ctx}", ctx=ctx, style="detailed", count=count,
            cases_from=None if ctx == max(CONTEXTS) else largest / "prompts.json"))
    rows = json.loads((largest / "prompts.json").read_text())
    suite = dict(checkpoint=str(Path(args.checkpoint).resolve()), contexts=list(CONTEXTS),
        case_ids=[r["id"] for r in rows], instruction=DETAILED_INSTRUCTION,
        protocol="dense 8B workload validation; batch 1; one process per context; thinking off; EOS or 512-token cap",
        cohort=f"first {count} validation cases with a short reference and enough source tokens for the largest input",
        runs=[dict(context=ctx, plan=f"ctx-{ctx}/plan.json", plan_sha256=sha(out / f"ctx-{ctx}/plan.json"))
              for ctx in CONTEXTS])
    save(out / "suite.json", suite)
    print(json.dumps(dict(status="suite_prepared", suite=str(out / "suite.json"), cases=suite["case_ids"])), flush=True)


def audit_records(plan, prompts, results):
    """Audit recorded outputs without dropping early EOS or capped cases."""
    if plan["thinking"] is not False or plan["ignore_eos"] is not False or plan["min_tokens"] != 0:
        raise ValueError("thinking or EOS policy differs from the suite")
    count = plan.get("cases", 10)
    if len(prompts) != count or len({p["id"] for p in prompts}) != count:
        raise ValueError(f"expected {count} distinct cases")
    if [r["id"] for r in results] != [p["id"] for p in prompts]:
        raise ValueError("incomplete or reordered results")
    for prompt, row in zip(prompts, results):
        ids = prompt["prompt_token_ids"]
        if len(ids) != plan["context"] or digest(ids) != prompt["prompt_sha256"]:
            raise ValueError("prompt length or token hash changed")
        check = output_audit(row["output_token_ids"], plan["eos_token_ids"],
                             plan["max_output_tokens"], row["finish_reason"])
        if any(row[k] != value for k, value in check.items()):
            raise ValueError("recorded counts differ from output tokens")
        if row["input_tokens"] != len(ids) or row["timed_decode_tokens"] != len(row["output_token_ids"]) - 1:
            raise ValueError("input or timed token count differs")
        if not row["text"].strip() or "<think>" in row["text"] or "</think>" in row["text"]:
            raise ValueError("empty text or thinking markers in output")
        if row["decode_s"] < 0 or (row["timed_decode_tokens"] and row["decode_s"] <= 0):
            raise ValueError("invalid decode time")
    lengths = [r["output_tokens"] for r in results]
    tokens = sum(r["timed_decode_tokens"] for r in results)
    seconds = sum(r["decode_s"] for r in results)
    return dict(context=plan["context"], cases=len(results),
        eos_stopped=sum(r["eos_position"] is not None for r in results),
        capped=sum(r["cap_hit"] for r in results), post_eos_tokens=0,
        output_tokens_min=min(lengths), output_tokens_median=statistics.median(lengths),
        output_tokens_max=max(lengths), output_tokens_total=sum(lengths),
        timed_decode_tokens=tokens, decode_seconds=seconds,
        decode_tok_s=tokens / seconds if seconds else None)


def report(args):
    path = Path(args.suite).resolve()
    suite = json.loads(path.read_text())
    if suite["contexts"] != list(CONTEXTS) or [r["context"] for r in suite["runs"]] != list(CONTEXTS):
        raise ValueError("suite must contain all three contexts in order")
    summaries, reference, previous = [], None, None
    table = ["# Multi-LexSum workload validation", "", suite["protocol"], "",
             f"Same {len(suite['case_ids'])} cases and nested source prefixes. No padding or artificial repetition. "
             "128K uses 130,560 input tokens, reserving 512 within the 131,072-token window.", "",
             "| Input tokens | Cases | EOS stops | Cap stops | Output min / median / max | Decode tok/s |",
             "|---:|---:|---:|---:|---:|---:|"]
    for entry in suite["runs"]:
        plan_path = path.parent / entry["plan"]
        if sha(plan_path) != entry["plan_sha256"]:
            raise ValueError("suite plan changed")
        plan = json.loads(plan_path.read_text())
        folder = plan_path.parent
        if json.loads((folder / "status.json").read_text())["status"] != "complete":
            raise ValueError("suite run is incomplete")
        if sha(folder / "prompts.json") != plan["prompts_sha256"]:
            raise ValueError("prompt file changed")
        for name, expected in plan["source_sha256"].items():
            if sha(folder / "source" / name) != expected:
                raise ValueError("frozen source changed")
        prompts = json.loads((folder / "prompts.json").read_text())
        results = json.loads((folder / "results.json").read_text())
        if plan["context"] != entry["context"] or [r["id"] for r in prompts] != suite["case_ids"]:
            raise ValueError("context or cohort differs from suite")
        if plan["instruction"] != suite["instruction"] or plan["checkpoint"] != suite["checkpoint"]:
            raise ValueError("instruction or model differs from suite")
        invariant = {k: plan[k] for k in ("checkpoint", "instruction", "suffix", "data_sha256", "runtime",
            "tokenizer_sha256", "chat_template", "max_output_tokens", "batch", "mode", "seed",
            "flash_attn_version", "source_sha256", "checkpoint_config_sha256")}
        if reference is not None and invariant != reference:
            raise ValueError("settings differ across contexts")
        reference = invariant
        prefixes = [p["prompt_token_ids"][plan["source_start"]:plan["source_start"] + plan["source_prefix_tokens"]]
                    for p in prompts]
        if previous is not None and any(a != b[:len(a)] for a, b in zip(previous, prefixes)):
            raise ValueError("source prefixes differ across contexts")
        previous = prefixes
        audit = audit_records(plan, prompts, results)
        save(folder / "audit.json", audit)
        summaries.append(audit)
        table.append(f"| {audit['context']:,} | {audit['cases']} | {audit['eos_stopped']} | {audit['capped']} | "
                     f"{audit['output_tokens_min']} / {audit['output_tokens_median']:g} / {audit['output_tokens_max']} | "
                     f"{audit['decode_tok_s']:.2f} |")
        text = [f"# Multi-LexSum outputs: {plan['context']:,} input tokens", ""]
        for row in results:
            text.extend([f"## {row['id']}: {row['case_name']}", "",
                         f"{row['output_tokens']} tokens; stop: {row['finish_reason']}.", "", row["text"], ""])
        (folder / "summaries.md").write_text("\n".join(text))
    summary = dict(status="complete", suite_sha256=sha(path), cases=suite["case_ids"], contexts=summaries,
                   purpose="workload validation, not a method speedup or saturation benchmark")
    save(path.parent / "summary.json", summary)
    table.extend(["", "Decode rates exclude each request's first output token, loading and prefill. "
                  "One process per context; no workload warm-up, repetitions or KV saturation. "
                  "Capped outputs are summary prefixes, not completed summaries. Factual accuracy is not established.", ""])
    (path.parent / "summary.md").write_text("\n".join(table))
    print(json.dumps(summary), flush=True)


def generate(llm, ids, params):
    from vllm.inputs import TokensPrompt

    request = str(next(llm.request_counter))
    first = completed = None
    started = time.perf_counter()
    llm.llm_engine.add_request(request, TokensPrompt(prompt_token_ids=ids), params)
    while llm.llm_engine.has_unfinished_requests():
        outputs = llm.llm_engine.step()
        now = time.perf_counter()
        for output in outputs:
            if output.request_id != request:
                raise RuntimeError("unexpected request")
            if output.outputs and output.outputs[0].token_ids and first is None:
                if len(output.outputs[0].token_ids) != 1:
                    raise RuntimeError("dense prefill did not emit exactly one token")
                first = now
            if output.finished:
                completed = output
                ended = now
    if completed is None or list(completed.prompt_token_ids) != ids:
        raise RuntimeError("missing output or changed input")
    return completed.outputs[0], dict(prefill_to_first_token_s=first - started,
                                      decode_s=ended - first, request_s=ended - started)


def run(args):
    plan_path = Path(args.plan).resolve()
    out = plan_path.parent
    plan = json.loads(plan_path.read_text())
    if args.gpu not in plan["allowed_gpus"]:
        raise ValueError("GPU not authorized")
    snapshot = Path(__file__).resolve().parents[1]
    for name, expected in plan["source_sha256"].items():
        if sha(snapshot / name) != expected:
            raise ValueError(f"source changed: {name}")
    if sha(out / "prompts.json") != plan["prompts_sha256"]:
        raise ValueError("prompts changed")
    if socket.gethostname() != plan["host"]:
        raise ValueError("host changed")
    for name, version in plan["runtime"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"runtime changed: {name}")
    if sha(Path(plan["checkpoint"]) / "config.json") != plan["checkpoint_config_sha256"]:
        raise ValueError("checkpoint config changed")
    if (out / "results.json").exists() or (out / "status.json").exists():
        raise ValueError("pilot already started; preserve evidence and use a fresh directory")
    state = dict(status="waiting_for_idle_gpu", gpu=args.gpu, pid=os.getpid(),
                 plan_sha256=sha(plan_path), completed=0)
    save(out / "status.json", state)
    lock = gpu_lock(Path(args.runtime_root), args.gpu)
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ.update(VLLM_USE_V2_MODEL_RUNNER="1", VLLM_USE_FLASHINFER_SAMPLER="0",
                          VLLM_RPC_BASE_PATH="/tmp")
        for key in ("TMPDIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "VLLM_CACHE_ROOT"):
            path = out / "cache" / key.lower()
            path.mkdir(parents=True, exist_ok=True)
            os.environ[key] = str(path)
        import spotattention  # register model and kernels before engine creation
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind
        from benchmarks import decode

        parser = argparse.ArgumentParser()
        decode.add_engine_args(parser)
        engine_args = parser.parse_args(["--checkpoint", plan["checkpoint"], "--mode", "dense"])
        engine_args.ctx = plan["context"]
        engine_args.gen, engine_args.batch = plan["max_output_tokens"], plan["batch"]
        if decode.yarn_factor(engine_args.ctx + engine_args.gen) != plan["yarn_factor"]:
            raise ValueError("context extension differs from frozen plan")
        state["status"] = "loading"
        save(out / "status.json", state)
        llm = decode.build_engine(engine_args, scheduler_cls=None)
        params = SamplingParams(temperature=0, top_p=1, top_k=0, min_p=0,
            presence_penalty=0, frequency_penalty=0, repetition_penalty=1,
            max_tokens=plan["max_output_tokens"], min_tokens=0, ignore_eos=False,
            stop_token_ids=plan["eos_token_ids"], skip_special_tokens=False,
            output_kind=RequestOutputKind.CUMULATIVE)
        rows = json.loads((out / "prompts.json").read_text())
        results = []
        for row in rows:
            ids = row["prompt_token_ids"]
            if digest(ids) != row["prompt_sha256"]:
                raise ValueError("prompt token hash mismatch")
            result, timing = generate(llm, ids, params)
            tokens = list(result.token_ids)
            audit = output_audit(tokens, plan["eos_token_ids"], plan["max_output_tokens"], result.finish_reason)
            results.append(dict(id=row["id"], case_name=row["case_name"],
                input_tokens=len(ids), output_token_ids=tokens, text=result.text,
                finish_reason=result.finish_reason, stop_reason=result.stop_reason,
                **audit, **timing, timed_decode_tokens=len(tokens) - 1,
                decode_tok_s=(len(tokens) - 1) / timing["decode_s"] if timing["decode_s"] else None))
            save(out / "results.json", results)
            state.update(status="running", completed=len(results))
            save(out / "status.json", state)
            print(json.dumps({k: results[-1][k] for k in
                              ("id", "output_tokens", "eos_position", "cap_hit", "decode_tok_s")}), flush=True)
        state.update(status="complete", eos_stopped=sum(r["eos_position"] is not None for r in results),
                     capped=sum(r["cap_hit"] for r in results), post_eos_tokens=0)
        save(out / "status.json", state)
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        save(out / "status.json", state)
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    for field in ("checkpoint", "data", "out"):
        prep.add_argument("--" + field, required=True)
    prep.add_argument("--style", choices=("concise", "detailed"), default="concise")
    prep.add_argument("--cases-from", help="original frozen prompts.json; reuse its ten cases")
    prep.add_argument("--ctx", type=int, choices=CONTEXTS, default=32768)
    prep.add_argument("--count", type=int, default=10)
    suite = sub.add_parser("prepare-suite")
    for field in ("checkpoint", "data", "out"):
        suite.add_argument("--" + field, required=True)
    suite.add_argument("--count", type=int, default=10)
    audit = sub.add_parser("report")
    audit.add_argument("--suite", required=True)
    worker = sub.add_parser("run")
    for field in ("plan", "runtime-root"):
        worker.add_argument("--" + field, required=True)
    worker.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "prepare-suite": prepare_suite, "run": run, "report": report}[args.action](args)
