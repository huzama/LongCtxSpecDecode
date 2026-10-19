"""Prepare and audit the EOS-stopped Multi-LexSum throughput campaign.

Preparation is CPU-only. Launch requires a separate, recorded GPU preflight.
Completed quality and selector campaigns are retained, not rerun here.
"""
import argparse
import copy
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import socket
import statistics
import subprocess
import time

from benchmarks.ffn_agreement import save, sha
from benchmarks.multilexsum import CONTEXTS, digest, output_audit

POOL = 32
REPETITIONS = 3
BATCHES = {"4b": (26, 13, 6), "8b": (25, 12, 6)}
METHODS = {"dense": ("dense", 0), "direct": ("direct", 0),
           "dv": ("highlight", 0), "sv50": ("highlight", .5),
           "sv40": ("highlight", .4), "sv25": ("highlight", .25)}
FFN = ("bf16-full", "int4-full", "bf16-highlight", "int4-highlight")
MARKER = "eos-throughput"


def useful_speculation(steps, drafts, cap):
    """Emitted tokens include the terminal round; acceptance excludes censored rounds.

    An unfinished, unrestricted greedy verification emits its accepted draft
    prefix and exactly one correction/bonus token. EOS/cap termination can
    remove that last token, so emitted-1 is not an acceptance estimator there.
    """
    tokens = rounds = accepted = complete = 0
    for step in steps:
        for before, after, finished in zip(step["before"], step["after"], step["finished"], strict=True):
            emitted = after - before
            if not 1 <= emitted <= drafts + 1 or after > cap:
                raise ValueError("invalid speculative emission count")
            tokens += emitted
            rounds += 1
            if not finished and cap - before >= drafts + 1:
                accepted += emitted - 1
                complete += 1
    return dict(useful_tokens=tokens, verification_rounds=rounds,
        useful_tokens_per_round=tokens / rounds if rounds else None,
        uncensored_rounds=complete, accepted_in_uncensored_rounds=accepted,
        uncensored_acceptance=accepted / (drafts * complete) if complete else None,
        uncensored_accepted_per_round=accepted / complete if complete else None,
        excluded_acceptance_rounds=rounds - complete)


def audit_records(job, records, prompts):
    """Recompute every reported count from frozen inputs, emitted IDs and step traces."""
    n, batch, cap = len(prompts), job["batch"], 512
    expected_rows = job.get("expected_rows", POOL)
    if (n != job.get("prompt_pool_size", expected_rows) or len({p["id"] for p in prompts}) != n
            or len({p["prompt_sha256"] for p in prompts}) != n or len(records) != expected_rows
            or batch > n or (job.get("single_batch") and expected_rows != 1)):
        raise ValueError("incomplete distinct prompt pool or batch grid")
    for p in prompts:
        if len(p["prompt_token_ids"]) != job["ctx"] or digest(p["prompt_token_ids"]) != p["prompt_sha256"]:
            raise ValueError("frozen prompt tokens changed")
    total_tokens = seconds = eos = capped = rounds = emitted = accepted = complete = 0
    for slot, row in enumerate(records):
        expected = [(slot + j) % n for j in range(batch)]
        if row["prompt_slots"] != expected or row["slot"] != slot:
            raise ValueError("cyclic prompt grouping changed")
        if row["prompt_sha256"] != [prompts[s]["prompt_sha256"] for s in expected]:
            raise ValueError("prompt identities changed")
        if (row["ctx"], row["batch"], row["gen"]) != (job["ctx"], batch, cap):
            raise ValueError("workload shape changed")
        if row["ignore_eos"] is not False or row["thinking"] is not False or row["temperature"] != 0:
            raise ValueError("generation protocol changed")
        if row["flash_attn_version"] != 4 or row["indexer_cache_format"] != job["format"]:
            raise ValueError("attention backend or cache format changed")
        if row["mode"] != job["mode"] or row["draft_attention"] != job.get("draft_attention"):
            raise ValueError("execution method changed")
        if row["draft_ffn_checkpoint"] != job.get("draft_ffn_checkpoint"):
            raise ValueError("FFN checkpoint changed")
        if row["mode"] != "dense":
            selection = row["selection"]
            if selection["molle_verify_fraction"] != job["verify_fraction"] or selection["molle_topk_fraction"] != .07:
                raise ValueError("selection budgets changed")
            if selection["molle_topk_recency"] != 64 or selection["molle_draft_select"] != ("step" if row["mode"] == "direct" else "round"):
                raise ValueError("recency or selection refresh changed")
        if not row["kv_capacity"]["fits"] or row["kv_capacity"]["batch"] != batch:
            raise ValueError("insufficient KV capacity")
        if job.get("draft_attention"):
            validation = row["ffn_draft_validation"]
            if not validation or not validation["graph_matches_eager"] or validation["validation_batch"] != batch:
                raise ValueError("FFN graph lacks full-batch validation")
        outputs = row["request_outputs"]
        if len(outputs) != batch or [o["prompt_slot"] for o in outputs] != expected:
            raise ValueError("missing request outputs")
        for output in outputs:
            check = output_audit(output["output_token_ids"], row["eos_token_ids"], cap, output["finish_reason"])
            if any(output[k] != value for k, value in check.items()):
                raise ValueError("EOS output audit differs")
            if not output["text"].strip() or "<think>" in output["text"] or "</think>" in output["text"]:
                raise ValueError("empty output or thinking markers")
            eos += check["eos_position"] is not None
            capped += check["cap_hit"]
        previous, elapsed = [1] * batch, 0
        steps = row["decode_steps"]
        if not steps:
            raise ValueError("no full-batch decode window")
        for i, step in enumerate(steps):
            if step["before"] != previous or len(step["after"]) != batch or len(step["finished"]) != batch:
                raise ValueError("decode trace is incomplete")
            if not math.isfinite(step["elapsed_seconds"]) or step["elapsed_seconds"] <= elapsed:
                raise ValueError("invalid decode timestamps")
            if any(step["finished"]) != (i == len(steps) - 1):
                raise ValueError("decode window continued after first finish")
            for j, (a, b, finished) in enumerate(zip(step["before"], step["after"], step["finished"], strict=True)):
                limit = 7 if row["mode"] == "highlight" else 1
                if not 1 <= b - a <= limit or b > len(outputs[j]["output_token_ids"]):
                    raise ValueError("decode trace exceeds emitted output")
                terminal = b == len(outputs[j]["output_token_ids"])
                if finished != terminal:
                    raise ValueError("finish flag differs from EOS/cap output")
            previous, elapsed = step["after"], step["elapsed_seconds"]
        tokens = sum(previous) - batch
        if row["decode_tokens"] != tokens or row["decode_seconds"] != elapsed:
            raise ValueError("decode totals differ from raw trace")
        if row["decode_window"] != "all_prefills_complete_to_first_request_finish":
            raise ValueError("decode window changed")
        if not math.isclose(row["decode_tok_s"], tokens / elapsed, rel_tol=1e-12):
            raise ValueError("throughput differs from raw counts")
        total_tokens += tokens
        seconds += elapsed
        if row["mode"] == "highlight":
            value = useful_speculation(steps, 6, cap)
            if row["speculation"] != value:
                raise ValueError("speculative accounting differs")
            # vLLM's native counters precede EOS clipping. Retain them only
            # as diagnostics, never as the published useful acceptance rate.
            native = row["native_spec_counters"]
            if not value["uncensored_rounds"] <= native["drafts"] <= value["verification_rounds"]:
                raise ValueError("native verification counters disagree with emission trace")
            if native["draft_tokens"] < 6 * value["uncensored_rounds"] or native["accepted"] < value["accepted_in_uncensored_rounds"]:
                raise ValueError("native verification counters disagree with emission trace")
            rounds += value["verification_rounds"]
            emitted += value["useful_tokens"]
            accepted += value["accepted_in_uncensored_rounds"]
            complete += value["uncensored_rounds"]
    return dict(batches=len(records), batch=batch, decode_tokens=total_tokens, decode_seconds=seconds,
        decode_tok_s=total_tokens / seconds, eos_stops=eos, cap_stops=capped, post_eos_tokens=0,
        kv_reserved_fraction=records[0]["kv_capacity"].get("reserved_fraction"),
        useful_tokens_per_round=emitted / rounds if rounds else None,
        uncensored_acceptance=accepted / (6 * complete) if complete else None,
        uncensored_rounds=complete, excluded_acceptance_rounds=rounds - complete)


def audit_job(job, dest):
    fingerprint = {name: sha(dest / name) for name in ("results.jsonl", "summary.json")}
    fingerprint["manifest"] = sha(job["manifest"])
    fingerprint["job"] = digest(job)
    cached = dest / "audit.json"
    if cached.exists():
        value = json.loads(cached.read_text())
        if value.get("evidence_sha256") == fingerprint:
            return value
    prompts = json.loads(Path(job["manifest"]).read_text())
    if sha(job["manifest"]) != job["manifest_sha256"]:
        raise ValueError("manifest changed")
    rows = [json.loads(s) for s in (dest / "results.jsonl").read_text().splitlines()]
    value = audit_records(job, rows, prompts)
    summary = json.loads((dest / "summary.json").read_text())
    for key in ("batches", "decode_tokens", "decode_seconds", "decode_tok_s"):
        if summary[key] != value[key]:
            raise ValueError("summary differs from audited records")
    if job.get("draft_attention"):
        info = summary["ffn_draft_info"]
        if info["graph_scope"] != "step" or info["step_graph_replays"] <= rows[0]["ffn_draft_validation"]["step_graph_replays"]:
            raise ValueError("draft CUDA graphs did not replay during timing")
    value["evidence_sha256"] = fingerprint
    save(dest / "audit.json", value)
    return value


def engine_flags(job, checkpoint, quant):
    flags = ["--checkpoint", checkpoint, "--mode", job["mode"], "--ctx", str(job["ctx"]),
        "--gen", "512", "--indexer-cache-format", job["format"], "--eos-aware",
        "--prompt-manifest", job["manifest"], "--samples", str(job.get("prompt_pool_size", job.get("expected_rows", POOL)))]
    if job.get("single_batch"):
        flags += ["--single-batch"]
    if job["mode"] == "highlight":
        flags += ["--draft-select", "round", "--verify-fraction", str(job["verify_fraction"])]
    if job.get("draft_attention"):
        flags += ["--draft-attention", job["draft_attention"]]
    if job.get("draft_ffn_checkpoint"):
        flags += ["--draft-ffn-checkpoint", quant]
    return flags


def make_jobs(root, out, checkpoints, quant, manifests):
    jobs = []
    python = str(root / ".venv/bin/python")
    for model in ("4b", "8b"):
        for ctx, requested in zip(CONTEXTS, BATCHES[model], strict=True):
            manifest = manifests[(model, ctx)]
            shared = dict(model=model, ctx=ctx, manifest=str(manifest), manifest_sha256=sha(manifest), requested_batch=requested)
            capacities = []
            for name in (["sv50", "int4-highlight"] if model == "8b" else ["sv50"]):
                job = dict(shared, name=f"capacity/{model}/{ctx}/{name}", kind="eos-capacity",
                    method=name, mode="highlight", format="bf16", verify_fraction=.5 if name == "sv50" else 0,
                    batch=requested)
                if name == "int4-highlight":
                    job.update(draft_attention="highlight", draft_ffn_checkpoint=quant)
                job["command"] = [python, "-m", "benchmarks.decode", *engine_flags(job, checkpoints[model], quant),
                    "--batch", str(requested), "--capacity-only", "--out", str(out / job["name"])]
                jobs.append(job)
                capacities.append(job["name"])
            variants = [("dense", "bf16")]
            variants += [(m, f) for f in ("bf16", "mxfp4") for m in ("direct", "dv", "sv50", "sv40")]
            variants += [("sv25", "mxfp4")]
            if model == "8b":
                variants += [(m, "bf16") for m in FFN]
            for rep in range(REPETITIONS):
                for method, fmt in variants:
                    mode, fraction = METHODS.get(method, ("highlight", 0))
                    name = f"throughput/{model}/{ctx}/{method}-{fmt}/rep{rep + 1}"
                    job = dict(shared, name=name, kind="eos-throughput", method=method, mode=mode,
                        format=fmt, verify_fraction=fraction, rep=rep + 1, capacity_jobs=capacities,
                        depends_on=list(capacities), expected_rows=POOL)
                    if method != "dense":
                        job["depends_on"].append(f"throughput/{model}/{ctx}/dense-bf16/rep{rep + 1}")
                    if method in FFN:
                        job["draft_attention"] = method.split("-", 1)[1]
                        if method.startswith("int4"):
                            job["draft_ffn_checkpoint"] = quant
                    job["command"] = [python, "-m", "benchmarks.decode", *engine_flags(job, checkpoints[model], quant),
                        "--batch", "{batch}", "--out", str(out / name)]
                    jobs.append(job)
    return jobs


def resolve_batch(job, out):
    capacities = [json.loads((out / name / "capacity.json").read_text())["maximum_batch"] for name in job["capacity_jobs"]]
    batch = min(job["requested_batch"], *capacities)
    if batch < 1:
        raise ValueError("no request fits at the requested context")
    return batch, [str(batch) if word == "{batch}" else word for word in job["command"]]


def ffn_priority_jobs(original, out):
    """Change only pending FFN repetitions and their separate dense reference."""
    jobs = copy.deepcopy(original)
    priority = []
    for job in jobs:
        if job.get("state") == "complete":
            continue
        job["command"][job["command"].index("--out") + 1] = str(out / job["name"])
        if job["kind"] == "eos-throughput" and job["method"] in FFN:
            if job.get("state", "pending") != "pending":
                raise ValueError("cannot change an FFN process that has already started")
            job.update(expected_rows=1, prompt_pool_size=POOL, single_batch=True)
            job["command"].append("--single-batch")
            job["depends_on"] = [name.replace("/dense-bf16/", "/ffn-dense-bf16/")
                                 for name in job["depends_on"]]
            priority.append(job["name"])
        elif job["model"] == "8b" and job["method"] == "dense":
            baseline = copy.deepcopy(job)
            baseline.update(name=job["name"].replace("/dense-bf16/", "/ffn-dense-bf16/"),
                method="ffn-dense", expected_rows=1, prompt_pool_size=POOL, single_batch=True)
            baseline["command"][baseline["command"].index("--out") + 1] = str(out / baseline["name"])
            baseline["command"].append("--single-batch")
            priority.append(baseline["name"])
            # Append after iteration, so the copied dense method cannot recur.
            job["_ffn_baseline"] = baseline
    jobs += [j.pop("_ffn_baseline") for j in list(jobs) if "_ffn_baseline" in j]
    rank = {"ffn-dense": 0, "int4-highlight": 1, "int4-full": 2, "bf16-highlight": 3, "bf16-full": 4}
    def order(job):
        is_capacity = job["kind"] == "eos-capacity" and job["model"] == "8b"
        is_priority = job["name"] in priority or is_capacity
        if not is_priority and job.get("state", "pending") == "pending":
            job["depends_on"] = list(dict.fromkeys([*job.get("depends_on", []), *priority]))
        return (0, job["ctx"], 0 if is_capacity else 1, job.get("rep", 0), rank.get(job["method"], 0)) if is_priority else (1,)
    jobs.sort(key=order)
    return jobs


def single_batch_jobs(original):
    """Shorten pending runs and share identical 8B dense controls with FFN."""
    jobs = copy.deepcopy(original)
    if any(j["state"] == "running" for j in jobs):
        raise ValueError("finish active jobs before revising their plan")
    for job in jobs:
        if job["state"] == "pending" and job["kind"] == "eos-throughput":
            job.update(expected_rows=1, prompt_pool_size=POOL, single_batch=True)
            if "--single-batch" not in job["command"]:
                job["command"].append("--single-batch")
    by_name = {j["name"]: j for j in jobs}
    aliases = {}
    for job in jobs:
        if job["kind"] != "eos-throughput" or job["model"] != "8b" or job["method"] != "dense":
            continue
        if job["state"] != "pending":
            raise ValueError("cannot remove a dense reference that has already run")
        name = job["name"].replace("/dense-bf16/", "/ffn-dense-bf16/")
        control = by_name[name]
        def normalized(command):
            command = list(command)
            index = command.index("--out")
            return command[:index] + command[index + 2:]
        if (normalized(job["command"]) != normalized(control["command"])
                or any(job[k] != control[k] for k in ("capacity_jobs", "requested_batch", "manifest_sha256"))):
            raise ValueError("dense controls do not have identical workloads")
        aliases[job["name"]] = name
    for job in jobs:
        job["depends_on"] = list(dict.fromkeys(aliases.get(name, name) for name in job.get("depends_on", [])))
    return [j for j in jobs if j["name"] not in aliases]


def revise_ffn(args):
    """Preserve completed evidence; put single-batch FFN runs ahead of other jobs."""
    from benchmarks.queue import source_digest
    parent, out = Path(args.campaign).resolve(), Path(args.out).resolve()
    prior = json.loads((parent / "plan.json").read_text())
    state = json.loads((parent / "status.json").read_text())
    if state["state"] != "superseded" or any(j["state"] == "running" for j in state["jobs"]):
        raise ValueError("stop the parent controller and reconcile its children first")
    if sha(parent / "plan.json") != state["plan_sha256"] or source_digest(parent / "source") != prior["source_sha256"]:
        raise ValueError("parent plan/source changed")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(parent / "source", out / "source", ignore=shutil.ignore_patterns("__pycache__"))
    live = Path(__file__).resolve().parents[1]
    for name in ("benchmarks/decode.py", "benchmarks/eos_throughput.py", "tests/test_eos_throughput.py"):
        shutil.copy2(live / name, out / "source" / name)
    shutil.copytree(parent / "preflight-inputs", out / "preflight-inputs")
    original = []
    for job in state["jobs"]:
        item = copy.deepcopy(job)
        if item["state"] == "complete":
            shutil.copytree(parent / item["name"], out / item["name"])
            item["inherited_from"] = str(parent / item["name"])
            item["inherited_source_sha256"] = prior["source_sha256"]
        else:
            item = copy.deepcopy(next(j for j in prior["jobs"] if j["name"] == item["name"]))
            item["state"] = "pending"
        original.append(item)
    jobs = ffn_priority_jobs(original, out)
    plan = copy.deepcopy(prior)
    plan.update(jobs=jobs, source_sha256=source_digest(out / "source"), ffn_priority=True,
        parent_campaign=str(parent), parent_plan_sha256=sha(parent / "plan.json"),
        parent_status_sha256=sha(parent / "status.json"))
    plan["protocol"].update(ffn_batches_per_process=1, ffn_dense_reference="ffn-dense",
        ffn_prompt_group="first B frozen cases, identical across all five methods and three repetitions",
        scheduling="finish all FFN cells and their dense controls before remaining main-method jobs",
        other_batches_per_process=POOL)
    save(out / "plan.json", plan)
    save(out / "status.json", dict(jobs=jobs, source_sha256=plan["source_sha256"],
        plan_sha256=sha(out / "plan.json"), allowed_gpus=[2, 3], idle_seconds_required=60,
        state="prepared_not_launched"))
    report(argparse.Namespace(run=out))
    print(json.dumps(dict(status="prepared_not_launched", out=str(out),
        ffn_runs=sum(j["method"] in (*FFN, "ffn-dense") and j["kind"] == "eos-throughput" for j in jobs))))


def prepare(args):
    from benchmarks.multilexsum import prepare as prepare_prompts
    from benchmarks.queue import source_digest
    out, runtime = Path(args.out).resolve(), Path(args.runtime_root).resolve()
    out.mkdir(parents=True, exist_ok=False)
    checkpoints = {"4b": str(Path(args.checkpoint_4b).resolve()), "8b": str(Path(args.checkpoint_8b).resolve())}
    if sha(args.data) != "6343f22c2e64eb97892608adb90e23c58e89fd60608cf6698f1fd518002d2822":
        raise ValueError("dataset differs from the validated Multi-LexSum revision")
    manifests, common = {}, None
    for model in ("8b", "4b"):
        for ctx in (max(CONTEXTS), *CONTEXTS[:-1]):
            folder = out / "prompts" / model / str(ctx)
            prepare_prompts(argparse.Namespace(checkpoint=checkpoints[model], data=args.data,
                out=folder, ctx=ctx, style="detailed", count=POOL, cases_from=common))
            manifests[model, ctx] = folder / "prompts.json"
            if common is None:
                common = folder / "prompts.json"
    kernel = json.loads(Path(args.kernel_validation).read_text())
    if not kernel["passed"] or len(kernel["layers"]) != 36:
        raise ValueError("missing full FFN kernel validation")
    # Reuse numerical kernel evidence only for exactly the same implementation.
    old_source = Path(args.kernel_validation).parent / "source"
    source_root = Path(__file__).resolve().parents[1]
    for filename in ("highlight_spec/ffn_draft.py", "highlight_spec/ffn_quant.py"):
        if sha(old_source / filename) != sha(source_root / filename):
            raise ValueError(f"FFN implementation changed since kernel validation: {filename}")
    agreement = json.loads(Path(args.agreement_plan).read_text())
    if agreement["checkpoint"] != checkpoints["8b"] or agreement["quant"] != str(Path(args.quant).resolve()):
        raise ValueError("FFN identity evidence uses different checkpoints")
    pilot = Path(args.eos_pilot).resolve()
    pilot_plan = json.loads((pilot / "plan.json").read_text())
    if pilot_plan["checkpoint"] != checkpoints["8b"] or pilot_plan["ignore_eos"] is not False or pilot_plan["context"] != 32768:
        raise ValueError("EOS preflight needs the validated dense 8B/32K pilot")
    pilot_rows = json.loads((pilot / "prompts.json").read_text())[:2]
    if sha(pilot / "prompts.json") != pilot_plan["prompts_sha256"]:
        raise ValueError("EOS pilot prompts changed")
    preflight_inputs = out / "preflight-inputs"
    preflight_inputs.mkdir()
    save(preflight_inputs / "eos-prompts.json", pilot_rows)
    source = out / "source"
    for folder in ("benchmarks", "highlight_spec", "tests"):
        shutil.copytree(source_root / folder, source / folder, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(runtime / "outputs/molle/spotattention", source / "spotattention",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    jobs = make_jobs(runtime, out, checkpoints, str(Path(args.quant).resolve()), manifests)
    inputs = {str(p): sha(p) for p in manifests.values()}
    inputs[str(preflight_inputs / "eos-prompts.json")] = sha(preflight_inputs / "eos-prompts.json")
    for checkpoint in (*checkpoints.values(), args.quant):
        for name in ("config.json", "generation_config.json"):
            path = Path(checkpoint) / name
            if path.exists():
                inputs[str(path)] = sha(path)
    for path in (args.agreement_plan, args.kernel_validation):
        inputs[str(Path(path).resolve())] = sha(path)
    plan = dict(runtime_root=str(runtime), host=socket.gethostname(), allowed_gpus=[2, 3],
        report_module="benchmarks.eos_throughput", source_sha256=source_digest(source), input_sha256=inputs,
        runtime={n: importlib.metadata.version(n) for n in ("torch", "vllm", "transformers")}, jobs=jobs,
        protocol=dict(dataset="Multi-LexSum v20230518 validation", prompts=POOL, process_repetitions=REPETITIONS,
            contexts=list(CONTEXTS), max_output_tokens=512, ignore_eos=False, thinking=False,
            batch="minimum of existing matched batch and BF16 SV50 / INT4 Highlight capacity probes",
            sampling="greedy", warmup_tokens=16, draft_tokens=6, draft_graph="one graph per draft step",
            throughput_window="all prefills complete to first EOS/cap finish; drain tail outside timing",
            acceptance="accepted draft tokens on nonterminal rounds with room for all six drafts plus correction only",
            useful_tokens="all emitted decode tokens per verification, including the terminal round",
            retained_evidence="completed RULER, LongBench v2 and selector campaigns"))
    save(out / "plan.json", plan)
    report(argparse.Namespace(run=out))
    print(json.dumps(dict(status="prepared_not_launched", jobs=len(jobs), plan=str(out / "plan.json"))))


def preflight(args):
    """Run real EOS, sparse verification and FFN paths before enabling the queue."""
    from benchmarks.ffn_agreement import gpu_lock
    from benchmarks.queue import source_digest
    out = Path(args.campaign).resolve()
    source = out / "source"
    plan = json.loads((out / "plan.json").read_text())
    runtime = Path(plan["runtime_root"])
    if args.gpu not in (2, 3) or args.gpu not in plan["allowed_gpus"] or socket.gethostname() != plan["host"]:
        raise ValueError("GPU or host is not authorized")
    if source_digest(source) != plan["source_sha256"] or Path(__file__).resolve().parents[1] != source:
        raise ValueError("preflight must run from the frozen campaign source")
    for name, expected in plan["input_sha256"].items():
        if sha(name) != expected:
            raise ValueError(f"preflight input changed: {name}")
    for name, version in plan["runtime"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"preflight runtime changed: {name}")
    dest = out / "preflight"
    dest.mkdir(exist_ok=False)
    python = str(runtime / ".venv/bin/python")
    env = {**os.environ, "PYTHONPATH": str(source), "CUDA_VISIBLE_DEVICES": "",
        "VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_RPC_BASE_PATH": "/tmp", "OMP_NUM_THREADS": "8", "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"}
    for variable, folder in (("TMPDIR", "tmp"), ("TRITON_CACHE_DIR", "triton"),
                             ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"), ("VLLM_CACHE_ROOT", "vllm")):
        path = dest / "cache" / folder
        path.mkdir(parents=True)
        env[variable] = str(path)
    state = dict(passed=False, status="cpu_tests", gpu=args.gpu, plan_sha256=sha(out / "plan.json"),
        source_sha256=plan["source_sha256"], checks=[])
    save(out / "preflight.json", state)
    lock = None
    try:
        with (dest / "tests.log").open("w") as log:
            subprocess.run([python, "-m", "pytest", "tests/test_eos_throughput.py", "tests/test_multilexsum.py",
                "tests/test_ffn_timing.py", "-q", f"--junitxml={dest / 'tests.xml'}"],
                cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        state["status"] = "waiting_for_idle_gpu"
        save(out / "preflight.json", state)
        lock = gpu_lock(runtime, args.gpu)
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        checks = (("8b", "dense", "bf16"), ("4b", "direct", "bf16"),
                  ("8b", "sv50", "mxfp4"), ("8b", "int4-highlight", "bf16"))
        for model, method, fmt in checks:
            original = next(j for j in plan["jobs"] if j["kind"] == "eos-throughput" and
                (j["model"], j["ctx"], j["method"], j["format"], j["rep"]) == (model, 32768, method, fmt, 1))
            job = {**original, "batch": 2, "expected_rows": 2, "prompt_pool_size": 2}
            if job.get("single_batch"):
                job["expected_rows"] = 1
            folder = dest / f"{model}-{method}-{fmt}"
            folder.mkdir()
            manifest = folder / "prompts.json"
            input_path = out / "preflight-inputs/eos-prompts.json" if method in ("dense", "int4-highlight") else Path(original["manifest"])
            save(manifest, json.loads(input_path.read_text())[:2])
            job.update(manifest=str(manifest), manifest_sha256=sha(manifest))
            checkpoint = original["command"][original["command"].index("--checkpoint") + 1]
            command = [python, "-m", "benchmarks.decode", *engine_flags(job, checkpoint, job.get("draft_ffn_checkpoint")),
                "--batch", "2", "--out", str(folder)]
            state.update(status="running", current=folder.name)
            save(out / "preflight.json", state)
            save(folder / "command.json", dict(argv=command, CUDA_VISIBLE_DEVICES=str(args.gpu)))
            with (folder / "engine.log").open("w") as log:
                subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT,
                    pass_fds=(lock.fileno(),), check=True)
            result = audit_job(job, folder)
            if method == "dense" and result["eos_stops"] == 0:
                raise ValueError("dense EOS preflight never exercised a natural stop")
            state["checks"].append(dict(name=folder.name, passed=True, audit=result))
            save(out / "preflight.json", state)
        if source_digest(source) != plan["source_sha256"]:
            raise ValueError("source changed during preflight")
        state.update(passed=True, status="complete")
        save(out / "preflight.json", state)
    except Exception as error:
        state.update(status="failed", error=str(error))
        save(out / "preflight.json", state)
        raise
    finally:
        if lock is not None:
            lock.close()


def report(args):
    out = Path(args.run).resolve()
    plan = json.loads((out / "plan.json").read_text())
    state = json.loads((out / "status.json").read_text()) if (out / "status.json").exists() else dict(state="prepared_not_launched", jobs=[])
    completed = [j for j in state["jobs"] if j["state"] == "complete" and j["kind"] == "eos-throughput"]
    values = {}
    for job in completed:
        value = audit_job(job, out / job["name"])
        key = (job["model"], job["ctx"], job["method"], job["format"])
        values.setdefault(key, []).append((job, value))
    cells = {}
    for key, entries in values.items():
        if len(entries) != REPETITIONS:
            continue
        if {j["rep"] for j, _ in entries} != {1, 2, 3} or len({v["batch"] for _, v in entries}) != 1:
            raise ValueError("repetition grid or batch differs")
        rates = [v["decode_tok_s"] for _, v in entries]
        cells[key] = dict(batch=entries[0][1]["batch"], median=statistics.median(rates),
            minimum=min(rates), maximum=max(rates), repetitions=rates,
            kv_reserved_fractions=[v.get("kv_reserved_fraction") for _, v in entries],
            eos_stops=sum(v["eos_stops"] for _, v in entries), cap_stops=sum(v["cap_stops"] for _, v in entries))
    lines = [f"<!-- {MARKER}:start -->", "### EOS-stopped Multi-LexSum throughput", "",
        f"Status: `{state['state']}`; {len(completed)}/{sum(j['kind'] == 'eos-throughput' for j in plan['jobs'])} process runs audited.", "",
        "Both models draw from the same 32 frozen cases, detailed-summary instruction, thinking off, "
        "and EOS or a 512-token cap. All prefills finish before timing; the window ends "
        "at the first request finish. The remaining requests drain outside timing. "
        "Three fresh processes per cell; median pooled tokens/s. All methods and indexer "
        "formats use the same capacity-checked batch within each model/context.", "",
        "Dense is tokens/s; other cells are ratios to its unrounded median. "
        "A cell stays empty until all three repetitions pass the raw-output audit.", ""]
    if plan.get("single_batch_remaining"):
        retained = sorted({(j["model"], j["ctx"]) for j in state["jobs"]
                           if j["kind"] == "eos-throughput" and j.get("expected_rows") == POOL})
        label = "; ".join(f"Qwen3-{model.upper()} at {ctx:,}" for model, ctx in retained)
        lines += ["Every new process measures one full batch: the first B frozen cases, "
            "identical across methods and repetitions. "
            f"Retained 32-batch results: {label or 'none'}. "
            "Each speedup uses a dense reference with the same batch protocol. "
            "The main 8B and FFN tables share their identical one-batch dense controls.", ""]
    for model in ("4b", "8b"):
        lines += [f"#### Qwen3-{model.upper()}", "", "| Input tokens | Batch | Indexer | Dense (tok/s) | Direct (×) | DV (×) | SV50 (×) | SV40 (×) | SV25 (×) |",
                  "|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
        for ctx in CONTEXTS:
            dense = cells.get((model, ctx, "dense", "bf16"))
            if model == "8b" and plan.get("single_batch_remaining"):
                dense = cells.get((model, ctx, "ffn-dense", "bf16"))
            for fmt in ("bf16", "mxfp4"):
                row = [f"{ctx:,}", str(dense["batch"]) if dense else "", fmt.upper(), f"{dense['median']:.2f}" if dense else ""]
                for method in ("direct", "dv", "sv50", "sv40", "sv25"):
                    cell = cells.get((model, ctx, method, fmt))
                    row.append(f"{cell['median'] / dense['median']:.3f}×" if cell and dense else "")
                lines.append("| " + " | ".join(row) + " |")
            lines.append("|  |  |  |  |  |  |  |  |  |")
        lines.append("")
    lines += ["#### 8B FFN comparison", "", "Dense BF16 verification for every FFN draft variant. BF16 indexer keys; backbone KV remains BF16.", ""]
    if plan.get("ffn_priority"):
        lines += ["Priority: one full batch per fresh process, three processes per cell. "
            "Use the first B frozen cases identically across all five methods. "
            + ("Dense controls are shared with the main 8B comparison. " if plan.get("single_batch_remaining") else
               "A separate one-batch dense baseline supplies these speedups; the main tables above retain their 32-batch protocol. ") +
            "All FFN comparisons finish before remaining main-method jobs start.", ""]
    lines += [
        "| Input tokens | Batch | Dense (tok/s) | BF16/full (×) | INT4/full (×) | BF16/Highlight (×) | INT4/Highlight (×) |",
        "|---:|---:|---:|---:|---:|---:|---:|"]
    for ctx in CONTEXTS:
        dense = cells.get(("8b", ctx, "ffn-dense" if plan.get("ffn_priority") else "dense", "bf16"))
        row = [f"{ctx:,}", str(dense["batch"]) if dense else "", f"{dense['median']:.2f}" if dense else ""]
        for method in FFN:
            cell = cells.get(("8b", ctx, method, "bf16"))
            row.append(f"{cell['median'] / dense['median']:.3f}×" if cell and dense else "")
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "Useful tokens per verification count only emitted tokens through EOS/cap. "
        "Draft acceptance excludes terminal rounds and rounds shortened by the output cap; "
        "their counts are retained in the audit. Native pre-clipping counters are diagnostic only.", "",
        f"<!-- {MARKER}:end -->"]
    text = "\n".join(lines) + "\n"
    temp = out / "report.md.tmp"
    temp.write_text(text)
    temp.replace(out / "report.md")
    save(out / "summary.json", dict(state=state["state"], complete_runs=len(completed),
        cells={"/".join(map(str, k)): v for k, v in cells.items()}))


def sync(args):
    """Pull evidence and update the primary HTML report without replacing prose."""
    import fcntl
    from benchmarks.report_html import ReportChangedError, update_report
    local, notes = Path(args.local).resolve(), Path(args.notes).resolve()
    if notes.suffix != ".html":
        raise ValueError("the primary report is HTML; pass notes/experiments.html")
    local.mkdir(parents=True, exist_ok=True)
    with (local / "report-sync.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                subprocess.run(["rsync", "-rlt", "--delay-updates", "--timeout=30",
                    "--exclude=cache", "--exclude=source", "--exclude=prompts", "--exclude=tokens-*.json",
                    "-e", "ssh -o BatchMode=yes -o ConnectTimeout=10",
                    args.remote.rstrip("/") + "/", str(local) + "/"], check=True, timeout=120)
                update_report(notes, local)
                status_path = local / "status.json"
                if status_path.exists():
                    state = json.loads(status_path.read_text())
                    summary = json.loads((local / "summary.json").read_text())
                    completed = sum(j["kind"] == "eos-throughput" and j["state"] == "complete" for j in state["jobs"])
                    if state["state"] in ("complete", "failed", "report_failed") and summary["complete_runs"] == completed:
                        if summary["state"] == state["state"] or state["state"] == "report_failed":
                            return
            except (subprocess.SubprocessError, ReportChangedError) as error:
                save(local / "sync-status.json", dict(status="retrying", error=str(error), updated=time.time()))
            time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", help="audit and render an existing campaign")
    sub = parser.add_subparsers(dest="action")
    prep = sub.add_parser("prepare")
    for name in ("runtime-root", "out", "checkpoint-4b", "checkpoint-8b", "quant", "data", "kernel-validation", "agreement-plan", "eos-pilot"):
        prep.add_argument("--" + name, required=True)
    check = sub.add_parser("preflight")
    check.add_argument("--campaign", required=True)
    check.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    revision = sub.add_parser("revise-ffn")
    revision.add_argument("--campaign", required=True)
    revision.add_argument("--out", required=True)
    mirror = sub.add_parser("sync")
    for name in ("local", "remote", "notes"):
        mirror.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if args.run:
        report(args)
    elif args.action == "prepare":
        prepare(args)
    elif args.action == "sync":
        sync(args)
    elif args.action == "preflight":
        preflight(args)
    elif args.action == "revise-ffn":
        revise_ffn(args)
    else:
        parser.error("choose prepare, sync, or --run")


if __name__ == "__main__":
    main()
