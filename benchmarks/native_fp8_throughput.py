"""Matched-batch native FP8 Highlight throughput on the parent's authorized GPUs."""
import argparse
import importlib.metadata
import math
import os
from pathlib import Path
import shutil
import socket
import statistics
import sys
import time

from benchmarks.main_results import ROOT, METHODS, common_batch, load, save, sha, audit_throughput, campaign_gpus
from benchmarks.queue import source_digest

METHOD = "direct-native-fp8"
CONTEXTS = (32768, 65536, 130560)


def campaign_scope(plan):
    scope = plan.get("scope", dict(models=["4b", "8b"], contexts=list(CONTEXTS)))
    models, contexts = tuple(scope["models"]), tuple(scope["contexts"])
    if (not models or not contexts or len(set(models)) != len(models)
            or len(set(contexts)) != len(contexts) or not set(models) <= {"4b", "8b"}
            or not set(contexts) <= set(CONTEXTS)):
        raise ValueError("invalid native FP8 campaign scope")
    return models, contexts


def make_jobs(runtime, run, batches, manifests, models=("4b", "8b"), contexts=CONTEXTS,
              repetition_gpus=None):
    repetition_gpus=repetition_gpus or {'1':2,'2':3,'3':2}
    return [dict(name=f"throughput/{model}/{ctx}/r{rep}", kind="native-fp8-throughput",
        stage="throughput", method=METHOD, model=model, ctx=ctx, batch=batches[f"{model}/{ctx}"],
        repetition=rep, required_gpu=repetition_gpus[str(rep)],
        manifest=manifests[f"{model}/{ctx}"],
        command=[str(runtime/".venv/bin/python"), "-P", "-m", "benchmarks.native_fp8_throughput",
                 "--run", str(run), "--job", f"throughput/{model}/{ctx}/r{rep}"])
        for rep in (1, 2, 3) for model in models for ctx in contexts]


def prepare(args):
    parent = args.parent.resolve()
    previous = load(parent/"plan.json")
    models, contexts = campaign_scope(previous)
    allowed,repetition_gpus=campaign_gpus(previous)
    runtime = Path(previous["runtime_root"])
    run = args.run.resolve()
    if run.exists():
        raise ValueError("use a new run directory")
    gate = runtime/"outputs/runs/native-fp8-validation/validation.json"
    if load(gate).get("passed") is not True:
        raise ValueError("native FP8 implementation validation has not passed")
    batches, manifests, hashes = {}, {}, {str(gate): sha(gate)}
    checkpoints = {m: previous["inputs"]["checkpoints"][m] for m in models}
    # Reuse the main campaign's pinned checkpoint digests, then verify them
    # before queue launch. Do not hash unrelated quality datasets again.
    for path, digest in previous["input_sha256"].items():
        if any(Path(path).parent == Path(c) for c in checkpoints.values()):
            hashes[path] = digest
    for model in models:
        for ctx in contexts:
            key = f"{model}/{ctx}"
            batches[key] = common_batch(parent, model, ctx)
            src = parent/f"prompts/{model}/ctx-{ctx}/prompts.json"
            rows = load(src)
            if len(rows) < batches[key] or any(len(r["prompt_token_ids"]) != ctx for r in rows):
                raise ValueError("frozen prompt pool cannot fill the matched batch")
            dest = run/f"prompts/{model}/{ctx}.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            manifests[key] = str(dest)
            hashes[str(dest)] = sha(dest)
            for method in METHODS:
                capacity = parent/f"capacity/{model}/{ctx}/{method}/capacity.json"
                hashes[str(capacity)] = sha(capacity)
    source = run/"source"
    source.mkdir()
    for name in ("benchmarks", "highlight_spec", "tests"):
        shutil.copytree(ROOT/name, source/name, ignore=shutil.ignore_patterns("__pycache__"))
    import spotattention
    shutil.copytree(Path(spotattention.__file__).parent, source/"spotattention",
                    ignore=shutil.ignore_patterns("__pycache__"))
    plan = dict(schema="native-fp8-throughput-v2", host=socket.gethostname(),
        source_root=str(source), runtime_root=str(runtime), source_sha256=source_digest(source),
        parent_campaign=str(parent), allowed_gpus=allowed, repetition_gpus=repetition_gpus, idle_seconds=60,
        scope=dict(models=list(models), contexts=list(contexts)),
        runtime={p:importlib.metadata.version(p) for p in ("torch", "vllm", "triton", "spotattention")},
        input_sha256=hashes, checkpoints=checkpoints, batches=batches,
        report_module="benchmarks.native_fp8_throughput",
        jobs=make_jobs(runtime, run, batches, manifests, models, contexts,repetition_gpus),
        protocol=dict(backbone_cache="native_fp8_e4m3", weights="bf16", indexer="bf16",
            selection_fraction=.07, recent_tokens=64, output_cap=512, eos_respected=True,
            batches_per_process=1, repetitions=3, async_scheduling=False, prefix_caching=False,
            graph="FULL_DECODE_ONLY", batch="same calibrated common batch as parent campaign",
            timing="all prefills complete through first request finish; native engine timestamps",
            scale_policy="vllm_calculate_kv_scales", speedup_reference=None,
            comparison_note="Absolute throughput; repetitions use the same GPUs as the parent dense controls."))
    save(run/"plan.json", plan)
    return plan


def verify_plan(run):
    plan = load(run/"plan.json")
    models, contexts = campaign_scope(plan)
    _,repetition_gpus=campaign_gpus(plan)
    expected = {(m, c, r) for m in models for c in contexts for r in (1, 2, 3)}
    if (plan["schema"] != "native-fp8-throughput-v2"
            or plan["host"] != socket.gethostname() or len(plan["jobs"]) != len(expected)
            or {(j.get("model"), j.get("ctx"), j.get("repetition")) for j in plan["jobs"]} != expected
            or any(j.get("repetition") not in (1, 2, 3)
                   or j.get("required_gpu") != repetition_gpus.get(str(j.get("repetition")))
                   for j in plan["jobs"])):
        raise ValueError("native FP8 campaign authorization or scope differs")
    if source_digest(Path(plan["source_root"])) != plan["source_sha256"]:
        raise ValueError("native FP8 frozen source changed")
    for name, version in plan["runtime"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("runtime changed: " + name)
    for path, digest in plan["input_sha256"].items():
        if sha(path) != digest:
            raise ValueError("frozen input changed: " + path)
    return plan


def run_job(run, name):
    from benchmarks import decode
    from benchmarks.hierarchical_run import capacity_info
    plan = load(run/"plan.json")
    job = next(j for j in plan["jobs"] if j["name"] == name)
    allowed,repetition_gpus=campaign_gpus(plan)
    if (plan["schema"] != "native-fp8-throughput-v2"
            or job["required_gpu"] not in allowed
            or job["required_gpu"]!=repetition_gpus.get(str(job.get('repetition')))
            or os.environ.get("CUDA_VISIBLE_DEVICES") != str(job["required_gpu"])):
        raise ValueError("job is not assigned to its authorized GPU")
    dest = run/name
    p = argparse.ArgumentParser()
    decode.add_engine_args(p)
    args = p.parse_args(["--checkpoint", plan["checkpoints"][job["model"]],
                        "--mode", "direct", "--native-fp8-highlight"])
    args.ctx, args.batch, args.gen, args.eos_aware, args.out = job["ctx"], job["batch"], 512, True, str(dest)
    rows = load(job["manifest"])[:args.batch]
    prompts = [r["prompt_token_ids"] for r in rows]
    started = time.monotonic()
    llm = decode.build_engine(args)
    capacity = capacity_info(llm, args)  # Same 512+6-slot reserve as the parent AR controls.
    save(dest/"capacity.json", capacity)
    if not capacity["fits"]:
        raise ValueError("matched batch does not fit native FP8; no automatic batch reduction")
    decode.generate_batch(llm, args, prompts, 16)
    initial = llm.collective_rpc("native_fp8_info", kwargs={"begin": True})[0]
    outputs, timing, before, after = decode.generate_batch(llm, args, prompts, 512)
    cache = llm.collective_rpc("native_fp8_info")[0]
    config = llm.llm_engine.vllm_config
    result = dict(status="recorded", method=METHOD, model=job["model"], ctx=args.ctx,
        batch=args.batch, batches=1, repetition=job["repetition"], gpu=int(os.environ["CUDA_VISIBLE_DEVICES"]),
        source_sha256=decode.source_sha256(), manifest_path=job["manifest"], manifest_sha256=sha(job["manifest"]),
        prompt_ids=[r["id"] for r in rows], prompt_sha256=[decode.digest(p) for p in prompts],
        eos_ids=decode.configured_eos_ids(args.checkpoint), timing=timing, capacity=capacity, cache=cache,
        execution=dict(target_execution=cache["target_execution"], target_graph_replays=cache["graph_replays"],
                       draft_graph_replays=0, peak_reserved_bytes=cache["peak_reserved_bytes"]),
        initial_cache=initial, speculation=None, selection=decode.selection_record(before["selection"], after["selection"]),
        outputs=[dict(token_ids=list(o.outputs[0].token_ids), text=o.outputs[0].text,
                      finish_reason=o.outputs[0].finish_reason) for o in outputs],
        engine=dict(kv_cache_dtype=config.cache_config.cache_dtype,
            calculate_kv_scales=config.cache_config.calculate_kv_scales,
            flash_attn_version=config.attention_config.flash_attn_version,
            async_scheduling=config.scheduler_config.async_scheduling,
            prefix_caching=config.cache_config.enable_prefix_caching),
        wall_seconds=time.monotonic()-started)
    save(dest/"throughput.json", result)
    validate(job, dest)
    result["status"] = "complete"
    save(dest/"throughput.json", result)


def validate(job, dest):
    result = load(dest/"throughput.json")
    cache, engine = result["cache"], result["engine"]
    if (result["batch"] != job["batch"] or result["gpu"] != job["required_gpu"]
            or result["model"] != job["model"] or result["repetition"] != job["repetition"]
            or result["manifest_path"] != job["manifest"]
            or cache["backbone_cache"] != "native_fp8_e4m3" or cache["indexer_cache"] != "bf16"
            or cache["kv_cache_groups"] != 1 or not cache["layers"]
            or engine != dict(kv_cache_dtype="fp8_e4m3", calculate_kv_scales=True,
                              flash_attn_version=4, async_scheduling=False, prefix_caching=False)):
        raise ValueError("native FP8 run configuration differs")
    if any(row["element_bytes"] != 1 or any(not math.isfinite(s) or s <= 0 for s in row["scales"].values())
           for row in cache["layers"]):
        raise ValueError("invalid native FP8 cache or scales")
    if not result["selection"] or not .06 < result["selection"]["retained_fraction"] < .08:
        raise ValueError("Highlight selection differs from the 7% protocol")
    audit = audit_throughput(job, result, speculative=False, check_cache_transactions=False)
    save(dest/"audit.json", audit)
    return audit


def report(run):
    plan = load(run/"plan.json")
    models, contexts = campaign_scope(plan)
    status = load(run/"status.json")
    rows = [dict(model=j["model"], ctx=j["ctx"], batch=j["batch"], repetition=j["repetition"],
                 gpu=j["required_gpu"], **validate(j, run/j["name"]))
            for j in status["jobs"] if j["state"] == "complete"]
    cells = []
    for model in models:
        for ctx in contexts:
            values = [r["decode_tok_s"] for r in rows if (r["model"], r["ctx"]) == (model, ctx)]
            cells.append(dict(model=model, ctx=ctx, repetitions=len(values),
                mean_tok_s=statistics.mean(values) if len(values) == 3 else None,
                std_tok_s=statistics.stdev(values) if len(values) == 3 else None))
    note = ("Absolute native FP8 throughput; matched GPU assignments with the dense controls."
            if plan["schema"] == "native-fp8-throughput-v2" else
            "Historical absolute native FP8 throughput on GPUs 0/1; dense controls ran on GPUs 2/3.")
    save(run/"report.json", dict(state=status["state"], measurements=rows, cells=cells,
        speedup_reference=None, note=note))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--job")
    args = parser.parse_args()
    if args.prepare:
        if args.parent is None: parser.error("--prepare requires --parent")
        prepare(args)
    elif args.check: verify_plan(args.run)
    elif args.job: run_job(args.run, args.job)
    else: report(args.run)


if __name__ == "__main__":
    main()
