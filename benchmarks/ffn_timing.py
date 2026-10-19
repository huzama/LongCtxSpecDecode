"""Paired FFN serving throughput at batch 1 or matched KV capacity, GPUs 1/2."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from benchmarks.ffn_agreement import gpu_lock, save, sha

VARIANTS = ("dense", "bf16-full", "int4-full", "bf16-highlight", "int4-highlight")


def prepare(args):
    prior = json.loads(Path(args.agreement_plan).read_text())
    prompts = prior["prompts"]
    if args.large_batch:
        folder = Path(prompts[0]["path"]).parent
        prompts = []
        for slot in range(31):
            path = folder / f"longbench2-32768-{slot}.json"
            ids = json.loads(path.read_text())
            if len(ids) != 32768 or not all(type(x) is int and 0 <= x < 151936 for x in ids):
                raise ValueError(f"invalid throughput prompt {path}")
            prompts.append(dict(slot=slot, path=str(path), sha256=sha(path), tokens=len(ids)))
    root = Path(__file__).resolve().parents[1]
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = out / "source"
    for folder in ("highlight_spec", "benchmarks"):
        (source / folder).mkdir(parents=True)
        for path in (root / folder).glob("*.py"):
            shutil.copy2(path, source / folder / path.name)
        (source / folder / "__init__.py").touch(exist_ok=True)
    jobs = []
    for rep in range(2):
        for i, variant in enumerate(VARIANTS):
            jobs.append(dict(name=f"{variant}-r{rep}", variant=variant, rep=rep,
                             gpu=1 + (i + rep) % 2))
    plan = dict(agreement_plan_sha256=sha(args.agreement_plan),
        checkpoint=prior["checkpoint"], quant=prior["quant"], prompts=prompts,
        runtime=prior["runtime"], host=prior["host"], jobs=jobs,
        source_sha256={p.relative_to(source).as_posix(): sha(p) for p in sorted(source.rglob("*.py"))},
        protocol=dict(context=32768, generated=128, batch=1, prompts=6, drafts=6,
            process_repetitions=2, gpu_swap=True, draft_graph="step", seed=42,
            indexer="bf16", verify="dense bf16", backbone_kv="bf16", ignore_eos=True,
            speedup_denominator="pooled dense AR decode tok/s, balanced across both GPUs",
            selection_fraction=.07, score_refresh="round", recent_tokens=64,
            window="all_prefills_complete_to_first_request_finish",
            excluded=["loading", "prefill", "compilation", "warmup"],
            included=["draft", "verification", "sampling", "scheduler", "launch and control overhead"]))
    if args.large_batch:
        plan["protocol"].update(generated=512, batch=None, requested_batch=25,
            prompts=31, matched_capacity=True,
            batch_rule="min(25, INT4 Highlight capacity with full output and lookahead reserve)",
            batches_per_process=31, prompt_grouping="31 cyclic batches; every prompt appears equally often")
    if args.reuse_kernel_validation:
        import ast
        old = Path(args.reuse_kernel_validation).resolve()
        old_source = old.parent / "source"
        def definitions(path):
            names = {"MergedTensorStore", "ServingFFN", "fused_silu", "NativeBF16FFN", "FFNDraftModel"}
            return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names}
        if definitions(old_source / "highlight_spec/ffn_draft.py") != definitions(source / "highlight_spec/ffn_draft.py"):
            raise ValueError("native FFN implementation changed; cannot reuse validation")
        if sha(old_source / "highlight_spec/ffn_quant.py") != sha(source / "highlight_spec/ffn_quant.py"):
            raise ValueError("quantization implementation changed; cannot reuse validation")
        evidence = json.loads(old.read_text())
        if not evidence["passed"] or len(evidence["layers"]) != 36:
            raise ValueError("incomplete kernel validation")
        plan["reused_kernel_validation"] = dict(path=str(old), sha256=sha(old))
        save(out / "kernel-validation.json", dict(passed=True, layers=evidence["layers"],
             reused_from=plan["reused_kernel_validation"]))
    save(out / "plan.json", plan)
    print(out / "plan.json")


def validate(args):
    import torch
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
    from highlight_spec.ffn_quant import TensorStore, PackedFFN, relative_l2
    from highlight_spec.ffn_draft import ServingFFN
    torch.set_num_threads(8)
    torch.manual_seed(42)
    with torch.inference_mode():
        store = TensorStore(args.quant)
        workspace = marlin_make_workspace_new(torch.device("cuda"))
        checks = []
        for layer in range(36):
            native, reference = ServingFFN(store, layer, workspace), PackedFFN(store, layer, workspace)
            x = torch.randn((1, 4096), device="cuda", dtype=torch.bfloat16)
            eager = native(x)
            error = relative_l2(eager, reference(x))
            if error > .01:
                raise AssertionError(f"fused gate/up FFN differs at layer {layer}: {error}")
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                captured = native(x)
            g.replay()
            torch.testing.assert_close(captured, eager, rtol=0, atol=0)
            checks.append(dict(layer=layer, fused_vs_audited_relative_l2=error, graph_matches_eager=True))
            del g, native, reference, captured, eager
        save(args.out, dict(passed=True, layers=checks))


def worker(args):
    plan_path = Path(args.plan).resolve()
    out, source = plan_path.parent, Path(__file__).resolve().parents[1]
    plan = json.loads(plan_path.read_text())
    for name, expected in plan["source_sha256"].items():
        if sha(source / name) != expected:
            raise ValueError(f"source changed: {name}")
    for prompt in plan["prompts"]:
        if sha(prompt["path"]) != prompt["sha256"]:
            raise ValueError("prompt changed")
    runtime_root = Path(args.runtime_root).resolve()
    state_path = out / f"gpu-{args.gpu}.json"
    state = dict(status="waiting_for_validation" if args.gpu == 2 else "waiting_for_idle_gpu",
                 gpu=args.gpu, pid=os.getpid(), started=time.time())
    save(state_path, state)
    validation = out / "kernel-validation.json"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu),
           "VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0",
           "VLLM_RPC_BASE_PATH": "/tmp", "HF_HUB_OFFLINE": "1", "PYTHONPATH": str(source)}
    for key, folder in (("TMPDIR", "tmp"), ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
                        ("TRITON_CACHE_DIR", "triton"), ("VLLM_CACHE_ROOT", "vllm")):
        cache = out / f"cache-{args.gpu}" / folder
        cache.mkdir(parents=True, exist_ok=True)
        env[key] = str(cache)
    def execute(argv, log):
        with Path(log).open("x") as stream:
            subprocess.run(argv, cwd=source, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    python = str(runtime_root / ".venv/bin/python")
    try:
        if args.gpu == 1 and not validation.exists():
            lock = gpu_lock(runtime_root, args.gpu)
            try:
                state.update(status="validating_native_kernels")
                save(state_path, state)
                execute([python, "-u", "-m", "benchmarks.ffn_timing", "validate", "--quant", plan["quant"],
                         "--out", str(validation)], out / "kernel-validation.log")
            finally:
                lock.close()
        else:
            while not validation.exists():
                other = out / "gpu-1.json"
                if other.exists() and json.loads(other.read_text())["status"] == "failed":
                    raise RuntimeError("kernel validation failed on GPU 1")
                time.sleep(10)
        if not json.loads(validation.read_text())["passed"]:
            raise ValueError("kernel validation failed")
        protocol = plan["protocol"]
        batch = protocol["batch"]
        def command(variant, dest, batch):
            argv = [python, "-u", "-m", "benchmarks.decode", "--checkpoint", plan["checkpoint"],
                "--mode", "dense" if variant == "dense" else "highlight",
                "--prompts", str(Path(plan["prompts"][0]["path"]).parent),
                "--ctx", str(protocol["context"]), "--gen", str(protocol["generated"]),
                "--batch", str(batch), "--samples", str(protocol["prompts"]),
                "--spec-tokens", "6", "--draft-select", "round", "--out", str(dest)]
            if variant != "dense":
                precision, attention = variant.split("-")
                argv += ["--draft-attention", attention]
                if precision == "int4":
                    argv += ["--draft-ffn-checkpoint", plan["quant"]]
            return argv
        if protocol.get("matched_capacity"):
            capacity_path = out / "matched-capacity.json"
            if args.gpu == 1:
                state.update(status="waiting_for_idle_gpu", job="capacity-probe")
                save(state_path, state)
                lock = gpu_lock(runtime_root, args.gpu)
                try:
                    state.update(status="checking_capacity", job_started=time.time())
                    save(state_path, state)
                    dest = out / "capacity-probe"
                    dest.mkdir(exist_ok=False)
                    argv = command("int4-highlight", dest, protocol["requested_batch"]) + ["--capacity-only"]
                    save(dest / "command.json", dict(argv=argv, gpu=args.gpu))
                    execute(argv, dest / "engine.log")
                    capacity = json.loads((dest / "capacity.json").read_text())
                    batch = min(protocol["requested_batch"], capacity["maximum_batch"])
                    if batch < 2:
                        raise ValueError("insufficient KV capacity for the requested large-batch comparison")
                    save(capacity_path, dict(batch=batch, limiting_variant="int4-highlight",
                        probe=capacity, probe_sha256=sha(dest / "capacity.json")))
                finally:
                    lock.close()
            else:
                state.update(status="waiting_for_capacity")
                save(state_path, state)
                while not capacity_path.exists():
                    if json.loads((out / "gpu-1.json").read_text())["status"] == "failed":
                        raise RuntimeError("capacity check failed on GPU 1")
                    time.sleep(10)
            batch = json.loads(capacity_path.read_text())["batch"]
        for job in (j for j in plan["jobs"] if j["gpu"] == args.gpu):
            dest = out / job["name"]
            dest.mkdir(exist_ok=False)
            state.update(status="waiting_for_idle_gpu", job=job["name"])
            save(state_path, state)
            lock = gpu_lock(runtime_root, args.gpu)
            try:
                state.update(status="running", job_started=time.time())
                save(state_path, state)
                argv = command(job["variant"], dest, batch)
                save(dest / "command.json", dict(argv=argv, gpu=args.gpu, rep=job["rep"]))
                execute(argv, dest / "engine.log")
                records = [json.loads(s) for s in (dest / "results.jsonl").read_text().splitlines()]
                if len(records) != protocol["prompts"]:
                    raise ValueError("missing prompt results")
                if job["variant"] != "dense" and not records[0]["ffn_draft_validation"].get("graph_matches_eager"):
                    raise ValueError("draft graphs not validated")
                if job["variant"] != "dense" and records[0]["ffn_draft_validation"]["validation_batch"] != batch:
                    raise ValueError("draft graph validation used a different batch")
            finally:
                lock.close()
        state.update(status="complete", finished=time.time())
        save(state_path, state)
    except BaseException as error:
        state.update(status="failed", error=repr(error), finished=time.time())
        save(state_path, state)
        raise


def report(args):
    """Pool emitted-token counts and elapsed seconds; never average speedups."""
    root = Path(args.plan).resolve().parent
    plan = json.loads((root / "plan.json").read_text())
    protocol = dict(plan["protocol"])
    if protocol.get("matched_capacity"):
        protocol["batch"] = json.loads((root / "matched-capacity.json").read_text())["batch"]
    batch, samples, generated = (protocol[k] for k in ("batch", "prompts", "generated"))
    if any(json.loads((root / f"gpu-{gpu}.json").read_text())["status"] != "complete" for gpu in (1, 2)):
        raise ValueError("both timing workers must finish before reporting")
    summaries, all_prompts, dense_tokens = {}, {}, {}
    for variant in VARIANTS:
        jobs = [j for j in plan["jobs"] if j["variant"] == variant]
        if len(jobs) != 2 or {j["gpu"] for j in jobs} != {1, 2}:
            raise ValueError("each variant needs two processes with GPUs swapped")
        rows, rates, infos, mismatches = [], [], [], []
        for job in jobs:
            folder = root / job["name"]
            records = [json.loads(s) for s in (folder / "results.jsonl").read_text().splitlines()]
            if len(records) != samples or sorted(r["prompt_slots"][0] for r in records) != list(range(samples)):
                raise ValueError("missing or duplicate timed prompts")
            for r in records:
                if not (0 < r["decode_tokens"] <= batch * (generated - 1)) or r["decode_seconds"] <= 0 or r["batch"] != batch:
                    raise ValueError("wrong decode window")
                if r["decode_window"] != "all_prefills_complete_to_first_request_finish":
                    raise ValueError("wrong timing boundary")
                if batch == 1 and r["decode_tokens"] != generated - 1:
                    raise ValueError("incomplete batch-1 decode")
                slot = r["prompt_slots"][0]
                if r["prompt_slots"] != [(slot + i) % samples for i in range(batch)]:
                    raise ValueError("wrong cyclic prompt batch")
                expected = all_prompts.setdefault(slot, r["prompt_sha256"])
                if expected != r["prompt_sha256"]:
                    raise ValueError("different prompts across variants")
            if variant != "dense":
                info = json.loads((folder / "summary.json").read_text())["ffn_draft_info"]
                if info["graph_scope"] != "step" or info["step_graph_replays"] <= 0:
                    raise ValueError("timing did not use standard per-step graphs")
                if not records[0]["ffn_draft_validation"]["graph_matches_eager"]:
                    raise ValueError("graph validation missing")
                if batch > 1 and records[0]["ffn_draft_validation"]["validation_batch"] != batch:
                    raise ValueError("draft validation did not cover the full batch")
                infos.append(info)
            rates.append(sum(r["decode_tokens"] for r in records) / sum(r["decode_seconds"] for r in records))
            rows.extend(records)
            paths = list(folder.glob("tokens-*.json"))
            if len(paths) != 1:
                raise ValueError("missing emitted tokens")
            emitted = json.loads(paths[0].read_text())["tokens"]
            if len(emitted) != samples * batch or any(len(x) != generated for x in emitted):
                raise ValueError("wrong number of emitted tokens")
            if variant == "dense":
                dense_tokens[job["gpu"]] = emitted
            else:
                mismatches.append(sum(a != b for ref, actual in zip(dense_tokens[job["gpu"]], emitted)
                                      for a, b in zip(ref, actual)))
        count, seconds = sum(r["decode_tokens"] for r in rows), sum(r["decode_seconds"] for r in rows)
        entry = dict(decode_tokens=count, decode_seconds=seconds, decode_tok_s=count / seconds,
                     process_decode_tok_s=rates, dense_output_mismatches=mismatches,
                     extra_ffn_weight_bytes=[v["extra_weight_bytes"] for v in infos])
        if "kv_capacity" in rows[0]:
            entry["reserved_kv_fraction"] = [rows[0]["kv_capacity"]["reserved_fraction"],
                                            rows[samples]["kv_capacity"]["reserved_fraction"]]
        if variant != "dense":
            drafted, accepted, rounds = (sum(r[k] for r in rows) for k in ("draft_tokens", "accepted", "drafts"))
            entry.update(proposed_tokens=drafted, accepted_tokens=accepted, draft_rounds=rounds,
                         acceptance_fraction=accepted / drafted, accepted_per_round=accepted / rounds)
        summaries[variant] = entry
    dense = summaries["dense"]["decode_tok_s"]
    for entry in summaries.values():
        entry["speedup_vs_dense_ar"] = entry["decode_tok_s"] / dense
    save(root / "summary.json", dict(protocol=protocol, variants=summaries))
    print(json.dumps(summaries, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("prepare")
    p.add_argument("--agreement-plan", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--reuse-kernel-validation", help="reuse passed checks only if native FFN code is unchanged")
    p.add_argument("--large-batch", action="store_true", help="match the largest batch fitting INT4 Highlight, capped at 25; 512 outputs and 31 existing prompts")
    p = subs.add_parser("validate")
    p.add_argument("--quant", required=True)
    p.add_argument("--out", required=True)
    p = subs.add_parser("worker")
    p.add_argument("--plan", required=True)
    p.add_argument("--runtime-root", required=True)
    p.add_argument("--gpu", type=int, choices=(1, 2), required=True)
    p = subs.add_parser("report")
    p.add_argument("--plan", required=True)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == "__main__":
    main()
