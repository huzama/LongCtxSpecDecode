"""Audit, freeze and run the agreed six-prompt FFN agreement diagnostic.

No serving throughput is measured. GPU indices must be explicitly supplied.
"""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

VARIANTS = ("bf16/full", "int4/full", "bf16/highlight", "int4/highlight")
SOURCES = ("benchmarks/ffn_agreement.py", "highlight_spec/ffn_quant.py", "highlight_spec/ffn_probe.py")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def audit(args):
    import torch
    from highlight_spec.ffn_quant import TensorStore, check_quant_config, digest
    torch.set_num_threads(8)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    backbone, original, quant = map(TensorStore, (args.checkpoint, args.original, args.quant))
    check_quant_config(args.quant)
    evidence = {}
    keys = sorted(k for k in backbone.files if ".indexer." not in k)
    if set(keys) != set(original.files):
        raise ValueError("Highlight and original backbone tensor names differ")
    for i, key in enumerate(keys):
        a, b = backbone.get(key), original.get(key)
        if not torch.equal(a, b):
            raise ValueError(f"Highlight backbone differs from original: {key}")
        evidence[key] = digest(a)
        if i % 50 == 0:
            print(f"backbone identity {i + 1}/{len(keys)}", flush=True)
    shared = sorted(set(keys) & set(quant.files))
    for key in shared:
        if not torch.equal(backbone.get(key), quant.get(key)):
            raise ValueError(f"quantized checkpoint's unquantized tensor differs: {key}")
    if not shared or "model.embed_tokens.weight" not in shared:
        raise ValueError("no unquantized embedding identity evidence")
    # Record every tensor used by the draft, including indexer and packed FFNs.
    packed = {k: digest(quant.get(k)) for k in sorted(quant.files) if ".mlp." in k}
    indexer = {k: digest(backbone.get(k)) for k in sorted(backbone.files) if ".indexer." in k}
    prompts = []
    for slot in range(6):
        path = Path(args.prompts).resolve() / f"longbench2-32768-{slot}.json"
        ids = json.loads(path.read_text())
        if len(ids) != 32768 or not all(type(x) is int and 0 <= x < 151936 for x in ids):
            raise ValueError(f"invalid frozen prompt {path}")
        prompts.append(dict(slot=slot, path=str(path), sha256=sha(path), tokens=len(ids)))
    root = Path(__file__).resolve().parents[1]
    snapshot = out / "source"
    for name in SOURCES:
        dest = snapshot / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, dest)
        (dest.parent / "__init__.py").touch()
    runtime = {n: importlib.metadata.version(n) for n in
               ("torch", "transformers", "vllm", "compressed-tensors", "safetensors")}
    import spotattention.serve.indexer as indexer_module
    plan = dict(checkpoint=str(Path(args.checkpoint).resolve()),
        original=str(Path(args.original).resolve()), quant=str(Path(args.quant).resolve()),
        prompts=prompts, source_sha256={n: sha(snapshot / n) for n in SOURCES},
        runtime=runtime, host=socket.gethostname(),
        indexer_source=str(Path(indexer_module.__file__).resolve()), indexer_sha256=sha(indexer_module.__file__),
        config_sha256={p: sha(Path(p) / "config.json") for p in (args.checkpoint, args.original, args.quant)},
        protocol=dict(context=32768, generated=128, offsets=list(range(0, 128, 16)), drafts=6,
            batch=1, yarn=2, seed=42, ignore_eos=True, variants=VARIANTS,
            indexer_format="bf16", backbone_kv="bf16", selection_fraction=.07,
            recent_tokens=64, score_refresh="once per six-token draft", allowed_gpus=[1, 2],
            packed_kernel_relative_l2_limit=.01, serving_throughput=False),
        identity=dict(backbone_equal_original=True, unquantized_quant_tensors_equal=shared,
                      backbone=evidence, indexer=indexer, packed_ffn=packed))
    save(out / "plan.json", plan)
    print(f"AUDIT PASSED: {len(keys)} backbone tensors; {len(shared)} unchanged quant-checkpoint tensors", flush=True)


def available(gpu):
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                                   "--format=csv,noheader,nounits"], text=True)
    occupied = set(subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid",
                                           "--format=csv,noheader"], text=True).splitlines())
    for row in raw.splitlines():
        index, uuid, memory, util = [x.strip() for x in row.split(",")]
        if int(index) == gpu:
            return uuid not in occupied and int(memory) <= 1000 and int(util) == 0
    return False


def gpu_lock(root, gpu):
    lock = (root / f"outputs/gpu-{gpu}-benchmark.lock").open("a")
    since = None
    while True:
        if not available(gpu):
            since = None
        else:
            since = time.monotonic() if since is None else since
            if time.monotonic() - since >= 60:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    if available(gpu):
                        return lock
                    fcntl.flock(lock, fcntl.LOCK_UN)
                except BlockingIOError:
                    pass
                since = None
        time.sleep(10)


def run(args):
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text())
    root = Path(args.runtime_root).resolve()
    snapshot = Path(__file__).resolve().parents[1]
    if args.gpu not in plan["protocol"]["allowed_gpus"]:
        raise ValueError("GPU not authorized in this plan")
    if len(set(args.slots)) != len(args.slots) or not set(args.slots) <= set(range(6)):
        raise ValueError("slots must be distinct values from 0 through 5")
    for name, expected in plan["source_sha256"].items():
        if sha(snapshot / name) != expected:
            raise ValueError(f"source differs from frozen plan: {name}")
    if sha(plan["indexer_source"]) != plan["indexer_sha256"]:
        raise ValueError("native indexer changed since audit")
    for path, expected in plan["config_sha256"].items():
        if sha(Path(path) / "config.json") != expected:
            raise ValueError(f"checkpoint config changed: {path}")
    for name, version in plan["runtime"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"runtime changed: {name}")
    if socket.gethostname() != plan["host"]:
        raise ValueError("host differs from the audited host")
    out = plan_path.parent / f"gpu-{args.gpu}"
    out.mkdir(exist_ok=False)
    state = dict(status="waiting_for_idle_gpu", gpu=args.gpu, slots=args.slots,
                 pid=os.getpid(), plan_sha256=sha(plan_path), started=time.time())
    save(out / "status.json", state)
    lock = gpu_lock(root, args.gpu)
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        for key, folder in (("TMPDIR", "tmp"), ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
                            ("TRITON_CACHE_DIR", "triton"), ("VLLM_CACHE_ROOT", "vllm")):
            path = out / folder
            path.mkdir(exist_ok=True)
            os.environ[key] = str(path)
        os.environ["HF_HUB_OFFLINE"] = "1"
        import torch
        from highlight_spec.ffn_probe import Probe
        from highlight_spec.ffn_quant import accepted_prefix
        torch.set_num_threads(8)
        torch.manual_seed(42)
        state.update(status="loading", gpu_name=torch.cuda.get_device_name())
        save(out / "status.json", state)
        probe = Probe(plan["checkpoint"], plan["quant"], 32768)
        for slot in args.slots:
            item = plan["prompts"][slot]
            if sha(item["path"]) != item["sha256"]:
                raise ValueError("prompt changed since audit")
            ids = json.loads(Path(item["path"]).read_text())
            state.update(status="prefilling", slot=slot)
            save(out / "status.json", state)
            backend_check = probe.validate_dense_backend(ids) if slot == args.slots[0] else None
            cache = probe.prefill(ids[:-1])
            anchor, continuation, rows = ids[-1], [], []
            probe.ffn_checks = []
            for offset in range(128):
                base = 32767 + offset
                if cache.get_seq_length() != base:
                    raise AssertionError("dense prefix length drifted")
                if offset in plan["protocol"]["offsets"]:
                    state.update(status="measuring_agreement", offset=offset)
                    save(out / "status.json", state)
                    if offset == 0:
                        probe.check_ffn = True
                        probe.set_variant("bf16/full")
                        probe.forward([anchor], cache)
                        probe.check_ffn = False
                        cache.crop(base)
                        save(out / f"ffn-{slot}.json", probe.ffn_checks)
                    reference = probe.rollout(cache, base, anchor, "bf16/full")
                    for variant in VARIANTS:
                        teacher = probe.rollout(cache, base, anchor, variant, teacher=reference)
                        drafts = probe.rollout(cache, base, anchor, variant)
                        # Actual six-position dense verification, including rewritten draft KV.
                        probe.set_variant("bf16/full")
                        verified = probe.forward([anchor, *drafts[:-1]], cache, all_logits=True).argmax(-1).tolist()
                        cache.crop(base)
                        if variant == "bf16/full" and (teacher != reference or drafts != reference or verified != reference):
                            raise AssertionError("BF16 dense control mismatch; resolve before interpreting INT4")
                        record = dict(slot=slot, offset=offset, variant=variant,
                            reference=reference, teacher_predictions=teacher, proposed=drafts, verified=verified,
                            matches=sum(a == b for a, b in zip(teacher, reference)),
                            accepted=accepted_prefix(drafts, verified))
                        rows.append(record)
                        with (out / "rounds.jsonl").open("a") as stream:
                            stream.write(json.dumps(record) + "\n")
                    restored = probe.rollout(cache, base, anchor, "bf16/full")
                    if restored != reference:
                        raise AssertionError("dense prefix restoration failed")
                    print(f"GPU {args.gpu}: prompt {slot}, offset {offset} complete", flush=True)
                probe.set_variant("bf16/full")
                anchor = int(probe.forward([anchor], cache)[-1].argmax())
                continuation.append(anchor)
            save(out / f"prompt-{slot}.json", dict(slot=slot, prompt_sha256=item["sha256"],
                continuation=continuation, rows=rows, ffn_checks=probe.ffn_checks,
                dense_backend_check=backend_check, prefix_restoration_passed=True))
            del cache
        state.update(status="complete", finished=time.time())
        save(out / "status.json", state)
    except BaseException as error:
        state.update(status="failed", error=repr(error), finished=time.time())
        save(out / "status.json", state)
        raise
    finally:
        lock.close()


def report(args):
    root = Path(args.plan).resolve().parent
    prompts = [json.loads(path.read_text()) for path in sorted(root.glob("gpu-*/prompt-*.json"))]
    if sorted(p["slot"] for p in prompts) != list(range(6)):
        raise ValueError("report requires all six distinct, completed prompts")
    rows = [r for p in prompts for r in p["rows"]]
    summary = {}
    for variant in VARIANTS:
        selected = [r for r in rows if r["variant"] == variant]
        if len(selected) != 48 or len({(r["slot"], r["offset"]) for r in selected}) != 48:
            raise ValueError("missing or duplicate rounds")
        summary[variant] = dict(matches=sum(r["matches"] for r in selected), predictions=288,
            agreement_percent=100 * sum(r["matches"] for r in selected) / 288,
            accepted=sum(r["accepted"] for r in selected), rounds=48,
            mean_accepted=sum(r["accepted"] for r in selected) / 48)
    save(root / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("audit")
    for name in ("checkpoint", "original", "quant", "prompts", "out"):
        p.add_argument("--" + name, required=True)
    p = commands.add_parser("run")
    p.add_argument("--plan", required=True)
    p.add_argument("--runtime-root", required=True)
    p.add_argument("--gpu", type=int, choices=(1, 2), required=True)
    p.add_argument("--slots", type=int, nargs="+", required=True)
    p = commands.add_parser("report")
    p.add_argument("--plan", required=True)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == "__main__":
    main()
