"""Run a frozen experiment plan on its authorized, idle, cooperatively locked GPUs."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def source_digest(root):
    paths = [*root.glob("benchmarks/*.py"), *root.glob("highlight_spec/*.py"),
             *root.glob("tests/*.py"), *root.glob("outputs/molle/spotattention/**/*.py"),
             *root.glob("spotattention/**/*.py")]
    h = hashlib.sha256()
    for path in sorted(paths):
        h.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    return h.hexdigest()


def idle_gpus(allowed=(2, 3)):
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                                   "--format=csv,noheader,nounits"], text=True)
    occupied = set(subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid",
                                           "--format=csv,noheader"], text=True).splitlines())
    result = set()
    for line in raw.splitlines():
        index, uuid, memory, utilization = [s.strip() for s in line.split(",")]
        if int(index) in allowed and uuid not in occupied and int(memory) <= 1000 and int(utilization) == 0:
            result.add(int(index))
    return result


def eligible(jobs, gpu=None):
    complete = {j["name"] for j in jobs if j["state"] == "complete"}
    return next((j for j in jobs if j["state"] == "pending"
                 and (gpu is None or j.get("required_gpu", gpu) == gpu)
                 and set(j.get("depends_on", [])) <= complete), None)


def validate_jobs(jobs, planned_jobs, allowed):
    """Reject altered or unlaunchable schedules before inspecting any GPU."""
    if (not allowed or len(set(allowed)) != len(allowed)
            or any(type(g) is not int or g not in range(4) for g in allowed)):
        raise ValueError("invalid authorized GPU list")
    names = [j["name"] for j in jobs]
    planned = {j["name"]: j for j in planned_jobs}
    if (len(set(names)) != len(names) or len(planned) != len(planned_jobs)
            or set(names) != set(planned)):
        raise ValueError("queue jobs differ from the frozen plan")
    dependencies = {}
    for job in jobs:
        name = job["name"]
        if (not isinstance(name, str) or not name
                or any(p in ("", ".", "..") for p in name.split("/"))):
            raise ValueError("job name must be a relative path without aliases")
        if job.get("state") == "running":
            raise ValueError("inspect existing children before restart; running jobs are never retried")
        if job.get("state") not in ("pending", "complete", "failed"):
            raise ValueError("unknown job state: " + name)
        for key, value in planned[name].items():
            # The EOS runner records its resolved common batch in status.json.
            if key == "batch" and job["kind"] == "eos-throughput":
                continue
            if job.get(key) != value:
                raise ValueError("job differs from the frozen plan: " + name + "/" + key)
        if "required_gpu" in job and (type(job["required_gpu"]) is not int
                                      or job["required_gpu"] not in allowed):
            raise ValueError("job requires an unauthorized GPU: " + name)
        deps = set(job.get("depends_on", []))
        if not deps <= set(planned):
            raise ValueError("unknown job dependency: " + name)
        dependencies[name] = deps
    resolved = set()
    while dependencies:
        ready = {name for name, deps in dependencies.items() if deps <= resolved}
        if not ready:
            raise ValueError("cyclic job dependencies")
        resolved.update(ready)
        for name in ready:
            del dependencies[name]


def require_preflight(out, source_hash):
    preflight = json.loads((out / "preflight.json").read_text())
    expected = {"8b-dense-bf16", "4b-direct-bf16", "8b-sv50-mxfp4", "8b-int4-highlight-bf16"}
    checks = preflight.get("checks", [])
    if (preflight.get("passed") is not True or preflight.get("source_sha256") != source_hash
            or preflight.get("plan_sha256") != hashlib.sha256((out / "plan.json").read_bytes()).hexdigest()
            or len(checks) != 4 or {c["name"] for c in checks} != expected
            or any(c.get("passed") is not True for c in checks)):
        raise ValueError("GPU preflight has not passed for this frozen source and plan")


def validate(job, dest):
    read = lambda name: json.loads((dest / name).read_text())
    kind = job["kind"]
    if kind == "main-quality":
        from benchmarks.main_quality import validate as validate_quality
        return validate_quality(job, dest)
    if kind == "native-fp8-throughput":
        from benchmarks.native_fp8_throughput import validate as validate_fp8
        validate_fp8(job, dest)
        return
    if kind.startswith("main-"):
        from benchmarks.main_results import validate as validate_main
        return validate_main(job, dest)
    if kind == "eos-capacity":
        capacity = read("capacity.json")
        if capacity["batch"] != job["batch"] or capacity["maximum_batch"] < 1:
            raise ValueError("invalid capacity probe")
        return
    if kind == "eos-throughput":
        from benchmarks.eos_throughput import audit_job
        audit_job(job, dest)
        return
    if kind == "validation":
        cases = list(ET.parse(dest / "validation.xml").iter("testcase"))
        assert len(cases) == job["expected_tests"] and all(not list(c) for c in cases), "skipped/failed GPU checks"
        return
    if kind == "selector":
        results = dest / "results"
        cache_format = job.get("indexer_cache_format", "mxfp4")
        selectors = job.get("selectors", ["highlight"])
        config = json.loads((results / "config.json").read_text())
        assert config["indexer_cache_format"] == cache_format
        assert config["selectors"] == selectors
        files = list(results.glob("slot*-layer*.json"))
        assert len(files) == job["expected_rows"]
        finished = json.loads((results / "finished.json").read_text())
        assert finished["records"] == job["expected_rows"] and finished["budget_records"] == 2 * job["expected_rows"]
        for path in files:
            row = json.loads(path.read_text())
            assert {v["budget"]["protocol"] for v in row["results"]} == {"matched", "paper"}
            for value in row["results"]:
                assert value.get("indexer_cache_format", "bf16") == cache_format
                assert all(value["validation"][name]["passed"] is True for name in selectors)
                assert value["validation"]["highlight"]["cache_format"] == cache_format
                expected_kernel = "native" if cache_format == "mxfp4" else "tma"
                assert value["validation"]["highlight"]["score_kernel"] == expected_kernel
        return
    records = [json.loads(line) for line in (dest / "results.jsonl").read_text().splitlines()]
    assert len(records) == job["expected_rows"], "incomplete results"
    summary = read("summary.json" if kind == "throughput" else "finished.json")
    assert summary[{"throughput": "batches", "ruler": "rows", "longbench": "questions"}[kind]] == job["expected_rows"]
    configs = records if kind == "throughput" else [read("config.json")]
    for config in configs:
        assert config["indexer_cache_format"] == "mxfp4"
        assert config["selection"]["molle_kcache_format"] == "mxfp4"
        assert config["flash_attn_version"] == 4
        assert config["selection"]["molle_verify_fraction"] == job["verify_fraction"]
    if kind == "throughput":
        assert all(r["batch"] == job["batch"] and r["gen"] == 512 for r in records)
    else:
        assert len({r["id"] for r in records}) == job["expected_rows"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    out = args.run.resolve()
    owner = (out / "controller.lock").open("a")
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((out / "plan.json").read_text())
    runtime_root = Path(plan.get("runtime_root", root))
    allowed = tuple(plan.get("allowed_gpus", (2, 3)))
    if len(set(allowed)) != len(allowed) or not allowed:
        raise ValueError("invalid authorized GPU list")
    report_module = plan.get("report_module", "benchmarks.quant_report")
    if report_module not in {"benchmarks.quant_report", "benchmarks.selector_report", "benchmarks.eos_throughput", "benchmarks.main_results", "benchmarks.native_fp8_throughput", "benchmarks.main_quality"}:
        raise ValueError("unknown report module")
    if report_module == "benchmarks.main_quality":
        from benchmarks.main_quality import verify_plan as verify_quality
        verify_quality(out)
    if report_module == "benchmarks.native_fp8_throughput":
        from benchmarks.native_fp8_throughput import verify_plan as verify_fp8
        verify_fp8(out)
    if report_module == "benchmarks.main_results":
        from benchmarks.main_results import verify_plan
        verify_plan(out)
    if report_module == "benchmarks.eos_throughput":
        import importlib.metadata
        import socket
        if allowed != (2, 3) or socket.gethostname() != plan["host"]:
            raise ValueError("EOS campaign host/GPU authorization changed")
        if source_digest(root) != plan["source_sha256"]:
            raise ValueError("frozen source changed")
        for name, version in plan["runtime"].items():
            if importlib.metadata.version(name) != version:
                raise ValueError(f"runtime changed: {name}")
        require_preflight(out, plan["source_sha256"])
    if report_module not in ("benchmarks.main_results", "benchmarks.native_fp8_throughput"):
        for filename, digest in plan.get("input_sha256", {}).items():
            if hashlib.sha256((root / filename).read_bytes()).hexdigest() != digest:
                raise ValueError("input changed: " + filename)
    plan_hash = hashlib.sha256((out / "plan.json").read_bytes()).hexdigest()
    state_path = out / "status.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(
        jobs=[dict(j, state="pending") for j in plan["jobs"]], source_sha256=source_digest(root),
        plan_sha256=plan_hash, allowed_gpus=list(allowed), idle_seconds_required=60)
    jobs = state["jobs"]
    if state["plan_sha256"] != plan_hash or state["source_sha256"] != source_digest(root):
        raise ValueError("plan/source changed")
    validate_jobs(jobs, plan["jobs"], allowed)
    for job in jobs:
        for filename, digest in job.get("input_sha256", {}).items():
            if hashlib.sha256((root / filename).read_bytes()).hexdigest() != digest:
                raise ValueError("input changed: " + filename)
    env = {**os.environ, "VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_RPC_BASE_PATH": "/tmp", "TMPDIR": str(runtime_root / "outputs/tmp"),
        "TORCHINDUCTOR_CACHE_DIR": str(runtime_root / "outputs/torchinductor"), "TRITON_CACHE_DIR": str(runtime_root / "outputs/triton"),
        "VLLM_CACHE_ROOT": str(runtime_root / "outputs/vllm-cache"), "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:" + os.environ["PATH"], "PYTHONPATH": str(root),
        "OMP_NUM_THREADS": "8", "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1",
        "HF_HOME": str(runtime_root / "outputs/hf"), "HF_HUB_CACHE": str(runtime_root / "outputs/hf/hub")}
    active, idle_since = {}, {}

    def save(label):
        state.update(state=label, updated=now(), controller_pid=os.getpid())
        atomic_json(state_path, state)

    def report():
        with (out / "report.log").open("a") as log:
            code = subprocess.call([str(runtime_root / ".venv/bin/python"), "-m", report_module,
                                    "--run", str(out)], stdout=log, stderr=subprocess.STDOUT,
                                   env={**env, "CUDA_VISIBLE_DEVICES": ""})
        state["report_exit_code"] = code
        if code:
            state["report_failed"] = True

    save("waiting_for_idle_gpu")
    while True:
        changed = False
        for gpu, (proc, job, lock, log) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            job.update(exit_code=code, finished=now(), state="complete" if code == 0 else "failed")
            if code == 0:
                try:
                    validate(job, out / job["name"])
                    if source_digest(root) != state["source_sha256"]:
                        raise ValueError("source changed during run")
                except Exception as error:
                    job.update(state="failed", validation_error=str(error))
            lock.close()
            log.close()
            del active[gpu]
            idle_since.pop(gpu, None)
            changed = True
            print(now(), job["name"], job["state"], flush=True)
        failure = any(j["state"] == "failed" for j in jobs) or state.get("report_failed")
        if changed:
            save("draining_after_failure" if failure else "running")
            report()
            failure = failure or state.get("report_failed")
        if failure and not active:
            save("failed")
            report()
            return 1
        if all(j["state"] == "complete" for j in jobs):
            save("complete")
            report()
            save("report_failed" if state.get("report_failed") else "complete")
            return 1 if state.get("report_failed") else 0
        if not failure:
            try:
                available = idle_gpus(allowed)
                for gpu in allowed:
                    if gpu not in available or gpu in active:
                        idle_since.pop(gpu, None)
                        continue
                    idle_since.setdefault(gpu, time.monotonic())
                    job = eligible(jobs, gpu)
                    if job is None or time.monotonic() - idle_since[gpu] < 60:
                        continue
                    if source_digest(root) != state["source_sha256"]:
                        state["report_failed"] = True
                        state["error"] = "source changed before job launch"
                        break
                    lock = (runtime_root / f"outputs/gpu-{gpu}-benchmark.lock").open("a")
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        lock.close()
                        continue
                    if gpu not in idle_gpus(allowed):
                        lock.close()
                        idle_since.pop(gpu, None)
                        continue
                    dest = out / job["name"]
                    command = job["command"]
                    if job["kind"] == "eos-throughput":
                        from benchmarks.eos_throughput import resolve_batch
                        job["batch"], command = resolve_batch(job, out)
                    dest.mkdir(parents=True, exist_ok=False)
                    atomic_json(dest / "command.json", dict(argv=command, CUDA_VISIBLE_DEVICES=str(gpu)))
                    log = (dest / "engine.log").open("w")
                    proc = subprocess.Popen(command, env={**env, "CUDA_VISIBLE_DEVICES": str(gpu)},
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),))
                    job.update(state="running", gpu=gpu, pid=proc.pid, started=now())
                    active[gpu] = (proc, job, lock, log)
                    print(now(), job["name"], "GPU", gpu, flush=True)
            except subprocess.CalledProcessError:
                idle_since.clear()
        save("draining_after_failure" if failure else "running" if active else "waiting_for_idle_gpu")
        time.sleep(15)


if __name__ == "__main__":
    sys.exit(main())
