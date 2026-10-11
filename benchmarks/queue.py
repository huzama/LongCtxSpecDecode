"""Run a frozen experiment plan on idle, cooperatively locked GPUs 2/3."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
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
             *root.glob("tests/*.py"), *root.glob("outputs/molle/spotattention/**/*.py")]
    h = hashlib.sha256()
    for path in sorted(paths):
        h.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    return h.hexdigest()


def idle_gpus():
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                                   "--format=csv,noheader,nounits"], text=True)
    occupied = set(subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid",
                                           "--format=csv,noheader"], text=True).splitlines())
    result = set()
    for line in raw.splitlines():
        index, uuid, memory, utilization = [s.strip() for s in line.split(",")]
        if int(index) in (2, 3) and uuid not in occupied and int(memory) <= 1000 and int(utilization) == 0:
            result.add(int(index))
    return result


def eligible(jobs):
    complete = {j["name"] for j in jobs if j["state"] == "complete"}
    return next((j for j in jobs if j["state"] == "pending"
                 and set(j.get("depends_on", [])) <= complete), None)


def validate(job, dest):
    read = lambda name: json.loads((dest / name).read_text())
    kind = job["kind"]
    if kind == "validation":
        cases = list(ET.parse(dest / "validation.xml").iter("testcase"))
        assert len(cases) == job["expected_tests"] and all(not list(c) for c in cases), "skipped/failed GPU checks"
        return
    if kind == "selector":
        results = dest / "results"
        files = list(results.glob("slot*-layer*.json"))
        assert len(files) == job["expected_rows"]
        finished = json.loads((results / "finished.json").read_text())
        assert finished["records"] == job["expected_rows"] and finished["budget_records"] == 2 * job["expected_rows"]
        for path in files:
            row = json.loads(path.read_text())
            assert {v["budget"]["protocol"] for v in row["results"]} == {"matched", "paper"}
            for value in row["results"]:
                assert value["indexer_cache_format"] == "mxfp4"
                assert value["validation"]["highlight"]["passed"] is True
                assert value["validation"]["highlight"]["score_kernel"] == "native"
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
    for filename, digest in plan.get("input_sha256", {}).items():
        assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == digest, filename
    plan_hash = hashlib.sha256((out / "plan.json").read_bytes()).hexdigest()
    state_path = out / "status.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(
        jobs=[dict(j, state="pending") for j in plan["jobs"]], source_sha256=source_digest(root),
        plan_sha256=plan_hash, allowed_gpus=[2, 3], idle_seconds_required=60)
    jobs = state["jobs"]
    assert state["plan_sha256"] == plan_hash and state["source_sha256"] == source_digest(root), "plan/source changed"
    assert not any(j["state"] == "running" for j in jobs), "inspect existing children before restart"
    assert len({j["name"] for j in jobs}) == len(jobs)
    for job in jobs:
        assert not Path(job["name"]).is_absolute() and ".." not in Path(job["name"]).parts
        assert all(p in {j["name"] for j in jobs} for p in job.get("depends_on", []))
        for filename, digest in job.get("input_sha256", {}).items():
            assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == digest, filename
    env = {**os.environ, "VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_RPC_BASE_PATH": "/tmp", "TMPDIR": str(root / "outputs/tmp"),
        "TORCHINDUCTOR_CACHE_DIR": str(root / "outputs/torchinductor"), "TRITON_CACHE_DIR": str(root / "outputs/triton"),
        "VLLM_CACHE_ROOT": str(root / "outputs/vllm-cache"), "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:" + os.environ["PATH"], "PYTHONPATH": str(root),
        "OMP_NUM_THREADS": "8", "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"}
    active, idle_since = {}, {}

    def save(label):
        state.update(state=label, updated=now(), controller_pid=os.getpid())
        atomic_json(state_path, state)

    def report():
        with (out / "report.log").open("a") as log:
            code = subprocess.call([str(root / ".venv/bin/python"), "-m", "benchmarks.quant_report",
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
                    assert source_digest(root) == state["source_sha256"], "source changed during run"
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
            return
        if all(j["state"] == "complete" for j in jobs):
            save("complete")
            report()
            save("report_failed" if state.get("report_failed") else "complete")
            return
        if not failure:
            try:
                available = idle_gpus()
                for gpu in (2, 3):
                    if gpu not in available or gpu in active:
                        idle_since.pop(gpu, None)
                        continue
                    idle_since.setdefault(gpu, time.monotonic())
                    job = eligible(jobs)
                    if job is None or time.monotonic() - idle_since[gpu] < 60:
                        continue
                    if source_digest(root) != state["source_sha256"]:
                        state["report_failed"] = True
                        state["error"] = "source changed before job launch"
                        break
                    lock = (root / f"outputs/gpu-{gpu}-benchmark.lock").open("a")
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        lock.close()
                        continue
                    if gpu not in idle_gpus():
                        lock.close()
                        idle_since.pop(gpu, None)
                        continue
                    dest = out / job["name"]
                    dest.mkdir(parents=True, exist_ok=False)
                    atomic_json(dest / "command.json", dict(argv=job["command"], CUDA_VISIBLE_DEVICES=str(gpu)))
                    log = (dest / "engine.log").open("w")
                    proc = subprocess.Popen(job["command"], env={**env, "CUDA_VISIBLE_DEVICES": str(gpu)},
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),))
                    job.update(state="running", gpu=gpu, pid=proc.pid, started=now())
                    active[gpu] = (proc, job, lock, log)
                    print(now(), job["name"], "GPU", gpu, flush=True)
            except subprocess.CalledProcessError:
                idle_since.clear()
        save("draining_after_failure" if failure else "running" if active else "waiting_for_idle_gpu")
        time.sleep(15)


if __name__ == "__main__":
    main()
