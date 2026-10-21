"""Queue restoration must preserve frozen commands, dependencies and GPU placement."""
import copy

import pytest

from benchmarks.queue import eligible, validate_jobs


def schedule():
    plan = [dict(name="capacity/model", kind="main-capacity", command=["python", "capacity"]),
            dict(name="throughput/model/r1", kind="main-throughput", command=["python", "decode"],
                 required_gpu=2, depends_on=["capacity/model"])]
    return plan, [dict(copy.deepcopy(j), state="pending") for j in plan]


def test_resume_preserves_dependencies_and_failure():
    plan, jobs = schedule()
    jobs[0].update(state="complete", exit_code=0, gpu=3, pid=123)
    validate_jobs(jobs, plan, [2, 3])
    assert eligible(jobs, 2)["name"] == "throughput/model/r1"
    assert eligible(jobs, 3) is None
    jobs[1]["state"] = "failed"
    validate_jobs(jobs, plan, [2, 3])
    assert eligible(jobs, 2) is None


@pytest.mark.parametrize("field,value", [
    ("command", ["python", "different"]), ("required_gpu", 3), ("depends_on", []),
])
def test_restored_job_cannot_change_the_plan(field, value):
    plan, jobs = schedule()
    jobs[1][field] = value
    with pytest.raises(ValueError, match="frozen plan"):
        validate_jobs(jobs, plan, [2, 3])


@pytest.mark.parametrize("state", ["running", "interrupted", "cancelled", "retry"])
def test_interrupted_or_unknown_jobs_are_not_relaunched(state):
    plan, jobs = schedule()
    jobs[1]["state"] = state
    with pytest.raises(ValueError, match="restart|unknown job state"):
        validate_jobs(jobs, plan, [2, 3])


@pytest.mark.parametrize("name", ["", "/absolute", "../escape", "a/../b", "./alias", "a//b", "a/"])
def test_job_paths_cannot_alias_or_escape(name):
    plan = [dict(name=name, kind="main-capacity", command=["python"])]
    with pytest.raises(ValueError, match="relative path"):
        validate_jobs([dict(plan[0], state="pending")], plan, [2, 3])


def test_missing_duplicate_and_cyclic_dependencies_fail_before_dispatch():
    plan, jobs = schedule()
    with pytest.raises(ValueError, match="frozen plan"):
        validate_jobs(jobs + [jobs[0]], plan, [2, 3])
    with pytest.raises(ValueError, match="frozen plan"):
        validate_jobs(jobs[:1], plan, [2, 3])
    for deps, message in [(["missing"], "unknown job dependency"),
                          ([plan[1]["name"]], "cyclic job dependencies")]:
        plan[0]["depends_on"] = deps
        jobs[0]["depends_on"] = deps
        with pytest.raises(ValueError, match=message):
            validate_jobs(jobs, plan, [2, 3])


@pytest.mark.parametrize("allowed", [[], [2, 2], [4], [True], [3]])
def test_unavailable_gpu_assignments_are_rejected(allowed):
    plan, jobs = schedule()
    with pytest.raises(ValueError, match="GPU"):
        validate_jobs(jobs, plan, allowed)


def test_legacy_eos_batch_resolution_is_preserved():
    plan = [dict(name="timing", kind="eos-throughput", batch=1, command=["python"])]
    validate_jobs([dict(plan[0], state="complete", batch=19)], plan, [2, 3])
