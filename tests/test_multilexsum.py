"""Reject post-EOS tokens and mismatched workload evidence."""
import copy

import pytest

from benchmarks.multilexsum import audit_records, digest, output_audit


def test_eos_at_cap_is_a_natural_stop():
    audit = output_audit([1, 2, 9], [9], 3, "stop")
    assert audit["eos_position"] == 3
    assert audit["cap_hit"] is False


@pytest.mark.parametrize("ids,reason", [
    ([1, 9, 2], "stop"), ([9, 9], "stop"), ([1], "stop"),
    ([9], "length"), ([], "length"), ([1, 2, 3, 4], "length"),
])
def test_invalid_stopping_is_rejected(ids, reason):
    with pytest.raises(ValueError):
        output_audit(ids, [9], 3, reason)


@pytest.fixture
def evidence():
    plan = dict(context=4, max_output_tokens=4, eos_token_ids=[9],
                thinking=False, ignore_eos=False, min_tokens=0)
    prompts = [dict(id=str(i), prompt_token_ids=[1, 2, 3, 4],
                    prompt_sha256=digest([1, 2, 3, 4])) for i in range(10)]
    results = [dict(id=str(i), input_tokens=4, output_token_ids=[5, 6, 9],
                    output_tokens=3, eos_position=3, cap_hit=False, post_eos_tokens=0,
                    finish_reason="stop", timed_decode_tokens=2, decode_s=1., text="Summary.")
               for i in range(10)]
    return plan, prompts, results


def test_early_eos_and_capped_cases_are_both_retained(evidence):
    plan, prompts, results = evidence
    results[0].update(output_token_ids=[5, 6, 7, 8], output_tokens=4,
                      eos_position=None, cap_hit=True, finish_reason="length", timed_decode_tokens=3)
    audit = audit_records(plan, prompts, results)
    assert (audit["cases"], audit["eos_stopped"], audit["capped"]) == (10, 9, 1)
    assert audit["timed_decode_tokens"] == 21
    assert audit["decode_tok_s"] == 2.1


@pytest.mark.parametrize("change", ["missing", "reordered", "duplicate", "prompt", "counts", "time", "thinking"])
def test_mismatched_evidence_is_rejected(evidence, change):
    plan, prompts, results = copy.deepcopy(evidence)
    if change == "missing":
        results.pop()
    elif change == "reordered":
        results.reverse()
    elif change == "duplicate":
        prompts[1] = prompts[0]
    elif change == "prompt":
        prompts[0]["prompt_token_ids"][0] = 100
    elif change == "counts":
        results[0]["output_tokens"] = 4
    elif change == "time":
        results[0]["decode_s"] = 0
    else:
        plan["thinking"] = True
    with pytest.raises(ValueError):
        audit_records(plan, prompts, results)
