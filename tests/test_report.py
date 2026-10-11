import hashlib
import json

import pytest

from benchmarks.report import agreement, read_json, summarize


def test_speedup_uses_ratio_of_median_rates():
    def rows(rates):
        return [[{"decode_tokens": n, "decode_seconds": 1}] for n in rates]

    # Median of paired ratios is 2; the protocol's ratio of medians is 4.
    assert summarize(rows([2, 100, 8]))["tok_s"] / summarize(rows([1, 2, 4]))["tok_s"] == 4


def test_agreement_requires_matching_prompts_and_detects_length():
    a = {"prompt_slots": [0], "prompt_sha256": ["a"], "tokens": [[1, 2]]}
    b = {**a, "tokens": [[1]]}
    assert agreement(a, b)["mismatches"] == [{"row": 0, "slot": 0, "position": 1}]
    with pytest.raises(ValueError, match="prompt_sha256"):
        agreement(a, {**b, "prompt_sha256": ["b"]})


def test_legacy_token_file_recovers_prompt_hashes(tmp_path):
    path = tmp_path / "tokens.json"
    path.write_text(json.dumps({"tokens": [[3]], "prompts": [[1, 2]], "prompt_slots": [0]}))
    assert read_json(path)["prompt_sha256"] == [hashlib.sha256(b"[1,2]").hexdigest()]


def test_agreement_separates_forced_post_eos_tokens():
    a = {"prompt_slots": [0, 1], "prompt_sha256": ["a", "b"], "tokens": [[1, 9, 2], [1, 2, 9]]}
    b = {**a, "tokens": [[1, 9, 3], [1, 3, 9]]}
    result = agreement(a, b, eos_token_id=9)
    assert result["identical"] == 0
    assert result["identical_through_eos"] == 1
