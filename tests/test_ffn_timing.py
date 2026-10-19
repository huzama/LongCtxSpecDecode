"""Audit full-batch timing without assuming every request finishes together."""
import argparse
import json
from pathlib import Path

import pytest

from benchmarks import ffn_timing


@pytest.fixture
def campaign(tmp_path):
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    jobs = []
    for variant in ffn_timing.VARIANTS:
        for rep in range(2):
            name = f"{variant}-r{rep}"
            jobs.append(dict(name=name, variant=variant, gpu=rep + 1))
            folder = tmp_path / name
            folder.mkdir()
            rows = []
            for slot in range(3):
                slots = [slot, (slot + 1) % 3]
                rows.append(dict(prompt_slots=slots, prompt_sha256=[str(s) for s in slots],
                    batch=2, decode_tokens=14 if variant == "dense" else 11,
                    decode_seconds=2, decode_window="all_prefills_complete_to_first_request_finish",
                    draft_tokens=12, accepted=9, drafts=2,
                    ffn_draft_validation=dict(graph_matches_eager=True, validation_batch=2)))
            (folder / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            write(folder / "summary.json", dict(ffn_draft_info=dict(
                graph_scope="step", step_graph_replays=12, extra_weight_bytes=0)))
            write(folder / "tokens-test.json", dict(tokens=[[1] * 8 for _ in range(6)]))
    for gpu in (1, 2):
        write(tmp_path / f"gpu-{gpu}.json", dict(status="complete"))
    write(tmp_path / "plan.json", dict(jobs=jobs, protocol=dict(batch=2, prompts=3, generated=8)))
    return argparse.Namespace(plan=str(tmp_path / "plan.json"))


def test_report_counts_only_tokens_emitted_before_first_finish(campaign):
    ffn_timing.report(campaign)
    summary = json.loads((Path(campaign.plan).parent / "summary.json").read_text())
    values = summary["variants"]
    assert values["dense"]["decode_tok_s"] == 7
    assert values["int4-highlight"]["decode_tok_s"] == 5.5
    assert values["int4-highlight"]["speedup_vs_dense_ar"] == 11 / 14
    assert values["int4-highlight"]["accepted_per_round"] == 4.5


@pytest.mark.parametrize("field,value,match", [
    ("prompt_slots", [0, 2], "cyclic prompt"),
    ("decode_tokens", 15, "decode window"),
    ("ffn_draft_validation", dict(graph_matches_eager=True, validation_batch=1), "full batch"),
])
def test_report_rejects_mismatched_batch_evidence(campaign, field, value, match):
    path = Path(campaign.plan).parent / "int4-highlight-r0/results.jsonl"
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    rows[0][field] = value
    path.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match=match):
        ffn_timing.report(campaign)
