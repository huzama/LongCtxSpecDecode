# SPDX-License-Identifier: Apache-2.0
"""Check speculative acceptance, EOS/caps and committed KV without a GPU."""
import unittest
from unittest.mock import patch
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from jakobi import generate as autoregressive
from jakobi_test import OracleEngine
from parallel_draft import audit, generate, sources
import jakobi as base


class ScriptedEngine(OracleEngine):
    """Force a desired draft prefix while retaining prefix-sensitive verifier KV."""
    def __init__(self, hits):
        self.hits = hits

    def forward(self, ids, positions=None, mask=None):
        actual = super().forward(ids, positions, mask)
        if self.calls % 2 == 0:
            return actual
        history = self.cache[:self.prefix] + [ids[0]]
        proposals = []
        for _ in ids:
            token = self.logits(history).argmax().item()
            proposals.append(token)
            history.append(token)
        if self.hits < len(proposals):
            proposals[self.hits] = (proposals[self.hits] + 1) % 23
        return torch.nn.functional.one_hot(torch.tensor(proposals), 23).float()


@patch("torch.cuda.synchronize", lambda: None)
class Speculation(unittest.TestCase):
    def test_full_generation_and_kv(self):
        for seed in range(12):
            for eos in ({99}, {7}, {0}):
                for cap in (1, 2, 6, 7, 35):
                    prompt = [seed, 4, 7, 2]
                    expected = autoregressive(OracleEngine(), prompt, "ar", eos, list(range(23)), seed=seed, cap=cap)
                    result = generate(OracleEngine(), prompt, eos, list(range(23)), seed=seed, cap=cap)
                    self.assertEqual(result["tokens"], expected["tokens"])
                    self.assertEqual(result["finish_reason"], expected["finish_reason"])
                    self.assertEqual(result["forward_calls"], 2 * result["rounds"])

    def test_rejection_at_each_position_and_bonus(self):
        for hits in range(7):
            engine = ScriptedEngine(hits)
            result = generate(engine, [2, 4, 7, 2], {99}, list(range(23)), cap=30)
            expected = autoregressive(OracleEngine(), [2, 4, 7, 2], "ar", {99}, list(range(23)), cap=30)
            self.assertEqual(result["tokens"], expected["tokens"])
            first = result["traces"][0]
            self.assertEqual(first["matching_prefix"], hits)
            self.assertEqual(len(first["emitted"]), hits + 1)
            self.assertEqual(first["bonus_emitted"], hits == 6)

    def test_eos_inside_accepted_block(self):
        prompt = [2, 4, 7, 2]
        full = autoregressive(OracleEngine(), prompt, "ar", {99}, list(range(23)), cap=12)["tokens"]
        for eos in ({token} for token in set(full)):
            result = generate(ScriptedEngine(6), prompt, eos, list(range(23)), cap=12)
            expected = autoregressive(OracleEngine(), prompt, "ar", eos, list(range(23)), cap=12)
            self.assertEqual(result["tokens"], expected["tokens"])
            self.assertEqual(result["finish_reason"], expected["finish_reason"])


class Audit(unittest.TestCase):
    def test_equal_scores_do_not_hide_different_tokens(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, baseline = root / "candidate", root / "baseline"
            dest, reference = run / "4B-4096", baseline / "4B-4096/ar"
            dest.mkdir(parents=True)
            reference.mkdir(parents=True)
            manifest = {"rows": [{"id": "0"}]}
            (baseline / "prompts-4B-4096.json").write_text(json.dumps(manifest))
            settings = {key: "same" for key in ("model_revision", "versions", "gpu", "host", "rope_scaling",
                                                "dtype", "batch", "seed", "cap", "kv_capacity")}
            original = {**settings, "source_sha256": base.sources(), "manifest_sha256": base.digest(manifest)}
            candidate = {**settings, "source_sha256": sources(), "base_source_sha256": base.sources(),
                         "manifest_sha256": base.digest(manifest)}
            row = {"id": "0", "tokens": [1, 2, 3], "finish_reason": "cap", "score": 1.0,
                   "decode_seconds": 1.0, "prefill_seconds": 2.0, "rounds": 1, "forward_calls": 2,
                   "accepted_draft_tokens": 1, "query_tokens": 13, "new_compiled_graphs": 0,
                   "peak_allocated_bytes": 1024}
            for folder, config in ((dest, candidate), (reference, original)):
                (folder / "config.json").write_text(json.dumps(config))
                (folder / "finished.json").write_text(json.dumps({"state": "complete", "rows": 1}))
                (folder / "results.jsonl").write_text(json.dumps(row) + "\n")
            with patch("parallel_draft.check_plan", return_value=({"source_sha256": sources()}, baseline)), \
                    patch("parallel_draft.update_notes"):
                self.assertEqual(audit(run, "4B", 4096)["state"], "passed")
                row["tokens"] = [1, 2, 4]
                (dest / "results.jsonl").write_text(json.dumps(row) + "\n")
                result = audit(run, "4B", 4096)
                self.assertEqual(result["state"], "failed")
                self.assertEqual(result["mismatches"], [{"row": "0", "position": 2}])
                row["tokens"], row["new_compiled_graphs"] = [1, 2, 3], 1
                (dest / "results.jsonl").write_text(json.dumps(row) + "\n")
                result = audit(run, "4B", 4096)
                self.assertEqual(result["state"], "failed")
                self.assertFalse(result["timing_valid"])


if __name__ == "__main__":
    unittest.main()
