# SPDX-License-Identifier: Apache-2.0
"""Check seed alignment, prompt boundaries, acceptance and committed KV."""
import random
import unittest
from unittest.mock import patch

from jakobi import generate as autoregressive
from jakobi_test import OracleEngine
from parallel_draft import generate as random_speculation
from parallel_draft_test import ScriptedEngine
from seeded_draft import Seeds, generate


class Initialization(unittest.TestCase):
    def test_carry_alignment_after_every_emission_count(self):
        for emitted in range(1, 8):
            state = Seeds("carry", [9, 2], list(range(23)), 42)
            state.update(list(range(10, 17)), emitted)
            guesses, origin = state.guesses([2], 6)
            expected = list(range(10 + emitted, 17))
            self.assertEqual(guesses[:len(expected)], expected)
            self.assertEqual(origin["seeded_input_positions"], min(len(expected), 5))

    def test_prompt_longest_then_latest_and_boundary(self):
        prompt = [1, 2, 3, 4, 9, 1, 2, 3, 4, 8, 7, 1, 2]
        state = Seeds("retrieval", prompt, list(range(23)), 42)
        guesses, origin = state.guesses([1, 2, 3, 4], 6)
        self.assertEqual(guesses[:4], [8, 7, 1, 2])
        self.assertEqual(origin["matched_ngram"], 4)
        self.assertEqual(origin["prompt_continuation_start"], 9)
        # A match ending at the final prompt token has no continuation.
        guesses, origin = state.guesses([1, 2], 6)
        self.assertEqual(origin["prompt_continuation_start"], 7)
        self.assertEqual(guesses, [3, 4, 8, 7, 1, 2])

    def test_fallback_and_fixed_random_stream(self):
        state = Seeds("retrieval", [1, 2, 3], list(range(23)), 42)
        rng = random.Random(42)
        for output in ([99], [98]):
            guesses, origin = state.guesses(output, 6)
            self.assertEqual(guesses, [rng.choice(list(range(23))) for _ in range(6)])
            self.assertEqual(origin["seeded_input_positions"], 0)
            self.assertIsNone(origin["prompt_continuation_start"])


@patch("torch.cuda.synchronize", lambda: None)
class Generation(unittest.TestCase):
    def test_greedy_and_committed_cache_for_all_policies(self):
        for seed in range(4):
            prompt = [seed, 4, 7, 2, seed, 4, 7, 2]
            for eos in ({99}, {0}, {7}):
                for cap in (1, 2, 6, 7, 35):
                    expected = autoregressive(OracleEngine(), prompt, "ar", eos, list(range(23)), cap=cap)
                    for mode in ("random", "carry", "retrieval"):
                        engine = OracleEngine()
                        result = generate(engine, prompt, eos, list(range(23)), mode=mode, seed=seed, cap=cap)
                        self.assertEqual(result["tokens"], expected["tokens"])
                        self.assertEqual(result["finish_reason"], expected["finish_reason"])
                        self.assertEqual(engine.cache, prompt + result["tokens"][:-1])

    def test_all_rejections_eos_and_cap(self):
        prompt = [2, 4, 7, 2]
        for hits in range(7):
            for mode in ("carry", "retrieval"):
                for eos in ({99}, {7}, {0}):
                    for cap in (2, 6, 7, 30):
                        result = generate(ScriptedEngine(hits), prompt, eos, list(range(23)), mode=mode, cap=cap)
                        expected = autoregressive(OracleEngine(), prompt, "ar", eos, list(range(23)), cap=cap)
                        self.assertEqual(result["tokens"], expected["tokens"])

    def test_random_control_matches_previous_algorithm(self):
        prompt = [2, 4, 7, 2]
        previous = random_speculation(OracleEngine(), prompt, {99}, list(range(23)), cap=35)
        current = generate(OracleEngine(), prompt, {99}, list(range(23)), cap=35)
        self.assertEqual(current["tokens"], previous["tokens"])
        for actual, expected in zip(current["traces"], previous["traces"]):
            for key in expected:
                self.assertEqual(actual[key], expected[key])


if __name__ == "__main__":
    unittest.main()
