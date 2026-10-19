# SPDX-License-Identifier: Apache-2.0
"""CPU algorithm checks. Optional reference path checks the released mask."""
import ast
import random
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from jakobi import LookaheadWindow, generate, lookahead_layout, truncate


class OracleEngine:
    """A prefix-sensitive oracle that rejects KV with a wrong computed prefix."""
    capture_margins = False
    bytes_per_position = 1

    @staticmethod
    def logits(prefix):
        token = sum((i + 3) * (x + 7) for i, x in enumerate(prefix)) % 23
        result = torch.zeros(23)
        result[token] = 1
        return result

    def prefill(self, ids):
        self.cache = list(ids)
        self.prefix = len(ids)
        self.calls = self.query_tokens = self.peak_query_tokens = 0
        return self.logits(ids)[None]

    def forward(self, ids, positions=None, mask=None):
        mask = torch.ones(len(ids),len(ids),dtype=torch.bool).tril() if mask is None else mask
        history = self.cache[:self.prefix]
        self.pending = [(token, history + [ids[j] for j in range(len(ids)) if mask[i,j]])
                        for i, token in enumerate(ids)]
        self.calls += 1
        self.query_tokens += len(ids)
        self.peak_query_tokens = max(self.peak_query_tokens, len(ids))
        return torch.stack([self.logits(path) for _, path in self.pending])

    def commit(self, indices):
        history = self.cache[:self.prefix]
        for i in indices:
            token, computed = self.pending[i]
            history.append(token)
            assert history == computed, (history, computed)
        self.cache = history
        self.prefix = len(history)


class Algorithms(unittest.TestCase):
    def test_window_and_pool(self):
        state = LookaheadWindow([1, 2, 3], random.Random(42))
        for sizes in ([9], [8, 9], [7, 8, 8], [6, 7, 7, 7]):
            self.assertEqual([len(x) for x in state.levels], sizes)
            state.update(42, list(range(sizes[-1])))
        self.assertEqual(len(state.pool[42][0]), 4)
        for i in range(20):
            state.add(42, [i] * 4)
        self.assertEqual(len(state.candidates(42)), 7)
        state.add(42, [19] * 4)
        self.assertEqual(len(state.candidates(42)), 7)

    def test_branches(self):
        levels = [[10] * 6, [11] * 7, [12] * 7, [13] * 7]
        ids, positions, mask, end = lookahead_layout(1, levels, [(2, 3, 4, 5), (6, 7, 8, 9)], 99)
        self.assertEqual(end, 28)
        self.assertEqual(positions[end:], [100, 101, 102, 103] * 2)
        self.assertTrue(mask[end:, 0].all())
        self.assertFalse(mask[end:, 1:end].any())
        self.assertFalse(mask[end:end+4, end+4:].any())
        self.assertFalse(mask[end+4:, end:end+4].any())
        self.assertTrue(torch.equal(mask[end:end+4, end:end+4], torch.ones(4,4,dtype=torch.bool).tril()))
        self.assertEqual(len(ids), 36)

    def test_eos_and_cap(self):
        self.assertEqual(truncate([1, 2, 3, 4], {3}, 4), [1, 2, 3])
        self.assertEqual(truncate([1, 2, 3, 4], {3}, 2), [1, 2])

    @patch('torch.cuda.synchronize', lambda: None)
    def test_full_generations_and_committed_cache(self):
        for seed in range(12):
            for eos in ({99}, {7}, {0}):
                prompt = [seed, 4, 7, 2]
                outputs = [generate(OracleEngine(), prompt, method, eos, list(range(23)),
                                    seed=seed, cap=35)['tokens']
                           for method in ('ar','jacobi','lookahead')]
                self.assertEqual(outputs[0], outputs[1])
                self.assertEqual(outputs[0], outputs[2])

    def test_released_reference_mask(self):
        source = Path('/tmp/jakobi-lookahead-reference/lade/models/modeling_llama.py')
        if not source.exists():
            self.skipTest('reference checkout not present')
        tree = ast.parse(source.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'j_make_causal_mask_multilevel')
        namespace = {'torch': torch}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
        reference = namespace[node.name]
        for sizes in ([9], [8,9], [7,8,8], [6,7,7,7]):
            for candidates in ([], [(1,2,3,4)], [(1,2,3,4)] * 7):
                if len(sizes) == 1 and candidates:
                    continue
                levels = [[1] * n for n in sizes]
                ids, _, actual, _ = lookahead_layout(1, levels, candidates, 0)
                guesses = [x for c in candidates for x in c] or None
                expected = reference(sizes, len(sizes)==1, 7, guesses, 4, False, False,
                                     torch.Size((1,len(ids))), torch.float32, 0, torch.device('cpu'))
                self.assertTrue(torch.equal(actual, expected[0,0] == 0), (sizes,len(candidates)))


if __name__ == '__main__':
    unittest.main()
