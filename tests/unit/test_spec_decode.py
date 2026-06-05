"""Unit tests for the speculative-decoding engine (demos/spec-decode/spec_decode.py)."""
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_DIR = REPO_ROOT / "demos" / "spec-decode"
for p in (REPO_ROOT, SPEC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import spec_decode as sd


class TestSpeculativeDecode(unittest.TestCase):
    def setUp(self):
        self.prompt = [1, 2, 3]
        self.T = list(range(10, 24))          # 14-token target continuation
        self.wrong = {2, 5}                    # absolute positions the draft gets wrong
        self.k = 4
        self.max_tokens = 11
        self.draft_next, self.target_next = sd.mock_models(
            prompt_len=len(self.prompt),
            target_continuation=self.T,
            draft_wrong_positions=self.wrong,
        )
        self.res = sd.speculative_decode(
            self.prompt, self.draft_next, self.target_next,
            k=self.k, max_tokens=self.max_tokens)

    def test_output_identical_to_plain_greedy(self):
        # The core correctness property of (greedy) speculative decoding.
        greedy = sd.greedy_decode(self.prompt, self.target_next, self.max_tokens)
        self.assertEqual(self.res.output, greedy)
        self.assertEqual(self.res.output, self.T[:self.max_tokens])

    def test_fewer_target_passes_than_tokens(self):
        # The whole point: many tokens per target forward pass.
        self.assertEqual(self.res.target_passes, 3)
        self.assertLess(self.res.target_passes, len(self.res.output))

    def test_draft_step_count(self):
        self.assertEqual(self.res.draft_steps, 3 * self.k)

    def test_acceptance_pattern_matches_wrong_positions(self):
        # Rejections happen at the rounds whose span contains a wrong position.
        self.assertEqual([r.num_accepted for r in self.res.rounds], [2, 2, 4])

    def test_all_accepted_when_draft_is_perfect(self):
        draft_next, target_next = sd.mock_models(
            prompt_len=len(self.prompt), target_continuation=self.T,
            draft_wrong_positions=set())
        res = sd.speculative_decode(self.prompt, draft_next, target_next, k=4, max_tokens=8)
        self.assertTrue(all(r.num_accepted == 4 for r in res.rounds))
        self.assertEqual(res.output, sd.greedy_decode(self.prompt, target_next, 8))


if __name__ == "__main__":
    unittest.main()
