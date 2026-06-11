"""Unit tests for the spec-decode capture harness + capture->trace converter.

run_spec.py wraps the real algorithm in demos/spec-decode/spec_decode.py (the
same speculative_decode the 4-server spec demo uses) as a capture harness in
the run_lora.py / run_coding_agent.py mold: --mock for CPU, --out for the
capture path, PROGRESS jsonl lines per round on stdout. The capture carries
the rounds shape the tracer already consumes plus the models' real configs,
and ``trace_spec_real(capture)`` converts it to a canonical trace — upgrading
the spec scenario from "ported rounds" to a real recorded run.
"""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACERS = REPO_ROOT / "demos" / "proof-compare" / "tracers"
SPEC_DEMO = REPO_ROOT / "demos" / "spec-decode"
HARNESS = REPO_ROOT / "demos" / "proof-compare" / "capture" / "run_spec.py"
for p in (REPO_ROOT, TRACERS, SPEC_DEMO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from modules.proof_server import flops as F
from modules.proof_server.graph import build_graph
import spec_decode as sd
import specdecode


class TestOnRoundCallback(unittest.TestCase):
    """speculative_decode grows an optional on_round hook for live progress."""

    def _models(self):
        T = [1000 + i for i in range(20)]
        return sd.mock_models(3, T, draft_wrong_positions={2, 7})

    def test_called_once_per_round_with_the_round(self):
        draft_next, target_next = self._models()
        seen = []
        res = sd.speculative_decode(
            [0, 1, 2], draft_next, target_next, k=4, max_tokens=8,
            on_round=lambda r: seen.append(r))
        self.assertEqual(len(seen), len(res.rounds))
        self.assertEqual(seen[0].num_accepted, res.rounds[0].num_accepted)

    def test_default_is_no_callback_and_identical_result(self):
        draft_next, target_next = self._models()
        a = sd.speculative_decode([0, 1, 2], draft_next, target_next, k=4, max_tokens=8)
        draft_next, target_next = self._models()
        b = sd.speculative_decode([0, 1, 2], draft_next, target_next, k=4,
                                  max_tokens=8, on_round=lambda r: None)
        self.assertEqual(a.output, b.output)
        self.assertEqual(len(a.rounds), len(b.rounds))


class TestSpecHarnessMock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        out = Path(tempfile.mkdtemp()) / "spec.json"
        proc = subprocess.run(
            [sys.executable, str(HARNESS), "--mock", "--out", str(out),
             "--prompt", "tell me a story", "--max-tokens", "12", "--k", "4"],
            capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        cls.stdout = proc.stdout
        cls.cap = json.loads(out.read_text())

    def test_capture_shape(self):
        c = self.cap
        self.assertEqual(c["kind"], "spec_decode_capture")
        self.assertEqual(c["k"], 4)
        self.assertEqual(c["max_tokens"], 12)
        self.assertTrue(c["rounds"])
        for rd in c["rounds"]:
            self.assertEqual(len(rd["drafts"]), 4)
            self.assertLessEqual(rd["num_accepted"], 4)
        # committed tokens cover max_tokens (last round may overshoot the cap)
        committed = sum(rd["num_accepted"] + 1 for rd in c["rounds"])
        self.assertGreaterEqual(committed, 12)
        self.assertEqual(len(c["output_ids"]), 12)

    def test_capture_carries_real_configs(self):
        self.assertEqual(self.cap["draft_model"], "Qwen/Qwen3-0.6B")
        self.assertEqual(self.cap["target_model"], "Qwen/Qwen3-1.7B")
        self.assertEqual(self.cap["draft_config"]["hidden_size"], 1024)
        self.assertEqual(self.cap["target_config"]["hidden_size"], 2048)

    def test_progress_lines_one_per_round(self):
        lines = [json.loads(l[len("PROGRESS "):]) for l in self.stdout.splitlines()
                 if l.startswith("PROGRESS ")]
        self.assertEqual(len(lines), len(self.cap["rounds"]))
        self.assertEqual(lines[0]["type"], "round")
        self.assertEqual(lines[0]["k"], 4)
        self.assertIn("accepted", lines[0])
        self.assertIn("committed", lines[0])

    def test_mock_run_is_deterministic(self):
        out2 = Path(tempfile.mkdtemp()) / "spec2.json"
        subprocess.run(
            [sys.executable, str(HARNESS), "--mock", "--out", str(out2),
             "--prompt", "tell me a story", "--max-tokens", "12", "--k", "4"],
            capture_output=True, text=True, timeout=60, check=True)
        self.assertEqual(json.loads(out2.read_text()), self.cap)


class TestTraceSpecReal(unittest.TestCase):
    def setUp(self):
        self.cap = {
            "kind": "spec_decode_capture",
            "draft_model": "Qwen/Qwen3-0.6B",
            "target_model": "Qwen/Qwen3-1.7B",
            "draft_config": dict(F.KNOWN_SHAPES["Qwen/Qwen3-0.6B"], hidden_size=999),
            "target_config": dict(F.KNOWN_SHAPES["Qwen/Qwen3-1.7B"]),
            "prompt": "p", "prompt_len": 6, "k": 2, "max_tokens": 5,
            "output": "x", "output_ids": [1, 2, 3, 4, 5],
            "rounds": [
                {"drafts": [" a", " b"], "num_accepted": 2, "correction": " c"},
                {"drafts": [" d", " e"], "num_accepted": 0, "correction": " f"},
            ],
            "draft_steps": 4, "target_passes": 2,
        }
        self.trace = specdecode.trace_spec_real(self.cap)

    def test_uses_captured_configs_not_known_shapes(self):
        # the deliberately-wrong hidden_size proves the capture's config wins
        self.assertEqual(self.trace["shapes"]["hf://Qwen/Qwen3-0.6B"]["hidden_size"], 999)

    def test_round_structure_matches_ported_tracer(self):
        evs = self.trace["events"]
        kinds = [e["kind"] for e in evs]
        self.assertEqual(kinds, ["draft", "draft", "verify", "draft", "draft", "verify"])
        v1 = evs[2]
        self.assertEqual(v1["tokens"], 3)          # k + 1 positions
        self.assertEqual(v1["payload"]["num_accepted"], 2)

    def test_builds_into_valid_graph(self):
        g = build_graph(self.trace)
        self.assertTrue(all(n["flops"] > 0 for n in g.nodes))

    def test_missing_rounds_rejected(self):
        with self.assertRaises(ValueError):
            specdecode.trace_spec_real(dict(self.cap, rounds=[]))


if __name__ == "__main__":
    unittest.main()
