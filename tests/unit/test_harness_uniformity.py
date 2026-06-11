"""The capture harnesses' shared workload-runner contract.

All four harnesses (run_inference / run_spec / run_lora / run_coding_agent)
must: support --mock and --out, print ``PROGRESS {json}`` lines on stdout, and
produce output that is bitwise-identical across re-runs (the 4-node protocol's
recomp cluster re-runs the whole workload and compares canonical digests).

The coding harness additionally normalizes unittest output BEFORE it is fed
back into LLM prompts: the tempdir path / wall times / object addresses would
otherwise make the recomp re-run genuinely diverge (different prompt tokens),
not just smudge the digest.
"""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CAPTURE = REPO_ROOT / "demos" / "proof-compare" / "capture"
TRACERS = REPO_ROOT / "demos" / "proof-compare" / "tracers"
for p in (REPO_ROOT, TRACERS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from modules.proof_server.graph import build_graph

_spec = importlib.util.spec_from_file_location(
    "run_coding_agent", CAPTURE / "run_coding_agent.py")
coding_harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(coding_harness)


def _run(harness: str, *args) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        [sys.executable, str(CAPTURE / harness), *args],
        capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc


def _progress_lines(stdout: str) -> list[dict]:
    return [json.loads(l[len("PROGRESS "):]) for l in stdout.splitlines()
            if l.startswith("PROGRESS ")]


class TestNormalizeTestOutput(unittest.TestCase):
    def test_rundir_path_is_replaced(self):
        rd = Path("/tmp/agent_run_abc123")
        out = f'File "{rd}/test_freivalds.py", line 7'
        self.assertEqual(coding_harness.normalize_test_output(out, rd),
                         'File "<rundir>/test_freivalds.py", line 7')

    def test_wall_time_is_replaced(self):
        out = "Ran 5 tests in 0.003s\n\nOK"
        self.assertEqual(coding_harness.normalize_test_output(out, Path("/x")),
                         "Ran 5 tests in N.NNNs\n\nOK")

    def test_object_address_is_replaced(self):
        out = "<rng.Xorshift object at 0x7f3a2b00fd60>"
        self.assertEqual(coding_harness.normalize_test_output(out, Path("/x")),
                         "<rng.Xorshift object at 0xADDR>")

    def test_two_runs_normalize_identically(self):
        a = coding_harness.normalize_test_output(
            "/tmp/agent_run_aaa/t.py ... ok\nRan 2 tests in 0.001s",
            Path("/tmp/agent_run_aaa"))
        b = coding_harness.normalize_test_output(
            "/tmp/agent_run_bbb/t.py ... ok\nRan 2 tests in 0.137s",
            Path("/tmp/agent_run_bbb"))
        self.assertEqual(a, b)


class TestInferenceHarnessMock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = Path(tempfile.mkdtemp()) / "inference.json"
        cls.proc = _run("run_inference.py", "--mock", "--out", str(cls.out),
                        "--prompt", "hello world", "--max-tokens", "6")
        cls.trace = json.loads(cls.out.read_text())

    def test_trace_is_valid_and_marked_mock(self):
        self.assertTrue(self.trace["mock"])
        g = build_graph(self.trace)
        self.assertEqual(len(g.nodes), 6)   # n tokens = n forward passes

    def test_progress_one_line_per_forward_pass(self):
        lines = _progress_lines(self.proc.stdout)
        self.assertEqual(len(lines), 6)
        self.assertEqual([l["i"] for l in lines], list(range(1, 7)))
        self.assertTrue(all(l["type"] == "token" for l in lines))

    def test_mock_run_is_deterministic(self):
        out2 = Path(tempfile.mkdtemp()) / "inference2.json"
        _run("run_inference.py", "--mock", "--out", str(out2),
             "--prompt", "hello world", "--max-tokens", "6")
        self.assertEqual(json.loads(out2.read_text()), self.trace)


class TestLoraHarnessMockProgress(unittest.TestCase):
    def test_mock_emits_step_and_eval_progress(self):
        out = Path(tempfile.mkdtemp()) / "training.json"
        proc = _run("run_lora.py", "--mock", "--out", str(out))
        lines = _progress_lines(proc.stdout)
        steps = [l for l in lines if l["type"] == "step"]
        evals = [l for l in lines if l["type"] == "eval"]
        cap = json.loads(out.read_text())
        self.assertEqual(len(steps), len(cap["steps"]))
        self.assertEqual(len(evals), len(cap["evals"]))
        self.assertEqual(steps[0]["step"], 0)
        self.assertIn("loss", steps[0])


class TestCodingHarnessMockProgress(unittest.TestCase):
    def test_mock_emits_call_and_tests_progress(self):
        out = Path(tempfile.mkdtemp()) / "coding.json"
        proc = _run("run_coding_agent.py", "--mock", "--out", str(out))
        lines = _progress_lines(proc.stdout)
        calls = [l for l in lines if l["type"] == "call"]
        tests = [l for l in lines if l["type"] == "tests"]
        cap = json.loads(out.read_text())
        self.assertEqual(len(calls), len(cap["calls"]))
        self.assertEqual([c["call"] for c in calls],
                         [c["id"] for c in cap["calls"]])
        self.assertTrue(tests and tests[0]["passed"])

    def test_capture_test_output_is_normalized(self):
        out = Path(tempfile.mkdtemp()) / "coding.json"
        _run("run_coding_agent.py", "--mock", "--out", str(out))
        tail = json.loads(out.read_text())["test_output_tail"]
        self.assertIn("in N.NNNs", tail)
        self.assertNotIn("agent_run_", tail)


if __name__ == "__main__":
    unittest.main()
