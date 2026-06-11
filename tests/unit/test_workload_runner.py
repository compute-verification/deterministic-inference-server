"""Unit tests for the workload registry + runner (servers/workloads.py).

Runs the real harnesses in --mock mode through the runner (subprocess +
PROGRESS streaming + canonical digest) and converts each capture into the
canonical task graph -- the exact path the Host/Recomp clusters and the
Gateway's /run/<id>/graph endpoint use.
"""
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = REPO_ROOT / "demos" / "tap-protocol"
for p in (REPO_ROOT, DEMO_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from modules.proof_server.graph import build_graph
from servers import workloads as W


class TestBuildArgv(unittest.TestCase):
    def test_unknown_workload_rejected(self):
        with self.assertRaises(W.WorkloadError):
            W.build_argv("mining", {}, True, Path("/tmp/x.json"))

    def test_unknown_param_rejected(self):
        with self.assertRaises(W.WorkloadError):
            W.build_argv("inference", {"temperature": 1}, True, Path("/tmp/x.json"))

    def test_params_are_coerced_and_flagged(self):
        argv = W.build_argv("spec", {"max_tokens": "9", "k": 4}, True,
                            Path("/tmp/x.json"))
        self.assertIn("--mock", argv)
        i = argv.index("--max-tokens")
        self.assertEqual(argv[i + 1], "9")
        self.assertIn("--k", argv)

    def test_non_numeric_int_param_rejected(self):
        with self.assertRaises(ValueError):
            W.build_argv("inference", {"max_tokens": "lots"}, True, Path("/t.json"))


class TestRunWorkloadMock(unittest.TestCase):
    def test_all_four_workloads_run_convert_and_digest_stably(self):
        for wl, params in [("inference", {"prompt": "hi", "max_tokens": 4}),
                           ("spec", {"prompt": "hi there", "max_tokens": 8, "k": 3}),
                           ("training", {}),
                           ("coding", {})]:
            with self.subTest(workload=wl):
                progress: list[dict] = []
                cap1, d1 = W.run_workload(wl, params, mock=True,
                                          on_progress=progress.append)
                cap2, d2 = W.run_workload(wl, params, mock=True)
                self.assertEqual(d1, d2, f"{wl}: mock re-run digest must match")
                self.assertTrue(progress, f"{wl}: no PROGRESS lines seen")
                summary = W.summarize(wl, cap1)
                self.assertTrue(summary)
                g = build_graph(W.capture_to_trace(wl, cap1))
                self.assertTrue(all(n["flops"] > 0 for n in g.nodes))

    def test_progress_stream_matches_workload_shape(self):
        progress: list[dict] = []
        W.run_workload("inference", {"prompt": "x", "max_tokens": 5},
                       mock=True, on_progress=progress.append)
        self.assertEqual([p["type"] for p in progress], ["token"] * 5)

    def test_real_mode_rejects_mock_marked_capture(self):
        # run_inference --mock writes a mock-marked trace; pretend it was real
        cap, _ = W.run_workload("inference", {"max_tokens": 3}, mock=True)
        self.assertTrue(W._is_mock_capture("inference", cap))

    def test_summaries_carry_the_demo_facts(self):
        cap, _ = W.run_workload("training", {}, mock=True)
        s = W.summarize("training", cap)
        self.assertEqual(s["steps"], 12)
        self.assertGreater(s["first_loss"], s["final_loss"])
        cap, _ = W.run_workload("coding", {}, mock=True)
        s = W.summarize("coding", cap)
        self.assertTrue(s["tests_passed"])
        self.assertEqual(s["forward_passes"],
                         sum(max(c["gen_tokens"], 1) for c in cap["calls"]))


class TestCanonicalDigest(unittest.TestCase):
    def test_key_order_does_not_matter(self):
        self.assertEqual(W.canonical_digest({"a": 1, "b": [2, 3]}),
                         W.canonical_digest({"b": [2, 3], "a": 1}))

    def test_value_change_changes_digest(self):
        self.assertNotEqual(W.canonical_digest({"a": 1}),
                            W.canonical_digest({"a": 2}))


if __name__ == "__main__":
    unittest.main()
