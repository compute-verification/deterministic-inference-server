"""Unit tests for the coding-agent tracer (stub)."""
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACERS = REPO_ROOT / "demos" / "proof-compare" / "tracers"
for p in (REPO_ROOT, TRACERS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from modules.proof_server.graph import build_graph
import coding


def _trace():
    return coding.trace_coding_stub(
        "agent",
        prompt={"tokens": 40, "label": "prompt"},
        retrievals=[{"kind": "search", "tokens": 600, "label": "search"},
                    {"kind": "fetch", "tokens": 6300, "label": "fetch"},
                    {"kind": "fetch", "tokens": 1500, "label": "fetch"}],
        plan={"tokens": 2900, "label": "plan"},
        codegens=[{"tokens": 1200, "label": "impl"}, {"tokens": 1400, "label": "test"}],
        verify={"tokens": 400, "label": "run tests"},
    )


class TestCodingTracer(unittest.TestCase):
    def setUp(self):
        self.trace = _trace()
        self.evs = self.trace["events"]
        self.byid = {e["id"]: e for e in self.evs}

    def test_root_prompt_has_no_inputs(self):
        prompt = next(e for e in self.evs if e["kind"] == "prompt")
        self.assertEqual(prompt["id"], 0)
        self.assertEqual(prompt["inputs"], [])

    def test_prompt_fans_out_to_retrievals(self):
        retr = [e for e in self.evs if e["kind"] in ("search", "fetch")]
        self.assertEqual(len(retr), 3)
        self.assertTrue(all(e["inputs"] == [0] for e in retr))

    def test_retrievals_fan_into_plan(self):
        plan = next(e for e in self.evs if e["kind"] == "plan")
        retr_ids = [e["id"] for e in self.evs if e["kind"] in ("search", "fetch")]
        self.assertEqual(sorted(plan["inputs"]), sorted(retr_ids))

    def test_codegens_fan_into_test(self):
        test = next(e for e in self.evs if e["kind"] == "test")
        cg_ids = [e["id"] for e in self.evs if e["kind"] == "codegen"]
        self.assertEqual(sorted(test["inputs"]), sorted(cg_ids))

    def test_stub_has_no_attention(self):
        # weight-only cost until the real tracer supplies context (Task 13).
        self.assertTrue(all(e["attended"] == 0 for e in self.evs))

    def test_builds_into_valid_graph_all_positive(self):
        g = build_graph(self.trace)
        self.assertTrue(all(n["flops"] > 0 for n in g.nodes))


if __name__ == "__main__":
    unittest.main()
