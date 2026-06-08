"""Unit tests for the training tracer (stub)."""
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACERS = REPO_ROOT / "demos" / "proof-compare" / "tracers"
for p in (REPO_ROOT, TRACERS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from modules.proof_server import flops as F
from modules.proof_server.graph import build_graph
import training


def _trace(max_steps=6, eval_steps=2):
    return training.trace_training_stub(
        "Qwen/Qwen3-1.7B", max_steps=max_steps, batch=4, seq_len=8,
        loss_trajectory=[2.0, 1.6, 1.3, 1.1, 0.95, 0.9], eval_steps=eval_steps,
        eval_gen=3)


class TestTrainingTracer(unittest.TestCase):
    def setUp(self):
        self.evs = _trace()["events"]

    def test_train_step_count_matches_max_steps(self):
        self.assertEqual(len([e for e in self.evs if e["kind"] == "train_step"]), 6)

    def test_each_eval_is_a_flattened_inference(self):
        # eval every 2 steps over 6 -> 3 evals, each 1 prefill + 3 decode.
        self.assertEqual(len([e for e in self.evs if e["kind"] == "eval_prefill"]), 3)
        self.assertEqual(len([e for e in self.evs if e["kind"] == "eval_decode"]), 9)

    def test_eval_branches_off_a_train_step(self):
        byid = {e["id"]: e for e in self.evs}
        for ep in [e for e in self.evs if e["kind"] == "eval_prefill"]:
            self.assertEqual(byid[ep["inputs"][0]]["kind"], "train_step")

    def test_train_steps_are_chained(self):
        steps = [e for e in self.evs if e["kind"] == "train_step"]
        for i in range(1, len(steps)):
            self.assertIn(steps[i - 1]["id"], steps[i]["inputs"])

    def test_lora_step_is_just_over_2x_forward(self):
        g = build_graph(_trace())
        step = next(n for n in g.nodes if n["kind"] == "train_step")
        shape = F.model_shape_from_config(F.KNOWN_SHAPES["Qwen/Qwen3-1.7B"])
        fwd = F.flops(shape, step["tokens"], step["attended"], "fwd", step["logits"])
        self.assertTrue(2.0 < step["flops"] / fwd < 2.01)  # small seq -> tiny attn correction

    def test_builds_into_valid_graph(self):
        self.assertTrue(build_graph(_trace()).nodes)


if __name__ == "__main__":
    unittest.main()
