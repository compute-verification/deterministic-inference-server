"""Unit tests for the canonical graph model + build_graph."""
import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.proof_server import flops as F
from modules.proof_server.graph import Event, build_graph

SHAPES = {"m": F.KNOWN_SHAPES["Qwen/Qwen3-1.7B"]}


def ev(**kw):
    """Build an event dict with sensible defaults."""
    base = dict(id=0, kind="decode", inputs=[], model="m", tokens=1, attended=10,
                mode="fwd", logits=1)
    base.update(kw)
    return base


class TestBuildGraph(unittest.TestCase):
    def test_builds_nodes_with_correct_flops(self):
        trace = {"shapes": SHAPES, "events": [
            ev(id=0, kind="prefill", tokens=5, attended=15, inputs=[]),
            ev(id=1, kind="decode", tokens=1, attended=6, inputs=[0]),
        ]}
        g = build_graph(trace)
        self.assertEqual(len(g.nodes), 2)
        shape = F.model_shape_from_config(SHAPES["m"])
        self.assertEqual(g.nodes[0]["flops"], F.flops(shape, 5, 15, "fwd", 1))
        self.assertEqual(g.nodes[1]["flops"], F.flops(shape, 1, 6, "fwd", 1))

    def test_inputs_become_edges(self):
        trace = {"shapes": SHAPES, "events": [
            ev(id=0, kind="prefill", inputs=[]),
            ev(id=1, kind="decode", inputs=[0]),
            ev(id=2, kind="verify", inputs=[0, 1]),
        ]}
        g = build_graph(trace)
        pairs = {(e.src, e.dst) for e in g.edges}
        self.assertEqual(pairs, {(0, 1), (0, 2), (1, 2)})

    def test_rejects_dangling_input(self):
        trace = {"shapes": SHAPES, "events": [ev(id=0, inputs=[9])]}
        with self.assertRaises(ValueError):
            build_graph(trace)

    def test_rejects_forward_reference(self):
        # input id >= event id is a forward ref (would break DAG layering).
        trace = {"shapes": SHAPES, "events": [
            ev(id=0, kind="prefill", inputs=[1]),
            ev(id=1, kind="decode", inputs=[]),
        ]}
        with self.assertRaises(ValueError):
            build_graph(trace)

    def test_rejects_input_greater_than_id_even_when_it_exists(self):
        # Non-sequential ids: input(10) > id(3) must be rejected by the id rule,
        # not merely by a "not seen yet" coincidence.
        trace = {"shapes": SHAPES, "events": [
            ev(id=10, kind="prefill", inputs=[]),
            ev(id=3, kind="decode", inputs=[10]),
        ]}
        with self.assertRaises(ValueError):
            build_graph(trace)

    def test_rejects_self_loop(self):
        trace = {"shapes": SHAPES, "events": [ev(id=0, inputs=[0])]}
        with self.assertRaises(ValueError):
            build_graph(trace)

    def test_rejects_duplicate_ids(self):
        trace = {"shapes": SHAPES, "events": [ev(id=0, inputs=[]), ev(id=0, inputs=[])]}
        with self.assertRaises(ValueError):
            build_graph(trace)

    def test_rejects_unknown_kind(self):
        trace = {"shapes": SHAPES, "events": [ev(id=0, kind="frobnicate", inputs=[])]}
        with self.assertRaises(ValueError):
            build_graph(trace)

    def test_rejects_unknown_model(self):
        trace = {"shapes": SHAPES, "events": [ev(id=0, model="ghost", inputs=[])]}
        with self.assertRaises(ValueError):
            build_graph(trace)

    def test_round_trips_through_canonical_json(self):
        trace = {"shapes": SHAPES, "events": [ev(id=0, kind="prefill", inputs=[])]}
        s = build_graph(trace).to_json()
        self.assertTrue(s.endswith("\n"))
        parsed = json.loads(s)
        self.assertEqual(len(parsed["nodes"]), 1)
        self.assertIn("flops", parsed["nodes"][0])

    def test_empty_trace_is_empty_graph(self):
        g = build_graph({"shapes": SHAPES, "events": []})
        self.assertEqual(g.nodes, [])
        self.assertEqual(g.edges, [])

    def test_event_payload_defaults_to_dict(self):
        self.assertEqual(Event(id=0, kind="decode").payload, {})


if __name__ == "__main__":
    unittest.main()
