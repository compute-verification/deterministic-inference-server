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



class TestWhitelist(unittest.TestCase):
    """Whitelisted inputs: exact strings that are free to pass into a node.

    A node whose recorded input text (payload.prompt) is byte-for-byte equal
    to a whitelist entry is stamped whitelisted=True so the viewer drops it
    from input-size accounting. FLOPs are untouched -- the pass still ran.
    """

    PROMPT = "Read the mini-paper in paper.md and implement Freivalds' check."

    def trace(self, **payload_kw):
        return {"shapes": SHAPES, "events": [
            ev(id=0, kind="prefill", tokens=10, attended=55, inputs=[],
               payload=dict(payload_kw)),
            ev(id=1, kind="decode", tokens=1, attended=11, inputs=[0]),
        ]}

    def test_exact_match_stamps_whitelisted(self):
        g = build_graph(self.trace(prompt=self.PROMPT), whitelist=[self.PROMPT])
        self.assertTrue(g.nodes[0].get("whitelisted"))
        self.assertNotIn("whitelisted", g.nodes[1])  # no prompt -> never free

    def test_flops_are_not_zeroed_by_the_whitelist(self):
        free = build_graph(self.trace(prompt=self.PROMPT), whitelist=[self.PROMPT])
        paid = build_graph(self.trace(prompt=self.PROMPT))
        self.assertEqual(free.nodes[0]["flops"], paid.nodes[0]["flops"])

    def test_near_miss_is_not_a_match(self):
        # exact string only: substring, superstring, and whitespace all differ
        for near in (self.PROMPT[:-1], self.PROMPT + " ", " " + self.PROMPT,
                     self.PROMPT.upper()):
            g = build_graph(self.trace(prompt=self.PROMPT), whitelist=[near])
            self.assertNotIn("whitelisted", g.nodes[0], near)

    def test_whitelist_falls_back_to_the_trace_key(self):
        trace = self.trace(prompt=self.PROMPT)
        trace["whitelist"] = [self.PROMPT]
        g = build_graph(trace)
        self.assertTrue(g.nodes[0].get("whitelisted"))

    def test_param_overrides_trace_whitelist(self):
        trace = self.trace(prompt=self.PROMPT)
        trace["whitelist"] = [self.PROMPT]
        g = build_graph(trace, whitelist=[])
        self.assertNotIn("whitelisted", g.nodes[0])

    def test_to_dict_carries_whitelist_only_when_set(self):
        with_wl = build_graph(self.trace(prompt=self.PROMPT),
                              whitelist=[self.PROMPT]).to_dict()
        self.assertEqual(with_wl["whitelist"], [self.PROMPT])
        without = build_graph(self.trace(prompt=self.PROMPT)).to_dict()
        self.assertNotIn("whitelist", without)  # old graphs stay byte-identical

    def test_non_string_entries_rejected(self):
        with self.assertRaises(ValueError):
            build_graph(self.trace(prompt=self.PROMPT), whitelist=[42])

if __name__ == "__main__":
    unittest.main()
