"""Unit tests for the bounded-cost partition statement (Python side).

Covers the planner, the reference checker (guest-assert semantics), the
canonical cost-view encoding, and the whitelist's effect on the S budget.
The Python<->Rust byte agreement is exercised by the SP1 smoke test.
"""
import json
import struct
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.proof_server.partition import (
    PARTITION_GRAPH_MAGIC,
    PartitionError,
    check_partition,
    graph_commitment,
    graph_partition_digest,
    partition_graph_bytes,
    plan_partition,
    sp1_input_json,
)

GRAPHS_JSON = REPO_ROOT / "demos" / "proof-compare" / "traces" / "graphs.json"


def g(nodes, edges=()):
    """Tiny graph literal: nodes = [(id, flops, tokens, whitelisted)]."""
    return {
        "nodes": [{"id": i, "flops": f, "tokens": t, **({"whitelisted": True} if w else {})}
                  for i, f, t, w in nodes],
        "edges": [{"src": s, "dst": d} for s, d in edges],
    }


CHAIN = g([(0, 100, 10, 0), (1, 100, 10, 0), (2, 100, 10, 0), (3, 100, 10, 0)],
          [(0, 1), (1, 2), (2, 3)])


class TestPlanner(unittest.TestCase):
    def test_everything_fits_in_one_part(self):
        parts = plan_partition(CHAIN, cap_flops=1000, cap_input=1000)
        self.assertEqual(parts, [0, 0, 0, 0])

    def test_flop_cap_splits(self):
        # C fits exactly two 100-FLOP nodes per part.
        parts = plan_partition(CHAIN, cap_flops=200, cap_input=1000)
        self.assertEqual(parts, [0, 0, 1, 1])

    def test_input_cap_splits(self):
        parts = plan_partition(CHAIN, cap_flops=10**9, cap_input=10)
        self.assertEqual(parts, [0, 1, 2, 3])

    def test_planner_output_always_passes_the_checker(self):
        for c, s in [(100, 10), (200, 20), (250, 35), (400, 40), (10**6, 10**6)]:
            parts = plan_partition(CHAIN, c, s)
            stats = check_partition(CHAIN, parts, c, s)
            self.assertLessEqual(stats["max_part_flops"], c)
            self.assertLessEqual(stats["max_part_input"], s)

    def test_infeasible_single_node(self):
        with self.assertRaises(PartitionError):
            plan_partition(CHAIN, cap_flops=99, cap_input=1000)
        with self.assertRaises(PartitionError):
            plan_partition(CHAIN, cap_flops=1000, cap_input=9)

    def test_whitelisted_node_is_free_on_S_but_not_C(self):
        wl = g([(0, 100, 1000, 1), (1, 100, 10, 0)], [(0, 1)])
        # S=10 would be infeasible if the whitelisted 1000-token input counted.
        parts = plan_partition(wl, cap_flops=200, cap_input=10)
        self.assertEqual(parts, [0, 0])
        # ...but its FLOPs still count toward C.
        with self.assertRaises(PartitionError):
            plan_partition(wl, cap_flops=99, cap_input=10)


class TestChecker(unittest.TestCase):
    def test_rejects_backward_edge_between_parts(self):
        with self.assertRaises(PartitionError):
            check_partition(CHAIN, [1, 0, 1, 1], cap_flops=10**6, cap_input=10**6)

    def test_rejects_non_contiguous_part_ids(self):
        with self.assertRaises(PartitionError):
            check_partition(CHAIN, [0, 0, 2, 2], cap_flops=10**6, cap_input=10**6)

    def test_rejects_over_budget_part(self):
        with self.assertRaises(PartitionError):
            check_partition(CHAIN, [0, 0, 0, 0], cap_flops=399, cap_input=10**6)
        with self.assertRaises(PartitionError):
            check_partition(CHAIN, [0, 0, 0, 0], cap_flops=10**6, cap_input=39)

    def test_rejects_wrong_length(self):
        with self.assertRaises(PartitionError):
            check_partition(CHAIN, [0, 0, 0], cap_flops=10**6, cap_input=10**6)

    def test_stats(self):
        stats = check_partition(CHAIN, [0, 0, 1, 1], cap_flops=200, cap_input=20)
        self.assertEqual(stats, {"n_nodes": 4, "n_parts": 2,
                                 "max_part_flops": 200, "max_part_input": 20})


class TestEncoding(unittest.TestCase):
    def test_layout_is_exactly_the_documented_struct(self):
        wl = g([(0, 7, 3, 1), (1, 9, 1, 0)], [(0, 1)])
        expect = (PARTITION_GRAPH_MAGIC
                  + struct.pack("<II", 2, 1)
                  + struct.pack("<QIB", 7, 3, 1)
                  + struct.pack("<QIB", 9, 1, 0)
                  + struct.pack("<II", 0, 1))
        self.assertEqual(partition_graph_bytes(wl), expect)

    def test_digest_is_stable_and_prefixed(self):
        d = graph_partition_digest(CHAIN)
        self.assertTrue(d.startswith("sha256:"))
        self.assertEqual(d, graph_partition_digest(json.loads(json.dumps(CHAIN))))

    def test_digest_binds_costs_and_whitelist(self):
        base = graph_partition_digest(CHAIN)
        bumped = json.loads(json.dumps(CHAIN))
        bumped["nodes"][2]["flops"] += 1
        self.assertNotEqual(base, graph_partition_digest(bumped))
        flagged = json.loads(json.dumps(CHAIN))
        flagged["nodes"][2]["whitelisted"] = True
        self.assertNotEqual(base, graph_partition_digest(flagged))

    def test_edges_dedupe_and_sort(self):
        a = g([(0, 1, 1, 0), (1, 1, 1, 0), (2, 1, 1, 0)], [(1, 2), (0, 1), (0, 1)])
        b = g([(0, 1, 1, 0), (1, 1, 1, 0), (2, 1, 1, 0)], [(0, 1), (1, 2)])
        self.assertEqual(partition_graph_bytes(a), partition_graph_bytes(b))

    def test_rejects_backward_edge_and_empty_graph(self):
        with self.assertRaises(PartitionError):
            partition_graph_bytes(g([(0, 1, 1, 0), (1, 1, 1, 0)], [(1, 0)]))
        with self.assertRaises(PartitionError):
            partition_graph_bytes({"nodes": [], "edges": []})

    def test_sp1_input_json_shape(self):
        doc = json.loads(sp1_input_json(CHAIN, [0, 0, 1, 1], 200, 20))
        self.assertEqual(doc["flops"], [100] * 4)
        self.assertEqual(doc["in_size"], [10] * 4)
        self.assertEqual(doc["whitelisted"], [0] * 4)
        self.assertEqual(doc["edges"], [[0, 1], [1, 2], [2, 3]])
        self.assertEqual(doc["parts"], [0, 0, 1, 1])
        self.assertEqual(doc["cap_flops"], 200)
        self.assertEqual(doc["auditor_nonce"], "00" * 32)
        self.assertEqual(doc["blind"], "00" * 32)
        with self.assertRaises(PartitionError):
            sp1_input_json(CHAIN, [0, 0, 1, 1], 200, 20, auditor_nonce="zz" * 32)
        with self.assertRaises(PartitionError):
            sp1_input_json(CHAIN, [0, 0, 1, 1], 200, 20, blind_hex="short")


class TestCommitment(unittest.TestCase):
    """The blinded commitment x = sha256(encoding || blind): binding in the
    graph, hiding via the blind."""

    def test_binding_same_blind_different_graph(self):
        bumped = json.loads(json.dumps(CHAIN))
        bumped["nodes"][2]["flops"] += 1
        b = "ab" * 32
        self.assertNotEqual(graph_commitment(CHAIN, b), graph_commitment(bumped, b))

    def test_hiding_same_graph_different_blind(self):
        self.assertNotEqual(graph_commitment(CHAIN, "aa" * 32),
                            graph_commitment(CHAIN, "bb" * 32))

    def test_zero_blind_reduces_to_appended_zeros_not_bare_digest(self):
        # x is never equal to the unblinded content digest, even for the
        # all-zero test blind — the 32 blind bytes are always hashed.
        self.assertNotEqual(graph_commitment(CHAIN, "00" * 32),
                            graph_partition_digest(CHAIN))

    def test_commitment_is_exactly_sha256_of_bytes_plus_blind(self):
        import hashlib
        b = "cd" * 32
        expect = hashlib.sha256(
            partition_graph_bytes(CHAIN) + bytes.fromhex(b)).hexdigest()
        self.assertEqual(graph_commitment(CHAIN, b), "sha256:" + expect)

    def test_rejects_malformed_blind(self):
        for bad in ("zz" * 32, "ab", 123):
            with self.assertRaises((PartitionError, TypeError)):
                graph_commitment(CHAIN, bad)


@unittest.skipUnless(GRAPHS_JSON.exists(), "bundled graphs.json missing")
class TestRealGraphs(unittest.TestCase):
    """The four real H100 scenes all plan + check under generous caps, and
    the whitelist (coding's task statement) measurably relaxes S."""

    @classmethod
    def setUpClass(cls):
        cls.scenes = json.loads(GRAPHS_JSON.read_text())

    def test_all_scenes_partition(self):
        for key in ("inference", "spec", "training", "coding"):
            graph = self.scenes[key]
            total = sum(n["flops"] for n in graph["nodes"])
            c = max(total // 7, max(n["flops"] for n in graph["nodes"]))
            parts = plan_partition(graph, c, cap_input=10**6)
            stats = check_partition(graph, parts, c, 10**6)
            self.assertGreaterEqual(stats["n_parts"], 2, key)
            self.assertLessEqual(stats["max_part_flops"], c, key)

    def test_coding_whitelist_relaxes_S(self):
        graph = self.scenes["coding"]
        wl_node = next(n for n in graph["nodes"] if n.get("whitelisted"))
        # One part holding everything, with S set to exactly the summed
        # non-whitelisted input: feasible only because the whitelist zeroes
        # the task statement's contribution.
        total_f = sum(n["flops"] for n in graph["nodes"])
        s = sum(0 if n.get("whitelisted") else n["tokens"] for n in graph["nodes"])
        parts = [0] * len(graph["nodes"])
        stats = check_partition(graph, parts, total_f, s)
        self.assertEqual(stats["max_part_input"], s)
        # Strip the flag: the same S is now short by the task statement.
        stripped = json.loads(json.dumps(graph))
        for n in stripped["nodes"]:
            n.pop("whitelisted", None)
        with self.assertRaises(PartitionError):
            check_partition(stripped, parts, total_f, s)
        self.assertGreater(wl_node["tokens"], 0)  # the relaxation is real


if __name__ == "__main__":
    unittest.main()
