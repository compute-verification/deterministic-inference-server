"""Merkle tree + digest tests over arbitrary JSON-dict ledger rows.

Goldens are pinned to flag any silent change in the canonical-JSON or
hash-tree encoding — the SP1 program depends on the encoding being
byte-stable across versions.
"""
from __future__ import annotations

import hashlib
import unittest

from modules.proof_server.ledger import (
    build_merkle_tree,
    leaf_hash,
    ledger_digest,
    ledger_digest_bytes,
    pubkey_set_digest,
    recompute_root,
)


# SignedEnvelope-shaped rows (the v0 demo uses these directly).
ROW_A = {"id": 1, "payload": {"prompt": "hello", "max_tokens": 8}}
ROW_B = {"id": 2, "payload": {"output": "world"}}
ROW_C = {"id": 3, "payload": {"prompt": "next", "max_tokens": 4}}
ROW_D = {"id": 4, "payload": {"output": "done"}}


class TestLeafHash(unittest.TestCase):
    def test_leaf_is_32_bytes(self):
        self.assertEqual(len(leaf_hash(ROW_A)), 32)

    def test_leaf_independent_of_insertion_order(self):
        # Same fields, different dict-insertion order, must yield same leaf hash.
        a1 = {"id": ROW_A["id"], "payload": ROW_A["payload"]}
        a2 = {"payload": ROW_A["payload"], "id": ROW_A["id"]}
        self.assertEqual(leaf_hash(a1), leaf_hash(a2))

    def test_leaf_accepts_arbitrary_dict(self):
        # We deliberately removed the strict-shape coupling; any
        # JSON-serialisable dict is a valid row.
        leaf_hash({})
        leaf_hash({"x": [1, 2, 3], "nested": {"y": True}})


class TestMerkleTree(unittest.TestCase):
    def test_single_leaf_tree(self):
        leaves = [leaf_hash(ROW_A)]
        tree = build_merkle_tree(leaves)
        self.assertEqual(tree.root, leaves[0])

    def test_two_leaf_tree_root_matches_manual(self):
        leaves = [leaf_hash(ROW_A), leaf_hash(ROW_B)]
        tree = build_merkle_tree(leaves)
        expected = hashlib.sha256(leaves[0] + leaves[1]).digest()
        self.assertEqual(tree.root, expected)

    def test_odd_count_duplicates_last(self):
        leaves = [leaf_hash(ROW_A), leaf_hash(ROW_B), leaf_hash(ROW_C)]
        tree = build_merkle_tree(leaves)
        n1_left = hashlib.sha256(leaves[0] + leaves[1]).digest()
        n1_right = hashlib.sha256(leaves[2] + leaves[2]).digest()
        expected_root = hashlib.sha256(n1_left + n1_right).digest()
        self.assertEqual(tree.root, expected_root)

    def test_four_leaf_proofs_roundtrip(self):
        leaves = [leaf_hash(ROW_A), leaf_hash(ROW_B), leaf_hash(ROW_C), leaf_hash(ROW_D)]
        tree = build_merkle_tree(leaves)
        for i, leaf in enumerate(leaves):
            proof = tree.proof_for(i)
            self.assertEqual(recompute_root(leaf, proof.leaf_index, proof.path), tree.root,
                             f"proof at leaf {i} did not round-trip")

    def test_three_leaf_proof_roundtrip_handles_duplicate_sibling(self):
        leaves = [leaf_hash(ROW_A), leaf_hash(ROW_B), leaf_hash(ROW_C)]
        tree = build_merkle_tree(leaves)
        proof = tree.proof_for(2)
        self.assertEqual(recompute_root(leaves[2], proof.leaf_index, proof.path), tree.root)

    def test_empty_tree_rejected(self):
        with self.assertRaises(ValueError):
            build_merkle_tree([])

    def test_non_32_byte_leaf_rejected(self):
        with self.assertRaises(ValueError):
            build_merkle_tree([b"\x00" * 31])


class TestDigests(unittest.TestCase):
    def test_ledger_digest_deterministic(self):
        rows = [ROW_A, ROW_B]
        d1 = ledger_digest(rows)
        d2 = ledger_digest(rows)
        self.assertEqual(d1, d2)
        self.assertTrue(d1.startswith("sha256:"))

    def test_ledger_digest_bytes_matches_prefixed(self):
        rows = [ROW_A, ROW_B]
        self.assertEqual(ledger_digest_bytes(rows).hex(), ledger_digest(rows).removeprefix("sha256:"))

    def test_pubkey_set_digest_dedupes_and_sorts(self):
        pks = ["bb" * 32, "aa" * 32, "bb" * 32]
        d1 = pubkey_set_digest(pks)
        d2 = pubkey_set_digest(["aa" * 32, "bb" * 32])
        self.assertEqual(d1, d2)


class TestGoldenDigests(unittest.TestCase):
    """Pin the exact digest bytes so a silent canonical-JSON drift would fail
    here before reaching the SP1 program."""

    GOLDEN_LEAF_A = "cc9e0a9b3fbb92528157920d9848c6230e5713abf164ddfaf5ee5859663734e6"
    GOLDEN_LEDGER_DIGEST_AB = "sha256:e041686fafea17f57a74cfbc1ba6b02750cce1cf6bf5b5a4b50c30e2ef46ed2a"

    def test_leaf_a_golden(self):
        actual = leaf_hash(ROW_A).hex()
        # If this fails on first land, update the golden and document why.
        self.assertEqual(actual, self.GOLDEN_LEAF_A, msg=f"got {actual}")

    def test_ledger_digest_ab_golden(self):
        actual = ledger_digest([ROW_A, ROW_B])
        self.assertEqual(actual, self.GOLDEN_LEDGER_DIGEST_AB, msg=f"got {actual}")


if __name__ == "__main__":
    unittest.main()
