"""Smoke test for the SP1 host binary.

Skipped when the host binary isn't compiled (``cargo-prove`` and a usable
``protoc`` aren't on every dev machine). When it IS available, this test
runs a one-row, one-signer witness through SP1's execute mode and verifies
that the program's committed ``ledger_digest`` and ``pubkey_set_digest``
match the Python side's byte-stable computation.

This is the single point in the test suite that actually exercises
Python↔Rust canonical-JSON agreement.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

from modules.core.common.deterministic import canonical_json_bytes
from modules.proof_server.api import (
    assemble_witness,
    build_merkle_tree,
    keypair_from_seed,
    leaf_hash,
    ledger_digest,
    pubkey_hex,
    pubkey_set_digest,
    sign_root,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
HOST_BIN = REPO_ROOT / "modules/proof_server/sp1/target/release/proof-server-host"


def _have_sp1() -> bool:
    if not HOST_BIN.exists():
        return False
    if shutil.which("cargo-prove") is None:
        return False
    return True


@unittest.skipUnless(_have_sp1(),
                     "SP1 host binary missing; install sp1up + protoc and rebuild")
class TestSP1Smoke(unittest.TestCase):
    def test_one_row_one_signer_matches_python(self):
        row = {"id": 1, "payload": {"prompt": "hello", "max_tokens": 4}}
        leaves = [leaf_hash(row)]
        tree = build_merkle_tree(leaves)
        sk, pk = keypair_from_seed(b"\x00" * 32)
        sig = sign_root(sk, tree.root)
        pk_hex = pubkey_hex(pk)

        leaf_index_of = {leaf_hash(row).hex(): 0}
        witnesses = assemble_witness(
            [row], [pk_hex],
            tree=tree,
            leaf_index_of=leaf_index_of,
            signed_root=tree.root,
            signature=sig,
            signer_pubkey_hex=pk_hex,
        )
        stdin_json = {
            "auditor_nonce": "00" * 32,
            "signer_pubkeys": [pk_hex],
            "ledger_rows_canon_hex": [
                w.row_canonical_json.rstrip(b"\n").hex() for w in witnesses
            ],
            "witnesses": [
                {
                    "signer_idx": w.signer_idx,
                    "leaf_index": w.leaf_index,
                    "merkle_path_hex": [s.hex() for s in w.merkle_path],
                    "signed_root": w.signed_root.hex(),
                    "signature": w.signature.hex(),
                }
                for w in witnesses
            ],
        }

        result = subprocess.run(
            [str(HOST_BIN), "--execute"],
            input=json.dumps(stdin_json).encode("utf-8"),
            capture_output=True,
            timeout=600,
        )
        if result.returncode != 0:
            self.fail(f"SP1 host exited {result.returncode}\n"
                      f"stderr: {result.stderr.decode('utf-8', errors='replace')[-400:]}")

        stdout = result.stdout.decode("utf-8", errors="replace").strip()
        last_line = stdout.splitlines()[-1] if stdout else "{}"
        public = json.loads(last_line)

        # The auditor's job: independently recompute both digests and
        # check that SP1 committed exactly those bytes.
        self.assertEqual(public["auditor_nonce"], "00" * 32)
        self.assertEqual(public["ledger_digest"], ledger_digest([row]))
        self.assertEqual(public["pubkey_set_digest"], pubkey_set_digest([pk_hex]))
        self.assertEqual(public["n_rows"], 1)
        self.assertEqual(public["n_signers"], 1)

    def test_bad_signature_aborts_guest(self):
        """A tampered signature must cause the SP1 program's assert! to fire,
        which surfaces as a non-zero exit from the host (the guest commits
        zero public-output bytes when it aborts mid-program)."""
        row = {"id": 1, "payload": {"prompt": "hello", "max_tokens": 4}}
        leaves = [leaf_hash(row)]
        tree = build_merkle_tree(leaves)
        sk, pk = keypair_from_seed(b"\x00" * 32)
        sig = bytearray(sign_root(sk, tree.root))
        sig[0] ^= 0x01  # flip a byte
        pk_hex = pubkey_hex(pk)

        leaf_index_of = {leaf_hash(row).hex(): 0}
        witnesses = assemble_witness(
            [row], [pk_hex],
            tree=tree,
            leaf_index_of=leaf_index_of,
            signed_root=tree.root,
            signature=bytes(sig),
            signer_pubkey_hex=pk_hex,
        )
        stdin_json = {
            "auditor_nonce": "00" * 32,
            "signer_pubkeys": [pk_hex],
            "ledger_rows_canon_hex": [
                w.row_canonical_json.rstrip(b"\n").hex() for w in witnesses
            ],
            "witnesses": [
                {
                    "signer_idx": w.signer_idx,
                    "leaf_index": w.leaf_index,
                    "merkle_path_hex": [s.hex() for s in w.merkle_path],
                    "signed_root": w.signed_root.hex(),
                    "signature": w.signature.hex(),
                }
                for w in witnesses
            ],
        }

        result = subprocess.run(
            [str(HOST_BIN), "--execute"],
            input=json.dumps(stdin_json).encode("utf-8"),
            capture_output=True,
            timeout=600,
        )
        # Host exits 10 when the guest produced zero public-output bytes.
        self.assertNotEqual(result.returncode, 0,
                            "tampered signature should have failed the guest's assert!")


if __name__ == "__main__":
    unittest.main()
