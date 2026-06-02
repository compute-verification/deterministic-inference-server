"""Ed25519 helper tests for the proof-server module."""
from __future__ import annotations

import unittest

from modules.proof_server.envelopes import (
    keypair_from_seed,
    pubkey_hex,
    sign_root,
    verify_root,
    verify_root_hex,
)


class TestKeypair(unittest.TestCase):
    def test_deterministic_pubkey_from_seed(self):
        # Zero seed produces a well-known Ed25519 pubkey (RFC 8032 test vector).
        _, pk = keypair_from_seed(b"\x00" * 32)
        self.assertEqual(pubkey_hex(pk), "3b6a27bcceb6a42d62a3a8d02a6f0d73653215771de243a63ac048a18b59da29")

    def test_rejects_wrong_seed_length(self):
        with self.assertRaises(ValueError):
            keypair_from_seed(b"\x00" * 31)


class TestSignVerifyRoot(unittest.TestCase):
    def setUp(self):
        self.sk, self.pk = keypair_from_seed(b"\x01" * 32)
        self.root = bytes(range(32))

    def test_roundtrip(self):
        sig = sign_root(self.sk, self.root)
        self.assertEqual(len(sig), 64)
        self.assertTrue(verify_root(self.pk, self.root, sig))

    def test_signature_deterministic(self):
        # Ed25519 is deterministic per RFC 8032 — repeated signing yields
        # identical 64-byte output. This is the "no Simmons subliminal
        # channel" property.
        sig1 = sign_root(self.sk, self.root)
        sig2 = sign_root(self.sk, self.root)
        self.assertEqual(sig1, sig2)

    def test_tampered_signature_rejected(self):
        sig = sign_root(self.sk, self.root)
        bad = bytearray(sig); bad[0] ^= 0x01
        self.assertFalse(verify_root(self.pk, self.root, bytes(bad)))

    def test_tampered_root_rejected(self):
        sig = sign_root(self.sk, self.root)
        bad_root = bytearray(self.root); bad_root[0] ^= 0x01
        self.assertFalse(verify_root(self.pk, bytes(bad_root), sig))

    def test_wrong_pubkey_rejected(self):
        sig = sign_root(self.sk, self.root)
        _, other_pk = keypair_from_seed(b"\x02" * 32)
        self.assertFalse(verify_root(other_pk, self.root, sig))

    def test_rejects_wrong_sized_inputs(self):
        sig = sign_root(self.sk, self.root)
        self.assertFalse(verify_root(self.pk, self.root[:-1], sig))
        self.assertFalse(verify_root(self.pk, self.root, sig[:-1]))

    def test_sign_rejects_wrong_sized_root(self):
        with self.assertRaises(ValueError):
            sign_root(self.sk, b"\x00" * 31)


class TestVerifyHex(unittest.TestCase):
    def test_hex_roundtrip(self):
        sk, pk = keypair_from_seed(b"\x03" * 32)
        root = bytes(range(32))
        sig = sign_root(sk, root)
        self.assertTrue(verify_root_hex(pubkey_hex(pk), root.hex(), sig.hex()))

    def test_invalid_hex_returns_false_not_raises(self):
        # Defensive: malformed inputs shouldn't crash the proof server's
        # validation loop, they should just fail to verify.
        self.assertFalse(verify_root_hex("not-hex", "a" * 64, "b" * 128))


if __name__ == "__main__":
    unittest.main()
