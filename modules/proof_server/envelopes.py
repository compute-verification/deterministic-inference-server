"""Ed25519 helpers used by the proof-server proxy.

RFC 8032 Ed25519 is deterministic by spec: no per-signature nonce, no
Simmons subliminal channel. ``cryptography``'s implementation produces the
RFC-test-vector outputs verbatim.

The proof server itself holds a single Ed25519 keypair which it uses to
sign each batch-commit's Merkle root. The auditor knows the proof
server's pubkey out-of-band; the per-batch signature is the hidden witness
the SP1 program verifies.
"""
from __future__ import annotations

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


def keypair_from_seed(seed: bytes) -> tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    """Derive an Ed25519 keypair from a 32-byte seed.

    Demos use fixed seeds so fixtures are reproducible; real deployments
    would use ``Ed25519PrivateKey.generate()``.
    """
    if len(seed) != 32:
        raise ValueError(f"Ed25519 seed must be 32 bytes; got {len(seed)}")
    sk = Ed25519PrivateKey.from_private_bytes(seed)
    return sk, sk.public_key()


def pubkey_hex(pk: Ed25519PublicKey) -> str:
    return pk.public_bytes_raw().hex()


def sign_root(sk: Ed25519PrivateKey, merkle_root: bytes) -> bytes:
    if len(merkle_root) != 32:
        raise ValueError("merkle_root must be 32 bytes")
    return sk.sign(merkle_root)


def verify_root(pk: Ed25519PublicKey, merkle_root: bytes, signature: bytes) -> bool:
    if len(merkle_root) != 32 or len(signature) != 64:
        return False
    try:
        pk.verify(signature, merkle_root)
        return True
    except InvalidSignature:
        return False


def verify_root_hex(pubkey_hex_: str, merkle_root_hex: str, signature_hex: str) -> bool:
    try:
        pk = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pubkey_hex_))
    except Exception:
        return False
    return verify_root(pk, bytes.fromhex(merkle_root_hex), bytes.fromhex(signature_hex))
