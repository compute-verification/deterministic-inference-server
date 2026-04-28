"""Crypto primitives shared by sim components.

ed25519 keys deterministically derived from string seeds for reproducibility.
sha256 helpers re-export the repo's canonical functions so the validator and
sim agree on byte-level digests.
"""
from __future__ import annotations

import hashlib

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
    PrivateFormat,
    NoEncryption,
)

from pkg.common.deterministic import canonical_json_bytes, sha256_prefixed


def derive_keypair(seed: str) -> tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    seed_bytes = hashlib.sha256(seed.encode("utf-8")).digest()
    sk = Ed25519PrivateKey.from_private_bytes(seed_bytes)
    return sk, sk.public_key()


def pubkey_hex(pk: Ed25519PublicKey) -> str:
    return pk.public_bytes(Encoding.Raw, PublicFormat.Raw).hex()


def privkey_bytes(sk: Ed25519PrivateKey) -> bytes:
    return sk.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())


def sign(sk: Ed25519PrivateKey, message: bytes) -> str:
    return sk.sign(message).hex()


def commitment(payload: bytes) -> str:
    return sha256_prefixed(payload)


__all__ = [
    "derive_keypair",
    "pubkey_hex",
    "privkey_bytes",
    "sign",
    "commitment",
    "canonical_json_bytes",
]
