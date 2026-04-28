"""Append-only Merkle transparency log (RFC 6962 hashing).

* Leaf hash:   sha256(0x00 || leaf_bytes)
* Node hash:   sha256(0x01 || left || right)
* Tree size n: balanced subtree on left, remainder recurses on right
              (matches RFC 6962-bis).
* STH:         (tree_size, root_hash) signed with the log's ed25519 key.

This implementation is the source of truth used by the Python sim and is
mirrored byte-for-byte by the Rust validator.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .crypto import sign


def _leaf_hash(leaf: bytes) -> bytes:
    return hashlib.sha256(b"\x00" + leaf).digest()


def _node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()


def _largest_power_of_two_less_than(n: int) -> int:
    # k = largest power of two strictly less than n (n > 1).
    k = 1
    while k * 2 < n:
        k *= 2
    return k


def _root_hash(leaves: list[bytes]) -> bytes:
    n = len(leaves)
    if n == 0:
        return hashlib.sha256(b"").digest()
    if n == 1:
        return _leaf_hash(leaves[0])
    k = _largest_power_of_two_less_than(n)
    return _node_hash(_root_hash(leaves[:k]), _root_hash(leaves[k:]))


def _inclusion_proof(leaves: list[bytes], index: int) -> list[bytes]:
    n = len(leaves)
    assert 0 <= index < n
    if n == 1:
        return []
    k = _largest_power_of_two_less_than(n)
    if index < k:
        return _inclusion_proof(leaves[:k], index) + [_root_hash(leaves[k:])]
    return _inclusion_proof(leaves[k:], index - k) + [_root_hash(leaves[:k])]


# RFC 6962 §2.1.2 PROOF/SUBPROOF for consistency proofs.
def _consistency_proof(leaves: list[bytes], m: int) -> list[bytes]:
    n = len(leaves)
    assert 0 < m <= n
    return _subproof(m, leaves, True)


def _subproof(m: int, D: list[bytes], b: bool) -> list[bytes]:
    n = len(D)
    if m == n:
        if b:
            return []
        return [_root_hash(D)]
    k = _largest_power_of_two_less_than(n)
    if m <= k:
        return _subproof(m, D[:k], b) + [_root_hash(D[k:])]
    return _subproof(m - k, D[k:], False) + [_root_hash(D[:k])]


def verify_consistency(
    old_size: int, new_size: int,
    old_root: bytes, new_root: bytes,
    proof: list[bytes],
) -> bool:
    """RFC 6962 §2.1.4.2 verifier."""
    if old_size > new_size:
        return False
    if old_size == 0:
        return proof == []
    if old_size == new_size:
        return proof == [] and old_root == new_root

    fn, sn = old_size - 1, new_size - 1
    while fn & 1:
        fn >>= 1
        sn >>= 1

    if fn == 0:
        fr = old_root
        sr = old_root
        pi = 0
    else:
        if not proof:
            return False
        fr = proof[0]
        sr = proof[0]
        pi = 1

    while fn > 0:
        if pi >= len(proof):
            return False
        if (fn & 1) or fn == sn:
            c = proof[pi]
            pi += 1
            fr = _node_hash(c, fr)
            sr = _node_hash(c, sr)
            while (fn & 1) == 0 and fn != 0:
                fn >>= 1
                sn >>= 1
        else:
            c = proof[pi]
            pi += 1
            sr = _node_hash(sr, c)
        fn >>= 1
        sn >>= 1

    while sn > 0:
        if pi >= len(proof):
            return False
        c = proof[pi]
        pi += 1
        sr = _node_hash(sr, c)
        sn >>= 1

    return pi == len(proof) and fr == old_root and sr == new_root


def _verify_recursive(
    h: bytes, index: int, n: int, proof: list[bytes], pi: int
) -> tuple[bytes | None, int]:
    if n == 1:
        return h, pi
    k = _largest_power_of_two_less_than(n)
    if index < k:
        sub, pi = _verify_recursive(h, index, k, proof, pi)
        if sub is None or pi >= len(proof):
            return None, pi
        return _node_hash(sub, proof[pi]), pi + 1
    sub, pi = _verify_recursive(h, index - k, n - k, proof, pi)
    if sub is None or pi >= len(proof):
        return None, pi
    return _node_hash(proof[pi], sub), pi + 1


def verify_inclusion(
    leaf: bytes, index: int, tree_size: int, proof: list[bytes], root: bytes
) -> bool:
    if not (0 <= index < tree_size):
        return False
    if tree_size == 0:
        return False
    h = _leaf_hash(leaf)
    final, pi = _verify_recursive(h, index, tree_size, proof, 0)
    return final is not None and pi == len(proof) and final == root


@dataclass
class SignedTreeHead:
    tree_size: int
    root_hash: bytes
    signature: str  # hex


@dataclass
class TransparencyLog:
    sk: Ed25519PrivateKey
    leaves: list[bytes] = field(default_factory=list)
    sth_history: list[SignedTreeHead] = field(default_factory=list)

    def append(self, leaf_bytes: bytes) -> int:
        idx = len(self.leaves)
        self.leaves.append(leaf_bytes)
        return idx

    def sign_sth(self) -> SignedTreeHead:
        size = len(self.leaves)
        root = _root_hash(self.leaves)
        msg = size.to_bytes(8, "big") + root
        sth = SignedTreeHead(tree_size=size, root_hash=root, signature=sign(self.sk, msg))
        self.sth_history.append(sth)
        return sth

    def inclusion_proof(self, index: int) -> list[bytes]:
        return _inclusion_proof(self.leaves, index)

    def consistency_proof(self, old_size: int) -> list[bytes]:
        return _consistency_proof(self.leaves, old_size)

    def root(self) -> bytes:
        return _root_hash(self.leaves)


__all__ = [
    "TransparencyLog",
    "SignedTreeHead",
    "verify_inclusion",
    "verify_consistency",
    "_leaf_hash",
    "_node_hash",
    "_root_hash",
    "_inclusion_proof",
    "_consistency_proof",
]
