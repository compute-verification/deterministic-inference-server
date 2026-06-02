"""Canonical Merkle tree + digest helpers for the proof-server.

A "ledger row" is any JSON-serialisable dict (typically a SignedEnvelope's
``EnvelopeData`` payload from ``demos/tap-protocol``). Both halves of the
demo agree byte-for-byte on what a leaf is and how the Merkle root is
computed; the Rust SP1 program mirrors this file exactly.

Leaf encoding
-------------
leaf(row) = sha256(canonical_json_bytes(row))

Canonical JSON: sort_keys=True, separators=(",", ":"), ensure_ascii=True,
trailing newline. (Same convention as ``modules.core.common.deterministic``.)

Merkle tree
-----------
Plain binary, no domain-separation prefix. Internal node = sha256(left || right).
Odd levels duplicate the rightmost node (standard "duplicate-last" padding).
Path from leaf ``i`` to root is determined by the bit pattern of ``i``: at
each level, bit = 0 means the leaf-side hash goes on the LEFT, bit = 1 on
the RIGHT, then we ascend.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

from modules.core.common.deterministic import canonical_json_bytes, sha256_prefixed


def leaf_hash(row: dict) -> bytes:
    """Return the 32-byte leaf hash for one ledger row.

    The row dict is canonicalised (sorted keys + ASCII) before hashing, so
    Python dict insertion order is irrelevant. The dict may have any
    structure — the proof server is responsible for choosing what to
    publish as a leaf (e.g. a full SignedEnvelope payload or a projection).
    """
    return hashlib.sha256(canonical_json_bytes(row)).digest()


@dataclass(frozen=True)
class MerkleProof:
    """Authentication path from one leaf to the root.

    ``path`` is the sequence of sibling 32-byte hashes from the leaf level
    upward. ``leaf_index`` selects which side each sibling lives on at each
    level (bit 0 of leaf_index = bottom level).
    """

    leaf_index: int
    path: list[bytes]


@dataclass(frozen=True)
class MerkleTree:
    """Materialised binary SHA-256 Merkle tree over leaf hashes."""

    leaves: list[bytes]
    levels: list[list[bytes]]  # levels[0] == leaves, levels[-1] == [root]

    @property
    def root(self) -> bytes:
        return self.levels[-1][0]

    def proof_for(self, leaf_index: int) -> MerkleProof:
        if not (0 <= leaf_index < len(self.leaves)):
            raise IndexError(f"leaf_index {leaf_index} out of range [0, {len(self.leaves)})")
        path: list[bytes] = []
        idx = leaf_index
        for level in self.levels[:-1]:  # all levels except the root level
            sibling_idx = idx ^ 1
            # On odd levels the right node was a duplicate of left, so the
            # sibling-index may equal len(level): clamp to last entry.
            if sibling_idx >= len(level):
                sibling_idx = len(level) - 1
            path.append(level[sibling_idx])
            idx //= 2
        return MerkleProof(leaf_index=leaf_index, path=path)


def build_merkle_tree(leaves: Iterable[bytes]) -> MerkleTree:
    """Materialise a binary SHA-256 Merkle tree over ``leaves``.

    Each leaf must be 32 bytes. Empty input is rejected (a "ledger" of zero
    rows would have no meaningful root for the proof to bind to).
    """
    lv0 = list(leaves)
    if not lv0:
        raise ValueError("Cannot build a Merkle tree with zero leaves")
    for h in lv0:
        if not isinstance(h, (bytes, bytearray)) or len(h) != 32:
            raise ValueError("Every leaf must be exactly 32 bytes")
    levels: list[list[bytes]] = [list(lv0)]
    while len(levels[-1]) > 1:
        prev = levels[-1]
        nxt: list[bytes] = []
        for i in range(0, len(prev), 2):
            left = prev[i]
            right = prev[i + 1] if i + 1 < len(prev) else prev[i]
            nxt.append(hashlib.sha256(left + right).digest())
        levels.append(nxt)
    return MerkleTree(leaves=lv0, levels=levels)


def recompute_root(leaf: bytes, leaf_index: int, path: list[bytes]) -> bytes:
    """Re-derive a Merkle root from a leaf + index + sibling path.

    Mirrors the SP1 program's verification loop exactly.
    """
    if len(leaf) != 32:
        raise ValueError("leaf must be 32 bytes")
    h = leaf
    idx = leaf_index
    for sibling in path:
        if len(sibling) != 32:
            raise ValueError("each sibling must be 32 bytes")
        if (idx & 1) == 0:
            h = hashlib.sha256(h + sibling).digest()
        else:
            h = hashlib.sha256(sibling + h).digest()
        idx //= 2
    return h


def ledger_digest(rows: list[dict]) -> str:
    """SHA-256 of canonical_json_bytes of the entire ledger ``rows`` array.

    Matches the digest the SP1 program commits as ``ledger_digest``.
    Returns the ``sha256:<hex>`` prefixed form for parity with the rest of
    the repo's digest conventions; raw 32 bytes are available via
    ``ledger_digest_bytes``.
    """
    return sha256_prefixed(canonical_json_bytes(rows))


def ledger_digest_bytes(rows: list[dict]) -> bytes:
    return hashlib.sha256(canonical_json_bytes(rows)).digest()


def pubkey_set_digest(pubkeys_hex: list[str]) -> str:
    """SHA-256 of canonical_json_bytes of the sorted+deduped pubkey hex list.

    Matches what the SP1 program commits as ``pubkey_set_digest``.
    """
    return sha256_prefixed(canonical_json_bytes(sorted(set(pubkeys_hex))))


def pubkey_set_digest_bytes(pubkeys_hex: list[str]) -> bytes:
    return hashlib.sha256(canonical_json_bytes(sorted(set(pubkeys_hex)))).digest()
