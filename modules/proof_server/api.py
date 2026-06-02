"""proof_server — stable public API.

The proof server's job is to turn (a) a public scrubbed compute ledger and
(b) signed Merkle attestation(s) over that ledger into a zero-knowledge
proof that every published row is backed by a valid signed Merkle leaf,
without revealing the signatures.

This module is the Python facade: it builds the ledger commitment,
assembles the SP1 witness, and the consumers (``demos/proof-server/servers/proof_server.py``,
``demos/proof-server/scripts/audit.py``) shell out to the host binary in
``modules/proof_server/sp1/host/``. The host binary runs the SP1
prover / executor.

See ``demos/proof-server/plan.md`` for the v0 design.
"""
from __future__ import annotations

from dataclasses import dataclass

from modules.proof_server.envelopes import (
    keypair_from_seed,
    pubkey_hex,
    sign_root,
    verify_root,
    verify_root_hex,
)
from modules.proof_server.ledger import (
    build_merkle_tree,
    leaf_hash,
    ledger_digest,
    ledger_digest_bytes,
    pubkey_set_digest,
    pubkey_set_digest_bytes,
    recompute_root,
)

__all__ = [
    # ledger
    "build_merkle_tree",
    "leaf_hash",
    "ledger_digest",
    "ledger_digest_bytes",
    "pubkey_set_digest",
    "pubkey_set_digest_bytes",
    "recompute_root",
    # envelopes / signatures
    "keypair_from_seed",
    "pubkey_hex",
    "sign_root",
    "verify_root",
    "verify_root_hex",
    # composite
    "WitnessRow",
    "assemble_witness",
]


@dataclass(frozen=True)
class WitnessRow:
    """One row's private witness for the SP1 program."""

    row_canonical_json: bytes
    signer_idx: int
    leaf_index: int
    merkle_path: list[bytes]
    signed_root: bytes
    signature: bytes


def assemble_witness(
    rows: list[dict],
    signer_pubkeys_hex: list[str],
    *,
    tree,
    leaf_index_of: dict[int, int],
    signed_root: bytes,
    signature: bytes,
    signer_pubkey_hex: str,
) -> list[WitnessRow]:
    """Build the per-row witness array the SP1 program consumes.

    v0 assumes a single signer (the proof server itself), so the witness is
    parameterised by one ``(tree, signed_root, signature, signer_pubkey_hex)``
    quad applied to every row in ``rows``. Multi-signer support is a v1
    follow-up; the ``signer_idx`` field in ``WitnessRow`` is already wired
    through so a v1 caller only needs to populate it per row.

    ``leaf_index_of`` maps each row's canonical-JSON leaf hash (lowercase
    hex string) to its position in the Merkle tree.
    """
    from modules.core.common.deterministic import canonical_json_bytes

    pubkeys_sorted = sorted(set(signer_pubkeys_hex))
    idx_of_pubkey = {pk: i for i, pk in enumerate(pubkeys_sorted)}
    if signer_pubkey_hex not in idx_of_pubkey:
        raise ValueError("signer_pubkey_hex not in declared pubkey set")
    signer_idx = idx_of_pubkey[signer_pubkey_hex]

    witnesses: list[WitnessRow] = []
    for row in rows:
        # ``leaf_index_of`` is keyed by the canonical-JSON leaf hash of each
        # row (hex). We use the content hash rather than any in-row field so
        # rows that share a logical id (e.g. a tap-protocol request envelope
        # and its response envelope, which both carry the same Gateway-
        # assigned id) end up at distinct positions in the tree.
        key = leaf_hash(row).hex()
        if key not in leaf_index_of:
            raise ValueError(f"Row leaf hash {key} not present in leaf_index_of map")
        leaf_index = leaf_index_of[key]
        proof = tree.proof_for(leaf_index)
        witnesses.append(
            WitnessRow(
                row_canonical_json=canonical_json_bytes(row),
                signer_idx=signer_idx,
                leaf_index=leaf_index,
                merkle_path=proof.path,
                signed_root=signed_root,
                signature=signature,
            )
        )
    return witnesses
