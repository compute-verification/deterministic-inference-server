//! SP1 guest: network-traffic consistency, soundness direction.
//!
//! For every row in the auditor-visible ledger, prove that the row's leaf
//! (under our canonical-JSON encoding) lives in some gateway's signed Merkle
//! tree, where the gateway is one of the auditor-declared public keys, and
//! the signature is a valid Ed25519 signature over that tree's root.
//!
//! Public inputs (committed at the end of `main`, in this order):
//!   - 32 bytes: auditor_nonce
//!   - 32 bytes: ledger_digest      = sha256(canonical_json_bytes(rows))
//!   - 32 bytes: pubkey_set_digest  = sha256(canonical_json_bytes(sorted_hex_pubkeys))
//!   - 4 bytes:  n_rows (le u32)
//!   - 4 bytes:  n_gateways (le u32)
//!
//! The auditor independently computes `ledger_digest` from the published
//! scrubbed ledger and `pubkey_set_digest` from its known pubkey set, and
//! rejects if either does not match what this program committed.

#![no_main]

sp1_zkvm::entrypoint!(main);

extern crate alloc;

use alloc::vec::Vec;

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use proof_server_lib::{ProofInput, RowWitness};
use sha2::{Digest, Sha256};

fn sha256_bytes(bytes: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().into()
}

fn sha256_concat(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(a);
    h.update(b);
    h.finalize().into()
}

/// Recompute a Merkle root from a leaf, the leaf's index, and the sibling
/// path. Mirrors `modules.proof_server.ledger.recompute_root` exactly.
fn recompute_root(leaf: [u8; 32], leaf_index: u64, path: &[[u8; 32]]) -> [u8; 32] {
    let mut h = leaf;
    let mut idx = leaf_index;
    for sibling in path {
        h = if idx & 1 == 0 {
            sha256_concat(&h, sibling)
        } else {
            sha256_concat(sibling, &h)
        };
        idx >>= 1;
    }
    h
}

/// Lowercase hex digit for a 4-bit value.
fn hex_digit(b: u8) -> u8 {
    if b < 10 { b + b'0' } else { b - 10 + b'a' }
}

/// Build the canonical JSON byte string of an array of pre-canonicalised rows.
///
/// Matches Python's `canonical_json_bytes(rows)` for `rows` a list of dicts:
///   b"[" + b",".join(per_row_no_newline) + b"]\n"
fn assemble_ledger_canon_json(per_row_canon: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    out.push(b'[');
    for (i, row) in per_row_canon.iter().enumerate() {
        if i > 0 {
            out.push(b',');
        }
        out.extend_from_slice(row);
    }
    out.push(b']');
    out.push(b'\n');
    out
}

/// Build the canonical JSON byte string of a sorted hex-pubkey list.
///
/// Matches Python's `canonical_json_bytes(sorted(set(pubkey_hex_list)))`:
///   b"[\"<hex0>\",\"<hex1>\",...]\n"
fn assemble_pubkey_set_canon_json(pubkeys: &[[u8; 32]]) -> Vec<u8> {
    let mut out = Vec::new();
    out.push(b'[');
    for (i, pk) in pubkeys.iter().enumerate() {
        if i > 0 {
            out.push(b',');
        }
        out.push(b'"');
        for byte in pk {
            out.push(hex_digit(byte >> 4));
            out.push(hex_digit(byte & 0x0f));
        }
        out.push(b'"');
    }
    out.push(b']');
    out.push(b'\n');
    out
}

pub fn main() {
    let input: ProofInput = sp1_zkvm::io::read();

    let n_rows = input.ledger_rows_canon.len();
    let n_signers = input.signer_pubkeys.len();

    assert_eq!(
        input.witnesses.len(),
        n_rows,
        "witness count must equal row count",
    );
    assert!(n_rows > 0, "ledger must contain at least one row");
    assert!(n_signers > 0, "must declare at least one signer pubkey");

    // Pre-build VerifyingKeys for each signer pubkey so we don't redo the
    // curve decoding on every signature verify.
    let verifying_keys: Vec<VerifyingKey> = input
        .signer_pubkeys
        .iter()
        .map(|pk_bytes| VerifyingKey::from_bytes(pk_bytes).expect("invalid Ed25519 pubkey"))
        .collect();

    // Per-row predicate: leaf -> path -> signed_root, then Ed25519 verify.
    for (row_bytes, w) in input.ledger_rows_canon.iter().zip(input.witnesses.iter()) {
        // Leaf = sha256(row_canon || "\n") -- matches Python leaf_hash.
        let mut leaf_input: Vec<u8> = Vec::with_capacity(row_bytes.len() + 1);
        leaf_input.extend_from_slice(row_bytes);
        leaf_input.push(b'\n');
        let leaf = sha256_bytes(&leaf_input);

        let recomputed = recompute_root(leaf, w.leaf_index, &w.merkle_path);
        assert_eq!(recomputed, w.signed_root, "merkle path does not match signed root");

        let gw_idx = w.signer_idx as usize;
        assert!(gw_idx < n_signers, "signer_idx out of range");
        assert_eq!(w.signature.len(), 64, "Ed25519 signature must be 64 bytes");
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(&w.signature);
        let sig = Signature::from_bytes(&sig_bytes);
        verifying_keys[gw_idx]
            .verify(&w.signed_root, &sig)
            .expect("Ed25519 signature did not verify");
    }

    // Row ordering: not enforced in-circuit. The auditor and the proof
    // server agree on the canonical ledger byte string via `ledger_digest`,
    // so any reordering would surface as a digest mismatch on the auditor
    // side (which already aborts before consulting the proof).

    // Compute the public-output digests inside the circuit so what we commit
    // is what we proved. The auditor independently recomputes these from
    // their own copies of the ledger + pubkey set and verifies equality.
    let ledger_canon = assemble_ledger_canon_json(&input.ledger_rows_canon);
    let ledger_digest = sha256_bytes(&ledger_canon);

    let pubkey_set_canon = assemble_pubkey_set_canon_json(&input.signer_pubkeys);
    let pubkey_set_digest = sha256_bytes(&pubkey_set_canon);

    // Commit public outputs in the documented layout (see proof_server_lib::PublicOutputs).
    sp1_zkvm::io::commit_slice(&input.auditor_nonce);
    sp1_zkvm::io::commit_slice(&ledger_digest);
    sp1_zkvm::io::commit_slice(&pubkey_set_digest);
    sp1_zkvm::io::commit_slice(&(n_rows as u32).to_le_bytes());
    sp1_zkvm::io::commit_slice(&(n_signers as u32).to_le_bytes());
}
