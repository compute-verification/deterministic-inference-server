//! Shared types between the SP1 guest program and the host harness.
//!
//! Keeping these in a dedicated crate guarantees the host's `SP1Stdin::write`
//! and the guest's `sp1_zkvm::io::read` agree byte-for-byte on the input
//! layout (both go through the same serde-derived `Deserialize`/`Serialize`).

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

/// One ledger row's private witness for the SP1 program.
///
/// Together with the row's pre-canonicalised bytes (held in `ProofInput`),
/// the SP1 program reconstructs the leaf hash, walks the Merkle path back
/// to `signed_root`, and verifies the gateway signature over that root.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RowWitness {
    /// Index into `ProofInput::signer_pubkeys`.
    pub signer_idx: u32,
    /// Leaf index in the signer's Merkle tree. Bits select left/right at each level.
    pub leaf_index: u64,
    /// Sibling hashes bottom-up (level 0 first).
    pub merkle_path: Vec<[u8; 32]>,
    /// The signer's Merkle root that was signed.
    pub signed_root: [u8; 32],
    /// Ed25519 signature over `signed_root` (raw 64 bytes).
    /// Serde derives only support fixed-size arrays up to `[u8; 32]` out of
    /// the box; we transport the 64-byte signature as a `Vec<u8>` and the
    /// guest asserts `signature.len() == 64` on read.
    pub signature: Vec<u8>,
}

/// All inputs to the SP1 program. Public + private fields are tracked in the
/// program's comments (everything in this struct is the program's *input*; what
/// becomes *public* is what the program calls `commit_slice` on at the end).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProofInput {
    /// 32-byte opaque nonce from the auditor. Committed back as a public output
    /// so the auditor binds the proof to the audit it requested.
    pub auditor_nonce: [u8; 32],
    /// Signer Ed25519 pubkeys, sorted by lowercase-hex representation.
    /// The hex-sorted ordering matches the Python `pubkey_set_digest` recipe so
    /// both sides agree on the committed `pubkey_set_digest`.
    /// v0 has a single signer (the proof server itself).
    pub signer_pubkeys: Vec<[u8; 32]>,
    /// Per-row canonical JSON bytes WITHOUT a trailing newline.
    /// `leaf_hash(row) = sha256(row_canon || b"\n")` matches Python's
    /// `canonical_json_bytes(row)` (which includes the trailing newline).
    pub ledger_rows_canon: Vec<Vec<u8>>,
    /// One witness per row, same order as `ledger_rows_canon`.
    pub witnesses: Vec<RowWitness>,
}

/// Layout of the SP1 program's public output bytes.
///
/// The program calls `commit_slice` on exactly these 104 bytes, in this order.
/// The host and auditor parse them positionally.
pub const PUBLIC_OUTPUT_LEN: usize = 32 + 32 + 32 + 4 + 4;

pub struct PublicOutputs {
    pub auditor_nonce: [u8; 32],
    pub ledger_digest: [u8; 32],
    pub pubkey_set_digest: [u8; 32],
    pub n_rows: u32,
    pub n_signers: u32,
}

impl PublicOutputs {
    pub fn from_bytes(b: &[u8]) -> Option<Self> {
        if b.len() != PUBLIC_OUTPUT_LEN {
            return None;
        }
        let mut nonce = [0u8; 32];
        nonce.copy_from_slice(&b[0..32]);
        let mut ledger_digest = [0u8; 32];
        ledger_digest.copy_from_slice(&b[32..64]);
        let mut pubkey_set_digest = [0u8; 32];
        pubkey_set_digest.copy_from_slice(&b[64..96]);
        let n_rows = u32::from_le_bytes(b[96..100].try_into().ok()?);
        let n_signers = u32::from_le_bytes(b[100..104].try_into().ok()?);
        Some(Self { auditor_nonce: nonce, ledger_digest, pubkey_set_digest, n_rows, n_signers })
    }
}
