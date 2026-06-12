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

// ---------------------------------------------------------------------------
// Bounded-cost partition proof (taskgraph-partition-v1)
// ---------------------------------------------------------------------------

/// Domain-separation prefix of the canonical graph encoding hashed in-guest.
pub const PARTITION_GRAPH_MAGIC: &[u8] = b"taskgraph-partition-v1\n";

/// All inputs to the partition SP1 program.
///
/// EVERYTHING about the graph is private. The guest re-encodes the cost view
/// (`flops`, `in_size`, `whitelisted`, `edges`) and computes the blinded
/// commitment sha256(encoding || blind) *inside the circuit*, committing only
/// that 32-byte value — so the proof binds to exactly these numbers without
/// the verifier ever seeing them. The blind makes the commitment hiding: a
/// bare content hash would let an auditor confirm guesses about a
/// low-entropy graph. The `parts` assignment is likewise private: the proof
/// shows a valid bounded partition EXISTS without revealing it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PartitionInput {
    /// 32-byte opaque nonce from the auditor, committed back unchanged.
    pub auditor_nonce: [u8; 32],
    /// 32-byte random blinding factor for the graph commitment. Private;
    /// held by the prover (and whoever legitimately holds the graph).
    pub blind: [u8; 32],
    /// Per-node exact FLOPs, in node-id (= topological) order.
    pub flops: Vec<u64>,
    /// Per-node ingested input size (tokens), same order.
    pub in_size: Vec<u32>,
    /// Per-node whitelist flag (1 = input is a publicly-known constant,
    /// free to pass), same order. Values other than 0/1 abort the guest.
    pub whitelisted: Vec<u8>,
    /// Dependency edges as (src_idx, dst_idx) node-index pairs, strictly
    /// lexicographically sorted and deduped, with src < dst (ids ascend in
    /// topological order, so this also makes the graph a DAG).
    pub edges: Vec<(u32, u32)>,
    /// The witness: part id per node. Ids must cover exactly 0..n_parts and
    /// never decrease along an edge (parts are executable stages in order).
    pub parts: Vec<u32>,
    /// Per-part FLOP budget (C).
    pub cap_flops: u64,
    /// Per-part non-whitelisted input budget in tokens (S).
    pub cap_input: u64,
}

/// Canonical byte encoding of a task graph's cost view. The guest computes
/// the blinded commitment sha256(encoding || blind) over this; the Python
/// side (`modules.proof_server.partition.partition_graph_bytes`) must
/// produce identical bytes from the graph JSON.
pub fn partition_graph_bytes(
    flops: &[u64],
    in_size: &[u32],
    whitelisted: &[u8],
    edges: &[(u32, u32)],
) -> Vec<u8> {
    let mut out =
        Vec::with_capacity(PARTITION_GRAPH_MAGIC.len() + 8 + flops.len() * 13 + edges.len() * 8);
    out.extend_from_slice(PARTITION_GRAPH_MAGIC);
    out.extend_from_slice(&(flops.len() as u32).to_le_bytes());
    out.extend_from_slice(&(edges.len() as u32).to_le_bytes());
    for i in 0..flops.len() {
        out.extend_from_slice(&flops[i].to_le_bytes());
        out.extend_from_slice(&in_size[i].to_le_bytes());
        out.push(whitelisted[i]);
    }
    for &(s, d) in edges {
        out.extend_from_slice(&s.to_le_bytes());
        out.extend_from_slice(&d.to_le_bytes());
    }
    out
}

/// Layout of the partition program's public output bytes (84 bytes).
///
/// Deliberately NOT included: n_nodes (or anything else about the graph) —
/// the auditor learns only the commitment, the caps, and the stage count.
pub const PARTITION_PUBLIC_OUTPUT_LEN: usize = 32 + 32 + 8 + 8 + 4;

pub struct PartitionPublicOutputs {
    pub auditor_nonce: [u8; 32],
    /// Blinded commitment sha256(taskgraph-partition-v1 encoding || blind).
    pub graph_commitment: [u8; 32],
    pub cap_flops: u64,
    pub cap_input: u64,
    pub n_parts: u32,
}

impl PartitionPublicOutputs {
    pub fn from_bytes(b: &[u8]) -> Option<Self> {
        if b.len() != PARTITION_PUBLIC_OUTPUT_LEN {
            return None;
        }
        let mut nonce = [0u8; 32];
        nonce.copy_from_slice(&b[0..32]);
        let mut graph_commitment = [0u8; 32];
        graph_commitment.copy_from_slice(&b[32..64]);
        let cap_flops = u64::from_le_bytes(b[64..72].try_into().ok()?);
        let cap_input = u64::from_le_bytes(b[72..80].try_into().ok()?);
        let n_parts = u32::from_le_bytes(b[80..84].try_into().ok()?);
        Some(Self { auditor_nonce: nonce, graph_commitment, cap_flops, cap_input, n_parts })
    }
}
