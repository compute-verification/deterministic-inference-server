//! SP1 guest: bounded-cost partition of a COMMITTED (hidden) task graph.
//!
//! Statement: "The graph committed to by x has a partition into k stages
//! such that
//!   (a) every dependency edge flows from an earlier-or-equal stage to a
//!       later-or-equal stage — i.e. the stages can be executed in order;
//!   (b) each stage's summed FLOPs are <= C;
//!   (c) each stage's summed NON-WHITELISTED input tokens are <= S
//!       (whitelisted inputs are publicly-known constants and free to pass)."
//!
//! The graph AND the partition are PRIVATE witnesses. The guest re-encodes
//! the graph's cost view and computes the blinded commitment
//! x = sha256(taskgraph-partition-v1 encoding || blind) in-circuit, so the
//! proof binds to exactly those numbers while the auditor sees only x. The
//! blind keeps the commitment hiding — a bare content hash would let an
//! auditor confirm guesses about a low-entropy graph. The auditor checks x
//! against the commitment the prover published out-of-band (e.g. inside a
//! signed envelope) and never sees the graph.
//!
//! Public outputs (committed at the end of `main`, in this order):
//!   - 32 bytes: auditor_nonce
//!   - 32 bytes: graph_commitment x (blinded; see above)
//!   - 8 bytes:  cap_flops C (le u64)
//!   - 8 bytes:  cap_input S (le u64)
//!   - 4 bytes:  n_parts (le u32)
//! Deliberately absent: n_nodes or anything else describing the graph.

#![no_main]

sp1_zkvm::entrypoint!(main);

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use proof_server_lib::{partition_graph_bytes, PartitionInput, PARTITION_PUBLIC_OUTPUT_LEN};
use sha2::{Digest, Sha256};

pub fn main() {
    let input: PartitionInput = sp1_zkvm::io::read();

    let n = input.flops.len();
    assert!(n > 0, "graph must contain at least one node");
    assert!(n <= u32::MAX as usize, "node count exceeds u32");
    assert_eq!(input.in_size.len(), n, "in_size length must equal node count");
    assert_eq!(input.whitelisted.len(), n, "whitelisted length must equal node count");
    assert_eq!(input.parts.len(), n, "parts length must equal node count");
    for &w in &input.whitelisted {
        assert!(w <= 1, "whitelist flags must be 0 or 1");
    }

    // Edges: strictly lex-increasing (sorted + deduped — keeps the hashed
    // encoding canonical) with src < dst. Node ids ascend in topological
    // order (enforced by the graph builder), so src < dst everywhere makes
    // the graph a DAG by construction.
    let mut prev: Option<(u32, u32)> = None;
    for &(s, d) in &input.edges {
        assert!((d as usize) < n, "edge endpoint out of range");
        assert!(s < d, "edge must go forward in node order");
        if let Some(p) = prev {
            assert!((s, d) > p, "edges must be strictly lex-sorted (sorted + deduped)");
        }
        prev = Some((s, d));
    }

    // Parts: ids must cover exactly 0..n_parts (no gaps — otherwise the
    // committed part count is gameable) and never decrease along an edge.
    // Monotone-along-edges means the quotient graph is acyclic with this
    // numbering as a topological order: the stages really are executable
    // one after another.
    let mut n_parts: u32 = 0;
    for &p in &input.parts {
        if p >= n_parts {
            n_parts = p + 1;
        }
    }
    assert!((n_parts as usize) <= n, "more parts than nodes");
    let mut used = vec![false; n_parts as usize];
    for &p in &input.parts {
        used[p as usize] = true;
    }
    for u in &used {
        assert!(*u, "part ids must be contiguous 0..n_parts");
    }
    for &(s, d) in &input.edges {
        assert!(
            input.parts[s as usize] <= input.parts[d as usize],
            "edge crosses backward between parts",
        );
    }

    // Per-part budgets. u128 accumulators: no overflow for any u64 inputs.
    let mut flops_sum = vec![0u128; n_parts as usize];
    let mut input_sum = vec![0u128; n_parts as usize];
    for i in 0..n {
        let p = input.parts[i] as usize;
        flops_sum[p] += input.flops[i] as u128;
        if input.whitelisted[i] == 0 {
            input_sum[p] += input.in_size[i] as u128;
        }
    }
    for p in 0..n_parts as usize {
        assert!(flops_sum[p] <= input.cap_flops as u128, "part exceeds FLOP cap C");
        assert!(input_sum[p] <= input.cap_input as u128, "part exceeds input cap S");
    }

    // Bind the proof to the (hidden) graph: blinded commitment over the
    // canonical cost-view encoding. The encoding is self-delimiting (magic +
    // length fields), so appending the fixed-width blind is unambiguous.
    let graph_bytes =
        partition_graph_bytes(&input.flops, &input.in_size, &input.whitelisted, &input.edges);
    let mut hasher = Sha256::new();
    hasher.update(&graph_bytes);
    hasher.update(&input.blind);
    let graph_commitment: [u8; 32] = hasher.finalize().into();

    let mut out: Vec<u8> = Vec::with_capacity(PARTITION_PUBLIC_OUTPUT_LEN);
    out.extend_from_slice(&input.auditor_nonce);
    out.extend_from_slice(&graph_commitment);
    out.extend_from_slice(&input.cap_flops.to_le_bytes());
    out.extend_from_slice(&input.cap_input.to_le_bytes());
    out.extend_from_slice(&n_parts.to_le_bytes());
    sp1_zkvm::io::commit_slice(&out);
}
