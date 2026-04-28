//! sha256, ed25519, and the Merkle node/leaf hashing used by the
//! transparency log. Mirrors `scripts/sim/transparency_log.py` byte-for-byte.

use anyhow::{anyhow, Context, Result};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use sha2::{Digest, Sha256};

pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

/// "sha256:<hex>" canonical commitment of `data`.
pub fn commitment(data: &[u8]) -> String {
    format!("sha256:{}", hex::encode(sha256(data)))
}

pub fn parse_commitment(s: &str) -> Result<[u8; 32]> {
    let hex_str = s.strip_prefix("sha256:")
        .ok_or_else(|| anyhow!("commitment missing sha256: prefix: {}", s))?;
    let bytes = hex::decode(hex_str).context("commitment hex decode")?;
    if bytes.len() != 32 {
        return Err(anyhow!("commitment wrong length: {}", bytes.len()));
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes);
    Ok(out)
}

pub fn verify_ed25519(pubkey_hex: &str, message: &[u8], signature_hex: &str) -> Result<()> {
    let pk_bytes = hex::decode(pubkey_hex).context("pubkey hex decode")?;
    if pk_bytes.len() != 32 {
        return Err(anyhow!("pubkey wrong length: {}", pk_bytes.len()));
    }
    let mut pk_arr = [0u8; 32];
    pk_arr.copy_from_slice(&pk_bytes);
    let pk = VerifyingKey::from_bytes(&pk_arr).context("pubkey parse")?;

    let sig_bytes = hex::decode(signature_hex).context("signature hex decode")?;
    if sig_bytes.len() != 64 {
        return Err(anyhow!("signature wrong length: {}", sig_bytes.len()));
    }
    let mut sig_arr = [0u8; 64];
    sig_arr.copy_from_slice(&sig_bytes);
    let sig = Signature::from_bytes(&sig_arr);

    pk.verify(message, &sig).map_err(|e| anyhow!("ed25519 verify failed: {}", e))
}

// ---------- RFC 6962 Merkle ---------------------------------------------------

pub fn leaf_hash(leaf: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update([0u8]);
    hasher.update(leaf);
    hasher.finalize().into()
}

pub fn node_hash(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update([1u8]);
    hasher.update(left);
    hasher.update(right);
    hasher.finalize().into()
}

fn largest_pow2_lt(n: usize) -> usize {
    let mut k = 1usize;
    while k.saturating_mul(2) < n {
        k *= 2;
    }
    k
}

fn root_recursive(leaves: &[&[u8]]) -> [u8; 32] {
    match leaves.len() {
        0 => sha256(&[]),
        1 => leaf_hash(leaves[0]),
        n => {
            let k = largest_pow2_lt(n);
            let left = root_recursive(&leaves[..k]);
            let right = root_recursive(&leaves[k..]);
            node_hash(&left, &right)
        }
    }
}

pub fn merkle_root(leaves: &[&[u8]]) -> [u8; 32] {
    root_recursive(leaves)
}

/// Verify an inclusion proof. `proof` is the sequence of sibling hashes from
/// the leaf up to the root, matching the recursive shape produced by
/// `_inclusion_proof` in the Python reference.
pub fn verify_inclusion(
    leaf: &[u8],
    index: usize,
    tree_size: usize,
    proof: &[[u8; 32]],
    root: &[u8; 32],
) -> bool {
    if tree_size == 0 || index >= tree_size {
        return false;
    }
    let h = leaf_hash(leaf);
    let mut pi: usize = 0;
    let final_h = match verify_recursive(h, index, tree_size, proof, &mut pi) {
        Some(h) => h,
        None => return false,
    };
    pi == proof.len() && final_h == *root
}

fn verify_recursive(
    h: [u8; 32], index: usize, n: usize, proof: &[[u8; 32]], pi: &mut usize,
) -> Option<[u8; 32]> {
    if n == 1 {
        return Some(h);
    }
    let k = largest_pow2_lt(n);
    if index < k {
        let sub = verify_recursive(h, index, k, proof, pi)?;
        if *pi >= proof.len() {
            return None;
        }
        let sibling = proof[*pi];
        *pi += 1;
        Some(node_hash(&sub, &sibling))
    } else {
        let sub = verify_recursive(h, index - k, n - k, proof, pi)?;
        if *pi >= proof.len() {
            return None;
        }
        let sibling = proof[*pi];
        *pi += 1;
        Some(node_hash(&sibling, &sub))
    }
}

/// RFC 6962 §2.1.4.2 consistency-proof verification. Returns true iff the
/// proof shows that a tree of size `old_size` with root `old_root` is a
/// prefix of a tree of size `new_size` with root `new_root`.
pub fn verify_consistency(
    old_size: usize, new_size: usize,
    old_root: &[u8; 32], new_root: &[u8; 32],
    proof: &[[u8; 32]],
) -> bool {
    if old_size > new_size {
        return false;
    }
    if old_size == 0 {
        return proof.is_empty();
    }
    if old_size == new_size {
        return proof.is_empty() && old_root == new_root;
    }

    let mut fnn = old_size - 1;
    let mut sn = new_size - 1;
    while fnn & 1 == 1 {
        fnn >>= 1;
        sn >>= 1;
    }

    let (mut fr, mut sr, mut pi) = if fnn == 0 {
        (*old_root, *old_root, 0usize)
    } else {
        if proof.is_empty() {
            return false;
        }
        (proof[0], proof[0], 1usize)
    };

    while fnn > 0 {
        if pi >= proof.len() {
            return false;
        }
        if (fnn & 1) == 1 || fnn == sn {
            let c = proof[pi];
            pi += 1;
            fr = node_hash(&c, &fr);
            sr = node_hash(&c, &sr);
            while (fnn & 1) == 0 && fnn != 0 {
                fnn >>= 1;
                sn >>= 1;
            }
        } else {
            let c = proof[pi];
            pi += 1;
            sr = node_hash(&sr, &c);
        }
        fnn >>= 1;
        sn >>= 1;
    }

    while sn > 0 {
        if pi >= proof.len() {
            return false;
        }
        let c = proof[pi];
        pi += 1;
        sr = node_hash(&sr, &c);
        sn >>= 1;
    }

    pi == proof.len() && fr == *old_root && sr == *new_root
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merkle_size_7_round_trip() {
        let leaves: Vec<Vec<u8>> = (0..7).map(|i| format!("leaf-{}", i).into_bytes()).collect();
        let leaf_refs: Vec<&[u8]> = leaves.iter().map(|v| v.as_slice()).collect();
        let root = merkle_root(&leaf_refs);
        // generate inclusion proofs by mimicking the python recursion.
        for i in 0..7 {
            let proof = inclusion_proof(&leaf_refs, i);
            assert!(
                verify_inclusion(&leaf_refs[i], i, 7, &proof, &root),
                "leaf {i} should verify"
            );
        }
    }

    fn inclusion_proof(leaves: &[&[u8]], index: usize) -> Vec<[u8; 32]> {
        let n = leaves.len();
        if n == 1 {
            return vec![];
        }
        let k = largest_pow2_lt(n);
        if index < k {
            let mut p = inclusion_proof(&leaves[..k], index);
            p.push(root_recursive(&leaves[k..]));
            p
        } else {
            let mut p = inclusion_proof(&leaves[k..], index - k);
            p.push(root_recursive(&leaves[..k]));
            p
        }
    }

    fn subproof(m: usize, leaves: &[&[u8]], b: bool) -> Vec<[u8; 32]> {
        let n = leaves.len();
        if m == n {
            if b { return vec![]; }
            return vec![root_recursive(leaves)];
        }
        let k = largest_pow2_lt(n);
        if m <= k {
            let mut p = subproof(m, &leaves[..k], b);
            p.push(root_recursive(&leaves[k..]));
            p
        } else {
            let mut p = subproof(m - k, &leaves[k..], false);
            p.push(root_recursive(&leaves[..k]));
            p
        }
    }

    #[test]
    fn consistency_proofs_round_trip() {
        let leaves: Vec<Vec<u8>> = (0..7).map(|i| format!("leaf-{}", i).into_bytes()).collect();
        let leaf_refs: Vec<&[u8]> = leaves.iter().map(|v| v.as_slice()).collect();
        let new_root = merkle_root(&leaf_refs);
        // For each old size, build the consistency proof and verify it.
        for m in 1..=leaf_refs.len() {
            let old_root = merkle_root(&leaf_refs[..m]);
            let proof = subproof(m, &leaf_refs, true);
            assert!(
                verify_consistency(m, leaf_refs.len(), &old_root, &new_root, &proof),
                "consistency m={} should verify",
                m,
            );
            // Tampered: zero out the first proof element if non-empty.
            if !proof.is_empty() {
                let mut bad = proof.clone();
                bad[0] = [0u8; 32];
                assert!(
                    !verify_consistency(m, leaf_refs.len(), &old_root, &new_root, &bad),
                    "tampered consistency m={} must not verify",
                    m,
                );
            }
        }
    }
}
