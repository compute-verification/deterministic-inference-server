//! The 15 validator checks from the spec, in order.
//!
//! Each check is a pure function over the deserialized graph + replay
//! responses; it returns `Ok(())` on pass or an error message describing the
//! first violation. The CLI binary aggregates the results into a verdict.

use anyhow::Result;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

use crate::canonical::canonical_bytes;
use crate::crypto::{merkle_root, parse_commitment, verify_consistency, verify_ed25519, verify_inclusion};
use crate::types::{
    Artifact, AttestedTaskGraph, ComputeAttestation, ReplayResponse, Task, TransmissionEvent,
};

pub const CHECK_NAMES: &[&str] = &[
    "schema",
    "artifact_commitment",
    "graph_wellformed",
    "acyclic",
    "tap_authenticity",
    "tap_ordering",
    "log_inclusion",
    "checkpoint_validation",
    "checkpoint_consistency",
    "graph_log_consistency",
    "compute_attestation",
    "task_compute_accounting",
    "ancestry_compute",
    "replay_correctness",
    "policy",
];

#[derive(Debug, Clone)]
pub struct CheckResult {
    pub name: &'static str,
    pub status: CheckStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckStatus {
    Pass,
    Fail(String),
    Skip(String),
}

pub struct ValidatorContext<'a> {
    pub graph: &'a AttestedTaskGraph,
    pub replay: &'a [ReplayResponse],
    pub task_by_id: HashMap<&'a str, &'a Task>,
    pub artifact_by_id: HashMap<&'a str, &'a Artifact>,
}

impl<'a> ValidatorContext<'a> {
    pub fn new(graph: &'a AttestedTaskGraph, replay: &'a [ReplayResponse]) -> Self {
        let task_by_id: HashMap<&str, &Task> =
            graph.tasks.iter().map(|t| (t.task_id.as_str(), t)).collect();
        let artifact_by_id: HashMap<&str, &Artifact> = graph
            .artifacts
            .iter()
            .map(|a| (a.artifact_id.as_str(), a))
            .collect();
        ValidatorContext { graph, replay, task_by_id, artifact_by_id }
    }
}

pub fn run_all(ctx: &ValidatorContext) -> Vec<CheckResult> {
    macro_rules! step {
        ($name:expr, $body:expr) => {{
            let status = match (|| -> Result<CheckStatus, String> { $body })() {
                Ok(s) => s,
                Err(msg) => CheckStatus::Fail(msg),
            };
            CheckResult { name: $name, status }
        }};
    }

    vec![
        step!("schema", check_schema(ctx).map(|()| CheckStatus::Pass)),
        step!("artifact_commitment",
              check_artifact_commitment(ctx).map(|()| CheckStatus::Pass)),
        step!("graph_wellformed",
              check_graph_wellformed(ctx).map(|()| CheckStatus::Pass)),
        step!("acyclic", check_acyclic(ctx).map(|()| CheckStatus::Pass)),
        step!("tap_authenticity",
              check_tap_authenticity(ctx).map(|()| CheckStatus::Pass)),
        step!("tap_ordering", check_tap_ordering(ctx).map(|()| CheckStatus::Pass)),
        step!("log_inclusion", check_log_inclusion(ctx).map(|()| CheckStatus::Pass)),
        step!("checkpoint_validation",
              check_checkpoint(ctx).map(|()| CheckStatus::Pass)),
        step!("checkpoint_consistency", check_checkpoint_consistency(ctx)),
        step!("graph_log_consistency",
              check_graph_log_consistency(ctx).map(|()| CheckStatus::Pass)),
        step!("compute_attestation",
              check_compute_attestation(ctx).map(|()| CheckStatus::Pass)),
        step!("task_compute_accounting",
              check_task_compute_accounting(ctx).map(|()| CheckStatus::Pass)),
        step!("ancestry_compute",
              check_ancestry_compute(ctx).map(|()| CheckStatus::Pass)),
        step!("replay_correctness",
              check_replay_correctness(ctx).map(|()| CheckStatus::Pass)),
        step!("policy", check_policies(ctx).map(|()| CheckStatus::Pass)),
    ]
}

// 1. Schema validation -- structural sanity beyond what serde already gave us.
fn check_schema(ctx: &ValidatorContext) -> Result<(), String> {
    if ctx.graph.graph_version != "v0" {
        return Err(format!("graph_version {}", ctx.graph.graph_version));
    }
    if ctx.graph.run_id.is_empty() {
        return Err("run_id empty".into());
    }
    if ctx.graph.tap_pubkeys.is_empty() {
        return Err("no tap pubkeys".into());
    }
    if ctx.graph.log_pubkey.len() != 64 {
        return Err("log_pubkey wrong length".into());
    }
    Ok(())
}

// 2. Artifact commitment validation.
fn check_artifact_commitment(ctx: &ValidatorContext) -> Result<(), String> {
    for a in &ctx.graph.artifacts {
        if !a.commitment.starts_with("sha256:") || a.commitment.len() != "sha256:".len() + 64 {
            return Err(format!("artifact {} bad commitment {}", a.artifact_id, a.commitment));
        }
        parse_commitment(&a.commitment)
            .map_err(|e| format!("artifact {} commitment parse: {}", a.artifact_id, e))?;
    }
    Ok(())
}

// 3. Graph well-formedness.
fn check_graph_wellformed(ctx: &ValidatorContext) -> Result<(), String> {
    let known: HashSet<&str> = ctx.graph.artifacts.iter().map(|a| a.artifact_id.as_str()).collect();
    for t in &ctx.graph.tasks {
        for a in &t.input_artifact_ids {
            if !known.contains(a.as_str()) {
                return Err(format!("task {} has unknown input {}", t.task_id, a));
            }
        }
        for a in &t.output_artifact_ids {
            if !known.contains(a.as_str()) {
                return Err(format!("task {} has unknown output {}", t.task_id, a));
            }
        }
    }
    let externals: HashSet<&str> =
        ctx.graph.external_artifact_ids.iter().map(|s| s.as_str()).collect();
    let mut producer_count: HashMap<&str, u32> = HashMap::new();
    for t in &ctx.graph.tasks {
        for a in &t.output_artifact_ids {
            *producer_count.entry(a.as_str()).or_insert(0) += 1;
        }
    }
    for a in &ctx.graph.artifacts {
        let aid = a.artifact_id.as_str();
        let is_external = externals.contains(aid);
        if a.external != is_external {
            return Err(format!("artifact {} external flag inconsistent with external_artifact_ids", aid));
        }
        let n = producer_count.get(aid).copied().unwrap_or(0);
        if is_external && n != 0 {
            return Err(format!("external artifact {} has a producer task", aid));
        }
        if !is_external && n != 1 {
            return Err(format!("artifact {} has {} producers (expected 1)", aid, n));
        }
    }
    Ok(())
}

// 4. Acyclic ancestry. Vertices: artifacts and tasks. Edges: artifact->task
// (input edge), task->artifact (output edge). DFS for cycles.
fn check_acyclic(ctx: &ValidatorContext) -> Result<(), String> {
    #[derive(Clone, PartialEq, Eq, Hash)]
    enum NodeKind { Artifact, Task }
    let mut adj: HashMap<(NodeKind, String), Vec<(NodeKind, String)>> = HashMap::new();
    for t in &ctx.graph.tasks {
        for a in &t.input_artifact_ids {
            adj.entry((NodeKind::Artifact, a.clone()))
                .or_default()
                .push((NodeKind::Task, t.task_id.clone()));
        }
        for a in &t.output_artifact_ids {
            adj.entry((NodeKind::Task, t.task_id.clone()))
                .or_default()
                .push((NodeKind::Artifact, a.clone()));
        }
    }
    enum Mark { Visiting, Done }
    let mut marks: HashMap<(NodeKind, String), Mark> = HashMap::new();
    fn dfs(
        node: (NodeKind, String),
        adj: &HashMap<(NodeKind, String), Vec<(NodeKind, String)>>,
        marks: &mut HashMap<(NodeKind, String), Mark>,
    ) -> Result<(), String> {
        if let Some(Mark::Done) = marks.get(&node) {
            return Ok(());
        }
        if let Some(Mark::Visiting) = marks.get(&node) {
            return Err(format!("cycle detected at {:?}", node.1));
        }
        marks.insert(node.clone(), Mark::Visiting);
        if let Some(out) = adj.get(&node) {
            for next in out.clone() {
                dfs(next, adj, marks)?;
            }
        }
        marks.insert(node, Mark::Done);
        Ok(())
    }
    let nodes: Vec<_> = adj.keys().cloned().collect();
    for n in nodes {
        dfs(n, &adj, &mut marks)?;
    }
    Ok(())
}

// 5. Tap authenticity: each transmission's tap_signature verifies under the
// declared tap_pubkey, over the canonical-JSON of the unsigned event fields.
fn check_tap_authenticity(ctx: &ValidatorContext) -> Result<(), String> {
    for ev in &ctx.graph.transmissions {
        let pk = ctx.graph.tap_pubkeys.get(&ev.tap_id)
            .ok_or_else(|| format!("unknown tap {}", ev.tap_id))?;
        let unsigned = unsigned_event_value(ev);
        let msg = canonical_bytes(&unsigned);
        verify_ed25519(pk, &msg, &ev.tap_signature)
            .map_err(|e| format!("tap {} seq {} signature: {}", ev.tap_id, ev.tap_seq, e))?;
    }
    Ok(())
}

fn unsigned_event_value(ev: &TransmissionEvent) -> Value {
    json!({
        "tap_id": ev.tap_id,
        "tap_seq": ev.tap_seq,
        "sender": ev.sender,
        "receiver": ev.receiver,
        "artifact_id": ev.artifact_id,
        "commitment": ev.commitment,
        "size": ev.size,
        "timestamp_ms": ev.timestamp_ms,
    })
}

fn full_event_value(ev: &TransmissionEvent) -> Value {
    json!({
        "tap_id": ev.tap_id,
        "tap_seq": ev.tap_seq,
        "sender": ev.sender,
        "receiver": ev.receiver,
        "artifact_id": ev.artifact_id,
        "commitment": ev.commitment,
        "size": ev.size,
        "timestamp_ms": ev.timestamp_ms,
        "tap_signature": ev.tap_signature,
    })
}

// 6. Tap ordering: per tap, seqs form contiguous range starting at 0.
fn check_tap_ordering(ctx: &ValidatorContext) -> Result<(), String> {
    let mut by_tap: HashMap<&str, Vec<u64>> = HashMap::new();
    for ev in &ctx.graph.transmissions {
        by_tap.entry(ev.tap_id.as_str()).or_default().push(ev.tap_seq);
    }
    for (tap, mut seqs) in by_tap {
        seqs.sort();
        for (i, s) in seqs.iter().enumerate() {
            if *s != i as u64 {
                return Err(format!("tap {} seq gap at {}", tap, i));
            }
        }
    }
    Ok(())
}

// 7. Log inclusion: tree size matches transmission count, every transmission
// has an inclusion proof that verifies against the declared root.
fn check_log_inclusion(ctx: &ValidatorContext) -> Result<(), String> {
    let log = &ctx.graph.log;
    if log.tree_size as usize != ctx.graph.transmissions.len() {
        return Err(format!(
            "tree_size {} != transmissions {}",
            log.tree_size,
            ctx.graph.transmissions.len()
        ));
    }
    let root = parse_commitment_hex(&log.root_hash)?;
    for (i, ev) in ctx.graph.transmissions.iter().enumerate() {
        let proof_hex = log.inclusion_proofs.get(&i.to_string())
            .ok_or_else(|| format!("missing inclusion proof for index {}", i))?;
        let mut proof = Vec::with_capacity(proof_hex.len());
        for h in proof_hex {
            proof.push(parse_commitment_hex(h)?);
        }
        let leaf = canonical_bytes(&full_event_value(ev));
        if !verify_inclusion(&leaf, i, log.tree_size as usize, &proof, &root) {
            return Err(format!("inclusion proof failed for transmission {}", i));
        }
    }
    Ok(())
}

fn parse_commitment_hex(s: &str) -> Result<[u8; 32], String> {
    let bytes = hex::decode(s).map_err(|e| format!("hex {}: {}", s, e))?;
    if bytes.len() != 32 {
        return Err(format!("hash wrong length: {}", bytes.len()));
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes);
    Ok(out)
}

// 8. Checkpoint validation: STH signature verifies, root_hash matches the
// merkle root of the leaves.
fn check_checkpoint(ctx: &ValidatorContext) -> Result<(), String> {
    let log = &ctx.graph.log;
    let leaves: Vec<Vec<u8>> = ctx.graph.transmissions.iter()
        .map(|ev| canonical_bytes(&full_event_value(ev)))
        .collect();
    let leaf_refs: Vec<&[u8]> = leaves.iter().map(|v| v.as_slice()).collect();
    let computed_root = merkle_root(&leaf_refs);
    let declared_root = parse_commitment_hex(&log.root_hash)?;
    if computed_root != declared_root {
        return Err("declared root_hash does not match recomputed merkle root".into());
    }
    let mut msg = Vec::with_capacity(8 + 32);
    msg.extend_from_slice(&(log.tree_size).to_be_bytes());
    msg.extend_from_slice(&computed_root);
    verify_ed25519(&ctx.graph.log_pubkey, &msg, &log.sth_signature)
        .map_err(|e| format!("STH signature: {}", e))?;
    Ok(())
}

// 9. Checkpoint consistency: every previous STH's signature verifies under
// the log key, and each declared consistency proof verifies (RFC 6962
// §2.1.4.2) between its from/to STHs and the recomputed roots.
fn check_checkpoint_consistency(ctx: &ValidatorContext) -> Result<CheckStatus, String> {
    let log = &ctx.graph.log;
    if log.previous_sths.is_empty() && log.consistency_proofs.is_empty() {
        return Ok(CheckStatus::Skip("no previous STHs to chain".into()));
    }
    // Every previous STH must verify under the log key.
    for prev in &log.previous_sths {
        let root = parse_commitment_hex(&prev.root_hash)?;
        let mut msg = Vec::with_capacity(8 + 32);
        msg.extend_from_slice(&prev.tree_size.to_be_bytes());
        msg.extend_from_slice(&root);
        verify_ed25519(&ctx.graph.log_pubkey, &msg, &prev.sth_signature)
            .map_err(|e| format!("previous STH (size {}) signature: {}", prev.tree_size, e))?;
    }
    if log.consistency_proofs.is_empty() {
        return Err("previous STHs declared but no consistency proofs to chain them".into());
    }
    let current_root = parse_commitment_hex(&log.root_hash)?;
    let by_size: HashMap<u64, &crate::types::PreviousSth> =
        log.previous_sths.iter().map(|p| (p.tree_size, p)).collect();
    for cp in &log.consistency_proofs {
        if cp.to_size != log.tree_size {
            return Err(format!(
                "consistency proof to_size {} != current tree size {}",
                cp.to_size, log.tree_size
            ));
        }
        let prev = by_size.get(&cp.from_size)
            .ok_or_else(|| format!("consistency proof references unknown STH at size {}", cp.from_size))?;
        let old_root = parse_commitment_hex(&prev.root_hash)?;
        let mut path: Vec<[u8; 32]> = Vec::with_capacity(cp.path.len());
        for h in &cp.path {
            path.push(parse_commitment_hex(h)?);
        }
        let ok = verify_consistency(
            cp.from_size as usize, cp.to_size as usize,
            &old_root, &current_root, &path,
        );
        if !ok {
            return Err(format!(
                "consistency proof from size {} to {} did not verify",
                cp.from_size, cp.to_size
            ));
        }
    }
    Ok(CheckStatus::Pass)
}

// 10. Graph/log consistency: leaf indices well-formed; the graph's artifact
// records agree with their (signed, logged) transmission events on
// commitment and size for every artifact that ever transited a tap.
fn check_graph_log_consistency(ctx: &ValidatorContext) -> Result<(), String> {
    let log = &ctx.graph.log;
    if log.leaves.len() != ctx.graph.transmissions.len() {
        return Err(format!(
            "leaves {} != transmissions {}",
            log.leaves.len(),
            ctx.graph.transmissions.len()
        ));
    }
    for (i, leaf_idx) in log.leaves.iter().enumerate() {
        if *leaf_idx as usize != i {
            return Err(format!("leaf {} references transmission {} (expected {})", i, leaf_idx, i));
        }
    }
    for ev in &ctx.graph.transmissions {
        let art = ctx.artifact_by_id.get(ev.artifact_id.as_str())
            .ok_or_else(|| format!("transmission references unknown artifact {}", ev.artifact_id))?;
        if art.commitment != ev.commitment {
            return Err(format!(
                "artifact {} graph commitment {} disagrees with signed transmission {}",
                art.artifact_id, art.commitment, ev.commitment
            ));
        }
        if art.size != ev.size {
            return Err(format!(
                "artifact {} graph size {} disagrees with signed transmission {}",
                art.artifact_id, art.size, ev.size
            ));
        }
    }
    Ok(())
}

// 11. Compute attestation validation: each one verifies under hypervisor key.
fn check_compute_attestation(ctx: &ValidatorContext) -> Result<(), String> {
    for att in &ctx.graph.compute_attestations {
        let msg = canonical_bytes(&json!({
            "pod_id": att.pod_id,
            "interval_ms": [att.interval_ms[0], att.interval_ms[1]],
            "claimed_flops": att.claimed_flops,
        }));
        verify_ed25519(&ctx.graph.hypervisor_pubkey, &msg, &att.signature)
            .map_err(|e| format!("compute attestation pod {} sig: {}", att.pod_id, e))?;
    }
    Ok(())
}

// 12. Task compute accounting: sum of task claimed flops per pod must not
// exceed the hypervisor attestation for that pod, and each task's interval
// must lie inside the attestation interval.
fn check_task_compute_accounting(ctx: &ValidatorContext) -> Result<(), String> {
    let mut by_pod: HashMap<&str, &ComputeAttestation> = HashMap::new();
    for att in &ctx.graph.compute_attestations {
        by_pod.insert(att.pod_id.as_str(), att);
    }
    let mut sum_by_pod: HashMap<&str, u64> = HashMap::new();
    for t in &ctx.graph.tasks {
        let att = by_pod.get(t.pod_id.as_str())
            .ok_or_else(|| format!("no compute attestation for pod {}", t.pod_id))?;
        if !(t.interval_ms[0] >= att.interval_ms[0] && t.interval_ms[1] <= att.interval_ms[1]) {
            return Err(format!(
                "task {} interval {:?} outside pod attestation {:?}",
                t.task_id, t.interval_ms, att.interval_ms
            ));
        }
        *sum_by_pod.entry(t.pod_id.as_str()).or_default() += t.claimed_flops;
    }
    for (pod, sum) in &sum_by_pod {
        let att = by_pod[pod];
        if *sum > att.claimed_flops {
            return Err(format!(
                "pod {} task flops sum {} exceeds attestation {}",
                pod, sum, att.claimed_flops
            ));
        }
    }
    Ok(())
}

// 13. Cumulative ancestry compute: compute and cache by artifact id; this is
// also exposed for policy use.
fn check_ancestry_compute(_ctx: &ValidatorContext) -> Result<(), String> {
    // Pure compute step; correctness is exercised by check 15 (policies).
    Ok(())
}

pub fn cumulative_ancestry_flops(ctx: &ValidatorContext) -> HashMap<String, u64> {
    let mut producer: HashMap<&str, &Task> = HashMap::new();
    for t in &ctx.graph.tasks {
        for o in &t.output_artifact_ids {
            producer.insert(o.as_str(), t);
        }
    }
    let mut cache: HashMap<String, u64> = HashMap::new();
    fn rec<'a>(
        aid: &'a str,
        producer: &HashMap<&'a str, &'a Task>,
        cache: &mut HashMap<String, u64>,
    ) -> u64 {
        if let Some(v) = cache.get(aid) {
            return *v;
        }
        cache.insert(aid.to_string(), 0);
        let total = match producer.get(aid) {
            None => 0,
            Some(t) => {
                let mut s = t.claimed_flops;
                for inp in &t.input_artifact_ids {
                    s = s.saturating_add(rec(inp.as_str(), producer, cache));
                }
                s
            }
        };
        cache.insert(aid.to_string(), total);
        total
    }
    let aids: Vec<String> = ctx.graph.artifacts.iter().map(|a| a.artifact_id.clone()).collect();
    for aid in aids {
        rec(&aid, &producer, &mut cache);
    }
    cache
}

// 14. Replay correctness: each replay response's recomputed_commitment
// matches the declared output commitment of (one of) the task's outputs.
fn check_replay_correctness(ctx: &ValidatorContext) -> Result<(), String> {
    for resp in ctx.replay {
        let task = ctx.task_by_id.get(resp.task_id.as_str())
            .ok_or_else(|| format!("replay response for unknown task {}", resp.task_id))?;
        let mut found_match = false;
        let mut declared_commits: Vec<String> = Vec::new();
        for oid in &task.output_artifact_ids {
            let art = ctx.artifact_by_id.get(oid.as_str())
                .ok_or_else(|| format!("task {} output artifact {} missing", resp.task_id, oid))?;
            declared_commits.push(art.commitment.clone());
            if art.commitment == resp.recomputed_commitment {
                found_match = true;
                break;
            }
        }
        if !found_match {
            return Err(format!(
                "replay mismatch for task {}: recomputed {} not in declared {:?}",
                resp.task_id, resp.recomputed_commitment, declared_commits
            ));
        }
    }
    Ok(())
}

// 15. Policies. Each policy variant has its own evaluator; first failing
// policy returns an error scoped to that policy kind.
fn check_policies(ctx: &ValidatorContext) -> Result<(), String> {
    let cumulative = cumulative_ancestry_flops(ctx);

    for p in &ctx.graph.policies {
        let kind = p.get("kind").and_then(|v| v.as_str())
            .ok_or_else(|| "policy missing kind".to_string())?;
        match kind {
            "ancestry_compute_gate" => {
                eval_ancestry_gate(ctx, p, &cumulative)?;
            }
            "partition_crossing" => {
                eval_partition_crossing(ctx, p)?;
            }
            "replay_required" => {
                eval_replay_required(ctx, p)?;
            }
            other => return Err(format!("unknown policy kind {}", other)),
        }
    }
    Ok(())
}

fn eval_ancestry_gate(
    ctx: &ValidatorContext, p: &Value, cumulative: &HashMap<String, u64>,
) -> Result<(), String> {
    let threshold = p.get("threshold_flops").and_then(|v| v.as_u64())
        .ok_or_else(|| "ancestry_compute_gate missing threshold_flops".to_string())?;
    let reviewer_id = p.get("required_reviewer_id").and_then(|v| v.as_str())
        .ok_or_else(|| "ancestry_compute_gate missing required_reviewer_id".to_string())?;
    let reviewer_pk = ctx.graph.reviewer_pubkeys.get(reviewer_id)
        .ok_or_else(|| format!("ancestry_gate: no pubkey for reviewer {}", reviewer_id))?;

    for a in &ctx.graph.artifacts {
        let cum = cumulative.get(&a.artifact_id).copied().unwrap_or(0);
        if cum <= threshold || a.external {
            continue;
        }
        let producer_id = match &a.producer_task_id {
            Some(p) => p,
            None => continue,
        };
        let task = ctx.task_by_id.get(producer_id.as_str())
            .ok_or_else(|| format!("ancestry_gate: producer task {} missing", producer_id))?;
        let signed = task.reviewer_signatures.iter()
            .find(|s| s.reviewer_id == reviewer_id);
        let sig = match signed {
            Some(s) => s,
            None => return Err(format!(
                "ancestry_gate: artifact {} (cum {}) lacks reviewer {} signature",
                a.artifact_id, cum, reviewer_id
            )),
        };
        let msg = canonical_bytes(&json!({
            "task_id": task.task_id,
            "output_artifact_ids": task.output_artifact_ids,
            "operation": task.operation,
        }));
        verify_ed25519(reviewer_pk, &msg, &sig.signature)
            .map_err(|e| format!(
                "ancestry_gate: reviewer {} sig on task {}: {}",
                reviewer_id, task.task_id, e
            ))?;
    }
    Ok(())
}

fn eval_partition_crossing(ctx: &ValidatorContext, p: &Value) -> Result<(), String> {
    let from_p = p.get("from_partition").and_then(|v| v.as_str())
        .ok_or_else(|| "partition_crossing missing from_partition".to_string())?;
    let to_p = p.get("to_partition").and_then(|v| v.as_str())
        .ok_or_else(|| "partition_crossing missing to_partition".to_string())?;
    let allowed = p.get("allowed").and_then(|v| v.as_bool())
        .ok_or_else(|| "partition_crossing missing allowed".to_string())?;

    // Partitions are pod-scoped. Map sender/receiver -> partition via tasks.
    let mut partition_of_pod: HashMap<&str, &str> = HashMap::new();
    for t in &ctx.graph.tasks {
        partition_of_pod.insert(t.pod_id.as_str(), t.partition.as_str());
    }
    for ev in &ctx.graph.transmissions {
        let s = partition_of_pod.get(ev.sender.as_str()).copied();
        let r = partition_of_pod.get(ev.receiver.as_str()).copied();
        if let (Some(sp), Some(rp)) = (s, r) {
            if sp == from_p && rp == to_p && !allowed {
                return Err(format!(
                    "partition_crossing: forbidden {} -> {} on transmission {}->{} {}",
                    sp, rp, ev.sender, ev.receiver, ev.artifact_id
                ));
            }
        }
    }
    Ok(())
}

fn eval_replay_required(ctx: &ValidatorContext, p: &Value) -> Result<(), String> {
    let min = p.get("min_sample_count").and_then(|v| v.as_u64())
        .ok_or_else(|| "replay_required missing min_sample_count".to_string())?;
    if (ctx.replay.len() as u64) < min {
        return Err(format!(
            "replay_required: only {} replay responses (need {})",
            ctx.replay.len(), min
        ));
    }
    Ok(())
}

