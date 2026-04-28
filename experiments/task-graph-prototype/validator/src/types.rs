//! Strongly-typed mirror of `attested_task_graph.v0.schema.json`.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestedTaskGraph {
    pub graph_version: String,
    pub run_id: String,
    pub tap_pubkeys: HashMap<String, String>,
    pub log_pubkey: String,
    pub hypervisor_pubkey: String,
    pub reviewer_pubkeys: HashMap<String, String>,
    pub external_artifact_ids: Vec<String>,
    pub tasks: Vec<Task>,
    pub artifacts: Vec<Artifact>,
    pub transmissions: Vec<TransmissionEvent>,
    pub log: TransparencyLog,
    pub compute_attestations: Vec<ComputeAttestation>,
    pub policies: Vec<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub task_id: String,
    pub pod_id: String,
    pub operation: String,
    #[serde(default)]
    pub operation_params: Value,
    pub input_artifact_ids: Vec<String>,
    pub output_artifact_ids: Vec<String>,
    pub claimed_flops: u64,
    pub interval_ms: [u64; 2],
    pub partition: String,
    #[serde(default)]
    pub reviewer_signatures: Vec<ReviewerSignature>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewerSignature {
    pub reviewer_id: String,
    pub signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    pub artifact_id: String,
    pub commitment: String,
    pub size: u64,
    pub external: bool,
    #[serde(default)]
    pub producer_task_id: Option<String>,
    #[serde(default)]
    pub consumer_task_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransmissionEvent {
    pub tap_id: String,
    pub tap_seq: u64,
    pub sender: String,
    pub receiver: String,
    pub artifact_id: String,
    pub commitment: String,
    pub size: u64,
    pub timestamp_ms: u64,
    pub tap_signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransparencyLog {
    pub log_id: String,
    pub tree_size: u64,
    pub root_hash: String,
    pub sth_signature: String,
    #[serde(default)]
    pub previous_sths: Vec<PreviousSth>,
    #[serde(default)]
    pub consistency_proofs: Vec<ConsistencyProof>,
    pub leaves: Vec<u64>,
    pub inclusion_proofs: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreviousSth {
    pub tree_size: u64,
    pub root_hash: String,
    pub sth_signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyProof {
    pub from_size: u64,
    pub to_size: u64,
    pub path: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeAttestation {
    pub pod_id: String,
    pub interval_ms: [u64; 2],
    pub claimed_flops: u64,
    pub signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayResponse {
    pub task_id: String,
    pub recomputed_commitment: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayResponses {
    pub responses: Vec<ReplayResponse>,
}
