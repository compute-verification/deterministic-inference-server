"""Mutators that turn an honest attested graph into one that fails a
specific validator check.

Each mutator returns (mutated_graph, expected_failure_check_name).
"""
from __future__ import annotations

import copy
import hashlib
from typing import Any


def missing_log_entry(graph: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Drop one tap event from the log (but keep it in the graph)."""
    g = copy.deepcopy(graph)
    log = g["log"]
    if not log["leaves"]:
        return g, "log_inclusion"
    log["leaves"].pop()
    log["tree_size"] -= 1
    log["inclusion_proofs"].pop(str(log["tree_size"]))
    # NOTE: we leave root_hash/sth_signature stale on purpose so the
    # validator catches either checkpoint-validation or log-inclusion.
    return g, "log_inclusion"


def bad_tap_signature(graph: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Flip one byte of the first transmission's tap_signature."""
    g = copy.deepcopy(graph)
    sig = g["transmissions"][0]["tap_signature"]
    flipped_byte = "0" if sig[0] != "0" else "1"
    g["transmissions"][0]["tap_signature"] = flipped_byte + sig[1:]
    return g, "tap_authenticity"


def graph_log_mismatch(graph: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Mutate the artifact entry's commitment for an artifact that DID transit
    a tap. The tap signature still verifies (we don't touch the transmission)
    but the graph's claim about the artifact disagrees with the signed log.
    """
    g = copy.deepcopy(graph)
    transited = {ev["artifact_id"] for ev in g["transmissions"]}
    target = next(a for a in g["artifacts"] if a["artifact_id"] in transited)
    h = hashlib.sha256(b"graph-log-forgery").hexdigest()
    target["commitment"] = f"sha256:{h}"
    return g, "graph_log_consistency"


def replay_mismatch(graph: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Change a TERMINAL artifact's commitment (one not transmitted, so it
    doesn't appear in any signed tap event). The graph's claim is the only
    source for its commitment, and replay must catch the divergence.
    """
    g = copy.deepcopy(graph)
    transited = {ev["artifact_id"] for ev in g["transmissions"]}
    target = next(
        a for a in g["artifacts"]
        if not a["external"] and a["producer_task_id"] is not None
        and a["artifact_id"] not in transited
    )
    h = hashlib.sha256(b"replay-forgery").hexdigest()
    target["commitment"] = f"sha256:{h}"
    return g, "replay_correctness"


def policy_gate_violation(graph: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Strip the reviewer signature off every gated task."""
    g = copy.deepcopy(graph)
    for t in g["tasks"]:
        if "reviewer_signatures" in t:
            del t["reviewer_signatures"]
    return g, "policy_ancestry_gate"


def replay_required_violation(graph: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Bump the replay_required policy past the orchestrator's sample size."""
    g = copy.deepcopy(graph)
    for p in g["policies"]:
        if p.get("kind") == "replay_required":
            p["min_sample_count"] = 999
    return g, "policy_replay_required"


def cyclic_graph(graph: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Add a downstream artifact as an input of the first task → cycle."""
    g = copy.deepcopy(graph)
    last_output = g["tasks"][-1]["output_artifact_ids"][0]
    g["tasks"][0]["input_artifact_ids"].append(last_output)
    return g, "acyclic"


def over_budget_compute(graph: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Inflate one task's claimed_flops past its pod's signed attestation."""
    g = copy.deepcopy(graph)
    target = g["tasks"][0]
    pod_att = next(
        a for a in g["compute_attestations"] if a["pod_id"] == target["pod_id"]
    )
    target["claimed_flops"] = pod_att["claimed_flops"] + 1_000_000
    return g, "task_compute_accounting"


def bad_compute_attestation(graph: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Corrupt the hypervisor signature on a compute attestation."""
    g = copy.deepcopy(graph)
    sig = g["compute_attestations"][0]["signature"]
    flipped = "0" if sig[0] != "0" else "1"
    g["compute_attestations"][0]["signature"] = flipped + sig[1:]
    return g, "compute_attestation"


def forbidden_partition_crossing(graph: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Flip research→deployment from allowed=True to allowed=False so every
    cross-partition transmission becomes a policy violation."""
    g = copy.deepcopy(graph)
    for p in g["policies"]:
        if (p.get("kind") == "partition_crossing"
            and p.get("from_partition") == "research"
            and p.get("to_partition") == "deployment"):
            p["allowed"] = False
    return g, "policy_partition_crossing"


def bad_sth_signature(graph: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Corrupt the signed-tree-head signature; root recomputes correctly
    but the STH no longer verifies under the log key."""
    g = copy.deepcopy(graph)
    sig = g["log"]["sth_signature"]
    flipped = "0" if sig[0] != "0" else "1"
    g["log"]["sth_signature"] = flipped + sig[1:]
    return g, "checkpoint_validation"


def dangling_input(graph: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Add a reference to an artifact that doesn't exist."""
    g = copy.deepcopy(graph)
    g["tasks"][0]["input_artifact_ids"].append("artifact:does_not_exist")
    return g, "graph_wellformed"


SCENARIOS = {
    "missing_log_entry": missing_log_entry,
    "bad_tap_signature": bad_tap_signature,
    "bad_sth_signature": bad_sth_signature,
    "graph_log_mismatch": graph_log_mismatch,
    "replay_mismatch": replay_mismatch,
    "dangling_input": dangling_input,
    "cyclic_graph": cyclic_graph,
    "over_budget_compute": over_budget_compute,
    "bad_compute_attestation": bad_compute_attestation,
    "forbidden_partition_crossing": forbidden_partition_crossing,
    "policy_gate_violation": policy_gate_violation,
    "replay_required_violation": replay_required_violation,
}


__all__ = ["SCENARIOS"]
