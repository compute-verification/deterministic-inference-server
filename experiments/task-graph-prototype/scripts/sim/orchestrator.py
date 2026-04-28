"""Orchestrator: schedules a small workload across N pods, captures every
transmission via taps, appends events to the transparency log, signs compute
attestations with the hypervisor key, attaches reviewer signatures, and
emits the canonical AttestedTaskGraph JSON consumed by the Rust validator.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .crypto import (
    canonical_json_bytes,
    derive_keypair,
    pubkey_hex,
    sign,
)
from .pods import Pod, Artifact, TaskRecord
from .taps import NetworkTap, TransmissionEvent, sign_payload
from .transport import Transport
from .transparency_log import TransparencyLog


# ---------- key bundle -------------------------------------------------------

@dataclass
class KeyBundle:
    log_sk: Ed25519PrivateKey
    log_pk_hex: str
    hypervisor_sk: Ed25519PrivateKey
    hypervisor_pk_hex: str
    tap_sks: dict[str, Ed25519PrivateKey]
    tap_pks_hex: dict[str, str]
    reviewer_sks: dict[str, Ed25519PrivateKey]
    reviewer_pks_hex: dict[str, str]


def make_keys(tap_ids: list[str], reviewer_ids: list[str]) -> KeyBundle:
    log_sk, log_pk = derive_keypair("log:global")
    hv_sk, hv_pk = derive_keypair("hypervisor:auditor-trusted")
    tap_sks: dict[str, Ed25519PrivateKey] = {}
    tap_pks: dict[str, str] = {}
    for tid in tap_ids:
        sk, pk = derive_keypair(f"tap:{tid}")
        tap_sks[tid] = sk
        tap_pks[tid] = pubkey_hex(pk)
    rev_sks: dict[str, Ed25519PrivateKey] = {}
    rev_pks: dict[str, str] = {}
    for rid in reviewer_ids:
        sk, pk = derive_keypair(f"reviewer:{rid}")
        rev_sks[rid] = sk
        rev_pks[rid] = pubkey_hex(pk)
    return KeyBundle(
        log_sk=log_sk, log_pk_hex=pubkey_hex(log_pk),
        hypervisor_sk=hv_sk, hypervisor_pk_hex=pubkey_hex(hv_pk),
        tap_sks=tap_sks, tap_pks_hex=tap_pks,
        reviewer_sks=rev_sks, reviewer_pks_hex=rev_pks,
    )


# ---------- orchestrator -----------------------------------------------------

@dataclass
class Orchestrator:
    run_id: str
    pods: dict[str, Pod]
    taps: dict[str, NetworkTap]
    log: TransparencyLog
    keys: KeyBundle
    transport: Transport
    transmissions: list[TransmissionEvent] = field(default_factory=list)
    leaf_indices: list[int] = field(default_factory=list)  # parallel to transmissions
    artifacts: dict[str, Artifact] = field(default_factory=dict)
    tasks: list[TaskRecord] = field(default_factory=list)
    external_artifact_ids: set[str] = field(default_factory=set)
    reviewer_sigs_by_task: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    pod_intervals: dict[str, list[tuple[int, int]]] = field(default_factory=dict)

    @classmethod
    def build(
        cls, run_id: str,
        pod_specs: list[tuple[str, str]],   # (pod_id, partition)
        link_specs: list[str],              # tap_ids
        reviewer_ids: list[str],
    ) -> "Orchestrator":
        keys = make_keys(link_specs, reviewer_ids)
        pods = {pid: Pod(pod_id=pid, partition=part) for pid, part in pod_specs}
        taps = {tid: NetworkTap(tap_id=tid, sk=keys.tap_sks[tid]) for tid in link_specs}
        log = TransparencyLog(sk=keys.log_sk)
        transport = Transport(taps=taps)
        return cls(
            run_id=run_id, pods=pods, taps=taps, log=log,
            keys=keys, transport=transport,
        )

    # --- workload primitives -------------------------------------------------

    def inject_external(self, pod_id: str, artifact_id: str, blob: bytes) -> Artifact:
        art = Artifact(artifact_id=artifact_id, bytes_=blob, external=True)
        self.pods[pod_id].store(art)
        self.artifacts[artifact_id] = art
        self.external_artifact_ids.add(artifact_id)
        return art

    def run_task(
        self, *,
        pod_id: str,
        operation: str,
        operation_params: dict,
        input_artifact_ids: list[str],
        output_artifact_id: str,
        duration_ms: int = 1,
    ) -> tuple[TaskRecord, Artifact]:
        pod = self.pods[pod_id]
        clock = self.transport.tick()
        rec, art = pod.run_task(
            operation=operation,
            operation_params=operation_params,
            input_artifact_ids=input_artifact_ids,
            output_artifact_id=output_artifact_id,
            clock_ms=clock,
            duration_ms=duration_ms,
        )
        self.tasks.append(rec)
        self.artifacts[output_artifact_id] = art
        self.pod_intervals.setdefault(pod_id, []).append(rec.interval_ms)
        return rec, art

    def transmit(
        self, *, link_id: str, sender_pod_id: str, receiver_pod_id: str,
        artifact_id: str,
    ) -> TransmissionEvent:
        sender = self.pods[sender_pod_id]
        receiver = self.pods[receiver_pod_id]
        artifact = sender.get(artifact_id)
        ev = self.transport.send(
            link_id=link_id, sender=sender, receiver=receiver, artifact=artifact,
        )
        self.transmissions.append(ev)
        # Append leaf to transparency log: canonical-JSON of the (signed) event.
        leaf = canonical_json_bytes(ev.to_dict())
        idx = self.log.append(leaf)
        self.leaf_indices.append(idx)
        return ev

    def checkpoint_log(self) -> "SignedTreeHead":
        """Sign and remember an intermediate STH at the current log size.
        Used to prove append-only consistency between the intermediate
        and final tree heads at emit time.
        """
        return self.log.sign_sth()

    def add_reviewer_signature(self, task_id: str, reviewer_id: str) -> None:
        """Reviewer signs the canonical-JSON identity of the task being approved."""
        sk = self.keys.reviewer_sks[reviewer_id]
        # Find the task and serialize its identity (task_id + output ids).
        task = next(t for t in self.tasks if t.task_id == task_id)
        msg = canonical_json_bytes({
            "task_id": task.task_id,
            "output_artifact_ids": list(task.output_artifact_ids),
            "operation": task.operation,
        })
        sig = sign(sk, msg)
        self.reviewer_sigs_by_task.setdefault(task_id, []).append({
            "reviewer_id": reviewer_id, "signature": sig,
        })

    def hypervisor_attest(self, pod_id: str) -> dict[str, Any]:
        """Sign a single hypervisor attestation per pod covering all its
        intervals. claimed_flops is the sum of claimed_flops across the pod's
        tasks — for the prototype the hypervisor is simulated and trusts the
        pod's aggregate.
        """
        intervals = self.pod_intervals.get(pod_id, [])
        if not intervals:
            return {}
        start = min(s for s, _ in intervals)
        end = max(e for _, e in intervals)
        total_flops = sum(t.claimed_flops for t in self.tasks if t.pod_id == pod_id)
        msg = canonical_json_bytes({
            "pod_id": pod_id,
            "interval_ms": [start, end],
            "claimed_flops": total_flops,
        })
        return {
            "pod_id": pod_id,
            "interval_ms": [start, end],
            "claimed_flops": total_flops,
            "signature": sign(self.keys.hypervisor_sk, msg),
        }

    # --- emit attested graph -------------------------------------------------

    def emit(self, policies: list[dict[str, Any]]) -> dict[str, Any]:
        # Sign STH at current tree size (the "current" head).
        sth = self.log.sign_sth()

        # Previous STHs are everything in history *before* the final one.
        prev_sths_obj = self.log.sth_history[:-1]

        # Build a consistency proof from each previous STH to the current one.
        consistency_proofs: list[dict[str, Any]] = []
        for prev in prev_sths_obj:
            proof = self.log.consistency_proof(prev.tree_size)
            consistency_proofs.append({
                "from_size": prev.tree_size,
                "to_size": sth.tree_size,
                "path": [h.hex() for h in proof],
            })

        previous_sths_json = [
            {
                "tree_size": p.tree_size,
                "root_hash": p.root_hash.hex(),
                "sth_signature": p.signature,
            }
            for p in prev_sths_obj
        ]

        # Per-leaf inclusion proofs — keyed by string index for JSON.
        inclusion: dict[str, list[str]] = {}
        for i in range(len(self.log.leaves)):
            proof = self.log.inclusion_proof(i)
            inclusion[str(i)] = [h.hex() for h in proof]

        # Compute attestations across all pods that ran tasks.
        compute_atts = [
            self.hypervisor_attest(pid)
            for pid in sorted(self.pod_intervals.keys())
        ]

        tasks_json = []
        for t in self.tasks:
            d: dict[str, Any] = {
                "task_id": t.task_id,
                "pod_id": t.pod_id,
                "operation": t.operation,
                "operation_params": t.operation_params,
                "input_artifact_ids": t.input_artifact_ids,
                "output_artifact_ids": t.output_artifact_ids,
                "claimed_flops": t.claimed_flops,
                "interval_ms": list(t.interval_ms),
                "partition": t.partition,
            }
            if t.task_id in self.reviewer_sigs_by_task:
                d["reviewer_signatures"] = self.reviewer_sigs_by_task[t.task_id]
            tasks_json.append(d)

        # Artifact records reference producer/consumer derivable from tasks.
        consumers: dict[str, list[str]] = {}
        producer: dict[str, str | None] = {}
        for t in self.tasks:
            for a in t.input_artifact_ids:
                consumers.setdefault(a, []).append(t.task_id)
            for a in t.output_artifact_ids:
                producer[a] = t.task_id
        artifacts_json = []
        for aid in sorted(self.artifacts.keys()):
            art = self.artifacts[aid]
            artifacts_json.append({
                "artifact_id": aid,
                "commitment": art.commitment,
                "size": art.size,
                "external": art.external,
                "producer_task_id": producer.get(aid),
                "consumer_task_ids": consumers.get(aid, []),
            })

        return {
            "graph_version": "v0",
            "run_id": self.run_id,
            "tap_pubkeys": self.keys.tap_pks_hex,
            "log_pubkey": self.keys.log_pk_hex,
            "hypervisor_pubkey": self.keys.hypervisor_pk_hex,
            "reviewer_pubkeys": self.keys.reviewer_pks_hex,
            "external_artifact_ids": sorted(self.external_artifact_ids),
            "tasks": tasks_json,
            "artifacts": artifacts_json,
            "transmissions": [ev.to_dict() for ev in self.transmissions],
            "log": {
                "log_id": "log:global",
                "tree_size": sth.tree_size,
                "root_hash": sth.root_hash.hex(),
                "sth_signature": sth.signature,
                "previous_sths": previous_sths_json,
                "consistency_proofs": consistency_proofs,
                "leaves": list(range(len(self.log.leaves))),
                "inclusion_proofs": inclusion,
            },
            "compute_attestations": compute_atts,
            "policies": policies,
        }

    def replay_task(self, task_id: str) -> str:
        """Replay handler used by the validator's challenge.

        Re-execute the task in a clean pod context using ONLY the declared
        inputs, look up by id from the orchestrator's artifact store. Returns
        the new commitment of the recomputed output bytes.
        """
        from .crypto import commitment as _commitment
        task = next(t for t in self.tasks if t.task_id == task_id)
        inputs_by_id = {a: self.artifacts[a].bytes_ for a in task.input_artifact_ids}
        out_bytes = self.pods[task.pod_id].replay_task(task, inputs_by_id)
        return _commitment(out_bytes)


def write_attested_graph(path: str, graph: dict[str, Any]) -> None:
    with open(path, "w") as f:
        f.write(canonical_json_bytes(graph).decode("utf-8"))


__all__ = ["Orchestrator", "KeyBundle", "make_keys", "write_attested_graph"]
