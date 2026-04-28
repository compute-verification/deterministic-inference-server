"""The canonical demo workload.

Two partitions (research, deployment) and five pods. Fan-out from
pod_R0 to two parallel research pods; fan-in at pod_D0; one final
deployment task. The orchestrator signs an intermediate STH part-way
through so the validator has a consistency proof to verify.

Cumulative ancestry compute for the deployment artifacts crosses the
policy threshold, so the producer tasks carry reviewer signatures.
"""
from __future__ import annotations

from typing import Any

from .orchestrator import Orchestrator


def build_honest(run_id: str = "tg-demo-001") -> tuple[Orchestrator, dict[str, Any]]:
    orch = Orchestrator.build(
        run_id=run_id,
        pod_specs=[
            ("pod_R0", "research"),
            ("pod_R1", "research"),
            ("pod_R2", "research"),
            ("pod_D0", "deployment"),
            ("pod_D1", "deployment"),
        ],
        link_specs=["tap_R0R1", "tap_R0R2", "tap_R1D0", "tap_R2D0", "tap_D0D1"],
        reviewer_ids=["reviewer_alice"],
    )

    # External (source) artifacts injected at pod_R0.
    orch.inject_external("pod_R0", "external:prompt_a", b"hello world")
    orch.inject_external("pod_R0", "external:prompt_b", b"the quick brown fox")
    orch.inject_external("pod_R0", "external:model_seed", b"\x42" * 16)

    # Stage A: pod_R0 produces a single research artifact from the externals.
    orch.run_task(
        pod_id="pod_R0",
        operation="concat_hash",
        operation_params={"label": "stage-a"},
        input_artifact_ids=[
            "external:prompt_a", "external:prompt_b", "external:model_seed",
        ],
        output_artifact_id="research_artifact_0",
        duration_ms=2,
    )

    # Fan-out: same research artifact transmitted to TWO research pods.
    orch.transmit(
        link_id="tap_R0R1",
        sender_pod_id="pod_R0", receiver_pod_id="pod_R1",
        artifact_id="research_artifact_0",
    )
    orch.transmit(
        link_id="tap_R0R2",
        sender_pod_id="pod_R0", receiver_pod_id="pod_R2",
        artifact_id="research_artifact_0",
    )

    # Intermediate STH — pinned by the log here. The orchestrator later
    # emits a consistency proof from this checkpoint to the final tree
    # head, demonstrating the log is append-only between the two STHs.
    orch.checkpoint_log()

    # Stage B: two parallel summary computations on the two research pods.
    orch.run_task(
        pod_id="pod_R1",
        operation="xor_fold",
        operation_params={"width": 32},
        input_artifact_ids=["research_artifact_0"],
        output_artifact_id="research_summary_a",
        duration_ms=1,
    )
    orch.run_task(
        pod_id="pod_R2",
        operation="identity_pad",
        operation_params={"length": 64},
        input_artifact_ids=["research_artifact_0"],
        output_artifact_id="research_summary_b",
        duration_ms=1,
    )

    # Cross-partition transmissions — research → deployment.
    orch.transmit(
        link_id="tap_R1D0",
        sender_pod_id="pod_R1", receiver_pod_id="pod_D0",
        artifact_id="research_summary_a",
    )
    orch.transmit(
        link_id="tap_R2D0",
        sender_pod_id="pod_R2", receiver_pod_id="pod_D0",
        artifact_id="research_summary_b",
    )

    # Stage C: pod_D0 fans BOTH summaries in to one deployment input.
    orch.run_task(
        pod_id="pod_D0",
        operation="concat_hash",
        operation_params={"label": "merge"},
        input_artifact_ids=["research_summary_a", "research_summary_b"],
        output_artifact_id="deployment_input",
        duration_ms=1,
    )
    # Cumulative ancestry compute on deployment_input crosses 300 — needs
    # reviewer sign-off on its producer task.
    orch.add_reviewer_signature(
        task_id=orch.tasks[-1].task_id, reviewer_id="reviewer_alice",
    )
    orch.transmit(
        link_id="tap_D0D1",
        sender_pod_id="pod_D0", receiver_pod_id="pod_D1",
        artifact_id="deployment_input",
    )

    # Stage D: final deployment artifact.
    orch.run_task(
        pod_id="pod_D1",
        operation="concat_hash",
        operation_params={"label": "stage-final"},
        input_artifact_ids=["deployment_input"],
        output_artifact_id="final_output",
        duration_ms=1,
    )
    # Also above the gate — needs reviewer sign-off.
    orch.add_reviewer_signature(
        task_id=orch.tasks[-1].task_id, reviewer_id="reviewer_alice",
    )

    policies: list[dict[str, Any]] = [
        {
            "kind": "ancestry_compute_gate",
            "threshold_flops": 300,
            "required_reviewer_id": "reviewer_alice",
        },
        {
            "kind": "partition_crossing",
            "from_partition": "research",
            "to_partition": "deployment",
            "allowed": True,
        },
        {
            "kind": "partition_crossing",
            "from_partition": "deployment",
            "to_partition": "research",
            "allowed": False,
        },
        {
            "kind": "replay_required",
            "min_sample_count": 1,
        },
    ]
    graph = orch.emit(policies)
    return orch, graph


__all__ = ["build_honest"]
