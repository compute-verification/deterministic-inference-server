"""Simulated pods running cheap deterministic dummy tasks.

A task takes input artifact bytes, runs an `operation` (a pure function
keyed by name), and emits an output artifact whose bytes are entirely
determined by the inputs and operation parameters. Replay re-executes
the same operation on the same declared inputs and must recover the
declared output commitment.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from .crypto import canonical_json_bytes, commitment


# ---------- operation registry -----------------------------------------------

def _op_concat_hash(inputs: list[bytes], params: dict) -> bytes:
    """Concatenate inputs and parameters, sha256 to 32 bytes of output."""
    h = hashlib.sha256()
    h.update(canonical_json_bytes(params))
    for blob in inputs:
        h.update(b"\x00")
        h.update(len(blob).to_bytes(8, "big"))
        h.update(blob)
    return h.digest()


def _op_xor_fold(inputs: list[bytes], params: dict) -> bytes:
    """XOR-fold all inputs to a fixed-width digest specified in params."""
    width = int(params.get("width", 32))
    out = bytearray(width)
    for blob in inputs:
        for i, b in enumerate(blob):
            out[i % width] ^= b
    return bytes(out)


def _op_identity_pad(inputs: list[bytes], params: dict) -> bytes:
    """Concatenate inputs and pad to declared length."""
    length = int(params.get("length", 64))
    body = b"".join(inputs)
    if len(body) >= length:
        return body[:length]
    return body + b"\x00" * (length - len(body))


OPERATIONS = {
    "concat_hash": _op_concat_hash,
    "xor_fold": _op_xor_fold,
    "identity_pad": _op_identity_pad,
}


def execute(operation: str, inputs: list[bytes], params: dict) -> bytes:
    fn = OPERATIONS.get(operation)
    if fn is None:
        raise KeyError(f"unknown operation: {operation}")
    return fn(inputs, params)


def estimate_flops(operation: str, inputs: list[bytes], params: dict) -> int:
    """Cheap deterministic FLOP proxy. Used as 'claimed_flops' for the demo."""
    total_in = sum(len(b) for b in inputs)
    if operation == "concat_hash":
        return 64 + total_in
    if operation == "xor_fold":
        return total_in
    if operation == "identity_pad":
        return max(int(params.get("length", 64)), total_in)
    return total_in


# ---------- artifact + task records ------------------------------------------

@dataclass
class Artifact:
    artifact_id: str
    bytes_: bytes
    external: bool = False
    schema: str | None = None

    @property
    def commitment(self) -> str:
        return commitment(self.bytes_)

    @property
    def size(self) -> int:
        return len(self.bytes_)


@dataclass
class TaskRecord:
    task_id: str
    pod_id: str
    operation: str
    operation_params: dict
    input_artifact_ids: list[str]
    output_artifact_ids: list[str]
    claimed_flops: int
    interval_ms: tuple[int, int]
    partition: str


# ---------- Pod --------------------------------------------------------------

@dataclass
class Pod:
    pod_id: str
    partition: str
    storage: dict[str, Artifact] = field(default_factory=dict)
    tasks: list[TaskRecord] = field(default_factory=list)
    _next_task_seq: int = 0

    def store(self, artifact: Artifact) -> None:
        self.storage[artifact.artifact_id] = artifact

    def has(self, artifact_id: str) -> bool:
        return artifact_id in self.storage

    def get(self, artifact_id: str) -> Artifact:
        return self.storage[artifact_id]

    def run_task(
        self,
        operation: str,
        operation_params: dict,
        input_artifact_ids: list[str],
        output_artifact_id: str,
        clock_ms: int,
        duration_ms: int = 1,
    ) -> tuple[TaskRecord, Artifact]:
        inputs = [self.storage[a].bytes_ for a in input_artifact_ids]
        out_bytes = execute(operation, inputs, operation_params)
        flops = estimate_flops(operation, inputs, operation_params)
        artifact = Artifact(artifact_id=output_artifact_id, bytes_=out_bytes)
        self.store(artifact)
        task_id = f"{self.pod_id}:t{self._next_task_seq}"
        self._next_task_seq += 1
        rec = TaskRecord(
            task_id=task_id,
            pod_id=self.pod_id,
            operation=operation,
            operation_params=operation_params,
            input_artifact_ids=list(input_artifact_ids),
            output_artifact_ids=[output_artifact_id],
            claimed_flops=flops,
            interval_ms=(clock_ms, clock_ms + duration_ms),
            partition=self.partition,
        )
        self.tasks.append(rec)
        return rec, artifact

    def replay_task(self, task: TaskRecord, inputs_by_id: dict[str, bytes]) -> bytes:
        inputs = [inputs_by_id[a] for a in task.input_artifact_ids]
        return execute(task.operation, inputs, task.operation_params)


__all__ = [
    "Pod", "Artifact", "TaskRecord",
    "OPERATIONS", "execute", "estimate_flops",
]
