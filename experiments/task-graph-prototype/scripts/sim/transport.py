"""In-process simulated transport: pod -> tap -> pod.

One artifact per transmission. The tap records a signed transmission event;
the receiver pod gets the artifact bytes stored locally.
"""
from __future__ import annotations

from dataclasses import dataclass

from .pods import Pod, Artifact
from .taps import NetworkTap, TransmissionEvent


@dataclass
class Transport:
    taps: dict[str, NetworkTap]   # link_id -> tap
    clock_ms: int = 0

    def tick(self, dt: int = 1) -> int:
        self.clock_ms += dt
        return self.clock_ms

    def send(
        self, *,
        link_id: str,
        sender: Pod,
        receiver: Pod,
        artifact: Artifact,
    ) -> TransmissionEvent:
        tap = self.taps[link_id]
        ts = self.tick()
        ev = tap.observe(
            sender=sender.pod_id,
            receiver=receiver.pod_id,
            artifact_id=artifact.artifact_id,
            commitment=artifact.commitment,
            size=artifact.size,
            timestamp_ms=ts,
        )
        receiver.store(artifact)
        return ev


__all__ = ["Transport"]
