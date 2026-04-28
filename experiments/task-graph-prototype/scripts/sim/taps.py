"""Network tap: observes one artifact transmission and signs an event.

Each tap holds an ed25519 keypair and a monotonically increasing sequence
number per emitted event. The signed event becomes the leaf in the
transparency log.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .crypto import canonical_json_bytes, sign


@dataclass
class TransmissionEvent:
    tap_id: str
    tap_seq: int
    sender: str
    receiver: str
    artifact_id: str
    commitment: str
    size: int
    timestamp_ms: int
    tap_signature: str  # hex

    def to_dict(self) -> dict[str, Any]:
        return {
            "tap_id": self.tap_id,
            "tap_seq": self.tap_seq,
            "sender": self.sender,
            "receiver": self.receiver,
            "artifact_id": self.artifact_id,
            "commitment": self.commitment,
            "size": self.size,
            "timestamp_ms": self.timestamp_ms,
            "tap_signature": self.tap_signature,
        }


def sign_payload(
    *, tap_id: str, tap_seq: int, sender: str, receiver: str,
    artifact_id: str, commitment: str, size: int, timestamp_ms: int,
) -> bytes:
    """Canonical JSON of the unsigned event. Source of truth for the signed
    blob; mirrored byte-for-byte by the Rust validator.
    """
    return canonical_json_bytes({
        "tap_id": tap_id,
        "tap_seq": tap_seq,
        "sender": sender,
        "receiver": receiver,
        "artifact_id": artifact_id,
        "commitment": commitment,
        "size": size,
        "timestamp_ms": timestamp_ms,
    })


@dataclass
class NetworkTap:
    tap_id: str
    sk: Ed25519PrivateKey
    seq: int = 0
    emitted: list[TransmissionEvent] = field(default_factory=list)

    def observe(
        self, *,
        sender: str, receiver: str, artifact_id: str,
        commitment: str, size: int, timestamp_ms: int,
    ) -> TransmissionEvent:
        s = self.seq
        self.seq += 1
        msg = sign_payload(
            tap_id=self.tap_id, tap_seq=s, sender=sender, receiver=receiver,
            artifact_id=artifact_id, commitment=commitment,
            size=size, timestamp_ms=timestamp_ms,
        )
        ev = TransmissionEvent(
            tap_id=self.tap_id, tap_seq=s, sender=sender, receiver=receiver,
            artifact_id=artifact_id, commitment=commitment,
            size=size, timestamp_ms=timestamp_ms,
            tap_signature=sign(self.sk, msg),
        )
        self.emitted.append(ev)
        return ev


__all__ = ["NetworkTap", "TransmissionEvent", "sign_payload"]
