"""Prover-side capture log.

Lifts the *pattern* from cmd/server/main.py's CaptureLog (append-only JSONL,
monotonic seq, threadsafe) but writes a slightly different inner shape:
each entry has direction/endpoint/payload_digest/payload_path, mirroring
the verifier transcript shape and making prover↔verifier correlation easy.

Phase 3.2 will factor a shared JSONL base both this and TranscriptLog
inherit from. Until then, accept the duplication — designing the
abstraction before seeing both concrete shapes is premature.
"""

from __future__ import annotations

import threading
from pathlib import Path

from pkg.common.deterministic import (
    canonical_json_text,
    sha256_prefixed,
    utc_now_iso,
)


class ProverCaptureLog:
    """Append-only JSONL log of prover-side requests and responses."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate (start fresh) so successive runs don't accumulate.
        self.path.write_text("", encoding="utf-8")
        self._lock = threading.Lock()
        self._seq = 0

    def record(
        self,
        *,
        direction: str,
        endpoint: str,
        payload: bytes,
        status_code: int | None = None,
        payload_path: str | None = None,
    ) -> int:
        """Append one entry; return its seq."""
        if direction not in ("sent", "received"):
            raise ValueError(f"direction must be 'sent' or 'received', got {direction!r}")

        with self._lock:
            self._seq += 1
            seq = self._seq
            entry: dict[str, object] = {
                "seq": seq,
                "direction": direction,
                "endpoint": endpoint,
                "timestamp": utc_now_iso(),
                "payload_digest": sha256_prefixed(payload),
            }
            if status_code is not None:
                entry["status_code"] = status_code
            if payload_path is not None:
                entry["payload_path"] = payload_path

            with self.path.open("a", encoding="utf-8") as f:
                f.write(canonical_json_text(entry))
            return seq

    @property
    def seq(self) -> int:
        with self._lock:
            return self._seq
