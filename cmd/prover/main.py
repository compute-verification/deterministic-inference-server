#!/usr/bin/env python3
"""Prover server (prover-verifier-demo).

Stdlib HTTP server that owns the prover-side endpoints of the prover ↔
verifier protocol. Phase 2 lands the skeleton + /health; subsequent phases
add /graph, /replay, /workload/{start,stop}, /attestation/{id}, and an
optional /debug/emit-frames.

Usage:
    python3 cmd/prover/main.py \\
        --host 127.0.0.1 --port 0 --port-file /tmp/prover.port \\
        --run-id demo-001 --out-dir /tmp/prover-demo
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pydantic import ValidationError as PydanticValidationError  # noqa: E402

from pkg.common.contracts import ValidationError, validate_with_schema  # noqa: E402
from pkg.proverdet.capture import ProverCaptureLog  # noqa: E402
from pkg.proverdet.graph_builder import build_empty_graph  # noqa: E402
from pkg.proverdet.replay import stub_evidence  # noqa: E402
from pkg.proverdet.wire import ReplayRequest  # noqa: E402


class ProverState:
    """Shared mutable state for the prover server.

    Mirrors `cmd/server/main.py`'s `ServerState` shape but only owns what the
    prover actually needs: a run id, an output dir, a cross-handler lock.
    Workload thread, capture log, etc. land in later tasks.
    """

    def __init__(
        self,
        *,
        run_id: str,
        out_dir: Path,
        verifier_url: str | None = None,
        debug_mode: bool = False,
    ) -> None:
        self.run_id = run_id
        self.out_dir = out_dir
        self.verifier_url = verifier_url
        self.debug_mode = debug_mode
        self.lock = threading.Lock()
        self.capture_log = ProverCaptureLog(out_dir / "capture.jsonl")


class ProverHandler(BaseHTTPRequestHandler):
    """Stdlib request handler for the prover. One method per route family."""

    state: ProverState | None = None

    server_version = "ProverServer/0.1"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:
        # The default logger writes to stderr at every request; we want
        # the demo to be quiet. The capture log (Task 2.4) is the real audit
        # trail.
        return

    # -- helpers --

    def _send_json(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        if self.state is not None:
            self.state.capture_log.record(
                direction="sent",
                endpoint=self.path,
                payload=data,
                status_code=status,
            )

    # -- GET --

    def do_GET(self) -> None:
        if self.path == "/health":
            return self._send_json(200, {"ok": True})
        if self.path == "/graph":
            return self._handle_get_graph()
        return self._send_json(404, {"error": "not found"})

    def _handle_get_graph(self) -> None:
        if self.state is None:
            return self._send_json(500, {"error": "prover state not initialized"})
        graph = build_empty_graph(run_id=self.state.run_id)
        body = graph.model_dump(exclude_none=True)
        try:
            validate_with_schema("prover_graph.v1.schema.json", body)
        except ValidationError as exc:
            # Schema mismatch is a programmer error; surface 500 with the
            # message so it shows up in tests.
            return self._send_json(500, {"error": f"graph schema mismatch: {exc}"})
        return self._send_json(200, body)

    # -- POST --

    def do_POST(self) -> None:
        if self.path == "/replay":
            return self._handle_post_replay()
        return self._send_json(404, {"error": "not found"})

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return b""
        return self.rfile.read(length)

    def _handle_post_replay(self) -> None:
        raw = self._read_body()
        if self.state is not None:
            self.state.capture_log.record(
                direction="received",
                endpoint=self.path,
                payload=raw,
            )
        if not raw:
            return self._send_json(400, {"error": "empty request body"})
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            return self._send_json(400, {"error": f"invalid JSON: {exc}"})

        # Schema-validate first (fast 400 with a path), then Pydantic-validate
        # (which gives us a typed object).
        try:
            validate_with_schema("replay_request.v1.schema.json", payload)
        except ValidationError as exc:
            return self._send_json(400, {"error": str(exc)})

        try:
            req = ReplayRequest.model_validate(payload)
        except PydanticValidationError as exc:
            return self._send_json(400, {"error": str(exc)})

        evidence = stub_evidence(req)
        body = evidence.model_dump(exclude_none=True)
        # Defensive: belt-and-braces validate evidence too.
        try:
            validate_with_schema("replay_evidence.v1.schema.json", body)
        except ValidationError as exc:
            return self._send_json(500, {"error": f"evidence schema mismatch: {exc}"})
        return self._send_json(200, body)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _write_port_file(port_file: Path, port: int) -> None:
    port_file.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(port_file), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    try:
        os.write(fd, f"{port}\n".encode())
        os.fsync(fd)
    finally:
        os.close(fd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prover server (prover-verifier-demo)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument(
        "--port-file",
        type=Path,
        default=None,
        help="Write the bound port to this file, fsync, then serve.",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--verifier-url",
        default=None,
        help="Base URL of the verifier server (used by traffic publisher).",
    )
    parser.add_argument(
        "--debug-mode",
        action="store_true",
        help="Enable test-only endpoints like /debug/emit-frames.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    state = ProverState(
        run_id=args.run_id,
        out_dir=args.out_dir,
        verifier_url=args.verifier_url,
        debug_mode=args.debug_mode,
    )
    ProverHandler.state = state

    server = ThreadedHTTPServer((args.host, args.port), ProverHandler)
    bound_host, bound_port = server.server_address[0], server.server_address[1]

    if args.port_file:
        _write_port_file(args.port_file, bound_port)

    print(
        f"prover: serving on {bound_host}:{bound_port} run_id={args.run_id} out_dir={args.out_dir}",
        flush=True,
    )

    def shutdown(signum: int, _frame: Any) -> None:
        print(f"prover: caught signal {signum}, shutting down", flush=True)
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    server.serve_forever()
    server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
