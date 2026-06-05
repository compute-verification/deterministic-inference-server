"""Proof server (spec-decode): receives both clusters' spec-decode payloads,
bitwise-compares them (output ids + per-round trace), logs MATCH/MISMATCH, and
builds a speculative-decoding task graph from the host payload.

POST /compare {id, host, recomp} ; GET /health -> counters. The built graph is
written to the work dir; nothing consumes it yet -- building it is the point.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.proof_server.task_graph import build_spec_decode_task_graph  # noqa: E402

LABEL = "proof-server"


class State:
    work_dir = Path("/tmp/spec-decode-graphs")
    draft_model = "hf://Qwen/Qwen3-0.6B"
    target_model = "hf://Qwen/Qwen3-1.7B"
    lock = threading.Lock()
    compared = 0
    matches = 0
    mismatches = 0
    graphs_built = 0


STATE = State()


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, code: int, body: dict) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path == "/health":
            with STATE.lock:
                return self._send_json(200, {
                    "status": "ok", "compared": STATE.compared,
                    "matches": STATE.matches, "mismatches": STATE.mismatches,
                    "graphs_built": STATE.graphs_built,
                })
        return self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/compare":
            return self._send_json(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            req_id = int(body["id"])
            host = body["host"]
            recomp = body["recomp"]
        except Exception as exc:  # noqa: BLE001
            return self._send_json(400, {"error": f"bad compare body: {exc}"})

        # 1. Compare the two clusters' spec-decode results.
        is_match = (host.get("output_ids") == recomp.get("output_ids")
                    and host.get("rounds") == recomp.get("rounds"))
        verdict = "MATCH" if is_match else "MISMATCH"
        sys.stderr.write(f"[{LABEL}] id={req_id} {verdict} "
                         f"(out={len(host.get('output_ids', []))} tok, "
                         f"{host.get('target_passes')} target passes)\n")

        # 2. Build the spec-decode task graph from the host payload.
        graph_ok = False
        try:
            graph = build_spec_decode_task_graph(
                request_id=req_id,
                draft_model_source=STATE.draft_model,
                target_model_source=STATE.target_model,
                prompt_len=int(host["prompt_len"]),
                rounds=host["rounds"],
            )
            STATE.work_dir.mkdir(parents=True, exist_ok=True)
            out = STATE.work_dir / f"spec_graph_{req_id}.json"
            out.write_text(graph.to_json())
            graph_ok = True
            sys.stderr.write(f"[{LABEL}] id={req_id} spec graph: {len(graph.nodes)} nodes -> {out}\n")
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"[{LABEL}] id={req_id} graph build failed: {exc}\n")

        with STATE.lock:
            STATE.compared += 1
            STATE.matches += int(is_match)
            STATE.mismatches += int(not is_match)
            STATE.graphs_built += int(graph_ok)

        return self._send_json(200, {"verdict": verdict, "graph_built": graph_ok})

    def log_message(self, format, *args):  # noqa: A002
        sys.stderr.write(f"[{LABEL}] " + (format % args) + "\n")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> int:
    p = argparse.ArgumentParser(description="Spec-decode proof server")
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--work-dir", default="/tmp/spec-decode-graphs")
    p.add_argument("--draft-model-source", default=State.draft_model)
    p.add_argument("--target-model-source", default=State.target_model)
    args = p.parse_args()

    STATE.work_dir = Path(args.work_dir)
    STATE.draft_model = args.draft_model_source
    STATE.target_model = args.target_model_source

    server = ThreadedHTTPServer((args.host, args.port), Handler)

    def _shutdown(signum, frame):  # noqa: ARG001
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"[{LABEL}] listening on {args.host}:{args.port}; work_dir={STATE.work_dir}")
    sys.stdout.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
