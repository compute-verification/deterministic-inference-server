"""Proof server (compare + task-graph variant).

This is the "proof server" in the sense of the task-graph demo: a developer-
controlled entity that receives BOTH clusters' token responses for a request,
bitwise-compares them, and logs whether they match. It is distinct from
``demos/proof-server/`` (the SP1/Merkle-ledger proxy) -- this one does no SP1,
no ledger; it only compares and builds a task graph.

Data path (Tap forwards both):
    Tap --/verify--> Recomp Cluster        (recomp recomputes, returns its output)
    Tap --/compare-> Proof Server (here)    {id, prompt, host_output, recomp_output}

On ``POST /compare`` the server:
  1. bitwise-compares ``host_output`` vs ``recomp_output``, logs MATCH/MISMATCH,
  2. builds a task graph from (prompt, host_output) -- see
     ``modules.proof_server.task_graph`` -- and writes it to the work dir.
     Nothing consumes the graph yet; building it is the point.

``GET /health`` returns running counters so a demo/test can assert progress.
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
TRACERS = REPO_ROOT / "demos" / "proof-compare" / "tracers"
for _p in (REPO_ROOT, TRACERS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from modules.proof_server import flops as _flops  # noqa: E402
from modules.proof_server.graph import build_graph  # noqa: E402
from inference import trace_inference  # noqa: E402


class _State:
    work_dir: Path = Path("/tmp/proof-compare")
    model_source: str = ""
    lock = threading.Lock()
    compared = 0
    matches = 0
    mismatches = 0
    graphs_built = 0


STATE = _State()


class CompareHandler(BaseHTTPRequestHandler):
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
                    "status": "ok",
                    "compared": STATE.compared,
                    "matches": STATE.matches,
                    "mismatches": STATE.mismatches,
                    "graphs_built": STATE.graphs_built,
                })
        return self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/compare":
            return self._send_json(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            body = json.loads(raw)
            req_id = int(body["id"])
            prompt = str(body["prompt"])
            host_output = str(body["host_output"])
            recomp_output = str(body["recomp_output"])
        except Exception as exc:  # noqa: BLE001
            return self._send_json(400, {"error": f"bad compare body: {exc}"})

        # 1. Compare the two clusters' token responses.
        is_match = host_output == recomp_output
        verdict = "MATCH" if is_match else "MISMATCH"
        sys.stderr.write(f"[proof-server] id={req_id} {verdict}\n")

        # 2. Build the task graph from (prompt, host_output). Built, stored, not
        #    yet consumed -- a failure here must not break the compare path.
        graph_ok = False
        try:
            # Mock-mode demo: no tokenizer here, so whitespace-tokenize as a
            # coarse stand-in (the real per-token graph is the GPU capture path).
            prompt_toks = prompt.split() or [prompt]
            out_toks = host_output.split() or [host_output]
            out_iter = iter(out_toks)
            trace = trace_inference(
                prompt_ids=prompt_toks,
                next_token=lambda _ctx: next(out_iter),
                model_key=STATE.model_source,
                shape_config=_flops.shape_for(STATE.model_source),
                max_tokens=len(out_toks),
            )
            graph = build_graph(trace)
            STATE.work_dir.mkdir(parents=True, exist_ok=True)
            out_path = STATE.work_dir / f"task_graph_{req_id}.json"
            out_path.write_text(graph.to_json())
            graph_ok = True
            sys.stderr.write(
                f"[proof-server] id={req_id} task graph: "
                f"{len(graph.tasks)} tasks -> {out_path}\n"
            )
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"[proof-server] id={req_id} task graph build failed: {exc}\n")

        with STATE.lock:
            STATE.compared += 1
            if is_match:
                STATE.matches += 1
            else:
                STATE.mismatches += 1
            if graph_ok:
                STATE.graphs_built += 1

        return self._send_json(200, {"verdict": verdict, "graph_built": graph_ok})

    def log_message(self, format, *args):  # noqa: A002
        sys.stderr.write("[proof-server] " + (format % args) + "\n")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> int:
    parser = argparse.ArgumentParser(description="Proof server (compare + task graph)")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--work-dir", default="/tmp/proof-compare",
                        help="Where built task graphs are written.")
    parser.add_argument("--model-source", default="hf://Qwen/Qwen3-1.7B",
                        help="Manifest model.source; selects FLOPs model dims.")
    args = parser.parse_args()

    STATE.work_dir = Path(args.work_dir)
    STATE.model_source = args.model_source

    server = ThreadedHTTPServer((args.host, args.port), CompareHandler)

    def _shutdown(signum, frame):  # noqa: ARG001
        sys.stderr.write("[proof-server] shutting down\n")
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"[proof-server] listening on {args.host}:{args.port}; "
          f"work_dir={STATE.work_dir}; model={STATE.model_source}")
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
