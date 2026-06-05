"""Recomp Cluster (spec-decode): re-runs the same greedy speculative decode and
bitwise-compares its output ids + per-round trace against the host's response.

Exposes /verify (not /request). Returns {is_verified, reason?, recomp_payload}
so the Tap can forward both clusters' payloads to the proof server. `--mock`
runs the deterministic mock backend; both clusters in mock mode produce the same
trace so the compare passes.
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
DEMO_DIR = Path(__file__).resolve().parent.parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import spec_decode as sd
from servers.envelope import SignedEnvelope, SpecDecodeRequest, verify

LABEL = "recomp_cluster"


class State:
    is_warm = False
    mock = False
    draft_id = "Qwen/Qwen3-0.6B"
    target_id = "Qwen/Qwen3-1.7B"
    hf = None


STATE = State()


def _boot(args) -> None:
    if args.mock:
        STATE.is_warm = True
        sys.stderr.write(f"[{LABEL}] mock mode; warm immediately\n")
        return
    sys.stderr.write(f"[{LABEL}] loading HF models draft={STATE.draft_id} target={STATE.target_id}\n")
    STATE.hf = sd.hf_models(STATE.draft_id, STATE.target_id)
    STATE.is_warm = True
    sys.stderr.write(f"[{LABEL}] ready\n")


def _run(req: SpecDecodeRequest) -> dict:
    if STATE.mock:
        return sd.run_mock(req.prompt, req.max_tokens, req.k)
    draft_next, target_next, tok = STATE.hf
    return sd.run_hf(req.prompt, req.max_tokens, req.k, draft_next, target_next, tok)


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
            return self._send_json(200 if STATE.is_warm else 503,
                                   {"status": "ok" if STATE.is_warm else "warming"})
        return self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/verify":
            return self._send_json(404, {"error": "not found"})
        if not STATE.is_warm:
            return self._send_json(503, {"error": "not warm"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            req_env = SignedEnvelope.model_validate(body["request_data"])
            resp_env = SignedEnvelope.model_validate(body["response_data"])
        except Exception as exc:  # noqa: BLE001
            return self._send_json(400, {"error": f"bad verify body: {exc}"})

        if not verify(req_env) or not verify(resp_env):
            return self._send_json(200, {"is_verified": False, "reason": "bad_signature"})
        if req_env.data.id != resp_env.data.id:
            return self._send_json(200, {"is_verified": False, "reason": "id_mismatch"})

        try:
            inner = SpecDecodeRequest.model_validate(req_env.data.payload)
        except Exception as exc:  # noqa: BLE001
            return self._send_json(400, {"error": f"bad inner request: {exc}"})

        try:
            recomp = _run(inner)
        except Exception as exc:  # noqa: BLE001
            return self._send_json(500, {"error": f"recomp spec-decode failed: {exc}"})

        host = resp_env.data.payload
        ids_match = host.get("output_ids") == recomp["output_ids"]
        trace_match = host.get("rounds") == recomp["rounds"]
        verified = bool(ids_match and trace_match)
        if not verified:
            reason = "output_mismatch" if not ids_match else "trace_mismatch"
            sys.stderr.write(f"[{LABEL}] [ALARM] id={req_env.data.id} {reason}\n")
            return self._send_json(200, {"is_verified": False, "reason": reason,
                                         "recomp_payload": recomp})
        return self._send_json(200, {"is_verified": True, "recomp_payload": recomp})

    def log_message(self, format, *args):  # noqa: A002
        sys.stderr.write(f"[{LABEL}] " + (format % args) + "\n")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> int:
    p = argparse.ArgumentParser(description="Spec-decode Recomp Cluster")
    p.add_argument("--port", type=int, default=8030)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--draft-model", default=State.draft_id)
    p.add_argument("--target-model", default=State.target_id)
    args = p.parse_args()

    STATE.mock = args.mock
    STATE.draft_id = args.draft_model
    STATE.target_id = args.target_model

    server = ThreadedHTTPServer((args.host, args.port), Handler)

    def _shutdown(signum, frame):  # noqa: ARG001
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    threading.Thread(target=_boot, args=(args,), daemon=True).start()
    print(f"[{LABEL}] listening on {args.host}:{args.port}; mock={args.mock}")
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
