"""Host Cluster (spec-decode): runs deterministic greedy speculative decoding
over a draft + target model and returns a SignedEnvelope<SpecDecodeResponse>
carrying the full per-round trace.

`--mock` runs the deterministic mock backend (no GPU). Real mode loads two HF
models (draft + target) in-process and runs greedy argmax under the determinism
knobs. `/health` is 200 only once the backend is warm.
"""
from __future__ import annotations

import argparse
import os
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
from servers.envelope import (
    SignedEnvelope,
    SpecDecodeRequest,
    SpecDecodeResponse,
    sign,
    verify,
)

LABEL = "host_cluster"


class State:
    is_warm = False
    mock = False
    draft_id = "Qwen/Qwen3-0.6B"
    target_id = "Qwen/Qwen3-1.7B"
    hf = None  # (draft_next, target_next, tok)


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
        if self.path != "/request":
            return self._send_json(404, {"error": "not found"})
        if not STATE.is_warm:
            return self._send_json(503, {"error": "not warm"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            req_env = SignedEnvelope.model_validate(json.loads(self.rfile.read(length)))
        except Exception as exc:  # noqa: BLE001
            return self._send_json(400, {"error": f"bad envelope: {exc}"})
        if not verify(req_env):
            return self._send_json(401, {"error": "bad request signature"})
        try:
            inner = SpecDecodeRequest.model_validate(req_env.data.payload)
        except Exception as exc:  # noqa: BLE001
            return self._send_json(400, {"error": f"bad inner request: {exc}"})
        try:
            resp_dict = _run(inner)
        except Exception as exc:  # noqa: BLE001
            return self._send_json(500, {"error": f"spec-decode failed: {exc}"})

        resp = SpecDecodeResponse.model_validate(resp_dict)
        signed = sign(resp.model_dump(), req_env.data.id)
        return self._send_json(200, signed.model_dump())

    def log_message(self, format, *args):  # noqa: A002
        sys.stderr.write(f"[{LABEL}] " + (format % args) + "\n")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> int:
    p = argparse.ArgumentParser(description="Spec-decode Host Cluster")
    p.add_argument("--port", type=int, default=8020)
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
