"""Gateway (spec-decode): wraps a client SpecDecodeRequest in a signed envelope,
relays to the Tap, verifies + unwraps the SpecDecodeResponse, returns it.
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
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEMO_DIR = Path(__file__).resolve().parent.parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from servers.envelope import (
    SignedEnvelope,
    SpecDecodeRequest,
    SpecDecodeResponse,
    next_id,
    sign,
    verify,
)

LABEL = "gateway"


class Handler(BaseHTTPRequestHandler):
    tap_url = ""

    def _send_json(self, code: int, body: dict) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path == "/health":
            return self._send_json(200, {"status": "ok"})
        return self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/request":
            return self._send_json(404, {"error": "not found"})
        try:
            req = SpecDecodeRequest.model_validate(
                json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0")))))
        except Exception as exc:  # noqa: BLE001
            return self._send_json(400, {"error": f"bad request: {exc}"})

        signed_req = sign(req.model_dump(), next_id())
        try:
            req_out = Request(f"{self.tap_url}/request",
                              data=json.dumps(signed_req.model_dump()).encode("utf-8"),
                              headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(req_out, timeout=300) as resp:
                resp_body = json.loads(resp.read())
        except HTTPError as exc:
            return self._send_json(502, {"error": f"tap HTTP {exc.code}"})
        except URLError as exc:
            return self._send_json(502, {"error": f"tap unreachable: {exc.reason}"})
        except Exception as exc:  # noqa: BLE001
            return self._send_json(502, {"error": f"tap call failed: {exc}"})

        try:
            signed_resp = SignedEnvelope.model_validate(resp_body)
        except Exception as exc:  # noqa: BLE001
            return self._send_json(502, {"error": f"bad response envelope: {exc}"})
        if not verify(signed_resp):
            return self._send_json(502, {"error": "response signature invalid"})
        try:
            inner = SpecDecodeResponse.model_validate(signed_resp.data.payload)
        except Exception as exc:  # noqa: BLE001
            return self._send_json(502, {"error": f"bad inner response: {exc}"})
        return self._send_json(200, inner.model_dump())

    def log_message(self, format, *args):  # noqa: A002
        sys.stderr.write(f"[{LABEL}] " + (format % args) + "\n")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> int:
    p = argparse.ArgumentParser(description="Spec-decode Gateway")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--tap-url", default="http://127.0.0.1:8010")
    args = p.parse_args()

    Handler.tap_url = args.tap_url.rstrip("/")
    server = ThreadedHTTPServer((args.host, args.port), Handler)

    def _shutdown(signum, frame):  # noqa: ARG001
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"[{LABEL}] listening on {args.host}:{args.port}; tap={Handler.tap_url}")
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
