"""Tap (spec-decode): relays the request to the Host Cluster and the response
back to the Gateway, then fires an async verify against the Recomp Cluster and
forwards BOTH clusters' payloads to the proof server's /compare.
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

from servers.envelope import SignedEnvelope, verify

LABEL = "tap"


def _post(url: str, body: dict, timeout: int):
    req = Request(url, data=json.dumps(body).encode("utf-8"),
                  headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _async_verify_and_compare(recomp_url, compare_url, request_env, response_env):
    """Recomp /verify, then forward both payloads to the proof server /compare."""
    try:
        verdict = _post(f"{recomp_url}/verify",
                        {"request_data": request_env, "response_data": response_env}, 600)
        sys.stderr.write(f"[{LABEL}] verify id={request_env['data']['id']}: "
                         f"{ {k:v for k,v in verdict.items() if k!='recomp_payload'} }\n")
    except (HTTPError, URLError) as exc:
        sys.stderr.write(f"[{LABEL}] verify failed: {exc}\n")
        return
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[{LABEL}] verify error: {exc}\n")
        return

    if not compare_url or "recomp_payload" not in verdict:
        return
    try:
        _post(f"{compare_url}/compare", {
            "id": request_env["data"]["id"],
            "host": response_env["data"]["payload"],
            "recomp": verdict["recomp_payload"],
        }, 30)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[{LABEL}] compare failed: {exc}\n")


class Handler(BaseHTTPRequestHandler):
    host_url = ""
    recomp_url = ""
    compare_url = ""

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
            req_env = SignedEnvelope.model_validate(
                json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0")))))
        except Exception as exc:  # noqa: BLE001
            return self._send_json(400, {"error": f"bad envelope: {exc}"})
        if not verify(req_env):
            return self._send_json(401, {"error": "bad request signature"})

        try:
            resp_body = _post(f"{self.host_url}/request", req_env.model_dump(), 300)
            resp_env = SignedEnvelope.model_validate(resp_body)
        except HTTPError as exc:
            return self._send_json(502, {"error": f"host HTTP {exc.code}"})
        except URLError as exc:
            return self._send_json(502, {"error": f"host unreachable: {exc.reason}"})
        except Exception as exc:  # noqa: BLE001
            return self._send_json(502, {"error": f"host call failed: {exc}"})
        if not verify(resp_env):
            return self._send_json(401, {"error": "bad response signature"})

        self._send_json(200, resp_env.model_dump())
        threading.Thread(
            target=_async_verify_and_compare,
            args=(self.recomp_url, self.compare_url, req_env.model_dump(), resp_env.model_dump()),
            daemon=True,
        ).start()

    def log_message(self, format, *args):  # noqa: A002
        sys.stderr.write(f"[{LABEL}] " + (format % args) + "\n")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> int:
    p = argparse.ArgumentParser(description="Spec-decode Tap")
    p.add_argument("--port", type=int, default=8010)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--host-url", default="http://127.0.0.1:8020")
    p.add_argument("--recomp-url", default="http://127.0.0.1:8030")
    p.add_argument("--compare-server-url", default="")
    args = p.parse_args()

    Handler.host_url = args.host_url.rstrip("/")
    Handler.recomp_url = args.recomp_url.rstrip("/")
    Handler.compare_url = args.compare_server_url.rstrip("/")

    server = ThreadedHTTPServer((args.host, args.port), Handler)

    def _shutdown(signum, frame):  # noqa: ARG001
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    cs = Handler.compare_url or "<disabled>"
    print(f"[{LABEL}] listening on {args.host}:{args.port}; host={Handler.host_url}; "
          f"recomp={Handler.recomp_url}; compare={cs}")
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
