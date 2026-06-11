"""Gateway: client-facing edge of the tap-protocol demo.

Accepts a plain `InferenceRequest` JSON from the client, wraps it in a signed
envelope with a monotonic id, relays to the Tap, verifies the response
envelope, unwraps the response, returns it to the client.

Also exposes two thin read-only pass-throughs from the Tap so a public
deployment only needs to expose the Gateway port:
- GET /events  → SSE stream proxied from Tap's /events
- GET /capture → JSON snapshot proxied from Tap's /capture

Both responses get CORS headers so a browser-side viewer can call them
cross-origin without a backend proxy.
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

# Allow `from servers.envelope import ...` when this file is executed directly
# from anywhere via `python3 demos/tap-protocol/servers/gateway.py`.
DEMO_DIR = Path(__file__).resolve().parent.parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from servers.envelope import (
    InferenceRequest,
    InferenceResponse,
    SignedEnvelope,
    WorkloadRequest,
    WorkloadResult,
    next_id,
    sign,
    verify,
)
from servers import workloads as W


# ---------------------------------------------------------------------------
# Async workload jobs
#
# POST /run returns immediately with the envelope id; a worker thread relays
# the signed request through the Tap (a coding-agent run takes tens of
# minutes -- a synchronous HTTP response would die in every proxy on the
# way). Clients follow progress on /events and poll GET /run/<id>.
# ---------------------------------------------------------------------------

JOBS: dict[int, dict] = {}
JOBS_LOCK = threading.Lock()


def _run_job(env_id: int, signed_req: SignedEnvelope, tap_url: str) -> None:
    try:
        data = json.dumps(signed_req.model_dump()).encode("utf-8")
        outbound = Request(f"{tap_url}/run", data=data,
                           headers={"Content-Type": "application/json"},
                           method="POST")
        with urlopen(outbound, timeout=7200) as resp:
            resp_body = json.loads(resp.read())
        signed_resp = SignedEnvelope.model_validate(resp_body)
        if not verify(signed_resp):
            raise ValueError("response envelope signature invalid")
        result = WorkloadResult.model_validate(signed_resp.data.payload)
    except Exception as exc:  # noqa: BLE001
        with JOBS_LOCK:
            JOBS[env_id].update(status="failed", error=str(exc))
        sys.stderr.write(f"[gateway] run {env_id} failed: {exc}\n")
        return
    with JOBS_LOCK:
        JOBS[env_id].update(status="done",
                            capture_digest=result.capture_digest,
                            summary=result.summary,
                            capture=result.capture)


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class GatewayHandler(BaseHTTPRequestHandler):
    tap_url: str = ""  # set on the class before serve_forever()

    def _send_json(self, code: int, body: dict) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path == "/health":
            return self._send_json(200, {"status": "ok"})
        if self.path == "/events":
            return self._proxy_sse(f"{self.tap_url}/events")
        if self.path.startswith("/capture"):
            return self._proxy_capture()
        if self.path.startswith("/run/"):
            return self._handle_run_get()
        return self._send_json(404, {"error": "not found"})

    def _handle_run_get(self) -> None:
        """GET /run/<id> | /run/<id>/capture | /run/<id>/graph"""
        parts = self.path.strip("/").split("/")
        try:
            env_id = int(parts[1])
        except (IndexError, ValueError):
            return self._send_json(400, {"error": "bad run id"})
        sub = parts[2] if len(parts) > 2 else ""
        with JOBS_LOCK:
            job = JOBS.get(env_id)
            job = dict(job) if job else None
        if job is None:
            return self._send_json(404, {"error": f"unknown run id {env_id}"})

        if sub == "":
            return self._send_json(200, {k: v for k, v in job.items()
                                         if k != "capture"})
        if job.get("status") != "done":
            return self._send_json(409, {"error": f"run is {job.get('status')}"})
        if sub == "capture":
            return self._send_json(200, job["capture"])
        if sub == "graph":
            try:
                from modules.proof_server.graph import build_graph
                trace = W.capture_to_trace(job["workload"], job["capture"])
                graph = build_graph(trace).to_dict()
            except Exception as exc:  # noqa: BLE001
                return self._send_json(500, {"error": f"graph build failed: {exc}"})
            # same file shape as the viz's graphs.json: {scene_key: graph}
            return self._send_json(200, {job["workload"]: graph})
        return self._send_json(404, {"error": "not found"})

    def do_OPTIONS(self) -> None:
        # CORS preflight (browser fetches sometimes send this)
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    def _proxy_capture(self) -> None:
        try:
            with urlopen(f"{self.tap_url}/capture", timeout=10) as r:
                body = r.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)
        except (HTTPError, URLError) as exc:
            return self._send_json(502, {"error": f"tap unreachable: {exc}"})

    def _proxy_sse(self, upstream_url: str) -> None:
        """Stream upstream SSE bytes back to the client until either side closes.

        Forwards LINE-AT-A-TIME (not chunk-at-a-time): SSE messages are
        line-delimited (`data: …\\n\\n`) and arrive at the Tap one event at
        a time. Reading fixed-size chunks via `upstream.read(N)` blocks
        until N bytes accumulate — a single 200-byte event would then sit
        in the read buffer until another event came along and pushed the
        byte count over the threshold. Using `readline()` instead, each
        event flushes through the proxy as soon as the Tap writes it.
        """
        try:
            # Build a Request with Accept: text/event-stream so the upstream
            # treats us like an SSE subscriber, not a one-shot GET.
            req = Request(upstream_url, headers={"Accept": "text/event-stream"})
            upstream = urlopen(req, timeout=600)
        except (HTTPError, URLError) as exc:
            return self._send_json(502, {"error": f"tap unreachable: {exc}"})

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        try:
            while True:
                line = upstream.readline()
                if not line:
                    break
                try:
                    self.wfile.write(line)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break
        finally:
            try:
                upstream.close()
            except Exception:
                pass

    def do_POST(self) -> None:
        if self.path == "/run":
            return self._handle_run_post()
        if self.path != "/request":
            return self._send_json(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            body = json.loads(raw)
            req = InferenceRequest.model_validate(body)
        except Exception as exc:  # noqa: BLE001
            return self._send_json(400, {"error": f"bad request: {exc}"})

        envelope_id = next_id()
        signed_req = sign(req.model_dump(), envelope_id)

        try:
            data = json.dumps(signed_req.model_dump()).encode("utf-8")
            outbound = Request(
                f"{self.tap_url}/request",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(outbound, timeout=300) as resp:
                resp_body = json.loads(resp.read())
        except HTTPError as exc:
            try:
                err_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                err_body = ""
            return self._send_json(502, {"error": f"tap returned HTTP {exc.code}", "body": err_body})
        except URLError as exc:
            return self._send_json(502, {"error": f"tap unreachable: {exc.reason}"})
        except Exception as exc:  # noqa: BLE001
            return self._send_json(502, {"error": f"tap call failed: {exc}"})

        try:
            signed_resp = SignedEnvelope.model_validate(resp_body)
        except Exception as exc:  # noqa: BLE001
            return self._send_json(502, {"error": f"bad response envelope: {exc}"})

        if not verify(signed_resp):
            return self._send_json(502, {"error": "response envelope signature invalid"})

        try:
            inner = InferenceResponse.model_validate(signed_resp.data.payload)
        except Exception as exc:  # noqa: BLE001
            return self._send_json(502, {"error": f"bad inner response: {exc}"})

        return self._send_json(200, inner.model_dump())

    def _handle_run_post(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            req = WorkloadRequest.model_validate(body)
            if req.workload not in W.WORKLOADS:
                raise ValueError(f"unknown workload {req.workload!r}; "
                                 f"expected one of {sorted(W.WORKLOADS)}")
            # surface bad params at submit time, not minutes later on the host
            W.build_argv(req.workload, req.params, True, Path("/dev/null"))
        except Exception as exc:  # noqa: BLE001
            return self._send_json(400, {"error": f"bad run request: {exc}"})

        env_id = next_id()
        signed_req = sign(req.model_dump(), env_id)
        with JOBS_LOCK:
            JOBS[env_id] = {"id": env_id, "status": "running",
                            "workload": req.workload, "params": req.params}
        threading.Thread(target=_run_job,
                         args=(env_id, signed_req, self.tap_url),
                         daemon=True).start()
        return self._send_json(202, {"id": env_id, "status": "running",
                                     "workload": req.workload})

    def log_message(self, format, *args):  # noqa: A002
        sys.stderr.write("[gateway] " + (format % args) + "\n")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> int:
    parser = argparse.ArgumentParser(description="Tap-protocol Gateway")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1",
                        help="Listen host. Pass 0.0.0.0 in start_servers.sh to expose to LAN/Internet.")
    parser.add_argument("--tap-url", default="http://127.0.0.1:8010")
    args = parser.parse_args()

    GatewayHandler.tap_url = args.tap_url.rstrip("/")
    server = ThreadedHTTPServer((args.host, args.port), GatewayHandler)

    def _shutdown(signum, frame):  # noqa: ARG001
        sys.stderr.write("[gateway] shutting down\n")
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"[gateway] listening on {args.host}:{args.port}; tap={GatewayHandler.tap_url}")
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
