"""Tap: relays SignedEnvelope between Gateway and Host Cluster.

After the response goes back to the Gateway, fires a daemon thread that POSTs
the (request, response) pair to the Recomp Cluster's /verify. Failures are
logged but do not propagate -- verification is async.

Also maintains an in-process ring buffer of protocol events and exposes them
at GET /events (SSE stream) and GET /capture (JSON snapshot). The Tap sees
every protocol step on its own — it receives the signed request from the
Gateway, forwards it to Host, relays the signed response back, and makes
the async /verify call — so it can synthesize the full event stream without
the other servers needing to participate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import signal
import sys
import threading
import time
from collections import deque
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


# ---------------------------------------------------------------------------
# Event ring buffer
# ---------------------------------------------------------------------------

EVENT_BUFFER_SIZE = 4096  # ~4k events; one full request is ~9 events


class EventBus:
    """Thread-safe ring buffer + SSE subscriber fan-out for protocol events.

    `events` is a deque keyed by sequence number; subscribers are notified
    via a Condition variable. Each event is a dict with at minimum
    {seq, ts, type, id}.
    """

    def __init__(self) -> None:
        self.events: deque = deque(maxlen=EVENT_BUFFER_SIZE)
        self._seq = 0
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)

    def emit(self, type_: str, id_: int, **payload) -> None:
        evt = {"ts": time.time(), "type": type_, "id": id_, **payload}
        with self._cv:
            self._seq += 1
            evt["seq"] = self._seq
            self.events.append(evt)
            self._cv.notify_all()
        # Helpful stderr breadcrumb when running interactively.
        sys.stderr.write(f"[tap.event] {type_} id={id_} {json.dumps(payload, default=str)}\n")

    def snapshot(self, since_seq: int = 0) -> list[dict]:
        with self._lock:
            return [e for e in self.events if e["seq"] > since_seq]

    def wait_for_new(self, since_seq: int, timeout: float) -> list[dict]:
        with self._cv:
            deadline = time.monotonic() + timeout
            while True:
                fresh = [e for e in self.events if e["seq"] > since_seq]
                if fresh:
                    return fresh
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return []
                self._cv.wait(timeout=remaining)


BUS = EventBus()


def _output_preview(env_dict: dict, n: int = 140) -> str:
    """Extract a short preview of the inference output from a response envelope dict."""
    try:
        out = env_dict["data"]["payload"]["output"] or ""
    except Exception:
        return ""
    if len(out) <= n:
        return out
    return out[: n - 1] + "…"


def _output_sha256(env_dict: dict) -> str:
    try:
        out = env_dict["data"]["payload"]["output"] or ""
    except Exception:
        return ""
    return "sha256:" + hashlib.sha256(out.encode("utf-8")).hexdigest()


def _signature_prefix(env_dict: dict, n: int = 12) -> str:
    sig = env_dict.get("signature") or ""
    return sig[:n]


# ---------------------------------------------------------------------------
# Async verify (now emits events to the bus)
# ---------------------------------------------------------------------------

def _async_verify(
    recomp_url: str,
    request_env: dict,
    response_env: dict,
    env_id: int,
    compare_server_url: str = "",
) -> None:
    """Fire-and-forget POST to recomp /verify. Logs verdict; ignores failures.

    Emits ``tap_verify_started`` / ``recomp_verified`` on the event bus. If
    ``compare_server_url`` is set and recomp returns its recomputed output,
    also forward both clusters' outputs to the proof server's /compare so it
    can compare them and build a task graph (see ``demos/proof-compare/``).
    """
    BUS.emit("tap_verify_started", env_id)
    try:
        body = json.dumps({
            "request_data": request_env,
            "response_data": response_env,
        }).encode("utf-8")
        req = Request(
            f"{recomp_url}/verify",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=600) as resp:
            verdict = json.loads(resp.read())
        sys.stderr.write(f"[tap] verify verdict for id={env_id}: {verdict}\n")
        # Surface the verdict on the event bus.
        BUS.emit(
            "recomp_verified",
            env_id,
            is_verified=bool(verdict.get("is_verified", False)),
            expected_sha256=_output_sha256(response_env),
            actual_sha256=verdict.get("actual_sha256") or _output_sha256(response_env),
            reason=verdict.get("reason"),
        )
    except HTTPError as exc:
        sys.stderr.write(f"[tap] verify HTTP {exc.code}: {exc.reason}\n")
        BUS.emit("recomp_verified", env_id, is_verified=False, reason=f"http_{exc.code}")
        return
    except URLError as exc:
        sys.stderr.write(f"[tap] verify unreachable: {exc.reason}\n")
        BUS.emit("recomp_verified", env_id, is_verified=False, reason="unreachable")
        return
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[tap] verify failed: {exc}\n")
        BUS.emit("recomp_verified", env_id, is_verified=False, reason=str(exc))
        return

    if compare_server_url and isinstance(verdict, dict) and "recomp_output" in verdict:
        _async_compare(compare_server_url, request_env, response_env, verdict["recomp_output"])


def _async_compare(
    compare_server_url: str,
    request_env: dict,
    response_env: dict,
    recomp_output: str,
) -> None:
    """POST {id, prompt, host_output, recomp_output} to the proof server /compare.

    The proof server bitwise-compares the two cluster outputs and builds a task
    graph. Fire-and-forget: failures never affect the client request.
    """
    try:
        rid = request_env.get("data", {}).get("id")
        prompt = request_env.get("data", {}).get("payload", {}).get("prompt", "")
        host_output = response_env.get("data", {}).get("payload", {}).get("output", "")
        body = json.dumps({
            "id": rid,
            "prompt": prompt,
            "host_output": host_output,
            "recomp_output": recomp_output,
        }).encode("utf-8")
        req = Request(
            f"{compare_server_url}/compare",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=30) as resp:
            resp.read()
    except HTTPError as exc:
        sys.stderr.write(f"[tap] compare HTTP {exc.code}: {exc.reason}\n")
    except URLError as exc:
        sys.stderr.write(f"[tap] compare unreachable: {exc.reason}\n")
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[tap] compare failed: {exc}\n")


def _async_verify_run(recomp_url: str, request_env: dict, response_env: dict,
                      env_id: int, workload: str) -> None:
    """Fire-and-forget POST to recomp /verify_run (a full workload re-run).

    The recomp re-executes the whole scenario and bitwise-compares canonical
    capture digests; its per-step progress arrives separately via POST /event.
    """
    BUS.emit("tap_verify_started", env_id, workload=workload)
    try:
        body = json.dumps({
            "request_data": request_env,
            "response_data": response_env,
        }).encode("utf-8")
        req = Request(
            f"{recomp_url}/verify_run",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # a coding-agent re-run takes tens of minutes on the GPU
        with urlopen(req, timeout=7200) as resp:
            verdict = json.loads(resp.read())
        sys.stderr.write(f"[tap] verify_run verdict for id={env_id}: "
                         f"{ {k: v for k, v in verdict.items() if k != 'capture'} }\n")
        BUS.emit(
            "recomp_verified",
            env_id,
            workload=workload,
            is_verified=bool(verdict.get("is_verified", False)),
            expected_sha256=verdict.get("expected_digest", ""),
            actual_sha256=verdict.get("actual_digest", ""),
            reason=verdict.get("reason"),
        )
    except HTTPError as exc:
        sys.stderr.write(f"[tap] verify_run HTTP {exc.code}: {exc.reason}\n")
        BUS.emit("recomp_verified", env_id, workload=workload,
                 is_verified=False, reason=f"http_{exc.code}")
    except URLError as exc:
        sys.stderr.write(f"[tap] verify_run unreachable: {exc.reason}\n")
        BUS.emit("recomp_verified", env_id, workload=workload,
                 is_verified=False, reason="unreachable")
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[tap] verify_run failed: {exc}\n")
        BUS.emit("recomp_verified", env_id, workload=workload,
                 is_verified=False, reason=str(exc))


def _async_proof_copy(proof_server_url: str, request_env: dict, response_env: dict) -> None:
    """Fire-and-forget POST of the verified (req, resp) pair to the proof server.

    The proof server is the developer-controlled single egress channel to the
    auditor (see ``demos/proof-server/plan.md``). Failures here never fail the
    client request — proof generation is strictly async.
    """
    try:
        body = json.dumps({
            "request_data": request_env,
            "response_data": response_env,
        }).encode("utf-8")
        req = Request(
            f"{proof_server_url}/tap-copy",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=30) as resp:
            resp.read()
    except HTTPError as exc:
        sys.stderr.write(f"[tap] proof-copy HTTP {exc.code}: {exc.reason}\n")
    except URLError as exc:
        sys.stderr.write(f"[tap] proof-copy unreachable: {exc.reason}\n")
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[tap] proof-copy failed: {exc}\n")


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class TapHandler(BaseHTTPRequestHandler):
    host_url: str = ""
    recomp_url: str = ""
    proof_server_url: str = ""  # empty disables the proof-server fan-out
    compare_server_url: str = ""  # empty disables the compare/task-graph fan-out

    def _send_json(self, code: int, body) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        # CORS so a Surge-deployed viewer can fetch /capture cross-origin.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def _send_sse_headers(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-Accel-Buffering", "no")  # disable proxy buffering
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/health":
            return self._send_json(200, {"status": "ok"})

        if self.path.startswith("/capture"):
            # /capture or /capture?since=N — return all buffered events as
            # one JSON array.
            try:
                qs = self.path.split("?", 1)[1] if "?" in self.path else ""
                since = 0
                for kv in qs.split("&"):
                    if kv.startswith("since="):
                        since = int(kv.split("=", 1)[1])
            except Exception:
                since = 0
            return self._send_json(200, BUS.snapshot(since_seq=since))

        if self.path.startswith("/events"):
            return self._handle_sse()

        return self._send_json(404, {"error": "not found"})

    def _handle_sse(self) -> None:
        """SSE stream: emit existing events then block waiting for new ones."""
        try:
            self._send_sse_headers()
            since = 0
            # Replay any already-buffered events so a late subscriber sees history.
            for evt in BUS.snapshot(since_seq=0):
                self._sse_write(evt)
                since = max(since, evt["seq"])
            # Block waiting for new events; flush each.
            while True:
                fresh = BUS.wait_for_new(since_seq=since, timeout=15.0)
                if not fresh:
                    # heartbeat comment keeps clients alive through proxies
                    try:
                        self.wfile.write(b": heartbeat\n\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    continue
                for evt in fresh:
                    try:
                        self._sse_write(evt)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    since = max(since, evt["seq"])
        except (BrokenPipeError, ConnectionResetError):
            return

    def _sse_write(self, evt: dict) -> None:
        self.wfile.write(b"data: " + json.dumps(evt).encode("utf-8") + b"\n\n")
        self.wfile.flush()

    def do_POST(self) -> None:
        if self.path == "/event":
            return self._handle_event_ingest()
        if self.path == "/run":
            return self._handle_run()
        if self.path != "/request":
            return self._send_json(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            body = json.loads(raw)
            req_env = SignedEnvelope.model_validate(body)
        except Exception as exc:  # noqa: BLE001
            return self._send_json(400, {"error": f"bad envelope: {exc}"})

        if not verify(req_env):
            return self._send_json(401, {"error": "bad request signature"})

        env_id = req_env.data.id
        req_env_dict = req_env.model_dump()
        # The Tap can synthesize the upstream events: by the time we have a
        # validated signed envelope here, the Gateway just sent it (so
        # request_sent + gateway_signed + tap_received all logically
        # happen at "now"). We emit them in sequence so the viewer can
        # show the visible animation, not because they happened apart.
        prompt = ""
        try:
            prompt = req_env_dict["data"]["payload"].get("prompt", "")
        except Exception:
            pass
        BUS.emit("request_sent", env_id, prompt=prompt)
        BUS.emit("gateway_signed", env_id, signature_prefix=_signature_prefix(req_env_dict))
        BUS.emit("tap_received", env_id)
        BUS.emit("tap_relayed_request", env_id)

        # Forward verbatim to host cluster
        try:
            outbound = Request(
                f"{self.host_url}/request",
                data=json.dumps(req_env_dict).encode("utf-8"),
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
            return self._send_json(502, {"error": f"host returned HTTP {exc.code}", "body": err_body})
        except URLError as exc:
            return self._send_json(502, {"error": f"host unreachable: {exc.reason}"})
        except Exception as exc:  # noqa: BLE001
            return self._send_json(502, {"error": f"host call failed: {exc}"})

        try:
            resp_env = SignedEnvelope.model_validate(resp_body)
        except Exception as exc:  # noqa: BLE001
            return self._send_json(502, {"error": f"bad response envelope: {exc}"})

        if not verify(resp_env):
            return self._send_json(401, {"error": "bad response signature"})

        resp_env_dict = resp_env.model_dump()
        BUS.emit(
            "host_completed",
            env_id,
            output_preview=_output_preview(resp_env_dict),
            output_sha256=_output_sha256(resp_env_dict),
        )
        BUS.emit("tap_relayed_response", env_id)

        # Return the verified response envelope to the Gateway first.
        self._send_json(200, resp_env_dict)
        BUS.emit("client_received", env_id, output_preview=_output_preview(resp_env_dict))

        # Then spawn the async verification tap-copy.
        threading.Thread(
            target=_async_verify,
            args=(self.recomp_url, req_env_dict, resp_env_dict, env_id,
                  self.compare_server_url),
            daemon=True,
        ).start()

        # Second fan-out: the proof server, if configured.
        if self.proof_server_url:
            threading.Thread(
                target=_async_proof_copy,
                args=(self.proof_server_url, req_env.model_dump(), resp_env.model_dump()),
                daemon=True,
            ).start()

    def _handle_event_ingest(self) -> None:
        """POST /event: the clusters stream workload progress through here.

        Only the two progress types are accepted -- the protocol lifecycle
        events are synthesized by the Tap itself and must not be spoofable
        via this ingest.
        """
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            type_ = body.pop("type")
            env_id = int(body.pop("id"))
        except Exception as exc:  # noqa: BLE001
            return self._send_json(400, {"error": f"bad event: {exc}"})
        if type_ not in ("host_progress", "recomp_progress"):
            return self._send_json(400, {"error": f"event type {type_!r} not ingestable"})
        body.pop("seq", None)
        body.pop("ts", None)
        BUS.emit(type_, env_id, **body)
        return self._send_json(200, {"ok": True})

    def _handle_run(self) -> None:
        """POST /run: relay a signed WorkloadRequest to the Host, then tap a
        copy to the Recomp for a full re-run. Same lifecycle events as
        /request (the viewer animation binds to them unchanged), plus a
        ``workload`` field on each."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            req_env = SignedEnvelope.model_validate(body)
        except Exception as exc:  # noqa: BLE001
            return self._send_json(400, {"error": f"bad envelope: {exc}"})

        if not verify(req_env):
            return self._send_json(401, {"error": "bad request signature"})

        env_id = req_env.data.id
        req_env_dict = req_env.model_dump()
        payload = req_env_dict["data"]["payload"]
        workload = payload.get("workload", "?")
        BUS.emit("request_sent", env_id, workload=workload,
                 params=payload.get("params", {}))
        BUS.emit("gateway_signed", env_id, workload=workload,
                 signature_prefix=_signature_prefix(req_env_dict))
        BUS.emit("tap_received", env_id, workload=workload)
        BUS.emit("tap_relayed_request", env_id, workload=workload)

        try:
            outbound = Request(
                f"{self.host_url}/run",
                data=json.dumps(req_env_dict).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            # a coding-agent run takes tens of minutes on the GPU
            with urlopen(outbound, timeout=7200) as resp:
                resp_body = json.loads(resp.read())
        except HTTPError as exc:
            try:
                err_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                err_body = ""
            return self._send_json(502, {"error": f"host returned HTTP {exc.code}", "body": err_body})
        except URLError as exc:
            return self._send_json(502, {"error": f"host unreachable: {exc.reason}"})
        except Exception as exc:  # noqa: BLE001
            return self._send_json(502, {"error": f"host call failed: {exc}"})

        try:
            resp_env = SignedEnvelope.model_validate(resp_body)
        except Exception as exc:  # noqa: BLE001
            return self._send_json(502, {"error": f"bad response envelope: {exc}"})

        if not verify(resp_env):
            return self._send_json(401, {"error": "bad response signature"})

        resp_env_dict = resp_env.model_dump()
        result = resp_env_dict["data"]["payload"]
        BUS.emit(
            "host_completed",
            env_id,
            workload=workload,
            capture_digest=result.get("capture_digest", ""),
            # the existing viewer's digest row binds to output_sha256
            output_sha256=result.get("capture_digest", ""),
            summary=result.get("summary", {}),
        )
        BUS.emit("tap_relayed_response", env_id, workload=workload)

        self._send_json(200, resp_env_dict)
        BUS.emit("client_received", env_id, workload=workload,
                 summary=result.get("summary", {}))

        threading.Thread(
            target=_async_verify_run,
            args=(self.recomp_url, req_env_dict, resp_env_dict, env_id, workload),
            daemon=True,
        ).start()

    def log_message(self, format, *args):  # noqa: A002
        sys.stderr.write("[tap] " + (format % args) + "\n")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> int:
    parser = argparse.ArgumentParser(description="Tap-protocol Tap")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--host-url", default="http://127.0.0.1:8020")
    parser.add_argument("--recomp-url", default="http://127.0.0.1:8030")
    parser.add_argument("--proof-server-url", default="",
                        help="Optional. If set, every verified envelope pair is "
                             "also POSTed to <url>/tap-copy.")
    parser.add_argument("--compare-server-url", default="",
                        help="Optional. If set, after recomp verifies, both "
                             "clusters' outputs are POSTed to <url>/compare.")
    args = parser.parse_args()

    TapHandler.host_url = args.host_url.rstrip("/")
    TapHandler.recomp_url = args.recomp_url.rstrip("/")
    TapHandler.proof_server_url = args.proof_server_url.rstrip("/")
    TapHandler.compare_server_url = args.compare_server_url.rstrip("/")

    server = ThreadedHTTPServer((args.host, args.port), TapHandler)

    def _shutdown(signum, frame):  # noqa: ARG001
        sys.stderr.write("[tap] shutting down\n")
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    ps = TapHandler.proof_server_url or "<disabled>"
    cs = TapHandler.compare_server_url or "<disabled>"
    print(f"[tap] listening on {args.host}:{args.port}; host={TapHandler.host_url}; "
          f"recomp={TapHandler.recomp_url}; proof_server={ps}; compare_server={cs}")
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
