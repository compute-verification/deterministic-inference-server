"""Recomp Cluster: re-runs the same inference and bitwise-compares.

Spawns its own deterministic vLLM child (same manifest, same c3 config) on
distinct ports. Exposes /verify (not /request). On mismatch, appends a JSON
line to ${OUT_DIR}/alarm.jsonl (opened with 'a', never truncated) and prints
a single [ALARM] line to stderr. The /verify HTTP response carries the
verdict regardless.

`--mock` skips the child Popen and returns the same canned string as the
host cluster, so the bitwise compare passes. `--mock-output-override <s>`
makes the recomp return <s> instead -- forces an alarm path for testing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
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

from modules.core.common.deterministic import canonical_json_text, utc_now_iso
from servers.envelope import (
    InferenceRequest,
    InferenceResponse,
    SignedEnvelope,
    WorkloadRequest,
    WorkloadResult,
    verify,
)
from servers import workloads as W


DEFAULT_MODEL_ID = "Qwen/Qwen3-1.7B"


# ---------------------------------------------------------------------------
# Cluster state
# ---------------------------------------------------------------------------

class ClusterState:
    def __init__(self) -> None:
        self.is_warm: bool = False
        self.proxy_port: int = 0
        self.mock: bool = False
        self.mock_output_override: str | None = None
        self.model_id: str = DEFAULT_MODEL_ID
        self.out_dir: Path = Path("/tmp/recomp-cluster")
        self.vllm_proc: subprocess.Popen | None = None
        self.alarm_lock = threading.Lock()
        self.lock = threading.Lock()
        self.tap_url: str = ""        # where to POST progress events
        self.no_vllm: bool = False
        self.force_run_divergence: bool = False
        self.run_lock = threading.Lock()  # one workload re-run at a time


STATE = ClusterState()


def _start_vllm_child(manifest_path: str, proxy_port: int, vllm_port: int, out_dir: str) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "modules" / "inference" / "server" / "main.py"),
        "--manifest", manifest_path,
        "--skip-boot-validation",
        "--port", str(proxy_port),
        "--vllm-port", str(vllm_port),
        "--out-dir", out_dir,
    ]
    sys.stderr.write(f"[recomp_cluster] launching child: {' '.join(cmd)}\n")
    return subprocess.Popen(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)


def _resolve_model_id(proxy_port: int) -> str:
    """Ask vLLM what name it's serving under. With RUNNER_MODEL_PATH set the
    served name is the snapshot path, not the HF hub id."""
    try:
        with urlopen(f"http://127.0.0.1:{proxy_port}/v1/models", timeout=10) as resp:
            payload = json.loads(resp.read())
        data = payload.get("data") or []
        if data and isinstance(data[0], dict) and data[0].get("id"):
            return data[0]["id"]
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[recomp_cluster] _resolve_model_id failed: {exc}; falling back to default\n")
    return DEFAULT_MODEL_ID


def _poll_child_health(proxy_port: int, deadline_s: float = 300.0) -> bool:
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        try:
            with urlopen(f"http://127.0.0.1:{proxy_port}/health", timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def _warm_up(proxy_port: int) -> bool:
    body = json.dumps({
        "model": STATE.model_id,
        "messages": [{"role": "user", "content": "warmup"}],
        "max_tokens": 4,
        "temperature": 0,
        "seed": 42,
    }).encode("utf-8")
    req = Request(
        f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=300) as resp:
            _ = resp.read()
        return True
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[recomp_cluster] warm-up failed: {exc}\n")
        return False


def _boot_thread(args: argparse.Namespace) -> None:
    if args.mock:
        STATE.is_warm = True
        sys.stderr.write("[recomp_cluster] mock mode; warm immediately\n")
        return
    if args.no_vllm:
        # /verify_run re-runs load their own models; skip the vLLM child
        STATE.is_warm = True
        sys.stderr.write("[recomp_cluster] --no-vllm; /verify_run-only, warm immediately\n")
        return

    proc = _start_vllm_child(args.manifest, args.proxy_port, args.vllm_port, args.out_dir)
    STATE.vllm_proc = proc

    if not _poll_child_health(args.proxy_port):
        sys.stderr.write("[recomp_cluster] child /health never became 200; exiting non-zero\n")
        os._exit(1)

    STATE.model_id = _resolve_model_id(args.proxy_port)
    sys.stderr.write(f"[recomp_cluster] child /health OK; served model_id={STATE.model_id!r}; sending warm-up\n")
    if not _warm_up(args.proxy_port):
        sys.stderr.write("[recomp_cluster] warm-up did not succeed; exiting non-zero\n")
        os._exit(1)

    STATE.is_warm = True
    sys.stderr.write("[recomp_cluster] ready\n")


# ---------------------------------------------------------------------------
# Inference paths
# ---------------------------------------------------------------------------

def _do_real_inference(prompt: str, max_tokens: int) -> str:
    body = json.dumps({
        "model": STATE.model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 42,
    }).encode("utf-8")
    req = Request(
        f"http://127.0.0.1:{STATE.proxy_port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=300) as resp:
        payload = json.loads(resp.read())
    return payload["choices"][0]["message"]["content"]


def _do_mock_inference(prompt: str, max_tokens: int) -> str:
    if STATE.mock_output_override is not None:
        return STATE.mock_output_override
    return f"MOCK[{prompt[:32]}|max={max_tokens}]"


# ---------------------------------------------------------------------------
# Alarm
# ---------------------------------------------------------------------------

def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _log_alarm(record: dict) -> None:
    path = STATE.out_dir / "alarm.jsonl"
    STATE.out_dir.mkdir(parents=True, exist_ok=True)
    with STATE.alarm_lock:
        # canonical_json_text already terminates with \n
        with open(path, "a", encoding="utf-8") as f:
            f.write(canonical_json_text(record))
    sys.stderr.write(f"[ALARM] id={record.get('id')} reason={record.get('reason')}\n")


# ---------------------------------------------------------------------------
# Workload re-runs (/verify_run)
# ---------------------------------------------------------------------------

def _post_event(payload: dict) -> None:
    """Fire-and-forget progress event to the Tap's /event ingest."""
    if not STATE.tap_url:
        return
    try:
        req = Request(f"{STATE.tap_url}/event",
                      data=json.dumps(payload).encode("utf-8"),
                      headers={"Content-Type": "application/json"},
                      method="POST")
        with urlopen(req, timeout=10) as resp:
            resp.read()
    except Exception as exc:  # noqa: BLE001 -- progress is best-effort
        sys.stderr.write(f"[recomp_cluster] event post failed: {exc}\n")


def _verify_run(env_id: int, inner_req: WorkloadRequest,
                claimed: WorkloadResult) -> dict:
    """Independently re-run the whole workload and compare canonical digests."""
    # Integrity of the claim itself: the digest the host signed must match the
    # capture it shipped. A mismatch means the response is internally bogus.
    host_digest = W.canonical_digest(claimed.capture)
    if host_digest != claimed.capture_digest:
        _log_alarm({"id": env_id, "workload": inner_req.workload,
                    "reason": "digest_claim_mismatch",
                    "claimed_digest": claimed.capture_digest,
                    "computed_digest": host_digest,
                    "verified_at": utc_now_iso()})
        return {"is_verified": False, "reason": "digest_claim_mismatch",
                "expected_digest": claimed.capture_digest,
                "actual_digest": host_digest}

    def on_progress(prog: dict) -> None:
        # nested: harness progress carries its own "type" key
        _post_event({"type": "recomp_progress", "id": env_id,
                     "workload": inner_req.workload, "progress": prog})

    with STATE.run_lock:
        capture, digest = W.run_workload(
            inner_req.workload, inner_req.params, mock=STATE.mock,
            on_progress=on_progress)
    if STATE.force_run_divergence:
        capture = dict(capture, _forced_divergence=True)
        digest = W.canonical_digest(capture)

    if digest == claimed.capture_digest:
        return {"is_verified": True, "expected_digest": claimed.capture_digest,
                "actual_digest": digest}

    # forensics: keep both captures next to the alarm line
    runs = STATE.out_dir / "mismatches"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / f"{env_id:04d}-{inner_req.workload}.host.json").write_text(
        json.dumps(claimed.capture))
    (runs / f"{env_id:04d}-{inner_req.workload}.recomp.json").write_text(
        json.dumps(capture))
    _log_alarm({"id": env_id, "workload": inner_req.workload,
                "reason": "capture_digest_mismatch",
                "expected_digest": claimed.capture_digest,
                "actual_digest": digest,
                "verified_at": utc_now_iso()})
    return {"is_verified": False, "reason": "capture_digest_mismatch",
            "expected_digest": claimed.capture_digest, "actual_digest": digest}


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

class RecompHandler(BaseHTTPRequestHandler):

    def _send_json(self, code: int, body: dict) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path == "/health":
            if STATE.is_warm:
                return self._send_json(200, {"status": "ok"})
            return self._send_json(503, {"status": "warming"})
        return self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path not in ("/verify", "/verify_run"):
            return self._send_json(404, {"error": "not found"})
        if not STATE.is_warm:
            return self._send_json(503, {"error": "not warm"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            body = json.loads(raw)
            req_env = SignedEnvelope.model_validate(body["request_data"])
            resp_env = SignedEnvelope.model_validate(body["response_data"])
        except Exception as exc:  # noqa: BLE001
            return self._send_json(400, {"error": f"bad verify body: {exc}"})

        if not verify(req_env) or not verify(resp_env):
            _log_alarm({
                "id": req_env.data.id if hasattr(req_env, "data") else None,
                "reason": "bad_signature",
                "verified_at": utc_now_iso(),
            })
            return self._send_json(200, {"is_verified": False, "reason": "bad_signature"})

        if req_env.data.id != resp_env.data.id:
            _log_alarm({
                "id": req_env.data.id,
                "response_id": resp_env.data.id,
                "reason": "id_mismatch",
                "verified_at": utc_now_iso(),
            })
            return self._send_json(200, {"is_verified": False, "reason": "id_mismatch"})

        if self.path == "/verify_run":
            return self._handle_verify_run(req_env, resp_env)

        try:
            inner_req = InferenceRequest.model_validate(req_env.data.payload)
            inner_resp = InferenceResponse.model_validate(resp_env.data.payload)
        except Exception as exc:  # noqa: BLE001
            return self._send_json(400, {"error": f"bad inner payload: {exc}"})

        try:
            if STATE.mock:
                recomp_output = _do_mock_inference(inner_req.prompt, inner_req.max_tokens)
            else:
                recomp_output = _do_real_inference(inner_req.prompt, inner_req.max_tokens)
        except Exception as exc:  # noqa: BLE001
            return self._send_json(500, {"error": f"recomp inference failed: {exc}"})

        expected = inner_resp.output
        actual = recomp_output
        if expected == actual:
            # Return the recomputed output too, so a downstream proof server
            # (the compare + task-graph variant) can do its own comparison.
            return self._send_json(200, {"is_verified": True, "recomp_output": actual})

        _log_alarm({
            "id": req_env.data.id,
            "prompt": inner_req.prompt,
            "max_tokens": inner_req.max_tokens,
            "expected_output_sha256": f"sha256:{_sha256_hex(expected)}",
            "actual_output_sha256": f"sha256:{_sha256_hex(actual)}",
            "expected_prefix": expected[:80],
            "actual_prefix": actual[:80],
            "reason": "output_mismatch",
            "verified_at": utc_now_iso(),
        })
        return self._send_json(200, {"is_verified": False, "reason": "output_mismatch", "recomp_output": actual})

    def _handle_verify_run(self, req_env: SignedEnvelope, resp_env: SignedEnvelope) -> None:
        try:
            inner_req = WorkloadRequest.model_validate(req_env.data.payload)
            claimed = WorkloadResult.model_validate(resp_env.data.payload)
        except Exception as exc:  # noqa: BLE001
            return self._send_json(400, {"error": f"bad inner payload: {exc}"})
        if inner_req.workload != claimed.workload:
            _log_alarm({"id": req_env.data.id, "reason": "workload_mismatch",
                        "request_workload": inner_req.workload,
                        "result_workload": claimed.workload,
                        "verified_at": utc_now_iso()})
            return self._send_json(200, {"is_verified": False,
                                         "reason": "workload_mismatch"})
        try:
            verdict = _verify_run(req_env.data.id, inner_req, claimed)
        except W.WorkloadError as exc:
            return self._send_json(500, {"error": f"recomp re-run failed: {exc}"})
        except Exception as exc:  # noqa: BLE001
            return self._send_json(500, {"error": f"recomp re-run failed: {exc}"})
        return self._send_json(200, verdict)

    def log_message(self, format, *args):  # noqa: A002
        sys.stderr.write("[recomp_cluster] " + (format % args) + "\n")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _install_shutdown(server: HTTPServer) -> None:
    def shutdown(signum, frame):  # noqa: ARG001
        sys.stderr.write("[recomp_cluster] shutting down\n")
        if STATE.vllm_proc and STATE.vllm_proc.poll() is None:
            STATE.vllm_proc.terminate()
            try:
                STATE.vllm_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                STATE.vllm_proc.kill()
        # server.shutdown() blocks until serve_forever returns; call it
        # from a daemon thread so the signal handler returns promptly.
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)


def main() -> int:
    parser = argparse.ArgumentParser(description="Tap-protocol Recomp Cluster")
    parser.add_argument("--port", type=int, default=8030)
    parser.add_argument("--proxy-port", type=int, default=8031)
    parser.add_argument("--vllm-port", type=int, default=8032)
    parser.add_argument("--manifest", default=str(DEMO_DIR / "qwen3-1.7b-tap.manifest.json"))
    parser.add_argument("--out-dir", default="/tmp/recomp-cluster")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--mock-output-override", default=None,
                        help="If set in mock mode, return this string from recomp inference, forcing a mismatch")
    parser.add_argument("--no-vllm", action="store_true",
                        help="Serve /verify_run re-runs only; skip the vLLM child")
    parser.add_argument("--tap-url", default="http://127.0.0.1:8010",
                        help="Tap base URL for POSTing recomp_progress events")
    parser.add_argument("--force-run-divergence", action="store_true",
                        help="Perturb the re-run capture before digesting -- "
                             "forces the /verify_run alarm path for testing")
    args = parser.parse_args()

    STATE.proxy_port = args.proxy_port
    STATE.mock = args.mock
    STATE.mock_output_override = args.mock_output_override
    STATE.no_vllm = args.no_vllm
    STATE.tap_url = args.tap_url.rstrip("/")
    STATE.force_run_divergence = args.force_run_divergence
    STATE.out_dir = Path(args.out_dir)
    STATE.out_dir.mkdir(parents=True, exist_ok=True)

    server = ThreadedHTTPServer((args.host, args.port), RecompHandler)
    _install_shutdown(server)

    threading.Thread(target=_boot_thread, args=(args,), daemon=True).start()

    print(f"[recomp_cluster] listening on {args.host}:{args.port}; proxy_port={args.proxy_port}; mock={args.mock}")
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
