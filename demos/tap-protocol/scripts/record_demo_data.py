"""Record the 4-scenario demo data for the web frontend.

Runs every workload through the 4-node stack (POST /run -> host executes ->
recomp re-runs + verdict), captures the per-run event stream from the Tap,
fetches the run's capture, builds its task graph, and writes everything the
Demo (replay) mode of web/index.html consumes:

    web/data/demo-manifest.json        index: per-scenario stats + provenance
    web/data/<workload>.events.json    ts-normalized event stream for replay
    web/data/protocol-graphs.json      {workload: graph} for the embedded viz
    web/data/runs/<workload>.capture.json   raw captures (forensics/inspection)

Two modes:
  * no --gateway-url : boots a local --mock stack on test ports (CPU; data is
    labeled source=mock -- placeholder wiring only, NEVER ship as real).
  * --gateway-url http://IP:PORT --source h100 : records from a live real
    stack (the H100 capture session).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEMO_DIR = Path(__file__).resolve().parents[1]
SERVERS = DEMO_DIR / "servers"
for p in (REPO_ROOT, DEMO_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from servers import workloads as W  # noqa: E402

GW, TAP, HOST, RECOMP = 28300, 28310, 28320, 28330

# The canonical demo runs. Prompts are part of the recorded artifact.
SCENARIOS = [
    ("inference", {"prompt": "The capital of France is", "max_tokens": 12}),
    ("spec", {"prompt": "The key idea behind bitwise-deterministic inference is",
              "max_tokens": 48, "k": 4}),
    ("training", {}),
    ("coding", {}),
]


def _get(url: str):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())


def _post(url: str, body: dict):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def _wait_health(url: str, deadline: float = 60.0) -> None:
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        try:
            _get(f"{url}/health")
            return
        except Exception:
            time.sleep(0.3)
    raise RuntimeError(f"{url} never became healthy")


def boot_mock_stack(tmp: Path) -> list[subprocess.Popen]:
    py = sys.executable
    procs = [
        subprocess.Popen([py, str(SERVERS / "host_cluster.py"), "--port", str(HOST),
                          "--mock", "--tap-url", f"http://127.0.0.1:{TAP}",
                          "--out-dir", str(tmp / "host")],
                         stdout=(tmp / "host.log").open("w"), stderr=subprocess.STDOUT),
        subprocess.Popen([py, str(SERVERS / "recomp_cluster.py"), "--port", str(RECOMP),
                          "--mock", "--tap-url", f"http://127.0.0.1:{TAP}",
                          "--out-dir", str(tmp / "recomp")],
                         stdout=(tmp / "recomp.log").open("w"), stderr=subprocess.STDOUT),
        subprocess.Popen([py, str(SERVERS / "tap.py"), "--port", str(TAP),
                          "--host-url", f"http://127.0.0.1:{HOST}",
                          "--recomp-url", f"http://127.0.0.1:{RECOMP}"],
                         stdout=(tmp / "tap.log").open("w"), stderr=subprocess.STDOUT),
        subprocess.Popen([py, str(SERVERS / "gateway.py"), "--port", str(GW),
                          "--tap-url", f"http://127.0.0.1:{TAP}"],
                         stdout=(tmp / "gateway.log").open("w"), stderr=subprocess.STDOUT),
    ]
    for port in (HOST, RECOMP, TAP, GW):
        _wait_health(f"http://127.0.0.1:{port}")
    return procs


def record_run(gateway: str, workload: str, params: dict,
               deadline_s: float) -> tuple[dict, dict, list[dict]]:
    """Submit one run; wait for host completion AND the recomp verdict.
    Returns (job, capture, events-for-this-id, ts-normalized)."""
    sub = _post(f"{gateway}/run", {"workload": workload, "params": params})
    rid = sub["id"]
    print(f"[record] {workload}: run id={rid} submitted")

    end = time.monotonic() + deadline_s
    job = None
    while time.monotonic() < end:
        job = _get(f"{gateway}/run/{rid}")
        if job["status"] in ("done", "failed"):
            break
        time.sleep(1.0)
    if not job or job["status"] != "done":
        raise RuntimeError(f"{workload}: job did not finish: {job}")

    verdict = None
    while time.monotonic() < end and verdict is None:
        evs = _get(f"{gateway}/capture")
        verdict = next((e for e in evs if e["type"] == "recomp_verified"
                        and e["id"] == rid), None)
        if verdict is None:
            time.sleep(1.0)
    if verdict is None:
        raise RuntimeError(f"{workload}: recomp_verified never arrived")
    print(f"[record] {workload}: verdict is_verified={verdict['is_verified']}")

    capture = _get(f"{gateway}/run/{rid}/capture")
    evs = [e for e in _get(f"{gateway}/capture") if e["id"] == rid]
    evs.sort(key=lambda e: e["seq"])
    t0 = evs[0]["ts"]
    for e in evs:
        e["ts"] = round(e["ts"] - t0, 3)
        e.pop("seq", None)
    return job, capture, evs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gateway-url", default="",
                    help="Record from an existing stack (e.g. the H100 box). "
                         "Default: boot a local --mock stack.")
    ap.add_argument("--source", default="",
                    help="Provenance label (mock|h100). Required with "
                         "--gateway-url; defaults to mock for the local stack.")
    ap.add_argument("--out-dir", default=str(DEMO_DIR / "web" / "data"))
    ap.add_argument("--deadline", type=float, default=5400.0,
                    help="Per-run wait (host run + recomp re-run), seconds")
    args = ap.parse_args()

    if args.gateway_url and not args.source:
        ap.error("--source is required with --gateway-url (mock|h100)")
    source = args.source or "mock"

    out = Path(args.out_dir)
    (out / "runs").mkdir(parents=True, exist_ok=True)

    procs: list[subprocess.Popen] = []
    if args.gateway_url:
        gateway = args.gateway_url.rstrip("/")
    else:
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="record_demo_"))
        print(f"[record] booting local mock stack (logs: {tmp})")
        procs = boot_mock_stack(tmp)
        gateway = f"http://127.0.0.1:{GW}"

    try:
        from modules.proof_server.graph import build_graph

        graphs: dict[str, dict] = {}
        manifest = {
            "source": source,
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "scenarios": {},
        }
        for workload, params in SCENARIOS:
            job, capture, evs = record_run(gateway, workload, params,
                                           args.deadline)
            verdict = next(e for e in evs if e["type"] == "recomp_verified")
            graph = build_graph(W.capture_to_trace(workload, capture)).to_dict()
            graphs[workload] = graph

            (out / f"{workload}.events.json").write_text(json.dumps(
                {"workload": workload, "source": source, "events": evs},
                indent=1) + "\n")
            (out / "runs" / f"{workload}.capture.json").write_text(
                json.dumps(capture))

            manifest["scenarios"][workload] = {
                "label": W.WORKLOADS[workload]["label"],
                "params": params,
                "forward_passes": len(graph["nodes"]),
                "total_flops": sum(n["flops"] for n in graph["nodes"]),
                "models": sorted(graph["shapes"]),
                "capture_digest": job["capture_digest"],
                "is_verified": verdict["is_verified"],
                "summary": job.get("summary", {}),
                "events_file": f"{workload}.events.json",
                "duration_s": evs[-1]["ts"],
            }

        (out / "protocol-graphs.json").write_text(json.dumps(graphs))
        (out / "demo-manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
        print(f"[record] wrote {out}/demo-manifest.json "
              f"({len(manifest['scenarios'])} scenarios, source={source})")
        bad = [w for w, s in manifest["scenarios"].items() if not s["is_verified"]]
        if bad:
            print(f"[record] WARNING: unverified scenarios: {bad}")
            return 1
        return 0
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    raise SystemExit(main())
