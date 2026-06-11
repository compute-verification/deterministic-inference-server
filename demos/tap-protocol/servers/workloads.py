"""Workload registry + runner shared by the Host and Recomp clusters.

A *workload* is one of the four proof-compare scenarios, executed by its
capture harness as a subprocess. The harness writes a capture JSON (--out) and
prints ``PROGRESS {json}`` lines on stdout; the runner streams those to an
``on_progress`` callback and returns the capture plus its canonical digest.

The digest is the protocol's verification object: the Recomp cluster re-runs
the SAME workload with the SAME params and bitwise-compares canonical capture
digests. Harnesses are responsible for being reproducible (deterministic
kernels, normalized tool output -- see demos/proof-compare/capture/*).

Conversion to the canonical task-graph trace (``capture_to_trace``) reuses the
proof-compare tracers unchanged; the Gateway exposes it at /run/<id>/graph.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.core.common.deterministic import canonical_json_bytes

CAPTURE_DIR = REPO_ROOT / "demos" / "proof-compare" / "capture"
TRACERS_DIR = REPO_ROOT / "demos" / "proof-compare" / "tracers"

# workload key -> harness + the params a client may set (whitelist: key ->
# (cli flag, coercion)). Anything not listed here is rejected before argv.
WORKLOADS: dict[str, dict[str, Any]] = {
    "inference": {
        "harness": CAPTURE_DIR / "run_inference.py",
        "params": {"prompt": ("--prompt", str), "max_tokens": ("--max-tokens", int)},
        "label": "Inference",
    },
    "spec": {
        "harness": CAPTURE_DIR / "run_spec.py",
        "params": {"prompt": ("--prompt", str), "max_tokens": ("--max-tokens", int),
                   "k": ("--k", int)},
        "label": "Speculative decoding",
    },
    "training": {
        "harness": CAPTURE_DIR / "run_lora.py",
        "params": {},
        "label": "LoRA training",
    },
    "coding": {
        "harness": CAPTURE_DIR / "run_coding_agent.py",
        "params": {},
        "label": "Coding agent",
    },
}


class WorkloadError(RuntimeError):
    pass


def canonical_digest(obj: Any) -> str:
    """sha256 over the canonical JSON bytes of obj."""
    return "sha256:" + hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


def build_argv(workload: str, params: dict, mock: bool, out_path: Path) -> list[str]:
    """Validate (workload, params) against the registry and build the argv."""
    if workload not in WORKLOADS:
        raise WorkloadError(f"unknown workload: {workload!r}; "
                            f"expected one of {sorted(WORKLOADS)}")
    spec = WORKLOADS[workload]
    argv = [sys.executable, str(spec["harness"]), "--out", str(out_path)]
    if mock:
        argv.append("--mock")
    for key, value in (params or {}).items():
        if key not in spec["params"]:
            raise WorkloadError(f"workload {workload!r} does not accept "
                                f"param {key!r}; allowed: {sorted(spec['params'])}")
        flag, coerce = spec["params"][key]
        argv.extend([flag, str(coerce(value))])
    return argv


def run_workload(
    workload: str,
    params: dict | None = None,
    mock: bool = False,
    on_progress: Callable[[dict], None] | None = None,
    timeout: float = 7200.0,
) -> tuple[dict, str]:
    """Run a workload harness to completion. Returns (capture, canonical digest).

    PROGRESS lines stream to ``on_progress`` as they arrive; all other harness
    output is forwarded to stderr (the server log).
    """
    out_path = Path(tempfile.mkdtemp(prefix=f"workload_{workload}_")) / "capture.json"
    argv = build_argv(workload, params or {}, mock, out_path)

    proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail: list[str] = []
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            tail.append(line)
            del tail[:-30]
            if line.startswith("PROGRESS "):
                if on_progress is not None:
                    try:
                        on_progress(json.loads(line[len("PROGRESS "):]))
                    except json.JSONDecodeError:
                        sys.stderr.write(f"[workloads] bad PROGRESS line: {line}\n")
            else:
                sys.stderr.write(f"[{workload}] {line}\n")
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise WorkloadError(f"workload {workload!r} timed out after {timeout}s")
    if proc.returncode != 0:
        raise WorkloadError(f"workload {workload!r} exited {proc.returncode}; "
                            "tail:\n" + "\n".join(tail))

    capture = json.loads(out_path.read_text())
    if not mock and _is_mock_capture(workload, capture):
        raise WorkloadError(f"workload {workload!r} produced a MOCK capture "
                            "in real mode")
    return capture, canonical_digest(capture)


def _is_mock_capture(workload: str, capture: dict) -> bool:
    if capture.get("mock"):
        return True
    kind = capture.get("kind", "")
    return kind.endswith("_mock")


def summarize(workload: str, capture: dict) -> dict:
    """A small, event-stream-friendly summary of a finished capture."""
    if workload == "inference":
        events = capture["events"]
        out = "".join(e["payload"].get("token", "") for e in events)
        return {"forward_passes": len(events), "output_preview": out[:140]}
    if workload == "spec":
        rounds = capture["rounds"]
        return {"rounds": len(rounds),
                "drafts_accepted": sum(r["num_accepted"] for r in rounds),
                "tokens": len(capture["output_ids"]),
                "output_preview": capture["output"][:140]}
    if workload == "training":
        steps = capture["steps"]
        return {"steps": len(steps),
                "first_loss": steps[0]["loss"], "final_loss": steps[-1]["loss"],
                "evals": len(capture["evals"]),
                "last_eval": (capture["evals"][-1]["text"][:140]
                              if capture["evals"] else "")}
    if workload == "coding":
        calls = capture["calls"]
        return {"calls": len(calls),
                "forward_passes": sum(max(c["gen_tokens"], 1) for c in calls),
                "tests_passed": capture["tests_passed"],
                "fix_rounds": capture["fix_rounds"]}
    raise WorkloadError(f"unknown workload: {workload!r}")


def capture_to_trace(workload: str, capture: dict) -> dict:
    """Convert a capture to the canonical trace via the proof-compare tracers."""
    if str(TRACERS_DIR) not in sys.path:
        sys.path.insert(0, str(TRACERS_DIR))
    if workload == "inference":
        if "events" not in capture or "shapes" not in capture:
            raise WorkloadError("inference capture is not a canonical trace")
        return capture            # run_inference.py already emits the trace
    if workload == "spec":
        import specdecode
        return specdecode.trace_spec_real(capture)
    if workload == "training":
        import training
        return training.trace_training_real(capture)
    if workload == "coding":
        import coding
        return coding.trace_coding_real(capture)
    raise WorkloadError(f"unknown workload: {workload!r}")
