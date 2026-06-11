"""Build all four canonical task graphs and emit them as JSON for the viz.

Each scenario's tracer -> a canonical trace -> build_graph -> a Graph dict.
We collect {inference, spec, training, coding} and write them to BOTH
traces/graphs.json (the canonical artifact) and viz/public/graphs.json, which the
React Flow app (demos/proof-compare/viz) fetches at runtime. No HTML is baked --
the frontend is a real Vite app now, not a single hand-edited file.

Run:  python3 demos/proof-compare/build_all.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (REPO_ROOT, HERE / "tracers"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from modules.proof_server.graph import build_graph
import specdecode as t_spec
import training as t_train
import coding as t_code

TRACES = HERE / "traces"
VIZ_PUBLIC = HERE / "viz" / "public"


def _inference_trace() -> dict:
    """Real captured H100 trace. No mock fallback: the viz labels every tab
    real, so a missing capture must FAIL the build, not silently degrade."""
    return json.loads((TRACES / "inference.real.json").read_text())


def _spec_trace() -> dict:
    data = json.loads((TRACES / "spec_rounds.json").read_text())
    return t_spec.trace_spec_decode(data["prompt_len"], data["rounds"])


def _training_trace() -> dict:
    """Real captured LoRA run (toy scale, H100) -> canonical trace."""
    capture = json.loads((TRACES / "training.real.json").read_text())
    return t_train.trace_training_real(capture)


def _coding_graph() -> dict:
    """Real captured coding-agent run (H100) -> canonical trace -> graph.

    The capture is one entry per actual LLM call (real prompt/gen token
    counts; capture/run_coding_agent.py); the tracer renders it as a DAG of
    forward passes -- one prefill per call + one decode per generated token,
    fanning out where the agent dispatched parallel calls (4 file reads, 3
    plan candidates, src||test codegen) and fanning back in at merges.

    The agent's task statement is a fixed, publicly-known constant, so it goes
    on the graph's whitelist: passing exactly that string into the orient call
    is free and shows no input size in the viewer (the orient FLOPs still
    count -- the forward pass ran).
    """
    capture = json.loads((TRACES / "coding.real.json").read_text())
    return build_graph(t_code.trace_coding_real(capture),
                       whitelist=[capture["prompt"]]).to_dict()


def build_all() -> dict:
    return {
        "inference": build_graph(_inference_trace()).to_dict(),
        "spec": build_graph(_spec_trace()).to_dict(),
        "training": build_graph(_training_trace()).to_dict(),
        "coding": _coding_graph(),
    }


def dump(data: dict) -> str:
    """Canonical JSON for the graphs payload."""
    return json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n"


def main() -> int:
    data = build_all()
    payload = dump(data)
    TRACES.mkdir(parents=True, exist_ok=True)
    VIZ_PUBLIC.mkdir(parents=True, exist_ok=True)
    (TRACES / "graphs.json").write_text(payload)
    (VIZ_PUBLIC / "graphs.json").write_text(payload)
    counts = {k: len(v["nodes"]) for k, v in data.items()}
    print(f"wrote {counts} to {TRACES / 'graphs.json'} and {VIZ_PUBLIC / 'graphs.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
