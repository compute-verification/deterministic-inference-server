"""Build all four canonical task graphs and bake them into the viz.

Each scenario's tracer -> a canonical trace -> build_graph -> a Graph dict.
We collect {inference, spec, training, coding} and (a) write them to
traces/graphs.json and (b) bake them into the `const DATA = ...;` line of
demos/proof-compare/viz/index.html.

Run:  python3 demos/proof-compare/build_all.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (REPO_ROOT, HERE / "tracers"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from modules.proof_server import flops as F
from modules.proof_server.graph import build_graph
import inference as t_inf
import specdecode as t_spec
import training as t_train
import coding as t_code

TRACES = HERE / "traces"
VIZ = HERE / "viz" / "index.html"


def _inference_trace() -> dict:
    """Real captured trace if present (Task 11), else a small labelled mock."""
    real = TRACES / "inference.real.json"
    if real.exists():
        return json.loads(real.read_text())
    # Placeholder until the GPU run: a short mock decode (clearly not real).
    tr = t_inf.trace_inference(
        prompt_ids=list(range(6)), next_token=t_inf.mock_next_token,
        model_key="Qwen/Qwen3-1.7B",
        shape_config=F.KNOWN_SHAPES["Qwen/Qwen3-1.7B"], max_tokens=8)
    tr["meta"] = {"real": False, "note": "mock placeholder; replaced by Task 11 GPU run"}
    return tr


def _spec_trace() -> dict:
    data = json.loads((TRACES / "spec_rounds.json").read_text())
    return t_spec.trace_spec_decode(data["prompt_len"], data["rounds"])


def _training_trace() -> dict:
    return t_train.trace_training_stub(
        "Qwen/Qwen3-1.7B", max_steps=6, batch=4, seq_len=8,
        loss_trajectory=[2.0, 1.6, 1.3, 1.1, 0.95, 0.9], eval_steps=2, eval_gen=3)


def _coding_trace() -> dict:
    # Hand-captured trace of the real p-less implementation run (STUB: tokens
    # are estimates; see demos/coding-agent). attended=0 until Task 13.
    return t_code.trace_coding_stub(
        "agent",
        prompt={"tokens": 40, "label": "prompt",
                "payload": {"text": "Summarize a paper that just came out, then implement it"}},
        retrievals=[
            {"kind": "search", "tokens": 620, "label": "search papers"},
            {"kind": "search", "tokens": 640, "label": "search truncation samplers"},
            {"kind": "fetch", "tokens": 1100, "label": "fetch arXiv abstract"},
            {"kind": "fetch", "tokens": 6300, "label": "fetch arXiv full"},
            {"kind": "fetch", "tokens": 1800, "label": "fetch repo"},
            {"kind": "fetch", "tokens": 1500, "label": "fetch reference code"},
        ],
        plan={"tokens": 2900, "label": "extract p-less algorithm"},
        codegens=[{"tokens": 1200, "label": "write p_less.py"},
                  {"tokens": 1400, "label": "write test_p_less.py"}],
        verify={"tokens": 400, "label": "run tests -> 9 passed"},
    )


def build_all() -> dict:
    return {
        "inference": build_graph(_inference_trace()).to_dict(),
        "spec": build_graph(_spec_trace()).to_dict(),
        "training": build_graph(_training_trace()).to_dict(),
        "coding": build_graph(_coding_trace()).to_dict(),
    }


def bake(html: str, data: dict) -> str:
    """Replace the `const DATA = ...;` line. lambda avoids \\u escape errors."""
    line = "const DATA = " + json.dumps(data) + ";"
    new, n = re.subn(r"const DATA = .*?;(?=\n)", lambda m: line, html, count=1)
    if n != 1:
        raise RuntimeError(f"expected exactly one `const DATA = ...;` line, replaced {n}")
    return new


def main() -> int:
    data = build_all()
    TRACES.mkdir(parents=True, exist_ok=True)
    (TRACES / "graphs.json").write_text(
        json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n")
    VIZ.write_text(bake(VIZ.read_text(), data))
    counts = {k: len(v["nodes"]) for k, v in data.items()}
    print(f"baked {counts} into {VIZ}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
