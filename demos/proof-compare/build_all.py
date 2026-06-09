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

from modules.proof_server import flops as F
from modules.proof_server.graph import build_graph
import inference as t_inf
import specdecode as t_spec
import training as t_train
import coding as t_code

TRACES = HERE / "traces"
VIZ_PUBLIC = HERE / "viz" / "public"


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
    # Forward-pass-granular trace of the real p-less implementation run (STUB:
    # token counts are estimates from that run; see demos/coding-agent). Each
    # turn = one prefill over the tokens READ this turn (the prompt, or a tool's
    # output) + one decode per GENERATED token. Tool calls (search/fetch/run-
    # tests) are not forward passes, so they appear only as the `via` tag +
    # prefill size of the turn that ingests their output. Counts are REALISTIC
    # (not shrunk for display): a real run of this task is several thousand
    # generated tokens, so the graph is several thousand decode nodes -- that is
    # the honest cost of "one node = one forward pass." prefill = bytes read
    # (search hits, fetched paper/repo, test output); gen = tokens written
    # (reasoning, the extracted plan, the two source files, the summary).
    return t_code.trace_coding_stub(
        "agent",
        prompt="Summarize a paper that just came out, then implement it",
        turns=[
            {"role": "reason",  "prefill": 40,   "gen": 80,   "label": "read prompt, decide to search"},
            {"role": "triage",  "prefill": 1260, "gen": 120,  "via": "search: recent papers", "label": "triage search hits"},
            {"role": "read",    "prefill": 7400, "gen": 300,  "via": "fetch: arXiv abstract + full text", "label": "read paper"},
            {"role": "plan",    "prefill": 3300, "gen": 2900, "via": "fetch: reference repo + code", "label": "extract p-less algorithm"},
            {"role": "codegen", "prefill": 0,    "gen": 1200, "label": "write p_less.py"},
            {"role": "codegen", "prefill": 0,    "gen": 1400, "label": "write test_p_less.py"},
            {"role": "test",    "prefill": 400,  "gen": 400,  "via": "run tests", "label": "read output -> 9 passed, summarize"},
        ],
    )


def build_all() -> dict:
    return {
        "inference": build_graph(_inference_trace()).to_dict(),
        "spec": build_graph(_spec_trace()).to_dict(),
        "training": build_graph(_training_trace()).to_dict(),
        "coding": build_graph(_coding_trace()).to_dict(),
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
