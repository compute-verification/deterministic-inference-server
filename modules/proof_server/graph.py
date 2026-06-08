"""Canonical task-graph model + the single builder.

A *trace* (emitted by a scenario's tracer) is turned into a *Graph* by
``build_graph``. The builder is generic: it computes each event's FLOPs via
``flops.py`` and turns ``inputs`` into edges. No scenario-specific logic lives
here -- that is the whole point of the unification (see
docs/plans/unified-task-graph-autogen.md).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from modules.proof_server import flops as F

# The fixed node-kind vocabulary. Every event.kind must be one of these.
KINDS = (
    "prefill", "decode",                    # inference
    "train_step", "eval_prefill", "eval_decode",  # training
    "draft", "verify",                      # speculative decoding
    "prompt", "search", "fetch", "plan", "codegen", "test",  # coding agent
)


@dataclass
class Event:
    """One traced task. Mirrors an inference forward pass: input, cost, output."""
    id: int                       # unique within a trace; inputs MUST be < id (DAG)
    kind: str                     # one of KINDS
    inputs: list[int] = field(default_factory=list)  # event ids this depends on
    model: str = ""               # key into the trace's `shapes` table
    tokens: int = 0               # tokens processed
    attended: int = 0             # total (token,key) attention pairs; 0 => no attention
    mode: str = "fwd"             # "fwd" | "lora_bwd"
    logits: int = 0               # positions taking an LM-head projection
    status: str = ""              # optional, e.g. "accepted"|"rejected" (styling)
    label: str = ""               # short human title
    payload: dict | None = None   # display extras (token text, loss, file, ...)

    def __post_init__(self):
        if self.payload is None:
            self.payload = {}


@dataclass
class Edge:
    src: int
    dst: int


@dataclass
class Graph:
    nodes: list[dict] = field(default_factory=list)   # event fields + computed flops
    edges: list[Edge] = field(default_factory=list)
    shapes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "nodes": self.nodes,
            "edges": [asdict(e) for e in self.edges],
            "shapes": self.shapes,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"


def build_graph(trace: dict) -> Graph:
    """Turn a canonical trace into a Graph, computing each node's exact FLOPs.

    ``trace`` = ``{"shapes": {model_key: <config dict>}, "events": [<event dict>...]}``.
    Validates: unique ids; every input id exists and is < this event's id
    (acyclic, no forward refs); known kind; known model.
    """
    shapes = trace.get("shapes", {})
    events = trace.get("events", [])

    seen: set[int] = set()
    nodes: list[dict] = []
    edges: list[Edge] = []

    for ev in events:
        eid = ev["id"]
        if eid in seen:
            raise ValueError(f"duplicate event id: {eid}")
        kind = ev["kind"]
        if kind not in KINDS:
            raise ValueError(f"unknown kind: {kind!r} (event {eid})")
        model = ev.get("model", "")
        if model not in shapes:
            raise ValueError(f"event {eid} references unknown model {model!r}")
        for src in ev.get("inputs", []):
            # Spec invariant: inputs must reference a strictly-earlier id. This
            # rejects self-loops (src == eid) and forward refs (src > eid)
            # regardless of list order, guaranteeing the DAG the renderer's
            # longest-path layering relies on.
            if src >= eid:
                raise ValueError(
                    f"event {eid} input {src} is not < event id (self-loop/forward ref)")
            if src not in seen:
                raise ValueError(f"event {eid} input {src} is not a known prior event id")
            edges.append(Edge(src=src, dst=eid))

        shape = F.model_shape_from_config(shapes[model])
        node = dict(ev)
        node["flops"] = F.flops(shape, ev.get("tokens", 0), ev.get("attended", 0),
                                ev.get("mode", "fwd"), ev.get("logits", 0))
        nodes.append(node)
        seen.add(eid)

    return Graph(nodes=nodes, edges=edges, shapes=shapes)
