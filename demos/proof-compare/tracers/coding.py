"""Coding-agent tracer (STUB).

Emits a canonical trace for the search -> plan -> codegen -> verify diamond from
a hand-captured trace. The real version (Task 13) is a minimal agent loop whose
tool calls emit this with real token counts + context lengths (so attention is
included). Until then, stub nodes use attended=0 (weight-only cost, matching the
prior coding graph) -- we do NOT fabricate a context/attention number.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.proof_server import flops as F
from modules.proof_server.tracer import Tracer


def trace_coding_stub(agent_key, prompt, retrievals, plan, codegens, verify):
    """Hand-captured agent run -> canonical trace. STUB (see module docstring).

    ``prompt``/``plan``/``verify`` are ``{"tokens", "label", ...}`` dicts;
    ``retrievals`` is a list of those plus ``"kind": "search"|"fetch"``;
    ``codegens`` is a list of ``{"tokens", "label", ...}``.
    """
    tr = Tracer()
    tr.add_shape(agent_key, F.shape_for(agent_key))

    def add(kind, spec, inputs):
        return tr.event(
            kind, model=agent_key, tokens=int(spec.get("tokens", 0)), attended=0,
            logits=spec.get("logits", 0), inputs=inputs,
            label=spec.get("label", kind), payload=spec.get("payload", {}),
        )

    prompt_id = add("prompt", prompt, [])
    retr_ids = [add(r["kind"], r, [prompt_id]) for r in retrievals]
    plan_id = add("plan", plan, retr_ids)
    cg_ids = [add("codegen", c, [plan_id]) for c in codegens]
    add("test", verify, cg_ids)

    return tr.trace()
