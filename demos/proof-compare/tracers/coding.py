"""Coding-agent tracer (STUB) — at FORWARD-PASS granularity.

Structurally identical to the inference tracer: every node is exactly one
forward pass — one ``prefill`` (reads a batch of tokens) or one ``decode``
(emits one token). A coding agent just alternates LLM turns with tool calls:

  * A tool call (search / fetch / run-tests) runs **no model forward pass**, so
    it is NOT a node. The tokens it returns are read by the NEXT turn's prefill
    (recorded in that prefill's ``payload.via``).
  * Each turn contributes one ``prefill`` over the tokens newly added to context
    this turn (the prompt for turn 0; a tool's output thereafter) followed by one
    ``decode`` per generated token.

So the whole run is a single chain — the same shape as plain inference, just
with periodic prefill "jumps" where tool output is ingested. STUB: the token
counts are estimates from a real "summarize a paper, then implement it" run
(demos/coding-agent); the real tracer (Task 13) plugs in measured counts. The
*shape* and FLOPs accounting (attention included via the growing context) are
already correct.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.proof_server import flops as F
from modules.proof_server.tracer import Tracer


def trace_coding_stub(agent_key, prompt, turns):
    """Render an agent run as a chain of prefill/decode forward passes.

    ``prompt`` is the user prompt text (stored in the first prefill's payload).
    ``turns`` is a list of dicts, one per LLM turn::

        {"role": str,          # phase tag: reason|triage|plan|codegen|test|...
         "prefill": int,       # context tokens read this turn (0 => pure continuation)
         "gen": int,           # tokens generated this turn (= number of decode nodes)
         "via": str|None,      # tool that produced the prefilled tokens (optional)
         "label": str}         # short title for the prefill node (optional)

    A turn with ``prefill == 0`` is a continuation (the agent keeps generating
    with no new tool context): it emits only decode nodes, no prefill.
    """
    if not turns:
        raise ValueError("a coding trace needs at least one turn")

    tr = Tracer()
    tr.add_shape(agent_key, F.shape_for(agent_key))

    ctx = 0            # running context length (drives causal attention)
    prev = None        # id of the previous node, to chain off
    first = True
    for turn in turns:
        role = turn["role"]
        # `phase` is the per-turn label ("write p_less.py"); it lets the viz
        # group a turn's forward passes together without merging adjacent turns
        # of the same role (e.g. the two codegen files).
        phase = turn.get("label") or role
        p = int(turn.get("prefill", 0))
        if p > 0:
            # New tokens occupy positions [ctx, ctx+p); each attends to all
            # tokens up to and including itself -> a triangle difference.
            attended = (ctx + p) * (ctx + p + 1) // 2 - ctx * (ctx + 1) // 2
            ctx += p
            payload = {"role": role, "phase": phase}
            if first:
                payload["prompt"] = prompt
            if turn.get("via"):
                payload["via"] = turn["via"]
            prev = tr.event(
                "prefill", model=agent_key, tokens=p, attended=attended, logits=1,
                inputs=([prev] if prev is not None else []),
                label=phase, payload=payload,
            )
        for _ in range(int(turn.get("gen", 0))):
            ctx += 1
            prev = tr.event(
                "decode", model=agent_key, tokens=1, attended=ctx, logits=1,
                inputs=([prev] if prev is not None else []),
                label=role, payload={"role": role, "phase": phase},
            )
        first = False

    return tr.trace()
