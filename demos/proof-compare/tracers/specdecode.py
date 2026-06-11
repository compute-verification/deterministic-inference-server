"""Spec-decode tracer: per-round draft/verify trace -> canonical trace.

Consumes the exact ``rounds`` shape the spec-decode runner already produces
(``[{"drafts":[str], "num_accepted":int, "correction":str}]``). Each round is a
chain of draft events (draft model) that all fan into one verify event (target
model); the committed context grows across rounds. Shapes come from KNOWN_SHAPES
(the proof server has only model source strings, no live config).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.proof_server import flops as F
from modules.proof_server.tracer import Tracer


def trace_spec_decode(prompt_len, rounds, draft_key="hf://Qwen/Qwen3-0.6B",
                      target_key="hf://Qwen/Qwen3-1.7B",
                      draft_config=None, target_config=None):
    """Build a canonical trace from a speculative-decoding run.

    ``draft_config``/``target_config`` (optional) are the models' real HF config
    dicts; when omitted, shapes fall back to KNOWN_SHAPES via the key.
    """
    tr = Tracer()
    tr.add_shape(draft_key, draft_config or F.shape_for(draft_key))
    tr.add_shape(target_key, target_config or F.shape_for(target_key))

    ctx = prompt_len            # committed context length entering this round
    prev_verify = None
    for rd in rounds:
        drafts = rd["drafts"]
        a = rd["num_accepted"]
        k = len(drafts)

        draft_ids = []
        for i, tok in enumerate(drafts):
            # draft i is proposed after the i earlier drafts this round; it
            # chains off the previous draft (or the prior round's verify).
            chain_in = [draft_ids[-1]] if draft_ids else ([prev_verify] if prev_verify is not None else [])
            did = tr.event(
                "draft", model=draft_key, tokens=1, attended=ctx + i, logits=1,
                inputs=chain_in, status="accepted" if i < a else "rejected",
                label=f"draft {i}", payload={"token": tok},
            )
            draft_ids.append(did)

        # verify: one parallel target pass over the k drafted positions; every
        # draft (accepted or rejected) fans in (the target ingests them all).
        attended = sum(ctx + j for j in range(k + 1))
        vid = tr.event(
            "verify", model=target_key, tokens=k + 1, attended=attended,
            logits=k + 1, inputs=list(draft_ids), label="verify",
            payload={"correction": rd["correction"], "num_accepted": a},
        )
        prev_verify = vid
        ctx += a + 1            # committed = accepted drafts + the correction

    return tr.trace()


def trace_spec_real(capture):
    """Convert a run_spec.py capture into the canonical trace.

    Same converter contract as trace_training_real / trace_coding_real: takes
    the harness's recorded capture, uses the models' REAL configs from it.
    """
    if capture.get("kind") != "spec_decode_capture":
        raise ValueError(f"not a spec_decode_capture: {capture.get('kind')!r}")
    rounds = capture.get("rounds") or []
    if not rounds:
        raise ValueError("capture has no rounds")
    return trace_spec_decode(
        int(capture["prompt_len"]),
        rounds,
        draft_key="hf://" + capture["draft_model"],
        target_key="hf://" + capture["target_model"],
        draft_config=capture["draft_config"],
        target_config=capture["target_config"],
    )
