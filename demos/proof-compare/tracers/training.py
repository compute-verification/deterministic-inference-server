"""Training tracer (STUB).

Emits a canonical trace from SIMULATED LoRA-training data. The real version
(Task 12) instruments workflows/deterministic_lora_training.py to emit per-step
loss + an eval+checkpoint every eval_steps from a real GPU run.

Design note: an eval is a real little inference, so it is FLATTENED into real
`eval_prefill` + `eval_decode` events linked off the checkpoint step (the
canonical model has no nesting; making the eval real events IS the unification).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.proof_server import flops as F
from modules.proof_server.tracer import Tracer


def trace_training_stub(model_key, max_steps, batch, seq_len, loss_trajectory,
                        eval_steps, eval_prompt_len=8, eval_gen=3):
    """Simulated LoRA run -> canonical trace. STUB (see module docstring)."""
    tr = Tracer()
    tr.add_shape(model_key, F.shape_for(model_key))

    tokens = batch * seq_len
    attended = batch * seq_len * (seq_len + 1) // 2
    prev_step = None
    for s in range(max_steps):
        loss = loss_trajectory[s] if s < len(loss_trajectory) else None
        step_id = tr.event(
            "train_step", model=model_key, tokens=tokens, attended=attended,
            mode="lora_bwd", logits=tokens, inputs=([prev_step] if prev_step is not None else []),
            label=f"step {s}", payload={"loss": loss},
        )
        prev_step = step_id

        # eval every eval_steps (after that step), flattened into real inference.
        if (s + 1) % eval_steps == 0:
            p = eval_prompt_len
            ep = tr.event(
                "eval_prefill", model=model_key, tokens=p, attended=p * (p + 1) // 2,
                logits=1, inputs=[step_id], label=f"eval@{s + 1}",
                payload={"checkpoint_digest": f"sha256:step{s + 1}", "metric": loss},
            )
            prev_eval = ep
            ctx = p
            for _ in range(eval_gen):
                ctx += 1
                prev_eval = tr.event(
                    "eval_decode", model=model_key, tokens=1, attended=ctx,
                    logits=1, inputs=[prev_eval], label="eval_decode",
                )

    return tr.trace()
