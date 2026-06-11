"""Capture a REAL inference trace on a GPU (the forcing function).

Loads a real HF model, greedy-decodes a prompt, and emits a canonical trace with
real token ids + real config dims (model.config.to_dict()) -> the inference tab
becomes a genuine run (no whitespace/mock fakery). Token text is decoded into
each node's payload. Default output is demos/proof-compare/traces/
inference.real.json, which build_all.py picks up automatically.

Emits one ``PROGRESS {json}`` line per forward pass on stdout so a workload
runner can stream live progress. --mock swaps the model for a deterministic
CPU stand-in (the trace gains a top-level ``"mock": true`` marker so it can
never be presented as a real run).

Run on the GPU box:  python3 run_inference.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# Deterministic cuBLAS workspace -- must be set before any torch import. The
# 4-node protocol re-runs this capture on a second process and bitwise-
# compares the traces, so the run itself must be reproducible.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REPO_ROOT = Path(__file__).resolve().parents[3]
TRACERS = REPO_ROOT / "demos" / "proof-compare" / "tracers"
for _p in (REPO_ROOT, TRACERS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from inference import trace_inference  # noqa: E402
from modules.proof_server import flops as F  # noqa: E402

MODEL_ID = "Qwen/Qwen3-1.7B"
PROMPT = "The capital of France is"
MAX_TOKENS = 12
DEFAULT_OUT = REPO_ROOT / "demos" / "proof-compare" / "traces" / "inference.real.json"


def _progress_wrap(next_token, max_tokens: int):
    """Wrap a next_token fn to print one PROGRESS line per forward pass."""
    state = {"i": 0}

    def wrapped(ids):
        t = next_token(ids)
        state["i"] += 1
        print("PROGRESS " + json.dumps(
            {"type": "token", "i": state["i"], "of": max_tokens, "token_id": t},
            sort_keys=True), flush=True)
        return t

    return wrapped


def real_backend():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # error out rather than silently pick a nondeterministic kernel
    torch.use_deterministic_algorithms(True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    # transformers >= 5: `torch_dtype=` was removed (use `dtype=`) and
    # device_map="cuda" requires accelerate (absent on the nix image).
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16).to("cuda").eval()
    cfg = model.config.to_dict()   # NOT dict(model.config) -- that drops keys

    @torch.inference_mode()
    def next_token(ids):
        t = torch.tensor([ids], device="cuda")
        return int(model(t).logits[0, -1].argmax().item())

    return tok, cfg, next_token


def mock_backend():
    """Deterministic CPU stand-in: token = sha256 of the context (plumbing only)."""
    class MockTok:
        def encode(self, text):
            return [1000 + (b % 100) for b in text.encode("utf-8")][:8] or [1000]

        def decode(self, ids):
            return "".join(f"<{i}>" for i in ids)

    def next_token(ids):
        h = hashlib.sha256(json.dumps(ids).encode()).digest()
        return 2000 + int.from_bytes(h[:2], "big") % 1000

    return MockTok(), dict(F.KNOWN_SHAPES[MODEL_ID]), next_token


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--mock", action="store_true",
                    help="deterministic CPU stand-in for the model")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    args = ap.parse_args()

    tok, cfg, next_token = mock_backend() if args.mock else real_backend()
    prompt_ids = tok.encode(args.prompt)
    trace = trace_inference(prompt_ids, _progress_wrap(next_token, args.max_tokens),
                            MODEL_ID, cfg, args.max_tokens)

    # decode token text into payloads for the viz tooltip. Every node carries
    # the token its forward pass PRODUCED -- including the prefill, whose last
    # position produces the first generated token.
    for ev in trace["events"]:
        text = tok.decode([ev["payload"]["token_id"]])
        ev["payload"]["token"] = text
        if ev["kind"] == "prefill":
            ev["payload"]["prompt"] = args.prompt
            ev["label"] = "prefill"
        else:
            ev["label"] = text.strip() or "·"

    if args.mock:
        trace["mock"] = True   # a mock trace must never pass as a real run

    out = Path(args.out)
    out.write_text(json.dumps(trace))

    decoded = tok.decode([e["payload"]["token_id"] for e in trace["events"]])
    print(f"PROMPT : {args.prompt!r}")
    print(f"OUTPUT : {decoded!r}")
    print(f"model  : {MODEL_ID}  ({cfg['num_hidden_layers']}L d={cfg['hidden_size']})")
    print(f"events : {len(trace['events'])}  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
