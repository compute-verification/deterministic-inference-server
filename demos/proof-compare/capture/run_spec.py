"""Capture a REAL speculative-decoding run on a GPU.

Wraps the real algorithm in demos/spec-decode/spec_decode.py (the same
``speculative_decode`` the 4-server spec demo serves): draft Qwen3-0.6B
proposes k greedy tokens per round, target Qwen3-1.7B verifies the longest
matching greedy prefix and emits one correction/bonus token. The recorded
``rounds`` (drafts / num_accepted / correction) are exactly what
tracers/specdecode.py consumes — this upgrades the spec scenario's provenance
from "ported rounds" to a real recorded capture.

Emits one ``PROGRESS {json}`` line per round on stdout so a workload runner
can stream live progress. --mock runs the deterministic CPU mock models.

Run on the GPU box:  python3 run_spec.py --out spec.real.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Determinism env must be set before any torch import.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SPEC_DEMO = REPO_ROOT / "demos" / "spec-decode"
for p in (REPO_ROOT, SPEC_DEMO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import spec_decode as sd  # noqa: E402
from modules.proof_server import flops as F  # noqa: E402

DRAFT_ID = "Qwen/Qwen3-0.6B"
TARGET_ID = "Qwen/Qwen3-1.7B"
DEFAULT_PROMPT = "The key idea behind bitwise-deterministic inference is"


def _progress_cb(state: dict, k: int, max_tokens: int):
    """Returns an on_round callback that prints one PROGRESS jsonl per round."""
    def on_round(rd) -> None:
        state["round"] += 1
        state["committed"] = min(state["committed"] + rd.num_accepted + 1, max_tokens)
        print("PROGRESS " + json.dumps({
            "type": "round", "round": state["round"], "k": k,
            "accepted": rd.num_accepted, "committed": state["committed"],
            "max_tokens": max_tokens,
        }, sort_keys=True), flush=True)
    return on_round


def spec_capture(prompt: str, max_tokens: int, k: int, mock: bool) -> dict:
    state = {"round": 0, "committed": 0}
    on_round = _progress_cb(state, k, max_tokens)

    if mock:
        prompt_len = max(1, len(prompt.split()))
        horizon = max_tokens + k + 2
        T = [1000 + i for i in range(horizon)]
        wrong = {i for i in range(horizon) if (i + len(prompt)) % 4 == 3}
        draft_next, target_next = sd.mock_models(prompt_len, T, wrong)
        res = sd.speculative_decode(list(range(prompt_len)), draft_next,
                                    target_next, k, max_tokens, on_round=on_round)
        resp = sd.to_response(prompt_len, res, sd._mock_text)
        draft_cfg = dict(F.KNOWN_SHAPES[DRAFT_ID])
        target_cfg = dict(F.KNOWN_SHAPES[TARGET_ID])
    else:
        from transformers import AutoConfig
        draft_next, target_next, tok = sd.hf_models(DRAFT_ID, TARGET_ID)
        prompt_ids = tok.encode(prompt)
        res = sd.speculative_decode(prompt_ids, draft_next, target_next, k,
                                    max_tokens, on_round=on_round)
        resp = sd.to_response(len(prompt_ids), res, lambda i: tok.decode([i]))
        draft_cfg = AutoConfig.from_pretrained(DRAFT_ID).to_dict()
        target_cfg = AutoConfig.from_pretrained(TARGET_ID).to_dict()

    return {
        "kind": "spec_decode_capture",
        "draft_model": DRAFT_ID,
        "target_model": TARGET_ID,
        "draft_config": draft_cfg,
        "target_config": target_cfg,
        "prompt": prompt,
        "prompt_len": resp["prompt_len"],
        "k": k,
        "max_tokens": max_tokens,
        "output": resp["output"],
        "output_ids": resp["output_ids"],
        "rounds": resp["rounds"],
        "draft_steps": resp["draft_steps"],
        "target_passes": resp["target_passes"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "spec.real.json"))
    ap.add_argument("--mock", action="store_true", help="CPU mock models (no GPU)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    cap = spec_capture(args.prompt, args.max_tokens, args.k, args.mock)
    Path(args.out).write_text(json.dumps(cap, indent=1, sort_keys=True) + "\n")
    n_acc = sum(r["num_accepted"] for r in cap["rounds"])
    print(f"wrote {args.out}: {len(cap['rounds'])} rounds, "
          f"{n_acc} drafts accepted, {len(cap['output_ids'])} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
