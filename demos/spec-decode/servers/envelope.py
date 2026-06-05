"""SignedEnvelope wire types for the spec-decode demo.

Same HMAC envelope as demos/tap-protocol, but the response carries a full
speculative-decoding trace (per-round drafts / accepted count / correction) so
the proof server can build a task graph from it. HMAC_KEY is a committed
constant -- integrity on the localhost channel only, not authentication (see
demos/tap-protocol/servers/envelope.py).
"""
from __future__ import annotations

import hashlib
import hmac
import sys
import threading
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.core.common.deterministic import canonical_json_bytes


HMAC_KEY: bytes = b"spec-decode-demo-key-do-not-use!"
assert len(HMAC_KEY) == 32, "HMAC_KEY must be exactly 32 bytes"


# ---------------------------------------------------------------------------
# Wire types
# ---------------------------------------------------------------------------

class SpecDecodeRequest(BaseModel):
    prompt: str
    max_tokens: int = 16
    k: int = 4                      # draft tokens proposed per round


class SpecRoundWire(BaseModel):
    drafts: list[str]               # K proposed token texts
    num_accepted: int               # 0..K
    correction: str                 # the target's correction/bonus token text


class SpecDecodeResponse(BaseModel):
    output: str                     # decoded committed text
    output_ids: list[int]           # committed token ids (the bitwise compare target)
    prompt_len: int                 # prompt length in tokens
    rounds: list[SpecRoundWire]     # the per-round trace
    draft_steps: int                # total draft forward passes
    target_passes: int              # total target forward passes (== len(rounds))


class EnvelopeData(BaseModel):
    id: int
    payload: dict[str, Any]


class SignedEnvelope(BaseModel):
    data: EnvelopeData
    signature: str = Field(description="hex HMAC-SHA256 over canonical_json_bytes(data)")


# ---------------------------------------------------------------------------
# Monotonic id counter (Gateway-only)
# ---------------------------------------------------------------------------

_id_lock = threading.Lock()
_id_counter = 0


def next_id() -> int:
    global _id_counter
    with _id_lock:
        _id_counter += 1
        return _id_counter


# ---------------------------------------------------------------------------
# Sign / verify
# ---------------------------------------------------------------------------

def _compute_signature(data: EnvelopeData) -> str:
    msg = canonical_json_bytes(data.model_dump())
    return hmac.new(HMAC_KEY, msg, hashlib.sha256).hexdigest()


def sign(payload: dict[str, Any], envelope_id: int) -> SignedEnvelope:
    data = EnvelopeData(id=envelope_id, payload=payload)
    return SignedEnvelope(data=data, signature=_compute_signature(data))


def verify(env: SignedEnvelope) -> bool:
    expected = _compute_signature(env.data)
    return hmac.compare_digest(expected, env.signature)
