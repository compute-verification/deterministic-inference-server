"""Deterministic greedy speculative decoding + run trace.

The algorithm here is the *real* thing; only the "models" are pluggable. A model
is a deterministic next-token function ``next_token(token_ids) -> token_id``. The
draft model proposes K tokens; the target model verifies the longest matching
prefix and emits one correction/bonus token.

Greedy speculative decoding is **output-identical to plain greedy target
decoding** -- the verify pass guarantees it. ``speculative_decode`` returns the
output plus a per-round trace (drafts / num_accepted / correction) that the
proof server turns into a task graph, and ``greedy_decode`` produces the
reference output the e2e test compares against.

Two backends:
  * ``mock_models`` -- deterministic token generators (no GPU). The target has a
    canned continuation T; the draft mirrors T but proposes a wrong token at a
    fixed set of absolute positions, so accept/reject is real and controllable.
  * ``hf_models`` (GPU) -- wraps two real Hugging Face causal LMs with greedy
    argmax under the determinism knobs. Imported lazily so the mock path needs
    no torch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# A model: maps a token-id context to the next token id, deterministically.
NextToken = Callable[[list[int]], int]


@dataclass
class SpecRound:
    drafts: list[int]          # the K proposed token ids
    num_accepted: int          # how many of them the target accepted (0..K)
    correction: int            # the target's correction/bonus token id


@dataclass
class SpecResult:
    output: list[int]          # committed token ids (excludes the prompt)
    rounds: list[SpecRound] = field(default_factory=list)
    draft_steps: int = 0       # total draft forward passes
    target_passes: int = 0     # total target forward passes (== len(rounds))


def speculative_decode(
    prompt_ids: list[int],
    draft_next: NextToken,
    target_next: NextToken,
    k: int,
    max_tokens: int,
) -> SpecResult:
    """Greedy speculative decoding. ``k`` drafts proposed per round."""
    output: list[int] = []
    rounds: list[SpecRound] = []
    draft_steps = 0
    ctx = list(prompt_ids)

    while len(output) < max_tokens:
        # 1. draft model proposes k tokens autoregressively.
        proposed: list[int] = []
        for _ in range(k):
            proposed.append(draft_next(ctx + proposed))
            draft_steps += 1

        # 2. target verifies the longest matching greedy prefix.
        accepted: list[int] = []
        for i in range(k):
            if proposed[i] == target_next(ctx + proposed[:i]):
                accepted.append(proposed[i])
            else:
                break

        # 3. correction token: the target's own token at the first divergence
        #    (or the bonus token, if all k were accepted).
        correction = target_next(ctx + accepted)
        rounds.append(SpecRound(drafts=list(proposed),
                                num_accepted=len(accepted),
                                correction=correction))

        # 4. commit accepted + correction (respecting max_tokens).
        for tok in accepted + [correction]:
            if len(output) >= max_tokens:
                break
            output.append(tok)
            ctx.append(tok)

    return SpecResult(output=output, rounds=rounds,
                      draft_steps=draft_steps, target_passes=len(rounds))


def greedy_decode(prompt_ids: list[int], target_next: NextToken, max_tokens: int) -> list[int]:
    """Plain greedy decoding with the target model -- the reference output."""
    out: list[int] = []
    ctx = list(prompt_ids)
    for _ in range(max_tokens):
        t = target_next(ctx)
        out.append(t)
        ctx.append(t)
    return out


# ---------------------------------------------------------------------------
# Mock backend (no GPU)
# ---------------------------------------------------------------------------

def mock_models(
    prompt_len: int,
    target_continuation: list[int],
    draft_wrong_positions: set[int],
    wrong_token_base: int = 9000,
) -> tuple[NextToken, NextToken]:
    """Two deterministic token generators for the mock demo.

    The target's greedy output is exactly ``target_continuation`` (independent of
    draft content, since greedy is position-determined here). The draft mirrors it
    but returns a distinct wrong token at each absolute position in
    ``draft_wrong_positions`` -- those are where speculation gets rejected.
    """
    T = target_continuation

    def target_next(ctx: list[int]) -> int:
        pos = len(ctx) - prompt_len
        return T[pos] if 0 <= pos < len(T) else T[-1]

    def draft_next(ctx: list[int]) -> int:
        pos = len(ctx) - prompt_len
        if pos in draft_wrong_positions:
            return wrong_token_base + pos
        return T[pos] if 0 <= pos < len(T) else T[-1]

    return draft_next, target_next


# ---------------------------------------------------------------------------
# HF backend (GPU) -- imported lazily so the mock path needs no torch.
# ---------------------------------------------------------------------------

def hf_models(draft_model_id: str, target_model_id: str):  # pragma: no cover - GPU only
    """Return (draft_next, target_next, tokenizer) backed by two real HF models.

    Greedy argmax under the determinism knobs; draft and target must share a
    tokenizer (Qwen3-0.6B + Qwen3-1.7B do). Env determinism flags must already be
    set before torch import (see the cluster server entrypoints).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(target_model_id)
    draft = AutoModelForCausalLM.from_pretrained(
        draft_model_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    target = AutoModelForCausalLM.from_pretrained(
        target_model_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    @torch.inference_mode()
    def _argmax(model, ctx: list[int]) -> int:
        ids = torch.tensor([ctx], device="cuda")
        logits = model(ids).logits[0, -1, :]
        return int(torch.argmax(logits).item())

    def draft_next(ctx: list[int]) -> int:
        return _argmax(draft, ctx)

    def target_next(ctx: list[int]) -> int:
        return _argmax(target, ctx)

    return draft_next, target_next, tok


# ---------------------------------------------------------------------------
# Shared run helpers (host AND recomp call these so their work is byte-identical)
# ---------------------------------------------------------------------------

_MOCK_WORDS = "the quick brown fox jumps over a lazy dog while two birds sing".split()


def _mock_text(tok_id: int) -> str:
    if tok_id >= 9000:                       # a rejected draft guess
        return f"⟂{tok_id - 9000}"
    if tok_id >= 1000:                       # a real continuation token
        return _MOCK_WORDS[(tok_id - 1000) % len(_MOCK_WORDS)] + " "
    return f"<{tok_id}>"


def to_response(prompt_len: int, res: SpecResult, idtext) -> dict:
    """Shape a SpecResult into the SpecDecodeResponse wire dict (text + ids)."""
    return {
        "output": "".join(idtext(i) for i in res.output),
        "output_ids": list(res.output),
        "prompt_len": prompt_len,
        "rounds": [
            {
                "drafts": [idtext(d) for d in r.drafts],
                "num_accepted": r.num_accepted,
                "correction": idtext(r.correction),
            }
            for r in res.rounds
        ],
        "draft_steps": res.draft_steps,
        "target_passes": res.target_passes,
    }


def run_mock(prompt: str, max_tokens: int, k: int) -> dict:
    """Deterministic mock spec-decode run (no GPU). Same prompt -> same trace."""
    prompt_len = max(1, len(prompt.split()))
    horizon = max_tokens + k + 2
    T = [1000 + i for i in range(horizon)]
    wrong = {i for i in range(horizon) if (i + len(prompt)) % 4 == 3}
    draft_next, target_next = mock_models(prompt_len, T, wrong)
    res = speculative_decode(list(range(prompt_len)), draft_next, target_next, k, max_tokens)
    return to_response(prompt_len, res, _mock_text)


def run_hf(prompt: str, max_tokens: int, k: int, draft_next, target_next, tok) -> dict:  # pragma: no cover - GPU only
    """Real spec-decode run over two HF models; decode tokens via the shared tokenizer."""
    prompt_ids = tok.encode(prompt)
    res = speculative_decode(prompt_ids, draft_next, target_next, k, max_tokens)
    return to_response(len(prompt_ids), res, lambda i: tok.decode([i]))
