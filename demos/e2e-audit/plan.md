# E2E Audit Verification — Implementation Plan

**Goal:** Build a standalone script that demonstrates an end-to-end audit loop:
a primary GPU inference run produces token commitments, an auditor randomly
challenges one token, and a verification run reproduces just that request to
prove the challenged token is correct.

**Date:** 2026-04-23
**Branch:** `multi-gpu-determinism` (or a new feature branch off it)

---

## Background you need

### What this project does

This repo implements **bitwise-deterministic LLM inference**. Given the same
model weights, the same prompts, and the same config flags, two independent
servers will produce *identical* token outputs. We've proven this across
millions of tokens on H100s.

### Why this script matters

The determinism guarantee enables a new thing: **spot-check auditing**. If
inference is deterministic, a third party can randomly pick any token from a
run, reproduce the inference independently, and verify the token matches. The
provider can't cheat — they'd need to know which token will be challenged
before generating it, and the auditor picks randomly after the fact.

This script is the MVP of that idea: everything on one machine, one GPU, no
network, no separate processes. Just prove the mechanical loop works.

### Security caveat

**This demo does NOT implement a cryptographically sound audit protocol.**
The commitment scheme uses HMAC with a hardcoded shared key. Since both the
provider and the verifier know the key, the provider could forge commitments
for tokens it never actually generated. A real audit protocol would need one
of:
- The auditor holds the key; the provider commits blindly (plain hash, no
  HMAC).
- An asymmetric scheme (provider signs with a private key, auditor verifies
  with public key).
- A commit-then-reveal protocol with a nonce the provider can't predict.

What this demo *does* prove is that the **deterministic replay** works — if
you run the same request twice with the same config, you get the same tokens.
That's the hard part. The cryptographic binding is a protocol-layer concern
that can be swapped in later without changing the inference machinery.

### The three determinism flags

The deterministic stack has three cumulative config levels. You'll use all
three (the "c3" config):

| Flag | What it does |
|------|-------------|
| `enforce_eager=True` | Disables CUDA Graphs and `torch.compile`. Prevents non-deterministic kernel autotuning across runs. |
| `CUBLAS_WORKSPACE_CONFIG=:4096:8` | Forces cuBLAS to use deterministic algorithm selection for matrix multiplications. |
| `VLLM_BATCH_INVARIANT=1` + `attention_backend=FLASH_ATTN` | Makes outputs independent of batch composition. Pins the attention backend. |

### How existing inference works

Look at `scripts/d6/benchmark_determinism_overhead.py` — it's the simplest
example of running vLLM with deterministic flags. The pattern is:

```python
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["VLLM_BATCH_INVARIANT"] = "1"
os.environ["PYTHONHASHSEED"] = "0"

from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    seed=42,
    dtype="auto",
    enforce_eager=True,
    attention_backend="FLASH_ATTN",
    gpu_memory_utilization=0.90,
    max_model_len=4096,
    trust_remote_code=True,
)
params = SamplingParams(temperature=0, max_tokens=16, seed=42)
outputs = llm.generate(["Hello world"], [params])
token_ids = list(outputs[0].outputs[0].token_ids)
```

Key points:
- `temperature=0` = greedy decoding. Required for determinism.
- `seed=42` in both `LLM()` and `SamplingParams()`.
- The deterministic env vars MUST be set before `from vllm import ...`.
- After you're done with an LLM instance, `del llm` + `torch.cuda.empty_cache()`
  + `gc.collect()` to free VRAM before creating another one.

### Existing test patterns

Tests use `unittest.TestCase`. They live under `tests/`. Run with:
```bash
python3 -m unittest tests.unit.test_your_module
```

Test helpers are in `tests/helpers.py` — `read_json()` and `write_json()` for
JSON fixture I/O. Tests don't use pytest.

### Repo path notes

- `pkg/__init__.py` already exists — do NOT recreate it.
- Scripts in `scripts/` are one directory deep from repo root, so the path
  idiom for repo root is `Path(__file__).resolve().parents[1]` (not
  `parents[2]`).
- Tests in `tests/unit/` are two directories deep, so they use
  `Path(__file__).resolve().parents[2]`.

---

## Architecture

```
scripts/e2e_verify.py          <-- the script (standalone, no manifest/lockfile)
pkg/e2e/__init__.py             <-- empty
pkg/e2e/crypto.py               <-- commit_token() + commit_token_stream()
tests/unit/test_e2e_crypto.py   <-- unit tests for the crypto module
```

### Why this structure

- The crypto logic goes in `pkg/` so it's testable in isolation, importable
  from the script, and could be reused later if we integrate with the runner.
- The script stays in `scripts/` — it's a demo, not production infrastructure.
- We do NOT modify any existing files. No manifest schema changes, no runner
  changes, no verifier changes.

---

## Tasks

All tasks below are meant to be done in order. Each task ends with a commit.
Each commit should be small and focused. Run the tests listed after each task
before committing.

---

### Task 1: Token commitment module

**What:** Create `pkg/e2e/crypto.py` with two functions.

**Files to create:**
- `pkg/e2e/__init__.py` (empty file)
- `pkg/e2e/crypto.py`

**Do NOT create `pkg/__init__.py` — it already exists.**

**Specification for `crypto.py`:**

```python
"""Deterministic token commitment via HMAC-SHA256.

We use HMAC rather than AES because we only need commitments (one-way),
not decryption. Both sides compute the same HMAC for the same token ID
and compare. Determinism is guaranteed: same key + same input = same output.

NOTE: This module uses a hardcoded shared key and does NOT provide
cryptographic binding against a malicious provider. See the security
caveat in docs/plans/e2e-audit-verification.md.
"""
from __future__ import annotations

import hashlib
import hmac

# Hardcoded key for the MVP. In production this would be held exclusively
# by the auditor, or replaced with an asymmetric scheme.
_DEFAULT_KEY = b"deterministic-verify-key-00000000"


def commit_token(token_id: int, *, key: bytes = _DEFAULT_KEY) -> str:
    """Return a hex HMAC-SHA256 commitment for a single token ID.

    Args:
        token_id: The integer token ID from the model's vocabulary.
            Must be non-negative.
        key: HMAC key (32 bytes). Defaults to the hardcoded MVP key.

    Returns:
        64-character lowercase hex string.

    Raises:
        ValueError: If token_id is negative.
    """
    ...


def commit_token_stream(token_ids: list[int], *, key: bytes = _DEFAULT_KEY) -> list[str]:
    """Commit a list of token IDs, preserving order.

    Returns a list of hex HMAC strings, one per token. The i-th output
    corresponds to the i-th input token.
    """
    ...
```

**Implementation notes:**
- `commit_token`: Validate `token_id >= 0`, then HMAC the 4-byte big-endian
  representation of the token ID.
  ```python
  if token_id < 0:
      raise ValueError(f"token_id must be non-negative, got {token_id}")
  return hmac.new(key, token_id.to_bytes(4, "big"), hashlib.sha256).hexdigest()
  ```
- `commit_token_stream`: map `commit_token` over the list.
- That's it. No classes, no config objects, no abstractions.

**How to test it — create `tests/unit/test_e2e_crypto.py`:**

```python
"""Unit tests for pkg.e2e.crypto."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pkg.e2e.crypto import commit_token, commit_token_stream


class TestCommitToken(unittest.TestCase):

    def test_returns_64_char_hex(self):
        """Output is a 64-char lowercase hex string (SHA-256 digest)."""
        result = commit_token(42)
        self.assertEqual(len(result), 64)
        self.assertRegex(result, r"^[0-9a-f]{64}$")

    def test_deterministic(self):
        """Same token ID + same key = same output, every time."""
        a = commit_token(1000)
        b = commit_token(1000)
        self.assertEqual(a, b)

    def test_different_tokens_differ(self):
        """Different token IDs produce different commitments."""
        a = commit_token(0)
        b = commit_token(1)
        self.assertNotEqual(a, b)

    def test_different_keys_differ(self):
        """Different keys produce different commitments for the same token."""
        a = commit_token(42, key=b"key-a" + b"\x00" * 27)
        b = commit_token(42, key=b"key-b" + b"\x00" * 27)
        self.assertNotEqual(a, b)

    def test_known_value(self):
        """Regression test: verify against a pre-computed HMAC.

        If you change the encoding (e.g. byte order, key), this breaks.
        Computed via:
            python3 -c "
                import hmac, hashlib
                print(hmac.new(
                    b'deterministic-verify-key-00000000',
                    (42).to_bytes(4, 'big'),
                    hashlib.sha256,
                ).hexdigest())
            "
        """
        expected = "d008610c21dc3edf8fcb0e0cdab97fd01895ab97e63531aecbb10a503137444b"
        result = commit_token(42)
        self.assertEqual(result, expected)

    def test_token_id_zero(self):
        """Token ID 0 (common padding token) commits without error."""
        result = commit_token(0)
        self.assertEqual(len(result), 64)

    def test_large_token_id(self):
        """Token IDs up to 2^31-1 (vLLM vocab range) work."""
        result = commit_token(2**31 - 1)
        self.assertEqual(len(result), 64)

    def test_negative_token_raises(self):
        """Negative token IDs are invalid and must raise."""
        with self.assertRaises(ValueError):
            commit_token(-1)


class TestCommitTokenStream(unittest.TestCase):

    def test_length_matches_input(self):
        tokens = [10, 20, 30]
        result = commit_token_stream(tokens)
        self.assertEqual(len(result), 3)

    def test_order_preserved(self):
        """The i-th output corresponds to commit_token(tokens[i])."""
        tokens = [100, 200, 300]
        stream = commit_token_stream(tokens)
        for i, tok in enumerate(tokens):
            self.assertEqual(stream[i], commit_token(tok))

    def test_empty_list(self):
        self.assertEqual(commit_token_stream([]), [])
```

**Run:**
```bash
python3 -m unittest tests.unit.test_e2e_crypto -v
```

**Before committing:** all 11 tests pass.

**Commit message:** `e2e: add deterministic token commitment module (HMAC-SHA256)`

---

### Task 2: The e2e verification script — primary run

**What:** Create `scripts/e2e_verify.py` with the primary inference run and
token commitment. This task does NOT yet include the challenge/verification
step — just run inference and print the committed outputs.

**Files to create:**
- `scripts/e2e_verify.py`

**File to read first (understand the pattern):**
- `scripts/d6/benchmark_determinism_overhead.py` — lines 128–204 show how to
  set up deterministic vLLM and tear it down.

**Specification:**

```python
#!/usr/bin/env python3
"""End-to-end audit verification demo.

Demonstrates the mechanical audit loop for deterministic LLM inference:
run inference, commit all output tokens, randomly challenge one token,
and verify it by replaying the request from scratch.

SECURITY NOTE: This demo uses HMAC with a hardcoded shared key. It proves
that deterministic replay works, but does NOT provide cryptographic binding
against a malicious provider. See docs/plans/e2e-audit-verification.md for
details on what a production protocol would need.

Prerequisites:
    - GPU with sufficient VRAM for the model (Qwen 2.5 1.5B needs ~4 GB)
    - vLLM installed (pip install vllm)
    - Model weights accessible from HuggingFace (auto-downloaded on first run)

Usage:
    # Default (Qwen 2.5 1.5B, seed 42, random challenge)
    python3 scripts/e2e_verify.py

    # Specific model and forced challenge
    python3 scripts/e2e_verify.py --model mistralai/Mistral-7B-Instruct-v0.3 --challenge req-1:5

    # Verbose output (shows plaintext token IDs)
    python3 scripts/e2e_verify.py --verbose

PASS means: the verification run produced the same token at the challenged
position as the primary run. The deterministic replay worked.

FAIL means: the tokens diverged. Something is wrong with the determinism
setup (env vars not set before vLLM import, engine teardown not clean, etc.).
"""
from __future__ import annotations

import argparse
import gc
import os
import random
import sys
import time
from pathlib import Path

# scripts/ is one level deep from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pkg.e2e.crypto import commit_token, commit_token_stream

# ── Prompts ──────────────────────────────────────────────────────────────
# A small, diverse set. Keep it short so the demo runs in under a minute.
PROMPTS = [
    {"id": "req-0", "prompt": "Explain how photosynthesis works in one paragraph.", "max_new_tokens": 16},
    {"id": "req-1", "prompt": "What is the difference between TCP and UDP?", "max_new_tokens": 16},
    {"id": "req-2", "prompt": "Describe the life cycle of a star.", "max_new_tokens": 16},
    {"id": "req-3", "prompt": "Why is the sky blue?", "max_new_tokens": 16},
    {"id": "req-4", "prompt": "What is a hash table?", "max_new_tokens": 16},
]


def setup_deterministic_env() -> None:
    """Set all env vars for full deterministic mode (c3).

    MUST be called before any `import vllm` or `import torch`.
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    os.environ["PYTHONHASHSEED"] = "0"


def run_inference(
    prompts: list[dict],
    *,
    model: str,
    seed: int,
) -> dict[str, list[int]]:
    """Run deterministic inference. Returns {request_id: [token_ids]}.

    Creates and destroys the LLM engine, freeing VRAM for the next call.
    """
    from vllm import LLM, SamplingParams
    import torch

    llm = LLM(
        model=model,
        seed=seed,
        dtype="auto",
        enforce_eager=True,
        attention_backend="FLASH_ATTN",
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        trust_remote_code=True,
    )

    prompt_texts = [p["prompt"] for p in prompts]
    params_list = [
        SamplingParams(temperature=0, max_tokens=p["max_new_tokens"], seed=seed)
        for p in prompts
    ]

    outputs = llm.generate(prompt_texts, params_list)

    result: dict[str, list[int]] = {}
    for prompt_def, output in zip(prompts, outputs):
        result[prompt_def["id"]] = list(output.outputs[0].token_ids)

    del llm
    torch.cuda.empty_cache()
    gc.collect()

    return result
```

**For this task, the `main()` function should:**
1. Parse `--model` (default `Qwen/Qwen2.5-1.5B-Instruct`) and `--seed`
   (default `42`) from CLI args.
2. Call `setup_deterministic_env()`.
3. Call `run_inference()` with `PROMPTS`.
4. Call `commit_token_stream()` on each request's tokens.
5. Print a summary: how many requests, total tokens, and the first 16 chars
   of each commitment (so you can eyeball it).

**Example output:**

```
=== E2E Audit Verification ===
Model: Qwen/Qwen2.5-1.5B-Instruct
Seed:  42

Phase 1: Primary inference run
  5 prompts
  Inference complete (2.3s)
  req-0: 16 tokens, commitment[0]=a3f8c91b...
  req-1: 16 tokens, commitment[0]=7d2e44f0...
  req-2: 16 tokens, commitment[0]=1bc09a3e...
  req-3: 16 tokens, commitment[0]=e5f712d8...
  req-4: 16 tokens, commitment[0]=9a01bf47...
  Total: 80 tokens committed
```

**How to test (manual, requires GPU):**
```bash
python3 scripts/e2e_verify.py
```

Verify it prints the summary without errors. Verify that running it twice
produces the same commitment hex prefixes (because inference is deterministic
and HMAC is deterministic).

**Commit message:** `e2e: primary inference run with token commitments`

---

### Task 3: Challenge and verification

**What:** Add the challenge selection and verification run to the script.

**Files to modify:**
- `scripts/e2e_verify.py`

**Add these functions:**

```python
def select_challenge(
    commitments: dict[str, list[str]],
) -> tuple[str, int]:
    """Pick a random (request_id, token_position) to challenge.

    Returns:
        (request_id, token_position) where token_position is 1-indexed
        (i.e. the number of tokens the verifier must generate to reach
        this position).
    """
    ...
```

Pick a random request ID from the keys, then a random position from
`1..len(commitments[request_id])` inclusive. Use `random.Random(None)` so the
challenge is different each run (this is the auditor — it should NOT be
deterministic).

```python
def verify_challenge(
    request_id: str,
    token_position: int,
    expected_commitment: str,
    prompts: list[dict],
    *,
    model: str,
    seed: int,
) -> dict:
    """Reproduce inference for one request and verify the challenged token.

    Args:
        token_position: 1-indexed position of the challenged token.

    Returns a dict with keys:
        - "pass": bool
        - "request_id": str
        - "token_position": int (1-indexed)
        - "expected": str (commitment from primary run)
        - "actual": str (commitment from verification run)
    """
    ...
```

This function should:
1. Find the prompt dict for `request_id` in `prompts`.
2. Make a copy with `max_new_tokens` overridden to `token_position`:
   ```python
   challenge_prompt = {**original_prompt, "max_new_tokens": token_position}
   ```
3. Call `run_inference()` with a list containing ONLY that one prompt.
4. Get the last token produced: `verification_tokens[-1]` (index `-1`).
   This is the token at position `token_position` (1-indexed).
5. Compute its commitment: `commit_token(verification_tokens[-1])`.
6. Compare against `expected_commitment`, which is looked up by the caller as
   `commitments[request_id][token_position - 1]` (0-indexed into the
   commitment list from Phase 1).

**Off-by-one summary (read this carefully):**
- `token_position` is **1-indexed** (matches `max_new_tokens` semantics).
- The commitment list from Phase 1 is **0-indexed** (a normal Python list).
- So the expected commitment is at `commitments[request_id][token_position - 1]`.
- The verification run generates `token_position` tokens total, and we
  check the last one (`[-1]`), which is the token at that position.

**Update `main()` to add phases 2 and 3:**

```
Phase 2: Challenge selection
  Challenging req-3, position 7 of 16

Phase 3: Verification run
  Replaying req-3 with max_new_tokens=7
  Inference complete (1.1s)
  Expected: e5f712d8b3a1c...
  Actual:   e5f712d8b3a1c...

PASS
```

If the verification fails, print `FAIL` and exit with code 1.
If it passes, exit with code 0.

**How to test (manual, requires GPU):**
```bash
# Run multiple times — the challenged position changes, but result is PASS
python3 scripts/e2e_verify.py
python3 scripts/e2e_verify.py
python3 scripts/e2e_verify.py
```

Each run should PASS. If any run fails, something is wrong with the
determinism setup (most likely env vars not set before vLLM import, or the
engine teardown isn't clean enough).

**Commit message:** `e2e: add challenge selection and verification run`

---

### Task 4: Edge cases and polish

**What:** Handle edge cases and add `--challenge` and `--verbose` flags.

**Files to modify:**
- `scripts/e2e_verify.py`
- `tests/unit/test_e2e_crypto.py` (already created in Task 1 — add tests)

**Add to the argparser:**

`--challenge` to force a specific challenge instead of random:
```
python3 scripts/e2e_verify.py --challenge req-2:3
```
Format: `request_id:position` (1-indexed). This makes debugging reproducible.
Parse it by splitting on `:`, validate that the request ID exists and the
position is in range. If invalid, print an error and exit 1.

`--verbose` to print the actual plaintext token IDs alongside the commitments.
Useful for debugging, off by default.

**Edge cases to cover when testing:**
1. **Token position 1** (first token only). The verification run generates
   exactly 1 token. This exercises the minimum path.
   ```bash
   python3 scripts/e2e_verify.py --challenge req-0:1
   ```
2. **Full length** (token position == max_new_tokens). The verification run
   generates all tokens. This should be equivalent to the primary run for
   that request.
   ```bash
   python3 scripts/e2e_verify.py --challenge req-4:16
   ```

**Run:**
```bash
python3 -m unittest tests.unit.test_e2e_crypto -v
```

**Commit message:** `e2e: add --challenge and --verbose flags`

---

## File inventory

| File | Action | Task |
|------|--------|------|
| `pkg/e2e/__init__.py` | Create (empty) | 1 |
| `pkg/e2e/crypto.py` | Create | 1 |
| `tests/unit/test_e2e_crypto.py` | Create | 1 |
| `scripts/e2e_verify.py` | Create | 2, expanded in 3–4 |

**Files you should NOT modify:** anything in `cmd/`, `pkg/manifest/`,
`pkg/networkdet/`, `pkg/common/`, `schemas/`, or existing tests.

**Files that already exist (do NOT recreate):** `pkg/__init__.py`.

## How to run the full test suite after all tasks

```bash
# Unit tests (no GPU required)
python3 -m unittest tests.unit.test_e2e_crypto -v

# E2E demo (requires GPU)
python3 scripts/e2e_verify.py
python3 scripts/e2e_verify.py --challenge req-0:1    # first token
python3 scripts/e2e_verify.py --challenge req-4:16   # last token
python3 scripts/e2e_verify.py --verbose               # show token IDs
```

All four GPU runs should print `PASS`. If run on a machine without a GPU or
without vLLM installed, the script should fail with a clear import error (do
NOT add a fallback/mock mode — this is a GPU demo).

## What success looks like

After all 4 tasks are committed, running `python3 scripts/e2e_verify.py`
on a machine with a GPU prints something like:

```
=== E2E Audit Verification ===
Model: Qwen/Qwen2.5-1.5B-Instruct
Seed:  42

Phase 1: Primary inference run
  5 prompts, 80 tokens generated (2.3s)

Phase 2: Challenge selection
  Challenging req-3, token position 7 of 16

Phase 3: Verification run
  Replaying req-3 with max_new_tokens=7 (1.1s)
  Expected: e5f712d8b3a1c...
  Actual:   e5f712d8b3a1c...

PASS
```

The entire run takes under 5 minutes on a single GPU (most of the time is
vLLM engine startup, which happens twice).
