# Spec-decode demo — design and implementation plan

Greedy **speculative decoding** under the same 5-process topology as the
inference demo, fed into the proof server, producing a new **task graph shape**:
a committed spine with pruned (rejected) draft branches and two FLOP
weight-classes (draft ≪ target).

```
client ─► Gateway (8000) ─► Tap (8010) ─► Host Cluster (8020)   [draft+target spec-decode]
                              ├─ verify ─► Recomp Cluster (8030)  [re-run, bitwise compare]
                              └─ compare ► Proof Server (8050)    [compare host vs recomp, build graph]
```

## Algorithm (real; only the models are pluggable)

Each round: the **draft** model (Qwen3-0.6B) proposes K tokens; the **target**
model (Qwen3-1.7B) verifies all K in one forward pass, accepts the longest
matching greedy prefix, and emits one correction/bonus token. Drafts past the
first mismatch are rejected. Greedy spec-decode is **output-identical to plain
greedy target decoding** — the e2e/unit tests assert `spec_output == greedy`.

`spec_decode.py` implements the loop over two `next_token` functions:
- `mock_models` — deterministic generators (no GPU); a canned target
  continuation T with the draft wrong at fixed positions → real accept/reject.
- `hf_models` — two real HF causal LMs, greedy argmax under the determinism
  knobs (shared Qwen3 tokenizer).

## Graph shape

`build_spec_decode_task_graph()` (in `modules/proof_server/task_graph.py`):
per round, K `draft` nodes (status accepted/rejected) + one `verify` node. The
committed spine threads accepted drafts + corrections via `next`; rejected
drafts are dead-ends (`next=None`, unreferenced). FLOPs: draft ≈ `2·N_draft`,
verify ≈ `2·N_target·(K+1)`.

## Determinism / verification

Host and recomp run the **same** `run_mock`/`run_hf` over the same models, so
output ids **and** the per-round trace are bitwise-identical; the proof server
compares both. This makes spec-decode a verification scenario, not just an
inference optimization.

## Staging

- **Phase 0 (local, mock, no GPU):** engine + servers + builder + tests +
  `demo.sh --mock`. `ALL PASS`.
- **Phase 1 (vast.ai H100):** `demo.sh --real` with the two Qwen models; capture
  real graphs (`SPEC_GRAPH_OUT`), feed the viz with real data, redeploy surge.
  Logged in `EXPERIMENT_LOG.md`.
