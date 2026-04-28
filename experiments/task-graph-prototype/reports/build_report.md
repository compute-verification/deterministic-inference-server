# Build report — task graph prototype

**Date:** 2026-04-28
**Scope:** session work on `experiments/task-graph-prototype/`

## Executive summary

The prototype moved from a 6-scenario, 14/15-PASS proof-of-concept to a
13-scenario, 15/15-PASS demonstration with an interactive HTML viewer.

Concretely:

- **Honest run now passes all 15 checks.** The previous Skip on
  check 9 (checkpoint consistency) is gone — RFC 6962 consistency
  proofs are implemented end-to-end on both Python (prover + verifier)
  and Rust (verifier), with a round-trip test over sizes 1..7
  including tampered proofs.
- **12 of 15 distinct checks have a dedicated failing test.** Added 7
  surgical adversarial mutators on top of the existing 5 so every
  check that can be cleanly broken via single-field mutation has one
  scenario that fails at it and no other. The remaining 3
  (schema, tap_ordering, ancestry_compute) are inherently
  weakly-failable.
- **Workload restructured** from a 4-task linear chain to a 5-pod /
  5-task DAG with fan-out at `pod_R0` and fan-in at `pod_D0`, so
  cumulative ancestry compute walks a real DAG and the policy gate
  has two artifacts above threshold (both signed).
- **Self-contained HTML viewer** (`viewer.html`, ~200 KB, no server,
  no CDN, no JS deps) renders the full DAG as SVG and answers all 8
  spec success-condition questions for any artifact in any of the 13
  scenarios. Verified end-to-end with playwright headless chromium:
  zero JS errors, all interactions populate correctly.

What was *not* done: real vLLM inference, model-shape-aware FLOP
estimation, encryption of artifact bodies, ZK-VM packaging. The vLLM
plan is sketched but not started.

## Chain of thought

### 1. Audit → load-bearing gaps

Reviewed the unstaged work against the one-day spec. Substrate looked
solid: real ed25519, real Merkle log, byte-mirrored canonical JSON
between Python and Rust (with unit tests pinning the encoder),
surgical adversarial mutators with documented retargeting decisions
in the experiment log.

Two material gaps surfaced:

1. **Check 9 was Skip, not Pass.** With only one STH there was
   nothing to chain. Acceptable for a one-day demo but the cheapest
   route to "all 15 PASS" was to actually implement RFC 6962
   consistency proofs.
2. **Adversarial coverage was thin.** 5 of 15 checks (3, 4, 11, 12,
   15.partition_crossing) had no failing test. Without that, "demo
   passes" doesn't prove the validator is doing real work — those
   checks could be commented out and the demo would still pass.

These were the two highest-leverage fixes: they convert the demo
from "looks complete" to "provably load-bearing."

### 2. RFC 6962 consistency proofs

Implemented because it (a) closed the only Skip and (b) exercised
the cross-language byte-mirroring discipline that the rest of the
crypto in this prototype depends on.

- **Python prover** (`transparency_log.py`): `_consistency_proof`
  and `_subproof` mirror RFC 6962 §2.1.2 verbatim.
- **Python and Rust verifier**: `verify_consistency` mirrors
  §2.1.4.2. The bit-iterative form is fiddly; I traced it manually
  on `(m=3, n=7)` to verify the algorithm before coding.
- **Round-trip test** (`crypto.rs`): for tree sizes 1..7, build the
  proof, verify positive case, then tamper one node and assert
  verification *fails*. Negative tests matter — verifiers that always
  return true also pass round-trip.
- **Orchestrator**: new `checkpoint_log()` signs an intermediate STH
  at the current log size; `emit()` collects all STHs into
  `previous_sths` and emits a consistency proof from each previous
  STH to the final tree head.
- **Validator check 9**: verifies each previous STH signature, then
  for each consistency proof verifies it via `verify_consistency`
  against the recomputed current root. Returns Skip only when no
  previous STHs are declared (truly vacuous).

Honest run: 14/15 + 1 SKIP → **15/15 PASS**.

### 3. Workload expansion

The 4-task linear chain was unconvincing for a viewer: no fan-out,
no fan-in, ancestry compute walked a chain not a DAG. Restructured:

```
external prompts (3) ─┐
                      v
                 pod_R0:t0 (concat_hash) ─→ research_artifact_0
                                                    │
                              ┌─────── transmit ────┴─── transmit ──────┐
                              v       (tap_R0R1)        (tap_R0R2)      v
                          pod_R1:t0                                 pod_R2:t0
                          (xor_fold)                              (identity_pad)
                              │                                        │
                              v                                        v
                       research_summary_a                     research_summary_b
                              │                                        │
                              └─── transmit ───┐    ┌── transmit ──────┘
                                  (tap_R1D0)   v    v   (tap_R2D0)
                                          pod_D0:t0 (concat_hash) ── FAN-IN
                                                │
                                                v
                                         deployment_input  (cum 476, gated, signed)
                                                │
                                                ├── transmit (tap_D0D1)
                                                v
                                          pod_D1:t0 (concat_hash)
                                                │
                                                v
                                         final_output  (cum 572, gated, signed, terminal)
```

Cumulative ancestry compute now walks a real DAG (the shared
ancestor `research_artifact_0` flows into both branches). Two
artifacts trip the 300-flop gate, both signed by `reviewer_alice`.
5 transmissions = 5 leaves in the log = enough for a meaningful
intermediate-STH-at-size-2 plus final-STH-at-size-5 with a real
consistency proof.

**Roadblock + fix**: the existing seeded random replay sampler
landed on a sample that missed the new terminal task `pod_D1:t0`,
which made `replay_mismatch` silently pass (the corrupted
commitment was on a task no replay covered). Replaced with "replay
every task" — a production deployment would sample, but the demo
wants comprehensive coverage so check 14 has something to bite on.

### 4. Seven new adversarial scenarios

Each mutator perturbs exactly one field and was verified to fail at
its named check and *not before*:

| Scenario | Check it should fail | What's mutated |
|---|---|---|
| `dangling_input` | 3 graph_wellformed | task input references nonexistent artifact |
| `cyclic_graph` | 4 acyclic | adds last task's output as first task's input |
| `bad_sth_signature` | 8 checkpoint_validation | corrupts STH sig (root still valid) |
| `bad_compute_attestation` | 11 compute_attestation | corrupts hypervisor sig |
| `over_budget_compute` | 12 task_compute_accounting | inflates one task's flops past pod budget |
| `forbidden_partition_crossing` | 15.partition_crossing | flips research→deployment to allowed=False |
| `replay_required_violation` | 15.replay_required | min_sample_count > actual sample size |

Two normalization tweaks were needed:
- `eval_ancestry_gate` error message changed from `policy_ancestry_gate:` to
  `ancestry_gate:` so the demo runner's regex consistently prepends
  `policy_` and produces e.g. `policy_partition_crossing` for all
  policy kinds.
- `demo.sh` first-failure parser now extracts the policy kind from the
  inner error message instead of bucketing everything as bare `policy`.

Result: 13 scenarios total (1 honest + 12 adversarial), 12 of 15 distinct
checks exercised. Demo runs ~5s, every scenario behaves as expected.

### 5. Pivot — interactive HTML viewer

User explicitly asked for an interactive HTML thing where they could
click any artifact and see the 8 spec questions answered. This became
the priority because it's the most useful thing for explaining the
system to a reviewer, and the substrate work above was groundwork.

**Architecture decisions:**

- **Self-contained HTML, no server, no CDN.** `python3 make_viewer.py`
  produces `viewer.html`; double-click opens it in a browser via
  `file://`. Easy to send to a reviewer or pin in a PR.
- **All crypto pre-computed in Python at build time.** The browser only
  looks up answers — no JS crypto, no library dependencies. Inclusion
  proofs verified, consistency proofs verified, replay matches checked,
  policies evaluated, all baked into the embedded JSON bundle.
- **Validator invoked per scenario in `--json` mode.** The viewer embeds
  the *actual* Rust validator's verdict + per-check status, so what
  shows in the UI is what the validator decided — not a Python re-imp.
- **Hand-rolled SVG layout.** Layered top-to-bottom by longest-path
  topology, x-spacing within layer, cubic beziers for input/output
  edges, dashed gold curves for tap arcs labeled with `tap_id #seq`.

**UI elements:**
- Scenario picker (13 scenarios) with verdict badge + expected-failure note
- Meta row: run_id, n_pods, n_tasks, n_artifacts, n_transmissions, n_taps, log_id, tree_size, n_policies
- Graph SVG with externals (orange), tasks (research blue / deployment purple), tap arcs
- Side panel populated on click with all 8 spec questions
- Wrong answers get red borders; right answers get green borders; N/A gets muted
- Expandable per-scenario "all 15 validator checks" list
- Expandable append-only log proof material

**Verification.** No browser was available locally, so I installed
playwright + chromium just to verify rendering. Headless run produced
zero JS errors, every artifact click populated the side panel, scenario
switching re-rendered the graph and re-bound the click handlers.
Screenshots at `/tmp/viewer-final3.png` (full graph),
`/tmp/viewer-honest-final-panel.png`, `/tmp/viewer-glm-panel.png`,
`/tmp/viewer-replay-panel.png` (panel states with red borders on the
failed Q6/Q7 answers).

A small layout regression — tap-arc labels overlapped artifact-row text
because the midpoint of a tap curve sometimes lands directly on an
artifact node — was fixed by bowing the curve out to the right and
positioning the label up near the sender, off the artifact row.

### 6. Integration

`demo.sh` is now a 4-phase pipeline:
1. Build the Rust validator
2. Generate the 13 bundles
3. Validate each, assert each behaves as expected
4. Build `viewer.html`

End message points at the viewer file. End-to-end runs in ~5 seconds.

## Verification

- `bash demo.sh`: 13/13 scenarios behave as expected, viewer rebuilt.
- `cargo test --release`: 5 tests (3 canonical-JSON, 1 Merkle inclusion
  round-trip, 1 consistency proof round-trip with positive + tampered).
- Honest run: 15/15 PASS.
- Every adversarial scenario fails at its named check, no earlier check
  fires first.
- Playwright headless render: zero JS errors, all interactions work.

## Files changed / added

**Modified:**
- `validator/src/types.rs` — typed `PreviousSth`, `ConsistencyProof`
- `validator/src/crypto.rs` — `verify_consistency` + round-trip test
- `validator/src/checks.rs` — implemented check 9, normalized policy
  error message
- `scripts/sim/transparency_log.py` — prover + verifier
- `scripts/sim/orchestrator.py` — `checkpoint_log()`, multi-STH emit
- `scripts/sim/workload.py` — 5-pod fan-out/fan-in DAG
- `scripts/sim/adversarial.py` — 7 new mutators
- `scripts/run_demo.py` — replay every task
- `demo.sh` — viewer build step, robust first-failure parser
- `EXPERIMENT_LOG.md` — appended session log

**New:**
- `scripts/make_viewer.py` — bundle builder (~330 LOC)
- `scripts/viewer/viewer_template.html` — UI template (~330 LOC)
- `viewer.html` — generated artifact (~200 KB, 13 scenarios embedded)
- `reports/build_report.md` — this file

## Not done

| Item | Why it's deferred | What it'd take |
|---|---|---|
| Real vLLM inference | No GPU available locally; needs Hyperbolic/Lambda H100. Plan sketched in chat. | New `vllm_completion` op + c3 determinism config + 1–2 days on GPU |
| Model-shape-aware FLOP estimator | Couples to vLLM work but can ship CPU-only first | Half day; pure stdlib; model registry of `(N, L, H)` |
| `flop_consistency` adversarial | Requires the FLOP estimator | Hours after the estimator |
| Encryption of artifact bodies | Validator already reads only commitments+sizes, so it's ortho | Orchestrator emit-path change only |
| ZK-VM (SP1) packaging | Rust core is structured for it (pure fns, no I/O), but standing up SP1 is a separate experiment | Whole separate effort |
| Per-task replay-cost gate | Avoids letting a malicious orchestrator force expensive replays | Few hours plus a new policy kind |

## What you can show someone now

`./demo.sh` then `xdg-open viewer.html`. They can:
- Pick `honest` from the dropdown — green PASS badge, green-bordered
  answers everywhere, click any artifact to walk through the 8
  questions.
- Pick `graph_log_mismatch`, click `deployment_input` — Q6 and Q7
  go red with the actual mismatching commitments shown.
- Pick `replay_mismatch`, click `final_output` — Q7 goes red.
- Pick `policy_gate_violation`, click `deployment_input` or
  `final_output` — Q8 shows the ancestry gate failing without a
  reviewer signature.
- Pick `forbidden_partition_crossing`, click `research_summary_a`
  or `research_summary_b` — Q8 shows partition_crossing violation
  on the actual transmission edge that's now forbidden.
- Expand "show all 15 validator checks for this scenario" to see
  the full check ladder per run.

That's the prototype's whole story in one HTML file.
