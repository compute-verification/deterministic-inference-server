# Task Graph Prototype — Demo Report

**Date:** 2026-04-27
**Spec:** one-day task graph prototype (in conversation, dated 2026-04-27)
**Status:** end-to-end demo passes; honest run + 5 adversarial mutations
behave as expected.

## What ships

```
experiments/task-graph-prototype/
  plan.md                 -- design
  EXPERIMENT_LOG.md       -- append-only milestone log
  schemas/                -- AttestedTaskGraph v0 JSON Schema
  scripts/sim/            -- Python simulator (pods, taps, transport,
                             transparency log, orchestrator, replay,
                             adversarial mutators, workload)
  scripts/run_demo.py     -- generates 6 bundles
  validator/              -- Rust crate: 15-check validator + CLI
  demo.sh                 -- end-to-end: build -> generate -> validate
  data/<scenario>/        -- generated bundles per scenario
  reports/memo.md         -- this file
```

`./demo.sh` runs in ~5 seconds on a laptop and prints:

```
  OK    bad_tap_signature        expected=tap_authenticity          got=tap_authenticity
  OK    graph_log_mismatch       expected=graph_log_consistency     got=graph_log_consistency
  OK    honest                   expected=PASS                      got=PASS
  OK    missing_log_entry        expected=log_inclusion             got=log_inclusion
  OK    policy_gate_violation    expected=policy_ancestry_gate      got=policy
  OK    replay_mismatch          expected=replay_correctness        got=replay_correctness

ALL PASS -- 6 scenarios behaved as expected
```

## Architecture

A single-process simulation. Four pods across two partitions
(`research`, `deployment`); three network taps on the inter-pod links;
one global RFC 6962 Merkle transparency log. Three external sources are
injected at `pod_R0`; four tasks chain through R0 → R1 → D0 → D1.

Each transmission is observed by exactly one tap which signs an
`ArtifactTransmissionEvent` with its ed25519 key. The log appends the
canonical-JSON of the signed event as a leaf, then signs an STH over
`(tree_size || root_hash)`. Inclusion proofs are emitted per leaf.

The orchestrator emits a single `AttestedTaskGraph v0` JSON file per run,
plus a `replay_responses.json` containing the orchestrator's response to
the validator's replay challenges. Both are consumed by the Rust
validator binary.

## Validator coverage (the 15 checks)

| # | Spec name              | Status      |
|---|------------------------|-------------|
| 1 | schema                 | implemented |
| 2 | artifact_commitment    | implemented |
| 3 | graph_wellformed       | implemented |
| 4 | acyclic                | implemented |
| 5 | tap_authenticity       | implemented (ed25519 over canonical-JSON unsigned event) |
| 6 | tap_ordering           | implemented |
| 7 | log_inclusion          | implemented (RFC 6962 inclusion proofs) |
| 8 | checkpoint_validation  | implemented (root recomputation + STH sig) |
| 9 | checkpoint_consistency | skipped in v0 (single STH; consistency proofs deferred) |
| 10 | graph_log_consistency | implemented (artifact ↔ signed transmission cross-check) |
| 11 | compute_attestation   | implemented (ed25519 over canonical-JSON pod attestation) |
| 12 | task_compute_accounting | implemented (interval containment + flops sum) |
| 13 | ancestry_compute      | computed via DFS, exposed to policies |
| 14 | replay_correctness    | implemented (commitment match) |
| 15 | policy                | three policy kinds: ancestry_compute_gate, partition_crossing, replay_required |

## Walkthrough on a single artifact

The eight spec questions, answered for `research_summary` from the
honest run:

```json
{
  "artifact_id": "research_summary",
  "commitment": "sha256:107d08c9a6a9a31e66214b6c46ac53aa9b42e7c6c7c0b7c0a51ca0e05f91a2e9",
  "size": 32,
  "external": false,
  "producer_task_id": "pod_R1:t0",
  "consumer_task_ids": ["pod_D0:t0"]
}
```

1. **What task produced it?** `pod_R1:t0` (operation `xor_fold`, partition
   `research`, claimed 32 flops, interval [3, 4] ms).
2. **What artifacts did that task depend on?** `research_artifact_0`
   (which itself ancestrally depends on the three external prompts
   injected at `pod_R0`).
3. **Was it transmitted over the simulated network?** Yes, once.
4. **Which tap observed the transmission?** `tap_R1D0` (seq 0,
   pod_R1 → pod_D0, timestamp 4 ms).
5. **Is the tap event included in the transparency log?** Yes — leaf
   index 1 in the global Merkle log of size 3, with a 2-hash inclusion
   proof against root
   `0ea3ba3aa45dda04...`.
6. **Did the graph's claim about the transmission match the logged
   event?** Yes — check 10 verifies that `graph.artifacts[research_summary].commitment == transmissions[*].commitment`
   for every event involving this artifact.
7. **If challenged, can the orchestrator rerun the producing task?** Yes
   — `pod_R1` re-executes `xor_fold` with `width=32` on the bytes of
   `research_artifact_0` and reproduces the exact 32-byte digest, whose
   commitment matches the declared one.
8. **Did the artifact and its producing task satisfy the toy policies?**
   Yes — `research_summary` has cumulative ancestry compute 142 flops
   (below the 300-flop gate), the only over-threshold task
   (`pod_D1:t0`, producing `final_output`, cum 334) carries
   `reviewer_alice`'s signature, the cross-partition transmission
   `pod_R1 → pod_D0` is in the allowed list, and replay sampled at
   least one task.

## Adversarial scenarios

All five mutators perturb the honest bundle in a single targeted way and
are caught by the expected validator check, with no other check firing
first:

- **bad_tap_signature** flips one byte of `transmissions[0].tap_signature`. Caught by check 5.
- **missing_log_entry** drops the trailing leaf and decrements
  `tree_size`. Caught by check 7 (length disagreement; would also fail
  the inclusion proof for the dropped index).
- **graph_log_mismatch** edits the `commitment` field of an artifact
  that DID transit a tap. The signed transmission is left untouched, so
  tap signature still verifies; check 10 catches the artifact-vs-
  transmission divergence.
- **replay_mismatch** edits the commitment of a TERMINAL artifact (one
  that never crossed a tap). Only the orchestrator's replay response
  reveals the lie; check 14 catches it.
- **policy_gate_violation** strips the reviewer signature from
  `pod_D1:t0` (whose output `final_output` exceeds the ancestry
  threshold). Check 15 evaluates the `ancestry_compute_gate` policy and
  fails.

## Path to SP1

The validator core is a pure-Rust crate with no I/O outside the binary.
The 15-check pipeline operates over deserialized structures and a
canonical-JSON encoder we control end-to-end. To prove the same
algorithm inside SP1:

1. Move the `bin/validate.rs` thin wrapper out of the proof program.
   Provide the graph and replay responses as committed inputs.
2. The cryptographic dependencies (`sha2`, `ed25519-dalek`, `hex`) are
   all `no_std`-friendly and have SP1-precompile-aware variants. SP1's
   ed25519 precompile delivers a >100x speedup on signature verification.
3. The `serde_json::Value` walking inside `canonical.rs` is the only
   step that allocates non-trivially; switching to a structured
   canonical encoder over the `types.rs` structs removes the dynamic
   value entirely if proof size becomes a constraint.

## What was deliberately deferred

- **Consistency proofs across STHs.** The demo emits one STH per run, so
  there's nothing to chain. Implementing the RFC 6962 consistency
  algorithm in both Python and Rust is a few hours; the algorithm is
  well-specified and the validator skips this check rather than
  silently passing.
- **Hypervisor compute attestations.** The `ComputeAttestation` here is a
  signed `(pod_id, interval, claimed_flops)` triple from a stub
  hypervisor key. Replacing the stub with real DCGM-derived flops is
  the next experiment.
- **Encryption / confidentiality.** The validator never reads artifact
  bytes — it only ever sees commitments, sizes, and timing metadata. So
  swapping plaintext artifact bodies for encrypted ones is a property
  of the orchestrator's emission path, not a validator change.
- **ZK wrapper.** Crate is structured for it; standing up the SP1 build
  is a separate task.

## Reproduce

```
./experiments/task-graph-prototype/demo.sh
```

Toolchain required: Python 3.10+, `cryptography` (already in repo env);
Rust 1.75+ (for the `ed25519-dalek` v2 dependency).
