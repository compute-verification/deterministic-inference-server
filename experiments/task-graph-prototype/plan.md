# Task Graph Prototype — Implementation Plan

**Goal:** Build a runnable simulated-datacenter demo of the Section 3 accounting layer
(paper) / one-day spec: declared task graph, network-tap attestations, transparency
log, validator with replay challenges, plus adversarial runs that fail in distinct ways.

**Status:** in-progress.

## Scope (in)

- N simulated pods running cheap deterministic dummy tasks (no ML, no GPU).
- In-process simulated transport. One artifact per transmission.
- Network taps emit ed25519-signed `ArtifactTransmissionEvent`s.
- One global transparency log (Merkle, RFC 6962 style) with signed tree heads,
  inclusion proofs, and consistency proofs.
- Orchestrator assembles a single attested task graph JSON.
- Rust validator crate runs the 15-check algorithm. Pure Rust, no_std-friendly,
  designed to drop into SP1 later.
- Replay challenge loop driven by validator → orchestrator → pod.
- Toy policies: cumulative-ancestry-compute threshold, reviewer-signature gate,
  partition crossing rule.
- Adversarial scenarios that produce bad bundles with distinct failure causes.

## Scope (out, deferred)

- Real ML / GPU compute. Compute attestations are simulated stubs (a signed
  `(pod, interval, claimed_flops)` record by an "auditor-trusted hypervisor"
  keypair). The hypervisor-flops experiment is a separate followup.
- Encryption of artifact contents — the demo carries plaintext blobs but commits
  via hash, so artifact bodies can be omitted from the auditor's view in a real
  deployment without touching the validator.
- ZKVM / SP1 packaging. The validator is structured to make that drop-in (no I/O
  in core, pure functions over deserialized inputs), but actually proving inside
  SP1 is a separate task.
- Cross-process or networked components. Everything runs in a single process
  for demo simplicity; the schemas are the boundary.

## Architecture

```
                  +-------------------+
                  |  orchestrator     |
                  +---------+---------+
                            |
              schedules     |    collects task records
                            v
        +----------+    +-------+    +----------+
        | pod_0    |--->| tap_0 |--->| pod_1    |
        | pod_1    |    | tap_1 |    | pod_2    |
        |  ...     |    |  ...  |    |  ...     |
        +----------+    +---+---+    +----------+
                            |  signs (tap_id, seq, sender, receiver,
                            |          artifact_id, commitment, size, ts)
                            v
                  +-----------------------+
                  | transparency log      |
                  | (Merkle, STH, proofs) |
                  +-----------------------+

Orchestrator emits  attested_task_graph.json
        |
        v
+--------------------+      replay challenge: re-run task tau
| validator (Rust)   | <--- response: claimed outputs + commitment
| 15 checks + replay |  --> verdict.json
+--------------------+
```

## Layout

```
experiments/task-graph-prototype/
  plan.md
  EXPERIMENT_LOG.md
  schemas/
    attested_task_graph.v0.schema.json
  scripts/
    sim/                     -- Python harness
      pods.py
      orchestrator.py
      taps.py
      transport.py
      transparency_log.py
      replay.py
      adversarial.py
    run_demo.py              -- entry point: honest run + 5 bad runs
  validator/                 -- Rust crate
    Cargo.toml
    src/lib.rs
    src/types.rs
    src/checks.rs
    src/crypto.rs
    src/bin/validate.rs
  data/                      -- generated bundles per run
  reports/memo.md
  demo.sh                    -- one-shot demo
```

## Data model

See `schemas/attested_task_graph.v0.schema.json`. Top-level:

```
{
  "graph_version": "v0",
  "run_id": "...",
  "tasks": [Task],
  "artifacts": [Artifact],
  "transmissions": [TransmissionEvent],          // signed by taps
  "log": { "log_id", "tree_size", "checkpoints":[STH], "inclusion_proofs":{idx: proof} },
  "compute_attestations": [HypervisorAttestation],  // simulated
  "policies": [Policy],
  "tap_pubkeys": {tap_id: hex},
  "log_pubkey": hex,
  "hypervisor_pubkey": hex
}
```

Artifact commitment: `sha256:<hex>` of canonical bytes (matches repo style).
Tap signature: ed25519 over canonical-JSON-serialized event.
Log STH signature: ed25519 over `(tree_size || root_hash)`.

## Validator checks (mapped from spec § Validator algorithm)

The 15 spec checks become numbered Rust functions in `checks.rs`. For the
prototype, all 15 are implemented but some lean on stubs (compute attestations,
partitions). Numbering follows the spec verbatim.

## Demo

`./demo.sh` runs:

1. **honest** — produces a bundle that passes all 15 checks.
2. **missing_log_entry** — drop one tap event from the log.
3. **bad_tap_signature** — flip a byte in a tap signature.
4. **graph_log_mismatch** — graph claims a different artifact size than logged.
5. **replay_mismatch** — task's claimed output differs from re-execution.
6. **policy_gate_violation** — high-ancestry artifact lacks reviewer signature.

Validator prints a one-line verdict per scenario. Success = honest passes and
each bad run fails with the expected check name.

## Success condition

For the honest bundle, given any artifact ID, we can answer the eight spec
questions from `# Success condition` of the spec doc. The memo demonstrates
this on one chosen artifact.
