# Plan: Prover ↔ Verifier E2E Demo

**Status:** shipped (2026-05-04). All 37 tasks landed on branch `prover-verifier-demo`. Run `cd experiments/prover-verifier-demo && ./demo.sh --quick` to reproduce the headline `ALL PASS` outcome.
**Scope:** End-to-end demo of a prover–verifier protocol that detects training (or
exfiltration) hidden inside an inference workload, by combining a deterministic
serving server (the prover) with an active verifier that issues graph queries,
replay challenges, and observes streamed network traffic.
**Why:** Today, trust in a remote inference provider is binary — clients believe
the provider is doing inference because the provider says so. This demo gives
auditors a concrete, runnable artefact showing how a verifier can catch a
provider that secretly trains LoRAs/gradients or loads them from outside,
quantifying detection rate against attacker effort (FLOPs, bytes).

This document is the implementation plan **for the engineer doing the work**.
It assumes you have never opened this repo before. Read it linearly, top to
bottom. Don't skip the primer — half of the mistakes you can make on this
project come from not knowing the existing conventions.

---

## How to use this plan

- **One task = one commit.** If you find yourself bundling tasks, stop and split.
- **TDD is non-negotiable.** Every code-touching task has a "Tests first" step.
  Write the test, run it, see it fail with a useful message, then implement,
  then watch it go green. If you can't make the test fail before implementing,
  the test is wrong.
- **Read the linked files before writing code.** "Skim" is fine; "ignore" is not.
- **Do not invent abstractions.** If the existing code has a pattern, follow it.
  See [§ Conventions](#conventions) for the ones you'll trip over.
- **Update `EXPERIMENT_LOG.md`** at the end of each working session — what you
  did, what surprised you, what's blocked. This is for the next person, which
  is sometimes you in two weeks.
- **Don't try to design ahead.** Each task gives you the next bite. If a later
  task seems wrong by the time you reach it, that's normal — adjust the plan
  and note it in the log.

---

## Read this first (codebase primer)

You'll need to internalize about 15 files. Plan ~2 hours of reading before
writing any code.

### What this project does

Bitwise-deterministic LLM inference: same manifest + same container = identical
tokens, byte-for-byte, on different machines. Proven across 8.88M tokens on H100s.
That determinism is the foundation for verifying remote inference: if a third
party can reproduce your output exactly, they can audit you.

Top-level layout (you can also `tree -L 2 -I __pycache__` once locally):

```
cmd/         CLI entry points. Each subdir is a runnable script.
pkg/         Shared library code. Imported by cmd/ and tests/.
schemas/     JSON Schema definitions. Source of truth for wire formats.
manifests/   Sample model manifests.
tests/       unit/ integration/ e2e/ determinism/ fixtures/
experiments/ One subfolder per experiment. Read-only after the experiment ships.
docs/        ADRs, plans (this file lives here), conformance docs, diagrams.
```

### Key files to read (in order)

1. `CLAUDE.md` — top-level conventions and quick commands.
2. `cmd/server/main.py` — the existing deterministic-inference proxy server.
   Read all of it. The `BaseHTTPRequestHandler` pattern (`ProxyHandler`,
   ~line 526), `CaptureLog` (~line 482), `/run`, `/replay`, `/manifest`,
   and `ServerState` (~line 272) are all things you will extend or copy.
3. `cmd/coordinator/main.py` — multi-replica router. The threading,
   health-check, dispatch-log patterns you'll reuse for the **verifier
   server** are here. Same stdlib HTTP pattern; no FastAPI.
4. `cmd/verifier/main.py` — current "verifier" is *offline*: it diffs two
   run_bundle JSON files. Useful for the bundle-shape conventions, but not
   a server. You're building the server it lacks.
5. `pkg/manifest/model.py` — Pydantic models. **The manifest schema is
   generated from these.** When you add wire types, follow this pattern.
6. `pkg/common/deterministic.py` — `canonical_json_bytes`, `sha256_prefixed`,
   `utc_now_iso`. **Use these everywhere.** Don't roll your own JSON
   serialization.
7. `pkg/common/contracts.py` — `validate_with_schema(name, obj)` — the schema
   loader. Schemas are loaded from `schemas/` by filename.
8. `pkg/networkdet/__init__.py` plus `pkg/networkdet/warden.py` — the
   deterministic network stack and "warden" that normalizes frames. The
   prover already uses this in `/run` to produce a tap-style frame sequence.
   You'll plumb this stream out over HTTP.
9. `pkg/freivalds/__init__.py` (read the plan in `experiments/freivalds-attestation/plan.md`
   for context) — the matmul attestation primitive. You'll call into it for
   "proof of work" evidence on the prover side.
10. `experiments/memory_wipe/README.md` — PoSE-DB protocol. You'll re-use the
    challenge/response loop for "proof of secure erasure" evidence.
11. `experiments/task-graph-prototype/plan.md` and
    `experiments/task-graph-prototype/scripts/sim/orchestrator.py` —
    the data model for the attested task graph. **Your `/graph` endpoint
    starts as a placeholder, but the schema you choose should be a strict
    subset of this.** You're not implementing the task graph in this demo;
    you're leaving the slot open for it.
12. `tests/helpers.py` — `read_json`, `write_json`, `run_cmd`. Tiny but used everywhere.
13. `tests/e2e/test_audit_replay.py` and `tests/e2e/test_manifest_endpoint_live.py`
    — examples of integration tests against a running server. Your end-to-end
    tests will look very similar.
14. `tests/unit/test_freivalds_protocol.py` — example of a clean
    in-process protocol test (round-trip prover→verifier→verdict).
15. `experiments/task-graph-prototype/scripts/run_demo.py` — example of a
    small "honest + adversarial" demo runner. The shape of *your* demo
    runner will be similar.

### Tooling

- **Python**, no framework. Stdlib for HTTP. Pydantic for schemas.
  vLLM/torch only on GPU machines.
- **`uv`** for Python (CLAUDE.md is firm: never `pip`/`pipx`/`apt` for Python).
- **`unittest.TestCase`**, not pytest. (Pytest is *imported* in some places via
  the helper but tests are written as `unittest.TestCase` subclasses.)
- **`make`** targets exist (`make test-fast`, `make schema`, `make lint`); you'll
  use `make test-fast` and `make schema` constantly.
- **No CI mock or local emulator for GPUs.** If a test needs vLLM, mark it
  GPU-only and gate it on an env var like the existing patterns do.

#### New tooling for this experiment

Existing repo `make lint` is intentionally minimal (`py_compile` + merge-marker
check). We add stricter tooling **scoped to the new code only** (`pkg/proverdet/`,
`cmd/prover/`, `cmd/verifier_server/`, `cmd/verifier_cli/`) so we don't churn
the existing tree:

- **Ruff** for lint + format. Replaces flake8/black/isort/pyupgrade in one
  fast tool. Config in `experiments/prover-verifier-demo/ruff.toml`, scoped
  via `--config` and explicit paths in the make target. Don't add a global
  `pyproject.toml` ruff section.
- **Pyright** for type checking. The wire types are where bugs hide;
  pyright traces them through your code and catches mismatches Ruff can't.
  Pyright > mypy for new code (faster, better Pydantic support, what most
  editors run under the hood).
- **Hypothesis** for property-based tests on schemas + canonical JSON. One
  property test (`canonical_round_trip_is_a_fixed_point`) replaces a dozen
  hand-rolled positive/negative cases and finds edge cases you wouldn't
  write. Use sparingly — best fit is wire-format invariants, not business
  logic.

Set up in Task 0.3.

### Run the existing test suite once

Before writing anything, do this from the repo root and confirm green:

```bash
python3 -m unittest discover -s tests/unit -v
bash scripts/ci/schema_gate.sh
```

If those don't pass on `main`, stop and ask.

---

## Architecture

```
                          random scheduler
                                  │
   ┌──────────────────────┐       │       ┌──────────────────────────┐
   │     PROVER server    │       │       │     VERIFIER server      │
   │  (cmd/prover/...)    │       │       │  (cmd/verifier_server/…) │
   │                      │       v       │                          │
   │  GET  /graph     ────┼──────────────►│  schedules requests      │
   │  POST /replay    ────┼──────────────►│  appends to transcript   │
   │                      │               │  runs verdict engine     │
   │  POST /traffic ──────┼──────────────►│  /traffic ingest         │
   │  (chunked stream     │   (plaintext) │                          │
   │   of normalized      │               │                          │
   │   frames from        │               │                          │
   │   networkdet)        │               │                          │
   │                      │               │                          │
   │  workload runner ────┼─► vLLM        │  GET /verdict (from CLI) │
   │     • benign         │               │                          │
   │     • mixed-LoRA     │               │                          │
   │     • LoRA-loading   │               │                          │
   └──────────────────────┘               └──────────────────────────┘
```

What the verifier does at the end of a run:

1. Reads the **transcript** (every prover↔verifier message it logged).
2. Reads the **latest committed graph** (placeholder for now).
3. Reads the **traffic capture**.
4. Decides: was this inference, or training/exfiltration?

For this demo the verdict logic is intentionally simple — see Phase 8.

### What's in scope, what isn't

In:

- Two stdlib-HTTP Python servers, runnable on one box (two ports) or two boxes.
- Placeholder graph endpoint (empty but typed). Real task-graph integration is
  a follow-up.
- Replay endpoint with **stub-then-real** evidence: PoSE-style erasure proof
  + Freivalds-style PoW.
- Plaintext traffic streaming. TLS is deferred.
- Three workloads: benign inference, mixed inference + LoRA training (size knob),
  inference using externally-loaded LoRAs (bandwidth knob).
- Detection logic that converts transcript+traffic into a verdict.
- One evaluation plot per knob (FLOPs vs detection-rate, bytes vs detection-rate).

Out (mention briefly in the experiment write-up):

- Encryption/TLS. Add a `--security-mode` flag stub but only `plaintext` works.
- Multi-pod orchestration, full task-graph integration, ZK proofs.
- Real-world LoRA training (we simulate matmuls; we don't train a model that
  improves).

---

## Conventions

The engineer's compass. Re-read before each task.

### Files & directories

- **Experiment artefacts** live in `experiments/prover-verifier-demo/`:
  ```
  experiments/prover-verifier-demo/
    plan.md                  # this file's experiment-side mirror (short)
    EXPERIMENT_LOG.md        # append-only
    scripts/
      run_demo.sh
      run_eval.py
      workloads/
        benign.py
        mixed_lora.py
        lora_loading.py
    data/                    # generated transcripts, traffic captures
    reports/                 # memo, plots
    figures/
  ```
- **New CLI entry points** live in `cmd/prover/main.py` and
  `cmd/verifier_server/main.py`. (We do *not* extend `cmd/server/main.py` in
  place — the prover here is a thin orchestrator that *uses* the existing
  server's machinery. See task 2.1.)
- **Shared library code** goes in `pkg/proverdet/` (new package). Tests in
  `tests/unit/test_proverdet_*.py`. Don't dump library code into `cmd/`.
- **Schemas** go in `schemas/<name>.v1.schema.json`. Bump to v2 only on a
  breaking change.

### Canonical JSON

Every byte you put on the wire or hash uses canonical JSON. Use the helpers:

```python
from pkg.common.deterministic import canonical_json_bytes, canonical_json_text, sha256_prefixed
```

Never write `json.dumps(...)` for anything that gets hashed or compared.

### Digests

Always prefixed: `sha256:<hex>`. Use `sha256_prefixed(bytes)` to produce them.

### Time

Use `utc_now_iso()`. Never `datetime.now()` without a timezone.

### Schemas as source of truth

For every wire object, a JSON Schema in `schemas/` validates it. The Pydantic
model in `pkg/proverdet/wire.py` is the *runtime* type. Both must agree.
`tests/unit/test_schema_files.py` already enforces that schemas exist and
parse — read it, your new schemas will be picked up automatically.

### HTTP server pattern

Stdlib `BaseHTTPRequestHandler` + `ThreadingMixIn(HTTPServer)`. Class
attributes for shared state (see `ProxyHandler.server_state` in
`cmd/server/main.py`). One handler method per `do_GET`/`do_POST` dispatch
table, route by `self.path`. Don't add Flask/FastAPI.

### Errors

Return JSON like `{"error": "<message>"}` with the right HTTP status.
`400` invalid input, `404` not found, `409` conflict, `422` invalid schema,
`500` internal, `502` upstream unreachable. Don't invent statuses.

### Test design (read this carefully — your taste here is questionable)

- **TDD loop**: write a failing test, run it, *read the failure message*,
  implement, run again, refactor. If your test passes the first time, it's
  almost certainly testing nothing.
- **Test behaviour, not implementation.** A test that calls a private function
  and checks its return is brittle and weak. A test that posts JSON to your
  server and checks the response is durable.
- **Boundaries are the units worth testing.** For HTTP servers, that's the
  request/response. For library code, the function's public input/output.
- **Real over mocked, when cheap.** Spawning your prover on `127.0.0.1:0` (OS
  picks a free port) and hitting it with `urllib` is cheap and worth doing in
  every integration test. Don't mock `urlopen`.
- **Each test asserts one thing in essence.** "Posts to /graph returns the
  active graph" — fine. "Posts to /graph returns active graph AND validates
  schema AND records transcript entry AND increments seq" — split it.
- **Negative tests are mandatory.** Invalid input. Missing field. Wrong type.
  These catch most regressions; people forget to add them.
- **Property-based tests for wire-format invariants.** Use `hypothesis` for
  things that should hold over a *space* of inputs (canonical-JSON
  round-trip is a fixed point; schema validation accepts every model
  Pydantic produces). Don't reach for hypothesis on business logic — its
  shrinker is great for byte-shaped invariants, less so for "did the
  verdict come out right".
- **Fixtures live in `tests/fixtures/` or `tests/proverdet/fixtures/`.** Not
  inline in test files unless they're trivial.
- **Skip GPU-only tests using the existing helpers** (e.g. `_has_gpu()` /
  `_has_vllm()` patterns from `tests/e2e/test_server_lifecycle.py`). Don't
  invent a new env var like `HAVE_GPU`. Don't `try: import torch except:
  pass` in the test body.
- **Test names describe behaviour, not method names.** `test_post_graph_returns_empty_graph_when_no_workload_started` is good. `test_graph_endpoint` is not.

### Commits

- One conceptual change per commit.
- Subject ≤ 70 chars, imperative: `prover-verifier: add /graph endpoint stub`.
- Prefix subject with `prover-verifier:` so the experiment is greppable in
  history.
- Body: *why*, not *what*. The diff says what.
- Run `make test-fast` and `make schema` before each commit. If you broke
  something unrelated, revert and split.
- **Frequent commits.** Aim for one per task in this plan, sometimes more.
  A 700-line commit is almost always wrong.
- Per CLAUDE.md: **do not** add Claude as co-author.

### Things you will be tempted to do — don't

- Add FastAPI for "ergonomics". Stdlib HTTP is the house style; mixing is worse than ugly.
- Add a database. Append-only JSONL files have served well; see `CaptureLog`.
- Wrap stdlib `urlopen` in a "client class". Use it directly.
- Write a `BaseWorkload` ABC with three subclasses before you have the
  second workload working.
- Write a docstring for every function. The conventions doc says: comments
  explain *why*, only when non-obvious. Most of your code needs none.
- Pre-commit a half-built feature behind a flag "for later". Either ship it
  or don't merge.
- Skip schema validation "to debug faster". The schemas catch real bugs;
  bypassing them is how you ship `seq: "5"` when a consumer expects `seq: 5`.

---

## Phases at a glance

| Phase | Theme                                | Tasks | Wall time est. |
|-------|--------------------------------------|-------|----------------|
| 0     | Setup: experiment skeleton + primer + tooling | 3 | 1 day          |
| 1     | Wire schemas (graph, replay, transcript) | 4 | 1 day          |
| 2     | Prover server scaffold               | 4     | 1.5 days       |
| 3     | Verifier server scaffold             | 4     | 1.5 days       |
| 4     | Wire traffic streaming               | 3     | 1 day          |
| 5     | Benign workload                      | 3     | 1 day          |
| 6     | Replay evidence (PoSE + Freivalds)   | 4     | 2 days         |
| 7     | Adversarial workloads                | 3     | 1.5 days       |
| 8     | Detection logic                      | 3     | 1.5 days       |
| 9     | Evaluation harness + plots + viewer  | 4     | 2 days         |
| 10    | Polish & memo                        | 2     | 0.5 day        |

Total: ~14 days for a careful pass. If you're flying through faster, check
that you're actually writing the negative tests.

---

# Phase 0 — Setup

## Task 0.1 — Create experiment skeleton

**Why.** Every experiment in this repo lives under `experiments/<name>/` with
a fixed shape. Doing this first makes everything else fit naturally.

**Read first.**

- `CLAUDE.md` § "Experiment organization"
- `experiments/task-graph-prototype/plan.md` (just the layout)
- `experiments/freivalds-attestation/plan.md` (just the layout)

**Files to create.**

```
experiments/prover-verifier-demo/
  plan.md                     # ≤ 100 lines, points to docs/plans/prover-verifier-demo.md
  EXPERIMENT_LOG.md           # one line: "Started 2026-XX-XX. See docs/plans/prover-verifier-demo.md."
  scripts/.gitkeep
  data/.gitkeep
  reports/.gitkeep
  figures/.gitkeep
```

`plan.md` content: a short summary (goal, status, link to this doc, scope-in,
scope-out) — copy the structure from `experiments/freivalds-attestation/plan.md`.

**Tests first.** None. This is a directory shuffle.

**Commit.**

```
prover-verifier: scaffold experiment directory
```

**Definition of done.** `ls experiments/prover-verifier-demo/` shows the four
subdirs and the two `.md` files. The plan file links back to this document.

---

## Task 0.2 — Read-this-first checklist (no commit)

Before you write a single line of code:

- [ ] Read every file listed in [§ Read this first](#read-this-first-codebase-primer).
- [ ] Run `python3 -m unittest discover -s tests/unit -v`. It should be green.
- [ ] Run `bash scripts/ci/schema_gate.sh`. Green.
- [ ] Sketch on paper the call sequence for one full run: prover starts →
  verifier starts → verifier polls `/graph` → prover starts a workload →
  verifier challenges `/replay` → verifier ingests `/traffic` → verifier
  emits a verdict. Three boxes connected by arrows. Don't skip this.
- [ ] If you see stray `__pycache__/` directories without sibling source
  files (often left over from worktree switches), nuke them safely:
  `find . -name __pycache__ -exec rm -rf {} +`.
- [ ] Append to `EXPERIMENT_LOG.md`: any surprises during reading.

If anything in the checklist isn't true, fix it before continuing.

---

## Task 0.3 — Wire ruff + pyright + hypothesis (scoped to new code)

**Why.** The existing `make lint` is just `py_compile`. We want stricter
tooling for the new code without rewriting the rest of the repo. Scoping
keeps the diff small and the existing CI green.

**Files to create.**

- `experiments/prover-verifier-demo/ruff.toml` — minimal config:

  ```toml
  line-length = 100
  target-version = "py311"

  [lint]
  select = ["E", "F", "I", "B", "UP", "SIM", "RUF"]
  ignore = ["E501"]   # line length is enforced by formatter

  [format]
  quote-style = "double"
  ```

- `experiments/prover-verifier-demo/pyrightconfig.json` — minimal:

  ```json
  {
    "include": [
      "pkg/proverdet",
      "cmd/prover",
      "cmd/verifier_server",
      "cmd/verifier_cli"
    ],
    "pythonVersion": "3.11",
    "typeCheckingMode": "strict",
    "reportMissingImports": "error",
    "reportUnknownMemberType": "none",
    "reportUnknownArgumentType": "none"
  }
  ```

  (`reportUnknown*` off because Pydantic v1/v2 dynamic attrs are noisy in
  strict mode. Strict-but-pragmatic.)

**Files to touch.**

- `Makefile` — add three targets at the bottom (don't disturb existing ones):

  ```make
  .PHONY: lint-proverdet typecheck-proverdet test-proverdet
  lint-proverdet:
  	uv run ruff check --config experiments/prover-verifier-demo/ruff.toml \
  	    pkg/proverdet cmd/prover cmd/verifier_server cmd/verifier_cli \
  	    tests/unit/test_proverdet_*.py tests/integration/test_prover_*.py \
  	    tests/integration/test_verifier_*.py tests/e2e/test_prover_verifier_*.py
  	uv run ruff format --check --config experiments/prover-verifier-demo/ruff.toml \
  	    pkg/proverdet cmd/prover cmd/verifier_server cmd/verifier_cli

  typecheck-proverdet:
  	uv run pyright --project experiments/prover-verifier-demo/pyrightconfig.json

  test-proverdet:
  	python3 -m unittest discover -s tests/unit -p 'test_proverdet_*.py' -v
  ```

- `experiments/prover-verifier-demo/EXPERIMENT_LOG.md` — note that
  `make lint-proverdet` and `make typecheck-proverdet` should be green at
  every commit.

**Install.** Use `uv` (per CLAUDE.md, never `pip`):

```bash
uv tool install ruff
uv tool install pyright
uv pip install hypothesis        # in the project venv used by tests
```

(If the repo has a `pyproject.toml`/`uv.lock`, add `hypothesis` to dev
dependencies there instead of installing ad-hoc. Check before guessing.)

**Tests first.** None — this is tooling. But verify:

- `make lint-proverdet` exits 0 on an empty project (it'll have nothing to
  lint yet — that's fine, ruff returns 0).
- `make typecheck-proverdet` exits 0 on an empty include set.

**Commit.**
```
prover-verifier: add ruff + pyright + hypothesis (scoped tooling)
```

**DoD.** From a fresh shell, all three of `make lint-proverdet`,
`make typecheck-proverdet`, `make test-proverdet` exit 0. Existing
`make lint`, `make schema`, `make test-fast` still pass — you haven't
touched them.

**Engineer's habit from now on.** Before every commit:
```
make lint-proverdet && make typecheck-proverdet && make test-proverdet
```
If any fail, fix before committing. If you find yourself adding
`# type: ignore` or `# noqa` more than twice in a session, stop and ask —
it usually means a wire type is wrong, not the linter.

---

# Phase 1 — Wire schemas

The schemas pin every cross-process object. Define them before any server code,
or you'll repeatedly tear up the implementation.

## Task 1.1 — Placeholder Graph schema

**Why.** The verifier polls `/graph`. We don't have a real task graph yet, so
the schema admits an empty-but-typed response. Future work will add fields;
the *shape* should already be a subset of
`experiments/task-graph-prototype/schemas/attested_task_graph.v0.schema.json`
so we can grow into it.

**Read first.**

- `experiments/task-graph-prototype/schemas/attested_task_graph.v0.schema.json`
- `schemas/freivalds_attestation.v1.schema.json` (for shape conventions)
- `tests/unit/test_schema_files.py` (so you know what auto-coverage exists)

**Files to create.**

- `schemas/prover_graph.v1.schema.json`

**Schema contents.** Required keys: `graph_version` (const `"v1-placeholder"`),
`run_id` (string), `produced_at` (ISO-8601 UTC), `tasks` (array, may be empty),
`artifacts` (array, may be empty), `transmissions` (array, may be empty).
Each `task` has `task_id`, `pod_id`, `operation`, `claimed_flops` (≥ 0).
Each `artifact` has `artifact_id`, `commitment` (sha256-prefixed), `size_bytes`.
Each `transmission` has `transmission_id`, `sender_pod_id`, `receiver_pod_id`,
`artifact_id`, `tap_signature` (hex). `additionalProperties: false`.

**Tests first.** Add `tests/unit/test_proverdet_schemas.py`:

```python
class TestGraphSchema(unittest.TestCase):
    def test_minimal_graph_validates(self): ...
    def test_graph_rejects_unknown_field(self): ...
    def test_graph_requires_run_id(self): ...
    def test_artifact_commitment_must_be_sha256_prefixed(self): ...
    def test_task_claimed_flops_rejects_string(self):
        # e.g. {"claimed_flops": "100"} should fail — wrong type.
        ...
```

Use `validate_with_schema("prover_graph.v1.schema.json", obj)` and assert
`ValidationError` is raised on the negatives. Cover three flavours of
negative test for each schema: **missing required**, **wrong type**, and
**unknown field** — these catch ~all real bugs in this kind of schema.

**Commit.**
```
prover-verifier: add prover_graph.v1 schema (placeholder)
```

**DoD.** `make schema` and `make test-fast` both green. Negative tests fail
*before* you add the schema and pass *after*.

---

## Task 1.2 — ReplayRequest + ReplayEvidence schemas

**Why.** This is the protocol contract for the verifier challenging the
prover. Get it wrong here and you re-do every server.

**Read first.**

- `cmd/server/main.py` lines around `_handle_post_replay` (the existing
  per-token replay endpoint — note its `request_id`, `token_position`, `side`
  contract).
- `experiments/memory_wipe/src/pose/protocol.py` (PoSE challenge schema).
- `pkg/freivalds/spec.py` (Freivalds challenge schema).
- `schemas/freivalds_challenge.v1.schema.json`.

**Files to create.**

- `schemas/replay_request.v1.schema.json`
- `schemas/replay_evidence.v1.schema.json`

**ReplayRequest fields.**

- `replay_id` (string)
- `pod_id` (string) — which pod the verifier wants the replay on
- `target` — discriminated union:
  - `{ "kind": "task", "task_id": "..." }`
  - `{ "kind": "artifact", "artifact_id": "..." }`
- `erasure` (object): `{ "challenge_seed": "<hex>", "deadline_ms": int, "rounds": int }`
- `proof_of_work` (object): `{ "matmul_dim": int, "dtype": "bf16" | "fp16" | "int8", "rounds": int, "report_every_ms": int }`
- `auxiliary` (array of strings) — list of additional evidence kinds (extension point).

**ReplayEvidence fields.**

- `replay_id` (matches request)
- `produced_at` (ISO-8601)
- `output` — `{ "commitment": "sha256:...", "bytes_b64": "..." }` (bytes are
  base64; deferred can be made optional later)
- `erasure_evidence` — `{ "rounds": int, "passed": int, "log_path": "..." }`
- `pow_stream` — array of `{ "t_ms": int, "freivalds_attestation_id": "...",
  "matmul_dim": int, "rounds": int, "dtype": "..." }`. The `matmul_dim` and
  `rounds` are duplicated from the spec on the prover side so the verdict
  engine in Phase 8 can compute observed FLOPs (`2 * dim^3 * rounds`) from
  the transcript alone, without an extra fetch. The `freivalds_attestation_id`
  is a handle the verifier uses to *confirm* the attestation when it wants
  to (Task 6.4 fetches via `GET /attestation/{id}` — a route added in 6.1).
- `errors` (optional array of strings)

**Tests first.** Extend `tests/unit/test_proverdet_schemas.py` with one test
class per schema. Cover: minimal valid, missing required, wrong type for
each field. Don't test every typo — pick the ones that are easy to make.

**Commit.**
```
prover-verifier: add replay_request/replay_evidence v1 schemas
```

**DoD.** Schema gate passes; positive and negative tests both pass; both
schemas are listed in the auto-discovered set.

---

## Task 1.3 — TranscriptEntry schema + helpers

**Why.** The verifier's transcript is the audit trail. Pin its shape so the
verdict engine can read it without surprises.

**Read first.**

- `cmd/server/main.py` `class CaptureLog` — your TranscriptLog will have the
  same shape.
- `cmd/coordinator/main.py` `class DispatchLog`.

**Files to create.**

- `schemas/verifier_transcript_entry.v1.schema.json`

**Schema fields.**

- `seq` (int, monotonic)
- `direction` — `"sent"` | `"received"`
- `endpoint` — string (e.g. `/graph`, `/replay`, `/traffic`)
- `timestamp` (ISO-8601)
- `payload_digest` (`sha256:...`)
- `status_code` (int, optional)
- `payload_path` (string, relative to verifier out-dir, optional — full
  payloads stored on disk; the transcript holds the digest + path)

**Tests first.** Same pattern as Tasks 1.1/1.2. Special test:
`payload_digest` must be sha256-prefixed.

**Commit.**
```
prover-verifier: add verifier_transcript_entry v1 schema
```

---

## Task 1.4 — `pkg/proverdet/wire.py` Pydantic models

**Why.** Schemas are the wire contract; Pydantic models are the runtime types.
Both must agree. The patterns in `pkg/manifest/model.py` show how.

**Read first.**

- `pkg/manifest/model.py` (top 100 lines)
- `pkg/freivalds/spec.py`

**Files to create.**

- `pkg/proverdet/__init__.py` (empty)
- `pkg/proverdet/wire.py`
- `tests/unit/test_proverdet_wire.py`

**`wire.py` contents.** Pydantic models for `Graph`, `Task`, `Artifact`,
`Transmission`, `ReplayRequest`, `ErasureSpec`, `ProofOfWorkSpec`,
`ReplayEvidence`, `TranscriptEntry`. Only the *top-level* models that get
hashed or sent on the wire need a `.to_canonical()` method —
`Graph`, `ReplayRequest`, `ReplayEvidence`, `TranscriptEntry`. Inner-only
models like `ErasureSpec`/`Task`/`Artifact` are never canonicalized
standalone, so don't add the method there.

**Tests first.** For each model:

- `test_<Model>_round_trip_via_canonical_bytes`: build, serialize, deserialize,
  assert equal.
- `test_<Model>_validates_against_schema`: serialize, then
  `validate_with_schema("<file>", obj)` must not raise.
- One negative per model (e.g., `test_replay_request_rejects_bad_dtype`).

Plus **one property test per top-level wire model** using hypothesis:

```python
from hypothesis import given, strategies as st

@given(st.builds(Graph, run_id=st.text(min_size=1, max_size=64)))
def test_graph_canonical_roundtrip_is_fixed_point(graph: Graph) -> None:
    once = canonical_json_bytes(graph.model_dump(exclude_none=True))
    twice = canonical_json_bytes(
        Graph.model_validate_json(once).model_dump(exclude_none=True)
    )
    assert once == twice
```

You'll likely need a `proverdet_strategies.py` with custom hypothesis
strategies for sha256-prefixed strings, ISO-8601 timestamps, etc. Don't
over-engineer — `st.from_regex(r"^sha256:[0-9a-f]{64}$", fullmatch=True)`
is enough.

**Commit.**
```
prover-verifier: add pkg/proverdet/wire.py Pydantic models
```

**DoD.** `make test-fast` green. The schema and model can each catch the
same kind of bug (wrong type), so the round-trip test exercises both.

---

# Phase 2 — Prover server scaffold

We won't extend `cmd/server/main.py` directly. Instead, the prover here is a
*shell* server that owns the new endpoints — `GET /graph`, `POST /replay`,
`POST /workload/start`, `POST /workload/stop` (Phase 5), and a debug
`POST /debug/emit-frames`. Inference happens elsewhere (in a workload
thread; in Phase 5+ that workload may call into vLLM). Outbound traffic to
the verifier is initiated by the prover itself in Phase 4 (no inbound
"tap-out" endpoint — the prover is the publisher).

**Note on `/replay`.** The existing `cmd/server/main.py` also has a
`POST /replay` endpoint, but it lives on a separate process/port and has a
different request/response shape (per-token HMAC commitments). We deliberately
reuse the path on the new server since they will never run on the same port.
If you find yourself grepping for `_handle_post_replay`, expect two hits —
they are unrelated handlers.

## Task 2.1 — Prover server skeleton with `/health`

**Why.** Stand up the smallest possible HTTP server we can hit from a test.
Everything else is added on top.

**Read first.**

- `cmd/server/main.py` lines 480 → end (the handler + main loop).
- `cmd/coordinator/main.py` lines 160 → end.

**Files to create.**

- `cmd/prover/__init__.py` (empty)
- `cmd/prover/main.py`

**Implementation.**

- `class ProverState`: holds `run_id`, `out_dir`, `lock`. Mirror
  `ServerState`, drop the vLLM-specific fields.
- `class ProverHandler(BaseHTTPRequestHandler)`: only `do_GET` for `/health`
  returning `{"ok": true}`. `log_message` no-op (silence default logging).
- `class ThreadedHTTPServer(ThreadingMixIn, HTTPServer): daemon_threads = True`.
- `def main()`: argparse for `--host`, `--port`, `--port-file`, `--run-id`,
  `--out-dir`, `--debug-mode` (boolean, default false; gates the
  `/debug/emit-frames` endpoint added in Task 4.3). Bind; if `--port-file`
  is set write the actual bound port to it and fsync; serve forever;
  signal-handle SIGINT/SIGTERM cleanly.

**Tests first.** Add `tests/integration/test_prover_server_lifecycle.py`:

```python
class TestProverLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.port_file = Path(self.tmp.name) / "bound.port"
        self.proc = subprocess.Popen(
            [sys.executable, "cmd/prover/main.py",
             "--host", "127.0.0.1", "--port", "0",  # 0 = let OS pick
             "--port-file", str(self.port_file),
             "--run-id", "test-run", "--out-dir", self.tmp.name],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=str(REPO_ROOT))
        self.port = _read_bound_port(self.port_file)
    def tearDown(self): self.proc.terminate(); self.tmp.cleanup()

    def test_health_returns_ok(self):
        with urlopen(f"http://127.0.0.1:{self.port}/health") as r:
            self.assertEqual(r.status, 200)
            self.assertEqual(json.loads(r.read())["ok"], True)
```

You'll need a `_read_bound_port(path)` helper. **Use a port file, not stdout.**
After the server binds, it writes the integer port to `--port-file` and then
fsyncs. The helper polls the file with a short timeout (≤ 5 s) and reads it.
This is far less flaky than reading subprocess stdout (which can stall on
buffering). Keep this helper in `tests/proverdet/_helpers.py`.

**Files to also create.** `tests/proverdet/__init__.py` (empty) — without
this, `from tests.proverdet._helpers import ...` will fail to import.

**Commit.**
```
prover-verifier: add prover server skeleton with /health
```

**DoD.** Integration test passes. `cmd/prover/main.py --host 127.0.0.1 --port 8100`
serves `/health` returning `{"ok": true}` and shuts down cleanly on SIGTERM.

---

## Task 2.2 — `GET /graph` returning empty placeholder graph

**Why.** First real protocol endpoint. Verifier will poll this.

**Read first.**

- `experiments/task-graph-prototype/scripts/sim/orchestrator.py` `emit()` —
  the *full* graph shape we'll grow into (don't copy more than the keys you
  declared in the schema).

**Files to touch.**

- `cmd/prover/main.py` — add `do_GET` for `/graph`.
- `pkg/proverdet/graph_builder.py` (new) — pure function `build_empty_graph(run_id) -> Graph`.
- `tests/unit/test_proverdet_graph_builder.py` (new).

**Implementation.**

- `build_empty_graph` returns a `Graph` Pydantic model with `tasks=[]`,
  `artifacts=[]`, `transmissions=[]`, `produced_at=utc_now_iso()`,
  `graph_version="v1-placeholder"`, `run_id=run_id`.
- The handler validates against the schema before serialization. If invalid,
  500 (this is a programmer error).

**Tests first.**

- Unit: `test_build_empty_graph_validates_against_schema`, `test_run_id_is_preserved`.
- Integration: extend lifecycle test — `test_get_graph_returns_empty_placeholder`.
  Assert run_id matches what we passed on the CLI; assert shape matches schema.

**Commit.**
```
prover-verifier: add GET /graph empty placeholder
```

---

## Task 2.3 — `POST /replay` with stub evidence

**Why.** End-to-end integration is fastest if `/replay` returns a *valid*
ReplayEvidence object early, even before PoSE/Freivalds are wired in.

**Read first.**

- `cmd/server/main.py` `_handle_post_replay` — existing replay handler is
  per-token; we're adding a different replay flow, so keep them distinct.
  Yours lives at `/replay` on the new server (different process/port).

**Files to touch.**

- `cmd/prover/main.py` — add `do_POST` dispatch + `_handle_post_replay`.
- `pkg/proverdet/replay.py` (new) — `def stub_evidence(req: ReplayRequest) -> ReplayEvidence`.
- `tests/unit/test_proverdet_replay_stub.py`.
- `tests/integration/test_prover_replay_endpoint.py`.

**Stub semantics.** `output.commitment` = sha256 of `b"stub:" + replay_id`,
`output.bytes_b64` = base64 of `b"stub-output"`, `erasure_evidence`
`{ rounds: req.erasure.rounds, passed: req.erasure.rounds }`, `pow_stream`
empty. The whole thing is a placeholder we'll replace in Phase 6, but it
validates cleanly against the schema today.

**Tests first.**

- Unit: build a request, call `stub_evidence`, assert schema validates,
  assert evidence.replay_id matches.
- Integration: POST a minimal request, get 200 + valid ReplayEvidence.
  Negative: POST with `pod_id` missing → 400 with `error` field.

**Commit.**
```
prover-verifier: add POST /replay returning stub evidence
```

---

## Task 2.4 — Prover capture log

**Why.** Mirror the prover side of the verifier's transcript. Helpful for
debugging mismatches.

**Read first.**

- `cmd/server/main.py` `class CaptureLog`. Reuse the pattern.

**Files to touch.**

- `pkg/proverdet/capture.py` (new) — `class ProverCaptureLog` lifted from
  `cmd/server/main.py`'s `CaptureLog` pattern, but writing canonical JSON
  entries with `direction`, `endpoint`, `payload_digest`, `payload_path`.
  **Don't import the existing class** — the inner shape differs slightly,
  lift the *pattern* not the code. The deliberately-different name avoids
  two `CaptureLog` classes in the codebase.
- `cmd/prover/main.py` — instantiate one in `ProverState`, append on every
  request/response.
- `tests/unit/test_proverdet_capture.py`.

**Heads-up for Task 3.2.** The verifier's `TranscriptLog` will look almost
identical. When you reach 3.2 and notice this, factor a shared
`pkg/proverdet/_jsonl_log.py` and have both classes inherit. Doing the
factor in 3.2 (after seeing both concrete shapes) is cheaper than designing
the abstraction here.

**Tests first.**

- `test_appends_each_entry_with_monotonic_seq`
- `test_payload_digest_is_sha256_prefixed`
- `test_writes_canonical_json_lines` (open the file, parse, assert sorted keys)

**Commit.**
```
prover-verifier: add prover-side capture log
```

**DoD.** After running 3 requests against the server, `out-dir/capture.jsonl`
has 3 lines, each canonical JSON, `seq` increments, every `payload_digest`
parses as `sha256:<64-hex>`.

---

# Phase 3 — Verifier server scaffold

## Task 3.1 — Verifier server skeleton with `/traffic` ingest

**Why.** The traffic stream is the only endpoint the prover initiates; the
others are verifier→prover. Standing up `/traffic` first lets us feed the
traffic capture in Phase 4 immediately.

**Read first.**

- `cmd/coordinator/main.py` (handler pattern + threading).
- `pkg/networkdet/capture.py` `class CaptureRing`.

**Files to create.**

- `cmd/verifier_server/__init__.py`
- `cmd/verifier_server/main.py`
- `pkg/proverdet/transcript.py` — `class TranscriptLog` (analogous to
  `CaptureLog`, but typed by `TranscriptEntry`).
- `tests/integration/test_verifier_server_lifecycle.py`.

**Implementation notes.**

- `POST /traffic` accepts `Content-Type: application/octet-stream` chunked
  bodies. Persist incoming bytes verbatim to `out-dir/traffic-<seq>.bin`,
  emit a transcript entry. Don't try to parse the bytes here.
- `do_GET` on `/health` returns `{"ok": true}`.
- argparse: `--host`, `--port`, `--out-dir`, `--prover-base-url`. The base
  URL is unused this task; required next.

**Tests first.**

- Lifecycle (start/stop, /health).
- `test_post_traffic_persists_bytes_and_logs_entry`: stream 1 KiB, assert
  file exists with that size and a transcript line was appended.
- `test_post_traffic_records_received_direction`: transcript entry has
  `direction == "received"`, `endpoint == "/traffic"`.

**Commit.**
```
prover-verifier: add verifier server with /traffic ingest
```

---

## Task 3.2 — Factor shared JSONL log + finish transcript log

**Why.** By now `ProverCaptureLog` (Task 2.4) and `TranscriptLog` (Task 3.1
inline) are 90% the same code. Factor the common base, validate both
against schema on append.

**Files to touch.**

- `pkg/proverdet/_jsonl_log.py` (new) — `class JsonlLog` with monotonic seq,
  `append(entry: dict)`, threadsafe.
- `pkg/proverdet/capture.py` — `ProverCaptureLog` becomes a thin subclass.
- `pkg/proverdet/transcript.py` — `TranscriptLog` becomes a thin subclass
  (validates each entry against `verifier_transcript_entry.v1.schema.json`).
- `tests/unit/test_proverdet_transcript.py`.

**Tests.** Mirror the capture-log tests. Add:
- `test_records_payload_path_when_provided`
- `test_validates_against_schema_on_append` — append a malformed entry
  (e.g. `seq: "5"` as a string) and assert it raises before the file write.
- `test_threadsafe_under_concurrent_appends` (10 threads, 100 entries each;
  sequence numbers must be unique and 1..1000)

**Commit.**
```
prover-verifier: factor JsonlLog and add transcript log
```

---

## Task 3.3 — Verifier scheduler (sends `/graph` and `/replay`)

**Why.** This is the **active** verifier behaviour. Without this it can only
listen.

**Read first.**

- `cmd/coordinator/main.py` `class DeterministicRouter`.
- Python's `threading.Timer` and `random.Random` (use a seeded Random for
  reproducible verifier behaviour).

**Files to touch.**

- `pkg/proverdet/scheduler.py` (new) — `class VerifierScheduler` with a
  seeded RNG, configurable cadences (mean ms between `/graph`, mean ms
  between `/replay`), and a `start(prover_base_url, transcript)`/`stop()` API.
- `cmd/verifier_server/main.py` — instantiate the scheduler.
- `tests/unit/test_proverdet_scheduler.py`.

**Implementation notes.**

- Decisions about *which* task/artifact to challenge are deferred: in this
  task the scheduler picks `replay_id` randomly and uses dummy targets. The
  `target.kind == "task"` with `task_id == "stub-task-0"` is fine.
- Scheduler logs every sent request to the transcript (direction `"sent"`)
  and every response (direction `"received"`).
- A seeded RNG means the scheduler's behaviour is reproducible across runs
  with the same seed.

**Tests first.**

- Unit: `test_scheduler_emits_at_expected_cadence` — **inject a fake clock**
  (callable `now() -> float`) and a fake `sleep(s)` (records calls, advances
  the fake clock). Assert the scheduled call sequence, not wallclock. Real
  `threading.Timer` is for production; pulling on it in tests = flaky CI.
- Unit: `test_seed_makes_request_pattern_reproducible`. Run with seed 7
  twice, capture `replay_id` sequence, assert equal.
- Integration: extend the verifier lifecycle test — start verifier with a
  fake prover (a tiny stdlib server in the test that replies 200 with a valid
  graph + evidence); assert the transcript shows `≥ 1 sent /graph` and
  `≥ 1 sent /replay` after a small bounded number of scheduler ticks.

**Commit.**
```
prover-verifier: add verifier scheduler
```

---

## Task 3.4 — Verdict CLI (separate file)

**Why.** A way to run the verdict engine after a run. Skeleton now (returns
a constant `"unknown"`); fleshed out in Phase 8.

**Read first.**

- `cmd/verifier/main.py` — the offline diff CLI. You're adding a sibling.

**Files to create.**

- `cmd/verifier_cli/__init__.py`
- `cmd/verifier_cli/main.py` — argparse: `--transcript <path> --out <path>`,
  later in Phase 8 also `--traffic-digest <path>`. Standalone CLI; not
  mixed into the long-running server's main loop.
- `pkg/proverdet/verdict.py` (new) — `def emit_verdict(transcript_path) -> dict`
  returning `{"verdict": "unknown", "reasons": []}` for now. Phase 8 will
  extend the signature.
- `tests/unit/test_proverdet_verdict_stub.py`.

**Tests first.** Unit-test that an empty transcript yields `"unknown"`, that
a non-empty transcript still yields `"unknown"`, that the JSON written to
disk matches the function output. (Stub-level only.)

**Commit.**
```
prover-verifier: add verdict CLI (stub)
```

---

# Phase 4 — Wire traffic streaming

## Task 4.1 — Prover-side traffic publisher (batched POSTs)

**Why.** The prover's `DeterministicNetStack` already emits frames; we plumb
those bytes to the verifier. Single-machine demo — perf is not the goal,
robustness is.

**Read first.**

- `pkg/networkdet/__init__.py` and `pkg/networkdet/warden.py` — see
  `cmd/server/main.py`'s `_handle_post_run` for usage.

**Files to touch.**

- `pkg/proverdet/traffic_publisher.py` — class with
  `start(verifier_url)`/`publish(frame_bytes)`/`stop()` API. `publish`
  enqueues onto a thread-safe queue; an internal worker thread drains it
  by sending **batched** POSTs (concatenate up to N frames or M bytes,
  whichever first; `urlopen` with `data=...` and
  `Content-Type: application/octet-stream`). On `stop()`, drain remaining
  frames with one final POST.
- `tests/unit/test_proverdet_traffic_publisher.py`.

**Why batched POSTs and not chunked streaming.** Python stdlib chunked-body
uploads have rough edges across versions; the demo doesn't need the
single-stream property and the verifier sees the same concatenated bytes
either way. We keep this option open for future work; a comment in the
publisher should say "batched-POST is intentional; see plan §4.1".

**Tests first.**

- `test_publisher_buffers_and_flushes`: spin up a tiny in-test HTTP server
  on `127.0.0.1:0` that records incoming POST bodies. Publish 100 frames
  (1 KiB each), `stop()`. Assert the server received them in order and
  the concatenated bytes equal the published bytes.
- `test_publisher_does_not_drop_on_stop`: publish 5 frames, immediately
  call `stop()` (no sleep). Assert all 5 land at the verifier.

(Recovery from transient connection drop is deferred — out of scope for the
demo.)

**Commit.**
```
prover-verifier: add prover traffic publisher
```

---

## Task 4.2 — Verifier-side traffic ingest finalizer

**Why.** Right now `/traffic` writes one file per POST. We want one file
per *run* (concatenation), with a digest at the end.

**Files to touch.**

- `cmd/verifier_server/main.py` — change `/traffic` handler to append to
  `out-dir/traffic.bin` and update a running `sha256` (`hashlib.sha256()`).
- Add a `/traffic/finalize` endpoint (POST, idempotent) that flushes the
  hasher and writes a `traffic.digest` file. The transcript records the
  digest.
- `tests/integration/test_verifier_traffic_ingest.py`.

**Tests first.**

- Stream 3 chunks, then call `/traffic/finalize`. Assert `traffic.bin` size
  equals total bytes, `traffic.digest` content matches a manual sha256
  computation.
- Negative: posting traffic after finalize returns 409.
- `test_finalize_is_idempotent`: calling `/traffic/finalize` twice returns
  200 both times, with the same digest. (Pick this semantics, not 409 on
  double-finalize — simpler for the eval harness in Phase 9 to retry.)

**Commit.**
```
prover-verifier: finalize traffic stream into digest
```

---

## Task 4.3 — End-to-end traffic loop integration test

**Why.** Reduce blast radius: prove the publisher and ingest work *together*
before we layer workloads on top.

**Files to create.**

- `tests/e2e/test_prover_verifier_traffic_e2e.py`.

**Test flow.**

1. Start verifier on a free port.
2. Start prover on a free port with `--verifier-url http://127.0.0.1:<v>`.
3. Trigger a synthetic publish from the prover via a debug-only endpoint
   `POST /debug/emit-frames` (gated behind `--debug-mode true` CLI flag —
   not enabled in production runs).
4. Sleep 1 s, call `POST /traffic/finalize` on the verifier.
5. Assert verifier `traffic.digest` exists and matches expected (you control
   the synthetic frames in step 3).

**Commit.**
```
prover-verifier: e2e test for traffic streaming
```

**DoD.** Test green. You can `tail -f` the verifier's transcript and see
both the streamed POST and the finalize call.

---

# Phase 5 — Benign workload

## Task 5.1 — `WorkloadContext` (concrete, not abstract)

**Why.** Workloads need a way to publish frames and record tasks. Don't
declare a `Workload` Protocol or ABC yet — write the first concrete
workload (Task 5.2) and let the duck-typed signature emerge. This honours
the "don't build a `BaseWorkload` ABC" rule from § Conventions.

**Files to create.**

- `experiments/prover-verifier-demo/scripts/workloads/__init__.py`
- `experiments/prover-verifier-demo/scripts/workloads/context.py` — concrete
  `WorkloadContext` dataclass with `publish_frame(bytes)`,
  `record_task(task_record: dict)`, `stop_event: threading.Event`. No ABC.

**Tests first.** Unit-test the context: a fake publisher captures
`publish_frame` calls, a fake task list captures `record_task`. Assert that
calling these methods routes correctly.

**Commit.**
```
prover-verifier: add WorkloadContext
```

---

## Task 5.2 — Benign inference workload (CPU-friendly stub for tests; real vLLM gated)

**Why.** Two-tier: the *unit* version returns synthetic frames so tests run
on CPU; the *real* version calls vLLM and is GPU-only.

**Read first.**

- `cmd/server/main.py` `_handle_post_run` — how vLLM is called.
- `experiments/e2e-audit/scripts/smoke.manifest.json` — a small manifest you
  can reuse.

**Files to create.**

- `experiments/prover-verifier-demo/scripts/workloads/benign.py`
- `tests/unit/test_proverdet_workload_benign.py`

**Implementation.**

- `class BenignInferenceWorkload`: takes `(prompts: list[str], use_vllm: bool)`.
- If `use_vllm`: import vLLM lazily, call the model, get tokens.
- If not: emit `len(prompts) * 10` synthetic frames of 256 bytes each, deterministic.
- Each frame is published via `ctx.publish_frame`.
- Each prompt → one task record via `ctx.record_task`.

**Tests first.**

- CPU: instantiate with `use_vllm=False`, run with a fake context that
  records calls. Assert N frames and M tasks recorded.
- GPU: gate this test the way the existing repo does (see
  `tests/e2e/test_server_lifecycle.py` for the `_has_gpu()` / `_has_vllm()`
  helpers). Don't invent a new env var. **You'll need a GPU instance for
  this test** — see § Cloud GPU provisioning below for how to get one.
  The unit-side `use_vllm=False` test runs on your laptop; you only need
  cloud GPU to validate the real path.

**Commit.**
```
prover-verifier: add benign inference workload
```

---

## Task 5.3 — Wire `POST /workload/start` and `POST /workload/stop`

**Why.** Verifier doesn't know which workload is running — but the prover's
operator (or our demo runner) starts one. This is the prover's control plane.

**Files to touch.**

- `cmd/prover/main.py`:
  - `POST /workload/start` body: `{ "name": "benign", "params": {...} }`.
    Loads the right module, instantiates the workload, runs it in a daemon
    thread. 409 if a workload is already running.
  - `POST /workload/stop`: sets the stop event, joins the thread.
- `pkg/proverdet/workload_runner.py` — thin glue: name → class lookup,
  thread management. The lookup is a literal dict; don't use entry points.

**Tests first.**

- Integration: start prover with `--debug-mode true`, POST a benign workload
  start (`use_vllm=False`), wait, POST stop. Assert verifier transcript shows
  *some* `/traffic` POSTs in that window.
- Negative: starting a second workload while one is running → 409.

**Commit.**
```
prover-verifier: wire /workload/start and /stop
```

---

# Phase 6 — Replay evidence (PoSE + Freivalds)

This phase replaces the stubs from Tasks 2.3.

## Task 6.1 — Wire Freivalds challenge into `/replay` + attestation store

**Why.** Replace the empty `pow_stream` with real Freivalds attestations.
Add a `GET /attestation/{id}` route so the verifier can fetch full
attestation bodies (matrix `C`, dims, dtype) when verifying — the streamed
`pow_stream` only carries handles + summary FLOPs.

**Read first.**

- `pkg/freivalds/prover.py`, `pkg/freivalds/verifier.py`, `pkg/freivalds/spec.py`.
- `experiments/freivalds-attestation/scripts/run_smoke.py` (verify it exists
  before relying on its pattern; if not, read whichever smoke/test file does
  exist under that experiment).

**Files to touch.**

- `pkg/proverdet/replay.py` — replace `stub_evidence` with
  `produce_evidence(req, *, freivalds_backend, attestation_store)`. The
  store accepts `put(attestation_id, attestation_body)` and `get(id)`.
- `pkg/proverdet/attestation_store.py` (new) — in-memory dict, threadsafe.
  Persists nothing; lifetimes match the prover process.
- `cmd/prover/main.py` — instantiate the store; pass it to handlers. Add
  `GET /attestation/{id}` returning the stored attestation (404 if unknown).
  On CPU the stdlib backend works for small dims (use `dim=64`, `dtype=int8`
  to keep tests fast).
- `tests/unit/test_proverdet_replay.py`.

**Tests first.**

- Unit: produce evidence on the stdlib backend; verify each `pow_stream`
  entry has correct `matmul_dim`, `rounds`, `dtype` populated (needed by
  Phase 8's compute-budget signal). Then run the Freivalds *verifier* over
  the stored attestation. Assert verdict = pass.
- Negative: flip one byte in the stored attestation's `C` matrix; assert
  the Freivalds verifier returns False on that attestation.
- Integration: `GET /attestation/<known-id>` returns 200 with the body;
  unknown id returns 404.

**Commit.**
```
prover-verifier: replace stub /replay with Freivalds-backed evidence
```

---

## Task 6.2 — Wire PoSE-style erasure evidence

**Why.** Round out the replay envelope with the erasure half.

**Read first.**

- `experiments/memory_wipe/src/pose/protocol.py` and `verifier.py` —
  challenge/response loop for erasure.

**Files to touch.**

- `pkg/proverdet/erasure.py` — wraps PoSE's protocol into a small "run K
  rounds" function suitable for a `/replay` call. Don't import PoSE's full
  pipeline — its modules expect a real GPU node. Lift the *protocol logic*
  (HMAC over (seed, block_index)) and give it a stub backend that responds
  honestly. The honest path is enough for the demo; the dishonest path is
  exercised by an adversarial workload in Phase 7.
- `pkg/proverdet/replay.py` — call the new helper.
- `tests/unit/test_proverdet_erasure.py`.

**Tests first.**

- `test_honest_erasure_passes_all_rounds`
- `test_corrupted_response_fails_at_least_one_round`

**Commit.**
```
prover-verifier: wire PoSE-style erasure evidence into /replay
```

---

## Task 6.3 — Streaming PoW evidence (cadence)

**Why.** The verifier wants periodic proofs *during* a long replay, not just
at the end.

**⚠ Contract change.** This task **breaks** the request/response contract
established in Task 2.3: `/replay` becomes a chunked streaming response
where each line is a separate JSON object (NDJSON). Any test code from
Task 3.3 that consumes `/replay` as a single JSON body must be updated as
part of this commit. Don't merge in pieces.

**Files to touch.**

- `cmd/prover/main.py` `_handle_post_replay` — change handler from
  request/response to NDJSON chunked response. Each line is one of:
  - `{"kind": "pow", ...pow_stream entry fields}`
  - `{"kind": "evidence", ...full ReplayEvidence}` — emitted last.
- `cmd/verifier_server/main.py` — change scheduler's `/replay` consumer to
  read line-by-line. Append each `pow` line to the transcript as it
  arrives; record the final `evidence` line as the response transcript
  entry.
- `tests/integration/test_replay_streaming.py`.

**Tests first.**

- Integration: send a `/replay` with `report_every_ms=100`, `rounds=3`.
  Read the streamed response. Assert at least 3 `kind=pow` lines arrive
  *before* the `kind=evidence` line. Total wall time ≤ 1.5 ×
  `rounds * report_every_ms`.

**Commit.**
```
prover-verifier: stream PoW evidence with cadence (NDJSON)
```

---

## Task 6.4 — Verifier verdict on replay evidence

**Why.** Verifier validates the evidence as it arrives. Without this, the
replay is theatre.

**Files to touch.**

- `pkg/proverdet/replay_verify.py` (new) — `def verify_evidence(req, ev) -> VerdictResult`
  with fields `(passed: bool, reasons: list[str])`. Runs Freivalds verifier,
  confirms erasure rounds count, checks `pow_stream` cadence.
- `cmd/verifier_server/main.py` — call this on each received evidence,
  append a `verdict` transcript entry.
- `tests/unit/test_proverdet_replay_verify.py`.

**Tests first.**

- `test_honest_evidence_passes`: build evidence with the real prover code,
  call `verify_evidence`, assert `passed == True`.
- `test_tampered_output_bytes_fails`: take the honest evidence, flip one
  byte in `output.bytes_b64` (decode → flip → re-encode), call
  `verify_evidence`. Assert `passed == False` and the reason string mentions
  the Freivalds check.
- `test_missing_pow_stream_fails_cadence`: build evidence with `pow_stream
  = []` and `req.proof_of_work.report_every_ms = 100, rounds = 3`. Assert
  `passed == False` and the reason string mentions cadence.

**Commit.**
```
prover-verifier: verifier-side verdict on replay evidence
```

**DoD.** End-to-end run with no workload still produces a clean transcript:
verifier issues some `/graph` and `/replay`, every replay verdict is `pass`.

---

# Phase 7 — Adversarial workloads

We add two more workloads. Each has a knob for "how much of the workload
is training/exfil". The detection rate vs knob is the deliverable.

## Task 7.1 — Mixed inference + LoRA training workload

**Read first.**

- `pkg/freivalds/prover.py` (using torch matmuls).
- Decide your knob: `lora_rank` or `gradient_steps_per_inference`. Pick
  the simpler one (`gradient_steps_per_inference`).

**Files to create.**

- `experiments/prover-verifier-demo/scripts/workloads/mixed_lora.py`
- `tests/unit/test_proverdet_workload_mixed_lora.py`

**Implementation.**

- For each "inference" prompt the workload normally serves, also do
  `gradient_steps` extra matmuls of dim D (the "LoRA training") that are
  *not* recorded as tasks via `ctx.record_task`. This is the cheating
  behaviour: real GPU work that doesn't appear in the graph.
- Frames are still published — the cheating is in the graph accounting,
  not the network bytes.

**Tests first.**

- Unit (CPU): run with `inference_count=3, gradient_steps=2, dim=64`. Assert:
  - `len(recorded_tasks) == 3` (only the inferences are reported).
  - The fake context's published-frame count matches inference frames only
    (gradients leave no traffic — they're pure compute cheating).
  - **The Freivalds-attested observed FLOPs from the workload's matmuls
    exceeds `sum(task.claimed_flops for task in recorded_tasks)`** by
    roughly `gradient_steps * 2 * dim^3` per inference. This is the gap
    Phase 8 detects; assert it exists at workload level.
- Determinism: same seed → same matmuls (sanity check).

**Commit.**
```
prover-verifier: add mixed inference + LoRA-training workload
```

---

## Task 7.2 — LoRA-loading workload

**Files to create.**

- `experiments/prover-verifier-demo/scripts/workloads/lora_loading.py`
- `tests/unit/test_proverdet_workload_lora_loading.py`

**Implementation.**

- Workload "downloads" `lora_bytes` (knob, default 64 KiB) from a fake
  external URL (a local in-test server) before running inference.
- Downloaded bytes show up as **incoming** frames the prover does *not*
  account for in the graph. Verifier's job is to spot this from the traffic
  capture vs the declared graph.

**Tests first.** Same shape as 7.1. Add: `test_download_bytes_match_knob`.

**Commit.**
```
prover-verifier: add LoRA-loading workload
```

---

## Task 7.3 — Workload-level integration test (adversarial)

**Files to create.**

- `tests/e2e/test_prover_verifier_adversarial_e2e.py`.

**Test flow.**

1. Start verifier and prover.
2. Start `mixed_lora` workload with `gradient_steps=4`.
3. Wait 2 s, stop workload, finalize traffic.
4. Run verdict CLI.
5. Assert verdict file exists. Until Phase 8 lands the verdict is
   `"unknown"` — that's fine. After Phase 8 lands, **edit this test in the
   8.3 commit** to assert `"training_or_exfil"`.

Don't use `@unittest.expectedFailure` — its semantics are confusing and
people forget to remove it.

**Commit.**
```
prover-verifier: e2e adversarial workload test
```

---

# Phase 8 — Detection logic

Now we wire the verdict. Three signals; combine into one decision.

## Task 8.1 — Replay-correctness signal

**Why.** Was every challenged replay passed?

**Files to touch.**

- `pkg/proverdet/verdict.py` — replace stub.
- Add `def replay_correctness(transcript_entries) -> SignalResult` reading
  every entry where `endpoint == "/replay"` and `direction == "received"`.
  `passed = all(verdict.passed)`.
- `tests/unit/test_proverdet_verdict_signals.py`.

**Tests first.** Two cases: all-pass transcript → passed; one fail → failed
with reason naming the replay_id.

**Commit.**
```
prover-verifier: add replay-correctness signal
```

---

## Task 8.2 — Compute-budget signal (graph vs Freivalds-attested FLOPs)

**Why.** This is the LoRA-training detector. The graph claims X FLOPs of
work; Freivalds attestations sample real FLOPs. If sampled work exceeds
the graph budget plus tolerance, flag it.

**Implementation.**

- For every `/replay` evidence in the transcript, parse its `pow_stream`
  entries' `matmul_dim` and `rounds` (populated by Task 6.1). Observed FLOPs
  per entry = `2 * dim**3 * rounds`.
- For every graph received, sum `claimed_flops` across tasks.
- Predicate: `observed_flops > (1 + tolerance) * claimed_flops` → fail.
  Equality (`observed == (1 + tolerance) * claimed`) **passes**. Default
  `tolerance = 0.10` (10%).

**Files to touch.**

- `pkg/proverdet/verdict.py` — `def compute_budget(transcript_entries, *, tolerance) -> SignalResult`.
- `tests/unit/test_proverdet_verdict_signals.py`.

**Tests first.** Synthesize a transcript where observed > claimed → fail;
where observed ≤ claimed → pass. **Boundary tests:**
- `observed == (1 + tolerance) * claimed` → pass (the predicate uses `>`).
- `observed == (1 + tolerance) * claimed + 1` → fail.

**Commit.**
```
prover-verifier: add compute-budget signal
```

---

## Task 8.3 — Bandwidth signal + final verdict combiner

**Why.** Detects LoRA-loading. Combines all signals.

**Implementation.**

- `def bandwidth_signal(traffic_digest, traffic_size, claimed_artifact_bytes)`:
  if `traffic_size > (1 + tolerance) * claimed_artifact_bytes` → fail.
- Extend `emit_verdict` from Task 3.4. Old signature
  `emit_verdict(transcript_path) -> dict` becomes
  `emit_verdict(transcript_path, traffic_digest_path) -> dict`. Update the
  verdict CLI in `cmd/verifier_cli/main.py` to pass the new arg. Don't add a
  second function — replace the stub.
- Verdict is `"training_or_exfil"` if any signal failed, `"inference"`
  otherwise; reasons concat.

**Files to touch.**

- `pkg/proverdet/verdict.py`.
- `cmd/verifier_cli/main.py` (extend args).
- `tests/e2e/test_prover_verifier_adversarial_e2e.py` (Task 7.3) — sharpen
  to assert `"training_or_exfil"`.
- `tests/unit/test_proverdet_verdict_combiner.py`.

**Tests first.** Each adversarial scenario (mixed_lora, lora_loading) crafted
synthetically → expected verdict. Honest scenario → "inference".

**Commit.**
```
prover-verifier: add bandwidth signal and verdict combiner
```

**DoD.** The Task 7.3 e2e test (now updated in this commit to assert the
sharper verdict) is green. Adversarial workloads produce
`"training_or_exfil"`.

---

# Phase 9 — Evaluation harness + plots

## Task 9.1 — Sweep harness

**Why.** Generate the data we plot.

**Files to create.**

- `experiments/prover-verifier-demo/scripts/run_eval.py`.
- `experiments/prover-verifier-demo/data/eval/` (output dir).

**Behaviour.**

- For each workload (`benign`, `mixed_lora`, `lora_loading`):
  - For each value of the workload's knob (e.g.
    `gradient_steps in [0, 1, 2, 4, 8, 16]`):
    - Spawn a prover + verifier on free ports.
    - Start workload, run 5 s, stop.
    - Run verdict, record `{workload, knob_value, verdict, signals,
      observed_flops, traffic_size, runtime}`.
- Write `eval/results.jsonl`.

Sweep is **deterministic** — same seed for the workload and the verifier
scheduler. Re-running should produce byte-identical results given the same
hardware and clock-tolerant signals.

**Tests.** None on the harness itself (it's a script). Add one smoke
integration `tests/e2e/test_eval_harness_smoke.py` that runs `run_eval.py
--smoke` (one knob value per workload) and asserts:
- The output JSONL has 3 rows (one per workload).
- Every row has the required fields `workload`, `knob_value`, `verdict`,
  `signals`, `observed_flops`, `traffic_size`.
- Every row's `verdict` is one of `{"inference", "training_or_exfil",
  "unknown"}`.

**Commit.**
```
prover-verifier: add eval sweep harness
```

---

## Task 9.2 — Plot generation

**Files to create.**

- `experiments/prover-verifier-demo/scripts/plot_results.py`.

**Behaviour.**

- Read `data/eval/results.jsonl`.
- Plot `detection_rate vs gradient_steps` (mixed_lora).
- Plot `detection_rate vs lora_bytes` (lora_loading).
- Save to `figures/` as `.png` only.

Use matplotlib. Don't open windows (`plt.savefig` only).

**Tests.** A unit test that the plotting script runs on a synthetic results
file without error and writes the expected `.png` files. Don't pixel-match
plots; that's brittle.

**Commit.**
```
prover-verifier: add plot generation
```

---

## Task 9.3 — Self-contained HTML viewer

**Why.** This repo has a precedent for one-file HTML viewers
(`experiments/task-graph-prototype/viewer.html`) that open via `file://`,
have no JS dependencies, and embed the run data inline. Reviewers love
them; they survive forever without a server. Build one for the eval results.

**Read first.**

- `experiments/task-graph-prototype/viewer.html` — pattern to follow.
- `experiments/task-graph-prototype/scripts/make_viewer.py` — the generator
  that embeds JSONL into the HTML at build time.

**Files to create.**

- `experiments/prover-verifier-demo/scripts/make_viewer.py` — reads
  `data/eval/results.jsonl`, embeds the rows as a JSON literal in a
  `<script>` block, writes a single `viewer.html`.
- `experiments/prover-verifier-demo/viewer.html` — the generated artefact
  (commit it; it's the deliverable).

**What the viewer shows.**

- Scenario picker (one entry per row in `results.jsonl`).
- For the selected scenario: verdict (big, color-coded), per-signal status
  (replay correctness, compute budget, bandwidth), the knob value, the
  observed FLOPs / traffic size with bars showing claimed vs observed.
- An inline SVG of the FLOPs-vs-detection-rate curve (recompute from the
  data; don't embed the matplotlib PNGs — keep it pure HTML/SVG).

**Constraints.**

- One file. No CDN, no fetch, no JS frameworks. Vanilla JS only.
- Works from `file://` (test it: open the file directly in a browser, not
  via a local server).
- Under 50 KB total. If it's getting bigger, you're putting too much in.

**Tests.**

- A unit test that the generator runs on a synthetic `results.jsonl` and
  produces a single self-contained file.
- A smoke test using `playwright` headless chromium (the task-graph-prototype
  experiment uses this pattern — copy its setup) that opens the file,
  asserts no JS errors, asserts clicking each scenario updates the verdict
  panel. If `playwright` isn't already in the dev dependencies, skip the
  smoke test rather than adding it; the unit test is enough.

**Commit.**
```
prover-verifier: add self-contained HTML viewer
```

---

## Task 9.4 — Memo

**Files to create.**

- `experiments/prover-verifier-demo/reports/memo.md`.

**Contents** (≤ 4 pages):

- Problem (3 sentences).
- Architecture — embed a **Mermaid sequence diagram** showing one honest
  run and one cheated run side-by-side. Example:

  ```mermaid
  sequenceDiagram
      participant V as Verifier
      participant P as Prover
      P->>V: POST /traffic (workload bytes, streamed)
      V->>P: GET /graph
      P-->>V: Graph { tasks, claimed_flops }
      V->>P: POST /replay { rounds, report_every_ms }
      P-->>V: NDJSON { kind: pow, dim, rounds } x N
      P-->>V: NDJSON { kind: evidence }
      V->>V: verdict = compute_budget + bandwidth + replay_correctness
  ```

  GitHub renders Mermaid inline. Two diagrams, one per scenario.
- Threat model (what we detect, what we don't).
- Results — embed the two PNGs and a link to the HTML viewer.
- Failure modes / limitations (no TLS, placeholder graph, single-pod, etc.).
- Next steps (full task graph integration; cross-machine; encryption).

**Commit.**
```
prover-verifier: write memo
```

---

# Phase 10 — Polish

## Task 10.1 — `demo.sh` (the headline deliverable)

**Why.** This is the proof. A reviewer clones the repo, runs one script, and
watches the protocol play out: two real server processes come up, exchange
real messages, run real workloads, and emit verdicts that distinguish honest
from cheating provers. The eval sweep (`run_eval.py`) and the plots are
*supplementary* — for a reviewer who wants to see one curve. `demo.sh`
is what they see if they have 90 seconds.

This is **not** a wrapper around `run_eval.py --smoke`. It's its own
top-level driver that spawns the servers, prints what's happening, and
verifies the verdicts match expectations.

**Files to create.**

- `experiments/prover-verifier-demo/demo.sh` — the entry point.
- `experiments/prover-verifier-demo/scripts/demo_driver.py` — the Python
  half (orchestrates two servers + three scenarios). The shell script is
  thin: arg parsing, venv prep, exec into Python.

**Architecture of the demo run.**

```
demo.sh
   │
   ├── prepare venv (uv sync)
   ├── exec demo_driver.py
              │
              ├── spawn cmd/verifier_server/main.py on $VERIFIER_PORT
              ├── spawn cmd/prover/main.py on $PROVER_PORT
              │     (--verifier-url http://$HOST:$VERIFIER_PORT, --debug-mode true)
              ├── wait for both /health to return 200 (timeout 30 s)
              │
              ├── print banner: "Prover @ host:port  Verifier @ host:port"
              │
              ├── for scenario in [benign, mixed_lora, lora_loading]:
              │     ├── print scenario header
              │     ├── POST /workload/start to the prover
              │     ├── tail the verifier transcript live for 5 s,
              │     │   printing each /graph, /replay, /traffic event with
              │     │   one-line summaries
              │     ├── POST /workload/stop
              │     ├── POST /traffic/finalize on verifier
              │     ├── run cmd/verifier_cli verdict, capture result
              │     └── print "  → verdict: <verdict>  (signals: ...)"
              │
              ├── print summary table comparing actual vs expected verdict
              ├── tear down both servers (SIGTERM, then SIGKILL after 5 s)
              └── exit 0 iff all three verdicts match expectations
```

**Configurable hosts (for the two-real-machines case).**

- Default: both servers on `127.0.0.1`, ports auto-assigned via port-files.
- Env-var override: `PROVER_HOST=10.0.0.1 PROVER_PORT=8000
  VERIFIER_HOST=10.0.0.2 VERIFIER_PORT=9000 ./demo.sh --remote`. In
  `--remote` mode, `demo_driver.py` does *not* spawn the servers — it
  assumes they're already running on the named hosts (one process per
  machine, started by the user via SSH). Everything else is identical.

This is the "two nodes on one box" → "two nodes on two boxes" swap. The
**§ Cloud GPU provisioning** section below has the full provisioning,
deployment, and networking recipe for Lambda, vast.ai, and Digital Ocean.
The memo's "Reproducing on two machines" section can summarize and link
back; don't duplicate the recipe.

**Output (sample, what a green run looks like).**

```
=== Prover ↔ Verifier demo ===
Prover @ 127.0.0.1:51201
Verifier @ 127.0.0.1:51202
Both healthy.

--- Scenario 1: benign inference ---
[t=0.1s] verifier → prover: GET /graph
[t=0.1s] prover → verifier: 200 (Graph: 0 tasks, run_id=demo-001)
[t=0.4s] prover → verifier: POST /traffic (1024 bytes)
[t=0.7s] verifier → prover: POST /replay (replay_id=r1, dim=64, rounds=3)
[t=0.8s] prover → verifier: NDJSON pow t=100ms ...
[t=0.9s] prover → verifier: NDJSON pow t=200ms ...
[t=1.0s] prover → verifier: NDJSON evidence (verdict=pass)
...
[t=5.0s] stopping workload
  → verdict: inference  (replay_correctness=pass, compute_budget=pass, bandwidth=pass)

--- Scenario 2: mixed inference + LoRA training ---
... (same shape) ...
  → verdict: training_or_exfil  (compute_budget=fail: observed 4.2× claimed)

--- Scenario 3: LoRA loading ---
... (same shape) ...
  → verdict: training_or_exfil  (bandwidth=fail: traffic 8.7× claimed)

=== Summary ===
  benign        expected=inference          actual=inference          ✓
  mixed_lora    expected=training_or_exfil  actual=training_or_exfil  ✓
  lora_loading  expected=training_or_exfil  actual=training_or_exfil  ✓

ALL PASS
```

**Implementation notes.**

- The "live tailing" of the verifier transcript: `demo_driver.py` opens the
  transcript JSONL file in tail mode (`open(path); seek(0, 2)`) and reads
  new lines every 100 ms. For each line, format it to a one-liner.
- Don't try to capture stderr from the spawned servers in the foreground —
  redirect to `/tmp/<run_id>.{prover,verifier}.log` and print the path
  on shutdown so the reviewer can `cat` them if something fails.
- Server stdout/stderr **must** be suppressed from the demo's terminal —
  the chatter of the underlying servers will drown out the demo narration.
  Servers log to files; demo prints to stdout.
- Each scenario runs in 5 s of wallclock by default. `--quick` flag drops
  to 2 s for development; `--long` extends to 15 s for a more thorough
  view. Default is the right one for a reviewer.
- Use the same workload modules the eval harness uses
  (`experiments/prover-verifier-demo/scripts/workloads/`), with fixed
  parameters chosen to make detection unambiguous — `gradient_steps=8` for
  mixed_lora, `lora_bytes=512_000` for lora_loading. Tweak only if the
  default doesn't reliably trip detection across runs.
- The script's exit code is **the** test signal. CI can run `./demo.sh
  --quick` and check exit 0. Don't add a separate test that re-implements
  the same checks.

**Tests.**

- `tests/e2e/test_demo_sh.py` — invokes `./demo.sh --quick` via
  subprocess. Asserts exit code 0, asserts stdout contains "ALL PASS",
  asserts stdout contains the three verdict lines. Skip with
  `@unittest.skipUnless(os.getenv("RUN_DEMO_SH_TEST"))` because it's slow
  (~15 s) — wire it into a nightly CI lane, not test-fast.
- Manual: run `./demo.sh` once locally, confirm the output matches the
  shape above.

**Commit.**
```
prover-verifier: add demo.sh — spins up two servers, runs three scenarios
```

**DoD.** A fresh clone followed by `cd
experiments/prover-verifier-demo && ./demo.sh` exits 0 and prints
ALL PASS. The script never leaves a zombie server process behind, even
on Ctrl-C.

---

## Task 10.2 — Top-level pointers

**Files to touch.**

- `README.md` (top of repo) — single line under "Demos" pointing to this
  experiment's memo.
- `docs/plans/prover-verifier-demo.md` (this file) — flip status to "shipped".

**Commit.**
```
prover-verifier: link demo from top-level README and mark plan shipped
```

---

# Cloud GPU provisioning

Most of this plan is CPU-runnable. You only need GPUs for a small set of
specific tasks. Read this before deciding whether to provision anything.

## When you actually need a GPU

| Task | GPU needed? | Why |
|------|-------------|-----|
| Phases 0–4 (schemas, server scaffolds, traffic streaming) | No | All stdlib + Pydantic. |
| Task 5.2 (benign workload, `use_vllm=False`) | No | Synthetic frames, runs anywhere. |
| Task 5.2 (benign workload, real vLLM path) | **Yes (1)** | Actual inference. Gated by `_has_gpu()` helper. |
| Phase 6 (replay evidence) — stdlib Freivalds backend | No | `dim=64, dtype=int8` runs on CPU in milliseconds. |
| Phase 6 — torch backend Freivalds | **Yes (1)** | Only if you want to demonstrate large-dim attestations. |
| Phase 7 (adversarial workloads) | No for unit tests | The "training" matmuls run small-dim on CPU. |
| Task 10.1 (`demo.sh`) — local mode | No | Two ports on one box; synthetic everything. |
| Task 10.1 (`demo.sh --remote`, two-machine) | **Yes (1 or 2)** | Real cross-host networking; recommended path is two GPU boxes. |

**Bottom line:** ~95% of the plan ships from a laptop. The two GPU moments
are: validating that the real-vLLM workload path works (single GPU box, one
afternoon), and the headline two-machine demo (two GPU boxes, one
afternoon). Don't burn money provisioning early.

## Picking a cloud

Three options, in increasing operational complexity.

| Cloud | Cost (rough) | Setup time | Notes |
|-------|--------------|------------|-------|
| **Lambda Cloud** | $1.85–3.50/h (H100 PCIe → SXM5) | ~2 min | Simplest. Already used by repo's `experiments/memory_wipe`. Default choice. |
| **Digital Ocean GPU droplets** | $2–5/h (H100/L40S) | ~5 min | Good if you already have a DO account; standard Ubuntu image. |
| **vast.ai** | $0.40–2.00/h (highly variable) | ~15 min | Cheapest, but our Nix container needs a custom launch path. Worth it for long runs, painful for quick experiments. |

For each, the typical demo shape is **one or two H100/H100-PCIe/GH200
instances** in the same region (low cross-host latency keeps the timing
signals clean). Don't mix instance types between prover and verifier —
even if both are "H100", driver/CUDA differences will produce noisy
Freivalds calibration.

## Lambda Cloud (recommended default)

Credentials and boilerplate commands are in your global `~/.claude/CLAUDE.md`
under "## Lambda Cloud". Use those — don't re-derive.

For this demo specifically:

```bash
# 1. Launch one instance (the prover) — see CLAUDE.md for the curl recipe.
#    GH200 (gpu_1x_gh200) is fine; H100-PCIe is cheaper and works.
#    Example name: "prover-verifier-demo-prover-001"

# 2. If running --remote two-machine demo: launch a second.
#    Same region, same instance type. Name: "...-verifier-001"

# 3. Wait for both to show "active" — usually 60–90 s.
curl -s -u "$LAMBDALABS_API_KEY:" \
  https://cloud.lambdalabs.com/api/v1/instances \
  | python3 -c "import sys,json; \
[print(i['name'], i['ip'], i['status']) for i in json.load(sys.stdin)['data']]"
```

Capacity sells out fast — keep a polling loop running if your preferred
type isn't available. The `experiments/memory_wipe/scripts/provision_gh200.sh`
script in this repo is a usable template.

**Termination is your responsibility.** Lambda does not auto-stop. The most
expensive bug in this experiment is forgetting to terminate. Stick a
calendar reminder.

## vast.ai

Full launch recipe (image build, `--args` mode, entrypoint hash extraction,
port mapping) is in your global `~/.claude/CLAUDE.md` under "## vast.ai".
Do not invent your own launch invocation — the `--args` vs `--ssh` distinction
matters and `--ssh` will silently crash-loop our Nix-only image.

Demo-specific notes:

- Use **H100 offers only**. The repo's torch is compiled SM_90 only;
  RTX 4090 (SM_89) boots but torch refuses the device.
- After SSH-ing in, run the `libcuda` symlink fixup loop from the
  CLAUDE.md vast section. Without it, `import torch` fails with a missing
  shared library. Re-run after every SSH session.
- For the two-machine demo, you need to make the verifier reachable from
  the prover. vast.ai port-maps `-p` flags from the `--env` field; see the
  CLAUDE.md recipe. Map the verifier's listening port (default 9000) on
  the verifier instance. The prover dials `http://<verifier-public-ip>:<mapped-port>`.

## Digital Ocean GPU droplets

DO offers GPU droplets via their standard `doctl` CLI and dashboard. This
repo doesn't have a recipe for DO yet — you're writing the first one.

Sketch (verify against current DO docs at
<https://docs.digitalocean.com/products/droplets/how-to/gpu/> before
running):

```bash
# Authenticate doctl (one-time)
doctl auth init

# List available GPU sizes
doctl compute size list | grep -i gpu

# Create a droplet (example — confirm the size slug from the list above)
doctl compute droplet create prover-001 \
  --image ubuntu-22-04-x64 \
  --size <gpu-size-slug> \
  --region <region-with-gpu-stock> \
  --ssh-keys <your-ssh-key-fingerprint> \
  --wait

# Get IP
doctl compute droplet list | grep prover-001
```

Differences from Lambda/vast worth knowing in advance:

- DO ships a stock Ubuntu image. **You install CUDA/drivers yourself**,
  unlike Lambda's pre-baked images. Easiest path: install NVIDIA's
  CUDA-on-Ubuntu meta-package, reboot, verify with `nvidia-smi`. Plan
  +30 min for first-time setup per box.
- DO's firewall defaults to "open" on the public interface for new
  droplets, which is convenient for the demo but is not what you want
  long-term. Tighten with `doctl compute firewall create` after the demo.
- DO charges by the hour with partial-hour rounding; terminate by
  destroying the droplet, not stopping (stopped droplets still bill for
  storage).

If you actually do this, **add a working recipe to `~/.claude/CLAUDE.md`**
under "## Digital Ocean" so the next person doesn't reinvent it. Treat
that addition as a separate commit outside this experiment.

## Code deployment

Same pattern across all three clouds:

```bash
# From your laptop, in repo root:
TARGET_IP=<instance-public-ip>
TARGET_USER=ubuntu   # Lambda; root on vast; root or whatever on DO

rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '.venv' \
    --exclude 'experiments/*/data' --exclude 'experiments/*/reports' \
    ./ $TARGET_USER@$TARGET_IP:~/deterministic_serving_stack/

# On the instance:
ssh $TARGET_USER@$TARGET_IP
cd ~/deterministic_serving_stack
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync   # or `uv pip install -e .` per existing repo pattern
```

For two-node, do this on **both** instances. Both must run the same git
commit — script the rsync + checkout to keep them in lockstep.

## Two-node networking

The verifier listens; the prover dials it (and vice versa for `/graph`,
`/replay`). Two things must be true:

1. **Both servers bind to `0.0.0.0`**, not `127.0.0.1`. The Task 2.1 and
   3.1 argparse already exposes `--host` — pass `0.0.0.0` for remote
   demos. The `127.0.0.1` default is a safety guard for local-mode.
2. **The cloud's firewall lets each instance reach the other on the
   server ports.** Default ports: prover 8000, verifier 9000.
   - Lambda: instances are publicly accessible on all TCP ports by
     default. No firewall config needed for the demo.
   - vast.ai: ports must be explicitly mapped via `--env "-p ..."`. See
     CLAUDE.md.
   - DO: open firewall rules on the relevant ports, scoped to the other
     droplet's IP if you want to be tidy, or 0.0.0.0/0 for the demo.

Test connectivity *before* trying to run the demo:

```bash
# From the prover instance
curl -s http://<verifier-ip>:9000/health   # expect {"ok": true}
# From the verifier instance
curl -s http://<prover-ip>:8000/health     # expect {"ok": true}
```

If those fail, the demo will fail in confusing ways. Fix networking first.

## Running the demo on two real machines

```bash
# On the verifier machine (call this VERIFIER_IP):
cd ~/deterministic_serving_stack
uv run python3 cmd/verifier_server/main.py \
    --host 0.0.0.0 --port 9000 \
    --out-dir /tmp/verifier-demo \
    --prover-base-url http://<PROVER_IP>:8000

# On the prover machine (call this PROVER_IP):
cd ~/deterministic_serving_stack
uv run python3 cmd/prover/main.py \
    --host 0.0.0.0 --port 8000 \
    --run-id remote-demo-001 --out-dir /tmp/prover-demo \
    --debug-mode true \
    --verifier-url http://<VERIFIER_IP>:9000

# From your laptop (orchestration):
PROVER_HOST=<PROVER_IP> PROVER_PORT=8000 \
VERIFIER_HOST=<VERIFIER_IP> VERIFIER_PORT=9000 \
./experiments/prover-verifier-demo/demo.sh --remote
```

The `--remote` flag tells `demo_driver.py` not to spawn the servers
itself — they're already running on the named hosts. Everything else
(scenario sequencing, transcript tailing, verdict checking) is identical
to local mode.

## Cleanup discipline

After every demo session:

```bash
# Lambda
curl -s -u "$LAMBDALABS_API_KEY:" -X POST \
  https://cloud.lambdalabs.com/api/v1/instance-operations/terminate \
  -H 'Content-Type: application/json' \
  -d "{\"instance_ids\":[\"<id>\"]}"

# vast.ai
vastai destroy instance <id>

# Digital Ocean
doctl compute droplet delete <name> --force
```

Add a line to `EXPERIMENT_LOG.md` for every instance you launch:
`launched <name> <id> at <timestamp>`. Add a matching `terminated` line
when you tear down. If those don't pair up at the end of a session, you
have a billing leak.

---

# Risks and known unknowns

These are notes the engineer should escalate, not silently work around:

- **Streaming POST in stdlib.** Python's `urlopen` streaming has rough
  edges. If Task 4.1's "raw socket" fallback doesn't work cleanly across
  the Python versions in CI, switch the prover's traffic publisher to
  open-and-write-many-small-POSTs and accept the overhead.
- **GPU-only Freivalds in CI.** The torch backend won't run in CI. Use the
  stdlib backend for everything in tests; gate any torch-backed test using
  the repo's existing GPU helpers (see Conventions § Test design).
- **Seeded reproducibility** depends on every randomness source being
  controlled — workloads, scheduler, and matmul inputs. If a test is flaky,
  audit for an unseeded `random.Random()` first.
- **Placeholder graph schema** is intentionally permissive. When the real
  task graph integration lands later, existing data will not validate
  against the new schema; that's expected. Keep the version number bump in
  mind.
- **Cross-machine demo.** Everything in this plan runs on one box (two
  ports). If you also want to run prover and verifier on two real boxes,
  pass `--host 0.0.0.0` and use `vast.ai` or `Lambda` per the setup in
  `~/.claude/CLAUDE.md`. Not in scope; mention it in the memo.

---

# What "done" looks like

**The headline deliverable** is `experiments/prover-verifier-demo/demo.sh`.
On a fresh clone, running it spawns two real server processes, exchanges
real messages over HTTP, runs three scenarios (benign + two adversarial),
prints a live narration of every verifier↔prover request, and ends with
`ALL PASS` plus a summary table showing the three verdicts match
expectations. Exit code 0 iff all three match.

Everything else exists to support that script — the unit tests prove the
pieces, the e2e tests prove combinations, the eval sweep proves the
detection rates over a parameter range, the viewer + memo make the result
legible to a reviewer who isn't running the script.

**Full checklist:**

- `experiments/prover-verifier-demo/demo.sh` exits 0, prints `ALL PASS`,
  prints the three "expected vs actual" lines.
- All tests in `tests/unit/test_proverdet_*` and
  `tests/integration/test_prover_*`, `tests/integration/test_verifier_*`,
  `tests/e2e/test_prover_verifier_*` are green.
- `make test-fast` and `make schema` are green.
- `make lint-proverdet`, `make typecheck-proverdet`, `make test-proverdet`
  all green.
- `experiments/prover-verifier-demo/figures/` has two PNGs that look
  like the paper's "FLOPs vs detection rate" / "bytes vs detection rate"
  curves.
- `experiments/prover-verifier-demo/viewer.html` opens via `file://`,
  shows the verdict for each scenario, has zero JS errors.
- `experiments/prover-verifier-demo/reports/memo.md` exists, with two
  Mermaid sequence diagrams and a "Reproducing on two machines" section.
- This document's status is updated to "shipped".

If you got here, you've shipped the demo. Update the experiment log one
last time and merge.
