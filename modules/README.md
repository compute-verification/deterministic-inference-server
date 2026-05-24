# Capability modules

The deterministic serving stack, organized **by function**. Each subdirectory is
one capability with a documented interface; the [`Pipeline`](pipeline.py)
composes them, and [`workflows/`](../workflows/) is the recipe book of runnable
compositions.

> These modules are the **curated public surface**. Module-specific assets live
> inside the module directory (e.g. `build/lockfiles/`, `utils/deploy/`); shared
> and deeply-coupled code (`pkg/`, `cmd/`, `schemas/`) stays central with an owner
> recorded in the [Asset ownership](#asset-ownership) map below. See
> [`docs/plans/repo-modularization.md`](../docs/plans/repo-modularization.md).

## The capability map

| Capability | What it does | Interface | Underlying code |
|---|---|---|---|
| [**build**](build/) | Hermetic, reproducible runtime + OCI image | `nix build .#oci` · `cmd/builder` | `build/lockfiles/`, `flake.nix`, `nix/`, `cmd/builder` |
| [**inference**](inference/) | Bitwise-deterministic vLLM (the c3 config) | `modules.inference` · `cmd/server` | `cmd/{server,runner}`, `pkg/manifest`, `manifests/` |
| [**network**](network/) | Deterministic L2 egress frames | `modules.network.egress_frames(...)` | `pkg/networkdet`, `native/libnetdet` |
| [**memory**](memory/) | PoSE memory wipe + erasure attestation | `modules.memory.load_pose(...)` | `experiments/memory_wipe/src/pose` |
| [**attestation**](attestation/) | Matmul / token / replay verification | `modules.attestation.attest_matmuls(...)` | `pkg/{freivalds,e2e,proverdet}`, `cmd/verifier` |
| [**utils**](utils/) | Canonical JSON, digests, schema validation | `modules.utils.canonical_json_bytes(...)` | `deploy/`, `pkg/common` |

A capability need not be a Python package — `build` and `utils` are nix + shell.
The contract is a **documented interface**, not a uniform implementation.

## Asset ownership

Every tracked asset has an owning module, is **shared core**, or is **repo-level**
(project-wide). Nothing is orphaned. Module-specific *data* moves into the module
directory; code/scripts with repo-root assumptions or cross-module use stay in place
with the owner recorded here — moving them would break the ~97 import sites, the
determinism gates, embedded path-depth assumptions (e.g. `deploy/`'s scripts compute
`REPO_ROOT` relative to their location), or — for `flake.nix` — Nix's requirement that
the flake live at the repo root.

| Asset | Owner | Lives in |
|---|---|---|
| `lockfiles/` | build | **`modules/build/lockfiles/`** ✅ moved |
| `flake.nix`, `flake.lock`, `nix/`, `Makefile`, `cmd/builder/` | build | repo root / `cmd/` (`flake.nix` must stay at root for Nix) |
| `cmd/{server,runner,capture,resolver}/`, `pkg/manifest/`, `manifests/` | inference | `cmd/`, `pkg/`, `manifests/` |
| `pkg/networkdet/`, `native/libnetdet/` | network | `pkg/`, `native/` |
| `pkg/{freivalds,e2e,proverdet}/`, `cmd/{verifier,verifier_cli,verifier_server,prover}/` | attestation | `pkg/`, `cmd/` |
| `experiments/memory_wipe/` | memory | `experiments/` (deployable `pose` package) |
| `deploy/`, `scripts/lambda/lambda_cli.py`, `demo/` | utils | repo root — `deploy/` scripts assume repo-root depth, so kept in place (designated) |
| `pkg/common/`, `schemas/` | shared core | `pkg/`, `schemas/` — used across all modules; schemas loaded by `pkg/common` |
| `workflows/` | shared (recipe book) | `workflows/` — module compositions via `modules.Pipeline` |
| `tests/` | per-module + shared | `tests/modules/` per capability; `unit/integration/e2e/determinism` cover the spine |
| `docs/`, `scripts/`, `scripts/ci/` | shared / platform | repo root |
| `experiments/{e2e-audit, prover-verifier-demo, freivalds-attestation, multinode-determinism}` | inference / attestation | `experiments/` — kept on `main` (gates/demos depend on them); other research experiments live on the `experiments` branch |
| `README.md`, `CLAUDE.md`, `LICENSE`, `CITATION.cff`, `.gitignore`, `.github/`, `.claude/`, `.internal/` | repo-level | repo root |

Code under `pkg/`/`cmd/` is owned by its module via the facade in `modules/<x>/`;
physically relocating it is the deferred Phase 3.

## The unified interface

Everything speaks the artifact spine:

```
manifest.v1  ──resolve──▶  lockfile.v1  ──build──▶  lockfile.v1(+closure)
     │                                                      │
     └──────────────────────── run ────────────────────────┘
                                 ▼
                          run_bundle.v1  ──verify──▶  verify_report.v1
```

Compose it in a few lines instead of bash:

```python
from modules import Pipeline

report = (Pipeline.from_manifest("manifests/qwen3-1.7b.manifest.json")
          .resolve()        # -> lockfile.v1
          .build()          # -> closure digest
          .run("/tmp/a")    # -> run_bundle.v1
          .run("/tmp/b")    # -> run_bundle.v1 (independent run)
          .verify())        # -> verify_report.v1  (status "conformant" iff identical)
```

## Status (Phase 1 + 2)

All six capabilities now have Python facades (`api.py`), plus the **Pipeline**.
Note: `build`'s nix wrappers need Nix, and `memory` re-exports the
separately-deployed `pose` package from its canonical location (it is *not*
relocated — that would break the remote `uv` install workflow). Smoke-tested in
`tests/modules/`. Recipe book: `deterministic_inference_server`,
`deterministic_lora_training`, `verified_inference`.

Remaining (Phase 3, deferred): optionally fold `pkg/` physically under
`modules/` — only once this API has stabilized in review (it would break the
`test_repo_layout` pinned-dir guardrail and churn every import).
