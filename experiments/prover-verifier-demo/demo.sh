#!/usr/bin/env bash
# prover-verifier demo entry point (Task 10.1).
#
# Spawns prover + verifier locally (or talks to a remote pair, see --remote),
# runs three scenarios — benign, mixed_lora, lora_loading — and exits 0 iff
# the actual verdicts match the expected outcomes.
#
# Usage:
#   ./demo.sh                # default: 5s per scenario
#   ./demo.sh --quick        # 2s per scenario (CI/dev)
#   ./demo.sh --long         # 15s per scenario (deep view)
#   PROVER_HOST=10.0.0.1 PROVER_PORT=8000 \
#   VERIFIER_HOST=10.0.0.2 VERIFIER_PORT=9000 ./demo.sh --remote

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PER_SCENARIO=5.0
REMOTE=""
for arg in "$@"; do
  case "$arg" in
    --quick)  PER_SCENARIO=2.0 ;;
    --long)   PER_SCENARIO=15.0 ;;
    --remote) REMOTE="--remote" ;;
    -h|--help)
      sed -n '2,12p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "Unknown arg: $arg" >&2
      exit 2
      ;;
  esac
done

# Pick a Python: prefer the repo's .venv if it exists (it has pydantic,
# jsonschema, hypothesis, matplotlib already installed), else assume the
# system python has pydantic + jsonschema available.
if [[ -x "${REPO_ROOT}/.venv/bin/python3" ]]; then
  PY="${REPO_ROOT}/.venv/bin/python3"
else
  PY="python3"
fi

cd "${REPO_ROOT}"
exec "${PY}" experiments/prover-verifier-demo/scripts/demo_driver.py \
  --per-scenario "${PER_SCENARIO}" ${REMOTE}
