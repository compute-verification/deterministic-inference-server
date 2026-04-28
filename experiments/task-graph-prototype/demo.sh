#!/usr/bin/env bash
# One-shot demo of the task graph prototype.
#
# 1. Builds the Rust validator if not already built.
# 2. Runs the Python sim, generating six attested-graph bundles:
#    honest + 5 adversarial mutations.
# 3. Validates each, asserts honest passes and each adversarial run fails
#    at the *expected* check name. Prints a green tick / red cross summary.

set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXP_DIR/../.." && pwd)"
VALIDATOR_DIR="$EXP_DIR/validator"
DATA_DIR="$EXP_DIR/data"

GREEN=$'\033[0;32m'
RED=$'\033[0;31m'
NC=$'\033[0m'

echo "[1/3] building validator..."
( cd "$VALIDATOR_DIR" && cargo build --release --quiet )
VALIDATE="$VALIDATOR_DIR/target/release/validate"

echo "[2/3] generating bundles..."
( cd "$REPO_ROOT" && python3 experiments/task-graph-prototype/scripts/run_demo.py )

echo "[3/3] validating..."
fail_count=0
for d in "$DATA_DIR"/*/; do
  name=$(basename "$d")
  expected=$(cat "$d/expected_failure.txt" 2>/dev/null || echo "PASS")
  set +e
  out=$("$VALIDATE" --graph "$d/graph.json" --replay "$d/replay_responses.json" 2>&1)
  rc=$?
  set -e
  verdict=$(printf '%s\n' "$out" | grep -E '^VERDICT:' | awk '{print $2}')
  first_fail=$(printf '%s\n' "$out" | grep -E '^  first failure:' \
                | sed -E 's/^  first failure: ([^:]+):.*/\1/' || true)
  # The policy check buckets all kinds under `policy:`; recover the kind
  # so the summary is readable.
  if [[ "$first_fail" == "policy" ]]; then
    sub=$(printf '%s\n' "$out" | grep -E '^  first failure:' \
           | sed -E 's/^  first failure: policy: ([^:]+):.*/\1/' || true)
    if [[ -n "$sub" && "$sub" != "policy" ]]; then
      first_fail="policy_${sub}"
    fi
  fi
  ok=0
  if [[ "$expected" == "PASS" && "$verdict" == "PASS" ]]; then ok=1; fi
  if [[ "$expected" != "PASS" && "$verdict" == "FAIL" ]]; then
    if [[ "$first_fail" == "$expected" ]]; then ok=1; fi
  fi
  if [[ $ok -eq 1 ]]; then
    printf "  ${GREEN}OK${NC}    %-24s expected=%-25s got=%s\n" "$name" "$expected" "${first_fail:-PASS}"
  else
    printf "  ${RED}FAIL${NC}  %-24s expected=%-25s got=%s\n" "$name" "$expected" "${first_fail:-PASS}"
    fail_count=$((fail_count + 1))
    printf '%s\n' "$out" | sed 's/^/        /'
  fi
done

echo
if [[ $fail_count -eq 0 ]]; then
  echo "${GREEN}ALL PASS${NC} -- $(ls -d "$DATA_DIR"/*/ | wc -l) scenarios behaved as expected"
else
  echo "${RED}${fail_count} scenario(s) misbehaved${NC}"
  exit 1
fi

echo
echo "[4/4] building interactive viewer..."
python3 "$EXP_DIR/scripts/make_viewer.py" --out "$EXP_DIR/viewer.html"
echo
echo "  open ${GREEN}$EXP_DIR/viewer.html${NC} in a browser to inspect any artifact"
echo "  in any of the $(ls -d "$DATA_DIR"/*/ | wc -l) scenarios."
