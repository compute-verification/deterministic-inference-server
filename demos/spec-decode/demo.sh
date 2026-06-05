#!/usr/bin/env bash
# spec-decode demo entry point.
#
# Same 5-process topology as the inference demo, but the workload is greedy
# speculative decoding (draft + target model). The Host Cluster runs spec-decode
# and returns the full per-round trace; the Recomp Cluster re-runs it and
# bitwise-compares; the Tap forwards both payloads to the proof server, which
# compares them and builds a speculative-decoding task graph.
#
# Usage:
#   ./demo.sh                 # --mock (CPU, no GPU): deterministic mock backend
#   ./demo.sh --real          # load real Qwen3-0.6B (draft) + Qwen3-1.7B (target) on GPU
#   ./demo.sh --help

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODE="mock"
for arg in "$@"; do
  case "$arg" in
    --mock) MODE="mock" ;;
    --real) MODE="real" ;;
    -h|--help) sed -n '2,14p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

PY=""
if [[ -x "${REPO_ROOT}/.venv/bin/python3" ]]; then PY="${REPO_ROOT}/.venv/bin/python3";
elif command -v uv >/dev/null 2>&1; then ( cd "${REPO_ROOT}" && uv sync --quiet ); PY="${REPO_ROOT}/.venv/bin/python3";
else PY="$(command -v python3)"; fi

WORK="$(mktemp -d -t spec-decode-demo-XXXXXX)"
LOGS="${WORK}/logs"; GRAPHS="${WORK}/graphs"; mkdir -p "${LOGS}" "${GRAPHS}"

PORTS=(8000 8010 8020 8030 8050)
for p in "${PORTS[@]}"; do pid="$(lsof -ti :"$p" 2>/dev/null || true)"; [[ -n "$pid" ]] && kill -9 $pid 2>/dev/null || true; done

PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  sleep 1
  for pid in "${PIDS[@]}"; do kill -9 "$pid" 2>/dev/null || true; done
  rm -rf "${WORK}"
}
trap cleanup EXIT

start_bg() { local name="$1"; shift; ( "$@" </dev/null >"${LOGS}/${name}.out" 2>&1 ) & PIDS+=("$!"); echo "[demo] started ${name} (pid $!)"; }
wait_health() {
  local url="$1"; local secs="${2:-60}"; local deadline=$(( $(date +%s) + secs ))
  until [[ $(date +%s) -ge $deadline ]]; do curl -sf "$url" >/dev/null 2>&1 && return 0; sleep 0.3; done
  echo "demo.sh: timed out waiting for $url" >&2; return 1
}

export PYTHONPATH="${REPO_ROOT}"
S="${REPO_ROOT}/demos/spec-decode/servers"

CLUSTER_ARGS=(); HEALTH_TIMEOUT=60
if [[ "${MODE}" == "mock" ]]; then CLUSTER_ARGS=(--mock); else HEALTH_TIMEOUT=600; fi

# Start the clusters one at a time: in --real mode each loads a draft+target
# model pair onto the GPU, and loading both pairs concurrently can abort the
# CUDA context. Sequential load is plenty fast.
start_bg host_cluster   "${PY}" "${S}/host_cluster.py"   --port 8020 "${CLUSTER_ARGS[@]}"
wait_health http://127.0.0.1:8020/health "${HEALTH_TIMEOUT}"
start_bg recomp_cluster "${PY}" "${S}/recomp_cluster.py" --port 8030 "${CLUSTER_ARGS[@]}"
wait_health http://127.0.0.1:8030/health "${HEALTH_TIMEOUT}"

start_bg proof_server   "${PY}" "${S}/proof_server.py"   --port 8050 --work-dir "${GRAPHS}"
wait_health http://127.0.0.1:8050/health

start_bg tap            "${PY}" "${S}/tap.py" --port 8010 \
  --host-url http://127.0.0.1:8020 --recomp-url http://127.0.0.1:8030 \
  --compare-server-url http://127.0.0.1:8050
wait_health http://127.0.0.1:8010/health

start_bg gateway        "${PY}" "${S}/gateway.py" --port 8000 --tap-url http://127.0.0.1:8010
wait_health http://127.0.0.1:8000/health

echo "=== sending requests through the gateway ==="
for prompt in "The quick brown fox" "Explain speculative decoding briefly"; do
  curl -sf -X POST http://127.0.0.1:8000/request -H 'Content-Type: application/json' \
    -d "{\"prompt\": \"${prompt}\", \"max_tokens\": 16, \"k\": 4}" \
    | "${PY}" -c "import json,sys; d=json.load(sys.stdin); print('  out:', repr(d['output'][:60]), '|', d['target_passes'], 'target passes for', len(d['output_ids']), 'tokens')"
done

# The verify+compare fan-out is async and, in --real mode, recomp re-runs the
# whole decode (seconds), so poll the proof server until both compares land.
echo "[demo] waiting for async verify+compare..."
field() { "${PY}" -c "import json,sys; print(json.load(sys.stdin).get('$1',0))"; }
COMPARE_DEADLINE=$(( $(date +%s) + 180 ))
while :; do
  HEALTH="$(curl -sf http://127.0.0.1:8050/health || echo '{}')"
  COMPARED="$(echo "${HEALTH}" | field compared)"
  [[ "${COMPARED:-0}" -ge 2 ]] && break
  [[ $(date +%s) -ge ${COMPARE_DEADLINE} ]] && break
  sleep 2
done
echo "[demo] proof server: ${HEALTH}"
MATCHES="$(echo "${HEALTH}" | field matches)"
GRAPHS_BUILT="$(echo "${HEALTH}" | field graphs_built)"
if [[ "${COMPARED}" -lt 2 || "${MATCHES}" -lt 2 || "${GRAPHS_BUILT}" -lt 2 ]]; then
  echo "demo.sh: expected >=2 compared/matches/graphs; got compared=${COMPARED} matches=${MATCHES} graphs=${GRAPHS_BUILT}" >&2
  tail -20 "${LOGS}/proof_server.out" >&2 || true; exit 4
fi

echo ""
echo "=== a built spec-decode graph (one request) ==="
GRAPH_FILE="$(ls "${GRAPHS}"/spec_graph_*.json | head -1)"
"${PY}" -m json.tool "${GRAPH_FILE}" | head -40

# Phase 1: copy graphs out for the viz when running on a GPU box.
if [[ -n "${SPEC_GRAPH_OUT:-}" ]]; then cp "${GRAPHS}"/spec_graph_*.json "${SPEC_GRAPH_OUT}/" 2>/dev/null || true; echo "[demo] copied graphs to ${SPEC_GRAPH_OUT}"; fi

echo ""
echo "ALL PASS"
