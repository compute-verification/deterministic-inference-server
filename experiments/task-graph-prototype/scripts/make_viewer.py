"""Build the interactive HTML viewer.

Reads every scenario directory under data/ and emits a single self-contained
HTML file. The viewer lets you pick a scenario, see the verdict, and click
on any artifact to get the eight success-condition answers about it.

Each scenario gets pre-computed in Python so the JavaScript only has to
look up answers — no crypto in the browser.

Usage:
    python3 scripts/make_viewer.py [--data-dir DIR] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

EXP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EXP_DIR / "scripts"))

from sim.transparency_log import (  # noqa: E402
    _leaf_hash,
    _node_hash,
    _root_hash,
    verify_consistency,
    verify_inclusion,
)
from sim.crypto import canonical_json_bytes  # noqa: E402

VIEWER_TEMPLATE = EXP_DIR / "scripts" / "viewer" / "viewer_template.html"


# --------------------------------------------------------------------- helpers

def _by_id(items: list[dict], key: str) -> dict[str, dict]:
    return {it[key]: it for it in items}


def _short(commit: str) -> str:
    if commit.startswith("sha256:"):
        return "sha256:" + commit.split(":", 1)[1][:10] + "…"
    return commit[:14] + "…" if len(commit) > 14 else commit


def _ev_full_dict(ev: dict[str, Any]) -> dict[str, Any]:
    # The leaf in the transparency log is the canonical JSON of the FULL
    # signed event (matches the orchestrator).
    return {
        "tap_id": ev["tap_id"],
        "tap_seq": ev["tap_seq"],
        "sender": ev["sender"],
        "receiver": ev["receiver"],
        "artifact_id": ev["artifact_id"],
        "commitment": ev["commitment"],
        "size": ev["size"],
        "timestamp_ms": ev["timestamp_ms"],
        "tap_signature": ev["tap_signature"],
    }


def _cumulative_flops(graph: dict[str, Any]) -> dict[str, int]:
    """Compute total FLOP-equivalent ancestry cost per artifact, mirroring
    the validator's `cumulative_ancestry_flops`."""
    producer = {}
    for t in graph["tasks"]:
        for o in t["output_artifact_ids"]:
            producer[o] = t
    cache: dict[str, int] = {}

    def rec(aid: str) -> int:
        if aid in cache:
            return cache[aid]
        cache[aid] = 0  # cycle guard
        t = producer.get(aid)
        if t is None:
            cache[aid] = 0
            return 0
        s = int(t["claimed_flops"])
        for inp in t["input_artifact_ids"]:
            s += rec(inp)
        cache[aid] = s
        return s

    for a in graph["artifacts"]:
        rec(a["artifact_id"])
    return cache


def _layered_layout(graph: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Position artifacts (circles) and tasks (rects) on a top-to-bottom
    layered grid. Layer of an external artifact is 0; layer of a task is
    max(input layer) + 1; layer of a non-external artifact is its
    producer task's layer + 1."""
    artifacts = _by_id(graph["artifacts"], "artifact_id")
    tasks = _by_id(graph["tasks"], "task_id")

    layer: dict[tuple[str, str], int] = {}
    # Externals at layer 0
    for aid, a in artifacts.items():
        if a["external"]:
            layer[("artifact", aid)] = 0

    # Iterate to a fixed point
    changed = True
    safety = 50
    while changed and safety > 0:
        changed = False
        safety -= 1
        for tid, t in tasks.items():
            if all(("artifact", inp) in layer for inp in t["input_artifact_ids"]):
                lvl = max((layer[("artifact", inp)] for inp in t["input_artifact_ids"]), default=0) + 1
                if layer.get(("task", tid)) != lvl:
                    layer[("task", tid)] = lvl
                    changed = True
        for aid, a in artifacts.items():
            if a["external"]:
                continue
            pid = a.get("producer_task_id")
            if pid and ("task", pid) in layer:
                lvl = layer[("task", pid)] + 1
                if layer.get(("artifact", aid)) != lvl:
                    layer[("artifact", aid)] = lvl
                    changed = True

    # Fall back: any unlayered nodes go to the bottom.
    max_layer = max(layer.values(), default=0)
    for aid in artifacts:
        layer.setdefault(("artifact", aid), max_layer + 1)
    for tid in tasks:
        layer.setdefault(("task", tid), max_layer + 1)

    # Group nodes by layer, assign x within layer.
    by_layer: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for k, v in layer.items():
        by_layer[v].append(k)
    # Sort within layer for stability.
    for v in by_layer.values():
        v.sort()

    width = 1180
    y_step = 95
    pos: dict[str, dict[str, float]] = {}
    for lvl, nodes in sorted(by_layer.items()):
        n = len(nodes)
        for i, (kind, ident) in enumerate(nodes):
            x = (i + 1) * width / (n + 1)
            y = 50 + lvl * y_step
            key = f"{kind}:{ident}"
            pos[key] = {"x": x, "y": y, "layer": lvl}

    return pos


# --------------------------------------------------------------------- per-scenario

def _verify_log(graph: dict[str, Any]) -> dict[str, Any]:
    """Verify the transparency log structure: per-leaf inclusion proofs,
    recomputed root, consistency proofs against previous STHs.

    Returns a JSON-serializable status object the viewer can render."""
    log = graph["log"]
    transmissions = graph["transmissions"]
    leaves_bytes = [canonical_json_bytes(_ev_full_dict(ev)) for ev in transmissions]

    declared_root = bytes.fromhex(log["root_hash"])
    recomputed_root = _root_hash(leaves_bytes) if leaves_bytes else b""
    root_match = (recomputed_root == declared_root)

    inclusion_status: dict[int, bool] = {}
    for i, ev in enumerate(transmissions):
        proof_hex = log["inclusion_proofs"].get(str(i), [])
        proof = [bytes.fromhex(h) for h in proof_hex]
        inclusion_status[i] = verify_inclusion(
            leaves_bytes[i], i, log["tree_size"], proof, declared_root,
        )

    consistency_status: list[dict[str, Any]] = []
    prev_by_size = {p["tree_size"]: p for p in log.get("previous_sths", [])}
    for cp in log.get("consistency_proofs", []):
        from_size = cp["from_size"]
        to_size = cp["to_size"]
        path = [bytes.fromhex(h) for h in cp["path"]]
        prev = prev_by_size.get(from_size)
        if prev is None or to_size != log["tree_size"]:
            consistency_status.append({
                "from_size": from_size, "to_size": to_size,
                "verified": False, "reason": "from STH not found or to_size != current",
            })
            continue
        old_root = bytes.fromhex(prev["root_hash"])
        ok = verify_consistency(from_size, to_size, old_root, declared_root, path)
        consistency_status.append({
            "from_size": from_size,
            "to_size": to_size,
            "verified": ok,
            "old_root_short": _short("sha256:" + prev["root_hash"]),
        })

    return {
        "tree_size": log["tree_size"],
        "root_short": _short("sha256:" + log["root_hash"]),
        "root_match": root_match,
        "inclusion_status": {str(k): v for k, v in inclusion_status.items()},
        "previous_sths": [
            {"tree_size": p["tree_size"], "root_short": _short("sha256:" + p["root_hash"])}
            for p in log.get("previous_sths", [])
        ],
        "consistency": consistency_status,
    }


def _eval_policies(graph: dict[str, Any], cumulative: dict[str, int],
                   replay: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mirror the validator's policy evaluation but per artifact, so we can
    show why each one passes or fails."""
    out: list[dict[str, Any]] = []
    tasks_by_id = _by_id(graph["tasks"], "task_id")
    partition_of_pod: dict[str, str] = {t["pod_id"]: t["partition"] for t in graph["tasks"]}
    for p in graph["policies"]:
        kind = p["kind"]
        rec: dict[str, Any] = {"kind": kind, "params": {k: v for k, v in p.items() if k != "kind"}, "violations": []}
        if kind == "ancestry_compute_gate":
            thr = int(p["threshold_flops"])
            req = p["required_reviewer_id"]
            for a in graph["artifacts"]:
                cum = cumulative.get(a["artifact_id"], 0)
                if a["external"] or cum <= thr:
                    continue
                pid = a.get("producer_task_id")
                if not pid:
                    continue
                t = tasks_by_id.get(pid, {})
                signed = any(s["reviewer_id"] == req for s in t.get("reviewer_signatures", []))
                if not signed:
                    rec["violations"].append({
                        "artifact_id": a["artifact_id"],
                        "reason": f"cum {cum} > {thr}, missing reviewer {req}",
                    })
        elif kind == "partition_crossing":
            from_p, to_p, allowed = p["from_partition"], p["to_partition"], p["allowed"]
            for ev in graph["transmissions"]:
                sp = partition_of_pod.get(ev["sender"])
                rp = partition_of_pod.get(ev["receiver"])
                if sp == from_p and rp == to_p and not allowed:
                    rec["violations"].append({
                        "artifact_id": ev["artifact_id"],
                        "reason": f"forbidden {sp}→{rp} on {ev['sender']}→{ev['receiver']}",
                    })
        elif kind == "replay_required":
            need = int(p["min_sample_count"])
            if len(replay) < need:
                rec["violations"].append({
                    "artifact_id": None,
                    "reason": f"only {len(replay)} replays, need {need}",
                })
        out.append(rec)
    return out


def _run_validator(scenario_dir: Path, validator_bin: Path) -> dict[str, Any]:
    """Invoke the Rust validator binary in --json mode."""
    res = subprocess.run(
        [
            str(validator_bin),
            "--graph", str(scenario_dir / "graph.json"),
            "--replay", str(scenario_dir / "replay_responses.json"),
            "--json",
        ],
        capture_output=True, text=True, check=False,
    )
    try:
        return json.loads(res.stdout)
    except json.JSONDecodeError:
        return {"verdict": "fail", "first_failure": "validator did not emit JSON",
                "checks": [], "stderr": res.stderr}


def _build_scenario(scenario_dir: Path, validator_bin: Path) -> dict[str, Any]:
    graph = json.loads((scenario_dir / "graph.json").read_text())
    replay = json.loads((scenario_dir / "replay_responses.json").read_text())["responses"]
    expected = ""
    if (scenario_dir / "expected_failure.txt").exists():
        expected = (scenario_dir / "expected_failure.txt").read_text().strip()

    validator_verdict = _run_validator(scenario_dir, validator_bin)

    cumulative = _cumulative_flops(graph)
    log_status = _verify_log(graph)
    policies = _eval_policies(graph, cumulative, replay)
    pos = _layered_layout(graph)
    partition_of_pod: dict[str, str] = {t["pod_id"]: t["partition"] for t in graph["tasks"]}

    # Build per-artifact answer maps (questions 1..8 from the spec).
    tasks_by_id = _by_id(graph["tasks"], "task_id")
    arts_by_id = _by_id(graph["artifacts"], "artifact_id")
    transmissions_by_artifact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for i, ev in enumerate(graph["transmissions"]):
        ev_with_idx = dict(ev)
        ev_with_idx["leaf_index"] = i
        ev_with_idx["inclusion_verified"] = log_status["inclusion_status"].get(str(i), False)
        transmissions_by_artifact[ev["artifact_id"]].append(ev_with_idx)

    replay_by_task = {r["task_id"]: r for r in replay}

    artifact_panels: dict[str, dict[str, Any]] = {}
    for aid, a in arts_by_id.items():
        producer_tid = a.get("producer_task_id")
        producer = tasks_by_id.get(producer_tid) if producer_tid else None

        # Q2 — inputs of producer
        inputs: list[dict[str, Any]] = []
        if producer:
            for inp in producer["input_artifact_ids"]:
                src = arts_by_id.get(inp, {})
                inputs.append({
                    "artifact_id": inp,
                    "commitment_short": _short(src.get("commitment", "")),
                    "external": src.get("external", False),
                    "size": src.get("size", 0),
                })

        # Q3..Q5 — transmissions involving this artifact
        evs = transmissions_by_artifact.get(aid, [])

        # Q6 — graph/log consistency for this artifact
        gl_status: dict[str, Any] = {"applies": False}
        if evs:
            mismatches = []
            for ev in evs:
                if ev["commitment"] != a["commitment"]:
                    mismatches.append(f"transmission seq {ev['tap_seq']} commitment {_short(ev['commitment'])} ≠ artifact {_short(a['commitment'])}")
                if ev["size"] != a["size"]:
                    mismatches.append(f"transmission seq {ev['tap_seq']} size {ev['size']} ≠ artifact {a['size']}")
            gl_status = {"applies": True, "matches": not mismatches, "mismatches": mismatches}

        # Q7 — replay
        replay_status: dict[str, Any] = {"applies": False}
        if producer_tid and producer_tid in replay_by_task:
            r = replay_by_task[producer_tid]
            replay_status = {
                "applies": True,
                "task_id": producer_tid,
                "recomputed_short": _short(r["recomputed_commitment"]),
                "matches": r["recomputed_commitment"] == a["commitment"],
            }

        # Q8 — policies
        policy_status: list[dict[str, Any]] = []
        for p in policies:
            relevant = [v for v in p["violations"] if v.get("artifact_id") == aid]
            if p["kind"] == "ancestry_compute_gate":
                cum = cumulative.get(aid, 0)
                thr = int(p["params"]["threshold_flops"])
                if a["external"]:
                    policy_status.append({"kind": p["kind"], "verdict": "n/a",
                                          "detail": "external artifact"})
                elif cum <= thr:
                    policy_status.append({"kind": p["kind"], "verdict": "below threshold",
                                          "detail": f"cum {cum} ≤ {thr}"})
                else:
                    if relevant:
                        policy_status.append({"kind": p["kind"], "verdict": "violation",
                                              "detail": relevant[0]["reason"]})
                    else:
                        req = p["params"]["required_reviewer_id"]
                        policy_status.append({"kind": p["kind"], "verdict": "satisfied",
                                              "detail": f"cum {cum} > {thr}, signed by {req}"})
            elif p["kind"] == "partition_crossing":
                # Only mention this policy if the artifact actually has a
                # crossing in this policy's direction.
                from_p, to_p = p["params"]["from_partition"], p["params"]["to_partition"]
                allowed = p["params"]["allowed"]
                actual_crossings = []
                for ev in graph["transmissions"]:
                    if ev["artifact_id"] != aid:
                        continue
                    sp = partition_of_pod.get(ev["sender"])
                    rp = partition_of_pod.get(ev["receiver"])
                    if sp == from_p and rp == to_p:
                        actual_crossings.append(f"{ev['sender']}→{ev['receiver']}")
                if not actual_crossings:
                    continue
                if not allowed:
                    policy_status.append({"kind": p["kind"], "verdict": "violation",
                                          "detail": f"forbidden {from_p}→{to_p} on {', '.join(actual_crossings)}"})
                else:
                    policy_status.append({"kind": p["kind"], "verdict": "ok",
                                          "detail": f"allowed {from_p}→{to_p} on {', '.join(actual_crossings)}"})

        artifact_panels[aid] = {
            "id": aid,
            "commitment": a["commitment"],
            "commitment_short": _short(a["commitment"]),
            "size": a["size"],
            "external": a["external"],
            "cumulative_flops": cumulative.get(aid, 0),
            "producer": (
                {
                    "task_id": producer["task_id"],
                    "pod_id": producer["pod_id"],
                    "partition": producer["partition"],
                    "operation": producer["operation"],
                    "operation_params": producer.get("operation_params", {}),
                    "claimed_flops": producer["claimed_flops"],
                    "interval_ms": producer["interval_ms"],
                    "reviewer_signatures": producer.get("reviewer_signatures", []),
                }
                if producer else None
            ),
            "inputs": inputs,
            "transmissions": [
                {
                    "tap_id": ev["tap_id"],
                    "tap_seq": ev["tap_seq"],
                    "sender": ev["sender"],
                    "receiver": ev["receiver"],
                    "timestamp_ms": ev["timestamp_ms"],
                    "leaf_index": ev["leaf_index"],
                    "tap_signature_short": ev["tap_signature"][:16] + "…",
                    "commitment_short": _short(ev["commitment"]),
                    "size": ev["size"],
                    "inclusion_verified": ev["inclusion_verified"],
                    "graph_log_match": (ev["commitment"] == a["commitment"] and ev["size"] == a["size"]),
                }
                for ev in evs
            ],
            "graph_log": gl_status,
            "replay": replay_status,
            "policies": policy_status,
        }

    # Build node + edge data for the SVG renderer.
    nodes = []
    for aid, a in arts_by_id.items():
        p = pos[f"artifact:{aid}"]
        nodes.append({
            "kind": "artifact",
            "id": aid,
            "label": aid,
            "x": p["x"],
            "y": p["y"],
            "external": a["external"],
        })
    for tid, t in tasks_by_id.items():
        p = pos[f"task:{tid}"]
        nodes.append({
            "kind": "task",
            "id": tid,
            "label": f"{tid}\n{t['operation']}",
            "x": p["x"],
            "y": p["y"],
            "partition": t["partition"],
        })
    edges = []
    for t in graph["tasks"]:
        for inp in t["input_artifact_ids"]:
            edges.append({"from": f"artifact:{inp}", "to": f"task:{t['task_id']}", "kind": "input"})
        for out in t["output_artifact_ids"]:
            edges.append({"from": f"task:{t['task_id']}", "to": f"artifact:{out}", "kind": "output"})
    transmission_edges = []
    for ev in graph["transmissions"]:
        transmission_edges.append({
            "artifact_id": ev["artifact_id"],
            "tap_id": ev["tap_id"],
            "sender": ev["sender"],
            "receiver": ev["receiver"],
            "tap_seq": ev["tap_seq"],
        })

    return {
        "scenario_name": scenario_dir.name,
        "expected_failure": expected,
        "verdict": validator_verdict,
        "graph_meta": {
            "run_id": graph.get("run_id", ""),
            "n_pods": len({t["pod_id"] for t in graph["tasks"]}),
            "n_tasks": len(graph["tasks"]),
            "n_artifacts": len(graph["artifacts"]),
            "n_transmissions": len(graph["transmissions"]),
            "n_taps": len(graph.get("tap_pubkeys", {})),
            "log_id": graph["log"]["log_id"],
            "tree_size": graph["log"]["tree_size"],
            "policies": [{"kind": p["kind"], **{k: v for k, v in p.items() if k != "kind"}}
                         for p in graph["policies"]],
        },
        "log_status": log_status,
        "nodes": nodes,
        "edges": edges,
        "transmissions": transmission_edges,
        "artifact_panels": artifact_panels,
        "replay_responses": replay,
    }


# --------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(EXP_DIR / "data"))
    parser.add_argument("--out", default=str(EXP_DIR / "viewer.html"))
    parser.add_argument("--validator-bin",
                        default=str(EXP_DIR / "validator" / "target" / "release" / "validate"))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    validator_bin = Path(args.validator_bin)
    if not validator_bin.exists():
        print(f"validator binary not built at {validator_bin}", file=sys.stderr)
        return 2

    scenarios: list[dict[str, Any]] = []
    for d in sorted(data_dir.iterdir()):
        if d.is_dir() and (d / "graph.json").exists():
            scenarios.append(_build_scenario(d, validator_bin))

    template = VIEWER_TEMPLATE.read_text()
    payload = json.dumps({"scenarios": scenarios}, separators=(",", ":"))
    html = template.replace("__BUNDLE_JSON__", payload)
    out_path.write_text(html)
    print(f"wrote {out_path} ({len(scenarios)} scenarios, {len(html)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
