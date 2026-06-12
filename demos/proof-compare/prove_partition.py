"""Prove a bounded-cost partition of a HIDDEN task-graph scene in SP1.

Two roles, both played here for the demo:

  PROVER  — holds the graph (a scene from traces/graphs.json), draws a random
            32-byte blind, publishes the blinded commitment
            x = sha256(canonical-encoding || blind), plans a partition under
            the given caps, and runs the SP1 program with graph + partition +
            blind as PRIVATE witnesses.

  AUDITOR — receives x out-of-band plus the proof; checks that the proof's
            public outputs are exactly (nonce, x, C, S, n_parts). The auditor
            NEVER sees the graph — not even its node count.

The statement proven: "the graph committed to by x has a partition into
stages where every dependency edge flows forward, each stage's ΣFLOPs <= C,
and each stage's Σ non-whitelisted input tokens <= S."

Examples:
  # interpreter only (fast; no proof bytes)
  python3 demos/proof-compare/prove_partition.py --scene coding \
      --cap-flops 5e13 --cap-input 2000

  # real SP1 proof + verification round-trip
  python3 demos/proof-compare/prove_partition.py --scene inference \
      --cap-flops 2e10 --cap-input 64 --prove --out /tmp/partition-proof
"""
from __future__ import annotations

import argparse
import json
import secrets
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.proof_server.partition import (
    PartitionError,
    check_partition,
    graph_commitment,
    plan_partition,
    sp1_input_json,
)

GRAPHS = HERE / "traces" / "graphs.json"
HOST_BIN = REPO_ROOT / "modules/proof_server/sp1/target/release/partition-host"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True,
                    choices=["inference", "spec", "training", "coding"])
    ap.add_argument("--cap-flops", required=True, type=float,
                    help="per-part FLOP budget C (float accepted, e.g. 5e13)")
    ap.add_argument("--cap-input", required=True, type=float,
                    help="per-part non-whitelisted input budget S, in tokens")
    ap.add_argument("--prove", action="store_true",
                    help="produce + verify a real SP1 proof (slow); default --execute")
    ap.add_argument("--out", type=Path, default=Path("/tmp/partition-proof"),
                    help="directory for proof bytes + public outputs (--prove)")
    args = ap.parse_args()

    cap_flops, cap_input = int(args.cap_flops), int(args.cap_input)

    # ---- PROVER side -------------------------------------------------------
    graph = json.loads(GRAPHS.read_text())[args.scene]
    total = sum(node["flops"] for node in graph["nodes"])
    print(f"[prover]  scene {args.scene}: {len(graph['nodes'])} nodes, "
          f"total {total:.3e} FLOPs (all private)")
    print(f"[caps]    C={cap_flops:.3e} FLOPs/part, S={cap_input} tokens/part")

    try:
        parts = plan_partition(graph, cap_flops, cap_input)
        stats = check_partition(graph, parts, cap_flops, cap_input)
    except PartitionError as exc:
        print(f"[plan]    INFEASIBLE: {exc}")
        return 2
    print(f"[plan]    {stats['n_parts']} parts; max part: "
          f"{stats['max_part_flops']:.3e} FLOPs, {stats['max_part_input']} input tokens")

    blind = secrets.token_hex(32)
    published_x = graph_commitment(graph, blind)
    print(f"[prover]  publishes commitment x = {published_x}")

    if not HOST_BIN.exists():
        print(f"[sp1]     SKIPPED — host binary missing at {HOST_BIN}\n"
              "          build: PROTOC=$(which protoc) cargo build --release \\\n"
              "                 --manifest-path modules/proof_server/sp1/Cargo.toml")
        return 0

    nonce = secrets.token_hex(32)
    stdin = sp1_input_json(graph, parts, cap_flops, cap_input, nonce,
                           blind_hex=blind).encode()

    if not args.prove:
        r = subprocess.run([str(HOST_BIN), "--execute"], input=stdin,
                           capture_output=True, timeout=1800)
        if r.returncode != 0:
            print(f"[sp1]     guest REJECTED: {r.stderr.decode(errors='replace')[-300:]}")
            return 1
        public = json.loads(r.stdout.decode().strip().splitlines()[-1])
        print("[sp1]     execute OK")
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        proof_path = args.out / f"{args.scene}.proof.bin"
        public_path = args.out / f"{args.scene}.public.json"
        r = subprocess.run([str(HOST_BIN), "--prove", "--proof", str(proof_path)],
                           input=stdin, capture_output=True, timeout=7200)
        if r.returncode != 0:
            print(f"[sp1]     prove FAILED: {r.stderr.decode(errors='replace')[-300:]}")
            return 1
        public = json.loads(r.stdout.decode().strip().splitlines()[-1])
        public_path.write_text(json.dumps(public, indent=1) + "\n")
        print(f"[sp1]     proof written: {proof_path} ({proof_path.stat().st_size} bytes)")
        v = subprocess.run([str(HOST_BIN), "--verify", "--proof", str(proof_path),
                            "--public", str(public_path)],
                           capture_output=True, timeout=1800)
        if v.returncode != 0:
            print(f"[sp1]     verify FAILED: {v.stderr.decode(errors='replace')[-300:]}")
            return 1
        print(f"[sp1]     verify: {v.stdout.decode().strip()}")

    # ---- AUDITOR side ------------------------------------------------------
    # Checks ONLY the public outputs against (x, nonce, caps). No graph access.
    print(f"[auditor] sees: x={public['graph_commitment'][:23]}…, "
          f"C={public['cap_flops']:.3e}, S={public['cap_input']}, "
          f"n_parts={public['n_parts']} — and nothing else")
    ok = (public["graph_commitment"] == published_x
          and public["auditor_nonce"] == nonce
          and public["cap_flops"] == cap_flops
          and public["cap_input"] == cap_input
          and public["n_parts"] == stats["n_parts"])
    print(f"[auditor] commitment+nonce+caps match: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
