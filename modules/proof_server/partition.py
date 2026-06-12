"""Bounded-cost partition of a HIDDEN task graph — Python side of the SP1
statement.

The SP1 program (``sp1/partition-program``) proves: *the graph committed to
by x has a partition into stages such that every dependency edge flows
forward (the stages are executable in order), each stage's summed FLOPs are
<= C, and each stage's summed NON-whitelisted input tokens are <= S*. Both
the graph and the partition are private witnesses; the public outputs are
only (nonce, graph_commitment, C, S, n_parts) — deliberately not even
n_nodes. The commitment is BLINDED, x = sha256(encoding || blind), so an
auditor holding x cannot confirm guesses about a low-entropy graph; the
prover publishes x out-of-band (e.g. inside a signed envelope) and keeps
graph + blind private.

This module owns everything the host needs around that statement:

  * ``partition_graph_bytes(graph)``   — the canonical cost-view encoding the
    guest re-builds in-circuit. MUST stay byte-identical to the Rust
    ``proof_server_lib::partition_graph_bytes``.
  * ``graph_commitment(graph, blind)`` — the blinded commitment x; what the
    prover publishes and the guest recomputes inside the proof.
  * ``graph_partition_digest(graph)``  — UNblinded content digest, for flows
    where the graph itself is public (transparent integrity binding only —
    it is not hiding).
  * ``plan_partition(graph, C, S)``    — greedy planner producing a valid
    witness (or raising if none can exist).
  * ``check_partition(...)``           — pure-Python reference checker with
    the exact semantics of the guest's asserts (fast pre-flight + tests).
  * ``sp1_input_json(...)``            — the stdin document for the
    ``partition-host`` binary (--execute / --prove).

A node's input size is its ``tokens`` (what the pass ingests — the same
number the viz annotates on incoming edges); ``whitelisted`` nodes ingest a
publicly-known constant and count 0 toward S. FLOPs always count toward C:
the whitelist makes *passing* a known input free, never the compute.
"""
from __future__ import annotations

import hashlib
import json
import struct

PARTITION_GRAPH_MAGIC = b"taskgraph-partition-v1\n"

_U32_MAX = 2**32 - 1
_U64_MAX = 2**64 - 1


class PartitionError(ValueError):
    """Raised for malformed graphs, invalid partitions, or infeasible caps."""


def _as_int(value, what: str, bound: int) -> int:
    """Exact non-negative integer in [0, bound] (rejects non-integral floats)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PartitionError(f"{what} must be a number, got {type(value).__name__}")
    if isinstance(value, float):
        if not value.is_integer():
            raise PartitionError(f"{what} must be integral, got {value}")
        value = int(value)
    if not 0 <= value <= bound:
        raise PartitionError(f"{what} out of range [0, {bound}]: {value}")
    return value


def graph_cost_view(graph: dict) -> tuple[list[int], list[int], list[int], list[tuple[int, int]]]:
    """Extract (flops, in_size, whitelisted, edges) in canonical form.

    Nodes are taken in ascending-id order (the builder emits ids in
    topological order); edges become (src_index, dst_index) pairs,
    deduplicated and lexicographically sorted, and must go forward
    (src < dst) — the same invariants the guest asserts.
    """
    nodes = sorted(graph.get("nodes") or [], key=lambda n: n["id"])
    if not nodes:
        raise PartitionError("graph has no nodes")
    index_of = {}
    for i, n in enumerate(nodes):
        if n["id"] in index_of:
            raise PartitionError(f"duplicate node id {n['id']}")
        index_of[n["id"]] = i

    flops = [_as_int(n.get("flops", 0), f"node {n['id']} flops", _U64_MAX) for n in nodes]
    in_size = [_as_int(n.get("tokens", 0), f"node {n['id']} tokens", _U32_MAX) for n in nodes]
    whitelisted = [1 if n.get("whitelisted") else 0 for n in nodes]

    edges = set()
    for e in graph.get("edges") or []:
        try:
            s, d = index_of[e["src"]], index_of[e["dst"]]
        except KeyError as exc:
            raise PartitionError(f"edge references unknown node id {exc}") from exc
        if s >= d:
            raise PartitionError(f"edge {e['src']}->{e['dst']} does not go forward in id order")
        edges.add((s, d))
    return flops, in_size, whitelisted, sorted(edges)


def partition_graph_bytes(graph: dict) -> bytes:
    """Canonical cost-view encoding — byte-identical to the Rust side."""
    flops, in_size, whitelisted, edges = graph_cost_view(graph)
    out = bytearray(PARTITION_GRAPH_MAGIC)
    out += struct.pack("<II", len(flops), len(edges))
    for f, t, w in zip(flops, in_size, whitelisted):
        out += struct.pack("<QIB", f, t, w)
    for s, d in edges:
        out += struct.pack("<II", s, d)
    return bytes(out)


def graph_partition_digest(graph: dict) -> str:
    """UNblinded ``sha256:<hex>`` content digest.

    Only for flows where the graph itself is public (transparent integrity
    binding). It is deterministic in the graph, hence NOT hiding — use
    ``graph_commitment`` whenever the auditor must not see the graph.
    """
    return "sha256:" + hashlib.sha256(partition_graph_bytes(graph)).hexdigest()


def _blind_bytes(blind_hex: str) -> bytes:
    if not (isinstance(blind_hex, str) and len(blind_hex) == 64):
        raise PartitionError("blind must be 64 hex chars (32 bytes)")
    try:
        return bytes.fromhex(blind_hex)
    except ValueError as exc:
        raise PartitionError(f"blind is not hex: {exc}") from exc


def graph_commitment(graph: dict, blind_hex: str) -> str:
    """Blinded commitment x = sha256(encoding || blind), as ``sha256:<hex>``.

    This is what the prover publishes (out-of-band) and what the SP1 guest
    recomputes in-circuit; the auditor compares the two without ever seeing
    the graph. The 32-byte ``blind`` makes the commitment hiding; the
    encoding is self-delimiting, so appending the fixed-width blind is
    unambiguous.
    """
    h = hashlib.sha256(partition_graph_bytes(graph) + _blind_bytes(blind_hex))
    return "sha256:" + h.hexdigest()


def _node_input(in_size: int, whitelisted: int) -> int:
    return 0 if whitelisted else in_size


def plan_partition(graph: dict, cap_flops: int, cap_input: int) -> list[int]:
    """Greedy planner: walk nodes in id (= topological) order and pack each
    into the current part until a budget would overflow, then open a new one.

    Contiguous-in-id-order parts automatically satisfy the guest's
    edge-monotonicity check (edges only go forward in id order). Raises
    ``PartitionError`` iff NO partition can exist: some single node exceeds
    a cap on its own (singleton parts are always available otherwise).
    """
    cap_flops = _as_int(cap_flops, "cap_flops", _U64_MAX)
    cap_input = _as_int(cap_input, "cap_input", _U64_MAX)
    flops, in_size, whitelisted, _ = graph_cost_view(graph)

    parts: list[int] = []
    part = 0
    acc_f = acc_i = 0
    for i, (f, t, w) in enumerate(zip(flops, in_size, whitelisted)):
        t_eff = _node_input(t, w)
        if f > cap_flops or t_eff > cap_input:
            raise PartitionError(
                f"infeasible: node index {i} alone exceeds a cap "
                f"(flops={f} vs C={cap_flops}, input={t_eff} vs S={cap_input})")
        if parts and (acc_f + f > cap_flops or acc_i + t_eff > cap_input):
            part += 1
            acc_f = acc_i = 0
        acc_f += f
        acc_i += t_eff
        parts.append(part)
    return parts


def check_partition(graph: dict, parts: list[int], cap_flops: int, cap_input: int) -> dict:
    """Reference checker mirroring the guest's asserts exactly.

    Returns summary stats ``{n_nodes, n_parts, max_part_flops,
    max_part_input}`` on success; raises ``PartitionError`` otherwise.
    """
    cap_flops = _as_int(cap_flops, "cap_flops", _U64_MAX)
    cap_input = _as_int(cap_input, "cap_input", _U64_MAX)
    flops, in_size, whitelisted, edges = graph_cost_view(graph)
    n = len(flops)
    if len(parts) != n:
        raise PartitionError(f"parts length {len(parts)} != node count {n}")
    parts = [_as_int(p, "part id", _U32_MAX) for p in parts]

    n_parts = max(parts) + 1
    if n_parts > n:
        raise PartitionError("more parts than nodes")
    if set(parts) != set(range(n_parts)):
        raise PartitionError("part ids must be contiguous 0..n_parts")
    for s, d in edges:
        if parts[s] > parts[d]:
            raise PartitionError(f"edge {s}->{d} crosses backward between parts")

    flops_sum = [0] * n_parts
    input_sum = [0] * n_parts
    for i in range(n):
        flops_sum[parts[i]] += flops[i]
        input_sum[parts[i]] += _node_input(in_size[i], whitelisted[i])
    for p in range(n_parts):
        if flops_sum[p] > cap_flops:
            raise PartitionError(f"part {p} exceeds FLOP cap: {flops_sum[p]} > {cap_flops}")
        if input_sum[p] > cap_input:
            raise PartitionError(f"part {p} exceeds input cap: {input_sum[p]} > {cap_input}")
    return {
        "n_nodes": n,
        "n_parts": n_parts,
        "max_part_flops": max(flops_sum),
        "max_part_input": max(input_sum),
    }


def sp1_input_json(graph: dict, parts: list[int], cap_flops: int, cap_input: int,
                   auditor_nonce: str = "00" * 32, blind_hex: str = "00" * 32) -> str:
    """The stdin document for the ``partition-host`` binary.

    ``blind_hex`` is the commitment blinding factor; real provers MUST draw
    it fresh from a CSPRNG (``secrets.token_hex(32)``) — the all-zero default
    is for tests only and makes the commitment equal to a non-hiding hash.
    """
    if not (isinstance(auditor_nonce, str) and len(auditor_nonce) == 64):
        raise PartitionError("auditor_nonce must be 64 hex chars")
    try:
        bytes.fromhex(auditor_nonce)
    except ValueError as exc:
        raise PartitionError(f"auditor_nonce is not hex: {exc}") from exc
    _blind_bytes(blind_hex)  # validate
    flops, in_size, whitelisted, edges = graph_cost_view(graph)
    return json.dumps({
        "auditor_nonce": auditor_nonce,
        "blind": blind_hex,
        "cap_flops": _as_int(cap_flops, "cap_flops", _U64_MAX),
        "cap_input": _as_int(cap_input, "cap_input", _U64_MAX),
        "flops": flops,
        "in_size": in_size,
        "whitelisted": whitelisted,
        "edges": [list(e) for e in edges],
        "parts": [_as_int(p, "part id", _U32_MAX) for p in parts],
    })
