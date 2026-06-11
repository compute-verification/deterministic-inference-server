// View-layer collapsing of long linear runs into a single "group" node.
//
// The atomic graph (graphs.json) is NEVER mutated — every node stays one forward
// pass. These pure functions derive a *display graph* from the atomic graph plus
// a set of expanded segment ids, so the UI can show ~12 group nodes for the
// coding agent while the data underneath is still ~6400 forward passes.

import { ctx0 } from "./graph-model.js";

export const COLLAPSE_MIN = 16;

// The key that nodes in a run must share to collapse together: (kind, phase).
// Kind is included so a turn's prefill (reading context) never merges into its
// decode run — the orange "jump" stays a separate node and only the decode
// stream collapses. Phase (the turn label, e.g. "write p_less.py") keeps
// adjacent same-kind turns apart; it falls back to role, then kind. We
// JSON.stringify the pair so the two fields can never collide regardless of
// what characters a label contains.
export function groupKey(n) {
  const p = n.payload || {};
  const phase = p.phase ?? p.role ?? n.kind;
  return JSON.stringify([n.kind, phase]);
}

// Human label for a collapsed run.
export function phaseLabel(n) {
  const p = n.payload || {};
  return p.phase ?? p.role ?? n.kind;
}

// Partition the atomic graph into ordered segments. A segment is a maximal run
// of nodes connected 1:1 (the previous node has out-degree 1, this node has
// in-degree 1, and this node's sole predecessor IS the previous node) that share
// a group key. Fan-out / fan-in nodes break runs and become singletons.
// Returns [{ id, key, kind, nodeIds, collapsible }] in topological (node) order.
export function segmentGraph(nodes, edges, opts = {}) {
  const min = opts.min ?? COLLAPSE_MIN;
  const indeg = new Map();
  const outdeg = new Map();
  const pred = new Map(); // dst -> src (unique only where indeg === 1)
  for (const n of nodes) {
    indeg.set(n.id, 0);
    outdeg.set(n.id, 0);
  }
  for (const e of edges) {
    outdeg.set(e.src, (outdeg.get(e.src) || 0) + 1);
    indeg.set(e.dst, (indeg.get(e.dst) || 0) + 1);
    pred.set(e.dst, e.src);
  }

  const segments = [];
  let cur = null;
  for (const n of nodes) {
    const k = groupKey(n);
    const last = cur && cur.nodeIds[cur.nodeIds.length - 1];
    const canContinue =
      cur &&
      cur.key === k &&
      indeg.get(n.id) === 1 &&
      outdeg.get(last) === 1 &&
      pred.get(n.id) === last;
    if (canContinue) {
      cur.nodeIds.push(n.id);
    } else {
      cur = { id: `seg:${n.id}`, key: k, kind: n.kind, nodeIds: [n.id] };
      segments.push(cur);
    }
  }
  for (const s of segments) s.collapsible = s.nodeIds.length >= min;
  return segments;
}

// All collapsible segment ids — for "expand all".
export function collapsibleSegmentIds(graph, opts = {}) {
  return segmentGraph(graph.nodes || [], graph.edges || [], opts)
    .filter((s) => s.collapsible)
    .map((s) => s.id);
}

// Derive the display graph: collapsed segments become one group node; everything
// else stays atomic. Node ids are strings throughout (group ids are "seg:<id>",
// atomic ids are String(node.id)). Edges are remapped to display ids, internal
// run edges dropped, and duplicates removed.
export function buildDisplayGraph(graph, expandedSet = new Set(), opts = {}) {
  const nodes = graph.nodes || [];
  const edges = graph.edges || [];
  const segments = segmentGraph(nodes, edges, opts);
  const byId = new Map(nodes.map((n) => [n.id, n]));

  const isCollapsed = (s) => s.collapsible && !expandedSet.has(s.id);
  const displayIdOf = new Map();
  for (const s of segments) {
    const collapsed = isCollapsed(s);
    for (const nid of s.nodeIds) {
      displayIdOf.set(nid, collapsed ? s.id : String(nid));
    }
  }

  const dNodes = [];
  for (const s of segments) {
    if (isCollapsed(s)) {
      const run = s.nodeIds.map((id) => byId.get(id));
      const first = run[0];
      const last = run[run.length - 1];
      dNodes.push({
        id: s.id,
        kind: "group",
        groupKind: s.kind,
        segId: s.id,
        count: run.length,
        flops: run.reduce((a, n) => a + (n.flops || 0), 0),
        // whitelisted inputs are free to pass — they don't count toward the
        // run's displayed input size (flops do: the passes still ran)
        tokens: run.reduce((a, n) => a + (n.whitelisted ? 0 : n.tokens || 0), 0),
        attended: run.reduce((a, n) => a + (n.whitelisted ? 0 : n.attended || 0), 0),
        // context range across the run, for the "in:" annotation — a sum of
        // attended over separate passes has no single starting context
        ctxFirst: ctx0(first.tokens, first.attended),
        ctxLast: ctx0(last.tokens, last.attended),
        label: phaseLabel(first),
        firstId: first.id,
        lastId: last.id,
        payload: { phase: phaseLabel(first), count: run.length },
      });
    } else {
      for (const nid of s.nodeIds) {
        dNodes.push({
          ...byId.get(nid),
          id: String(nid),
          segId: s.id,
          collapsibleSeg: s.collapsible, // true only when this run is expanded
        });
      }
    }
  }

  const seen = new Set();
  const dEdges = [];
  for (const e of edges) {
    const src = displayIdOf.get(e.src);
    const dst = displayIdOf.get(e.dst);
    if (src === undefined || dst === undefined) continue;
    if (src === dst) continue; // internal to a collapsed run
    const key = src + "->" + dst;
    if (seen.has(key)) continue;
    seen.add(key);
    dEdges.push({ src, dst });
  }

  return { nodes: dNodes, edges: dEdges, segments };
}
