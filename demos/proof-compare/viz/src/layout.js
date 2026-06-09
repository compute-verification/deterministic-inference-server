// Lay out a canonical graph ({nodes, edges}) into React Flow nodes/edges.
//
// Two strategies:
//   * Branching graphs (spec-decode fan-in, training spine+branches) use elkjs'
//     layered (Sugiyama) algorithm — proper crossing minimization.
//   * Pure chains (inference, coding agent) use a fast O(n) serpentine grid.
//     elk's layered layouter recurses per layer and overflows the stack on a
//     multi-thousand-node chain (the realistic coding agent is ~6.4k nodes), and
//     a chain needs no crossing minimization anyway. Column-major boustrophedon
//     keeps consecutive nodes vertically adjacent, matching the top/bottom
//     handles, and folds the chain to a readable aspect ratio.
import ELK from "elkjs/lib/elk.bundled.js";
import { edgeStyle, nodeColor, maxFlops } from "./graph-model.js";

const elk = new ELK();

export const NODE_W = 184;
export const NODE_H = 66;
const GAP_X = NODE_W + 44;
const GAP_Y = NODE_H + 34;
const ASPECT = 1.7; // target width/height for the wrapped chain

const ELK_OPTS = {
  "elk.algorithm": "layered",
  "elk.direction": "DOWN",
  "elk.layered.spacing.nodeNodeBetweenLayers": "48",
  "elk.spacing.nodeNode": "28",
  "elk.layered.nodePlacement.strategy": "NETWORK_SIMPLEX",
};

// A graph is a chain (union of simple paths) iff every node has in-degree <= 1
// and out-degree <= 1. Such graphs get the serpentine layout.
function isChain(nodes, edges) {
  if (nodes.length === 0) return false;
  const indeg = new Map();
  const outdeg = new Map();
  for (const e of edges) {
    outdeg.set(e.src, (outdeg.get(e.src) || 0) + 1);
    indeg.set(e.dst, (indeg.get(e.dst) || 0) + 1);
  }
  return nodes.every(
    (n) => (indeg.get(n.id) || 0) <= 1 && (outdeg.get(n.id) || 0) <= 1,
  );
}

// id->position for a chain, folded column-major so consecutive nodes stack
// vertically (down a column, then up the next). Nodes are already in
// topological order by id (build_graph guarantees inputs < id).
function serpentinePositions(nodes) {
  const ordered = [...nodes].sort((a, b) => a.id - b.id);
  const n = ordered.length;
  // Short chains stay a single straight column (clearest). Long ones fold to a
  // grid of ~ASPECT width/height so the cards stay legible.
  const rows = n <= 40 ? n : Math.max(1, Math.round(Math.sqrt((n * GAP_X) / (ASPECT * GAP_Y))));
  const pos = new Map();
  ordered.forEach((node, i) => {
    const col = Math.floor(i / rows);
    const within = i % rows;
    const row = col % 2 === 0 ? within : rows - 1 - within; // boustrophedon
    pos.set(node.id, { x: col * GAP_X, y: row * GAP_Y });
  });
  return pos;
}

async function elkPositions(nodes, edges) {
  const elkGraph = {
    id: "root",
    layoutOptions: ELK_OPTS,
    children: nodes.map((n) => ({ id: String(n.id), width: NODE_W, height: NODE_H })),
    edges: edges.map((e, i) => ({
      id: `e${i}`,
      sources: [String(e.src)],
      targets: [String(e.dst)],
    })),
  };
  const laid = await elk.layout(elkGraph);
  return new Map(laid.children.map((c) => [Number(c.id), { x: c.x, y: c.y }]));
}

// Returns { nodes, edges } ready for <ReactFlow>.
export async function layoutGraph(graph) {
  const gNodes = graph.nodes || [];
  const gEdges = graph.edges || [];
  const byId = new Map(gNodes.map((n) => [n.id, n]));
  const maxF = maxFlops(gNodes);

  const pos = isChain(gNodes, gEdges)
    ? serpentinePositions(gNodes)
    : await elkPositions(gNodes, gEdges);

  const nodes = gNodes.map((n) => ({
    id: String(n.id),
    type: "task",
    position: pos.get(n.id) || { x: 0, y: 0 },
    data: { ...n, color: nodeColor(n), barFrac: (n.flops || 0) / maxF },
    width: NODE_W,
    height: NODE_H,
  }));

  const edges = gEdges.map((e, i) => {
    const st = edgeStyle(byId.get(e.src), byId.get(e.dst));
    return {
      id: `e${i}`,
      source: String(e.src),
      target: String(e.dst),
      style: {
        stroke: st.stroke,
        strokeWidth: st.width,
        strokeDasharray: st.dashed ? "5 4" : undefined,
        opacity: st.opacity,
      },
      markerEnd: { type: "arrowclosed", color: st.stroke, width: 14, height: 14 },
    };
  });

  return { nodes, edges };
}
