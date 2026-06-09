// Lay out a canonical graph ({nodes, edges}) into React Flow nodes/edges using
// elkjs' layered (Sugiyama) algorithm — proper crossing minimization, which the
// spec-decode fan-in graph in particular needs.
import ELK from "elkjs/lib/elk.bundled.js";
import { edgeStyle, nodeColor, maxFlops } from "./graph-model.js";

const elk = new ELK();

export const NODE_W = 184;
export const NODE_H = 66;

const ELK_OPTS = {
  "elk.algorithm": "layered",
  "elk.direction": "DOWN",
  "elk.layered.spacing.nodeNodeBetweenLayers": "48",
  "elk.spacing.nodeNode": "28",
  "elk.layered.nodePlacement.strategy": "NETWORK_SIMPLEX",
  // Long chains (e.g. the 68-node coding agent) would otherwise be one
  // unreadable ribbon. Wrap them to a viewport-ish aspect ratio so cards stay
  // legible. Harmless for the short graphs (they don't reach the wrap width).
  "elk.layered.wrapping.strategy": "MULTI_EDGE",
  "elk.aspectRatio": "1.7",
};

// Returns { nodes, edges } ready for <ReactFlow>. Async because elk.layout is.
export async function layoutGraph(graph) {
  const gNodes = graph.nodes || [];
  const gEdges = graph.edges || [];
  const byId = new Map(gNodes.map((n) => [n.id, n]));
  const maxF = maxFlops(gNodes);

  const elkGraph = {
    id: "root",
    layoutOptions: ELK_OPTS,
    children: gNodes.map((n) => ({ id: String(n.id), width: NODE_W, height: NODE_H })),
    edges: gEdges.map((e, i) => ({
      id: `e${i}`,
      sources: [String(e.src)],
      targets: [String(e.dst)],
    })),
  };

  const laid = await elk.layout(elkGraph);
  const pos = new Map(laid.children.map((c) => [c.id, { x: c.x, y: c.y }]));

  const nodes = gNodes.map((n) => ({
    id: String(n.id),
    type: "task",
    position: pos.get(String(n.id)) || { x: 0, y: 0 },
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
