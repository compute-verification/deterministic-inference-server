import { useEffect, useMemo, useState } from "react";
import { ReactFlow, Background, Controls } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import TaskNode from "./TaskNode.jsx";
import { layoutGraph } from "./layout.js";
import { fmtFlops } from "./graph-model.js";

const nodeTypes = { task: TaskNode };

// Renders one scenario graph: lays it out with elk (async), then draws it with
// React Flow. Pan/zoom/drag come for free.
export default function GraphView({ graph, caption }) {
  const [laid, setLaid] = useState(null);

  useEffect(() => {
    let alive = true;
    setLaid(null);
    layoutGraph(graph).then((res) => {
      if (alive) setLaid(res);
    });
    return () => {
      alive = false;
    };
  }, [graph]);

  const total = useMemo(
    () => (graph.nodes || []).reduce((a, n) => a + (n.flops || 0), 0),
    [graph],
  );

  return (
    <div className="graph-view">
      <div className="meta">
        {caption} · {(graph.nodes || []).length} tasks · {(graph.edges || []).length} edges ·
        total ≈ {fmtFlops(total)}
      </div>
      <div className="canvas">
        {laid ? (
          <ReactFlow
            nodes={laid.nodes}
            edges={laid.edges}
            nodeTypes={nodeTypes}
            fitView
            fitViewOptions={{ padding: 0.15, maxZoom: 1.4 }}
            minZoom={0.02}
            proOptions={{ hideAttribution: true }}
            nodesDraggable={false}
            nodesConnectable={false}
            onlyRenderVisibleElements
            elevateNodesOnSelect={false}
          >
            <Background color="#21262d" gap={22} />
            <Controls showInteractive={false} />
          </ReactFlow>
        ) : (
          <div className="loading">laying out…</div>
        )}
      </div>
    </div>
  );
}
