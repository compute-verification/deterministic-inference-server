import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ReactFlow, Background, Controls } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import TaskNode from "./TaskNode.jsx";
import GroupNode from "./GroupNode.jsx";
import { layoutGraph } from "./layout.js";
import { fmtFlops } from "./graph-model.js";
import { buildDisplayGraph, collapsibleSegmentIds } from "./collapse.js";

const nodeTypes = { task: TaskNode, group: GroupNode };

// Renders one scenario graph. Long linear runs are collapsed into group nodes by
// default (collapse.js); the data underneath stays atomic. Clicking a group
// expands it; clicking any node of an expanded run collapses it again.
export default function GraphView({ graph, caption }) {
  const [expanded, setExpanded] = useState(() => new Set());
  const [laid, setLaid] = useState(null);
  const rfRef = useRef(null);

  // Reset the expand state whenever we switch scenarios.
  useEffect(() => {
    setExpanded(new Set());
  }, [graph]);

  const display = useMemo(() => buildDisplayGraph(graph, expanded), [graph, expanded]);

  useEffect(() => {
    let alive = true;
    setLaid(null);
    layoutGraph(display).then((res) => {
      if (alive) setLaid(res);
    });
    return () => {
      alive = false;
    };
  }, [display]);

  // Re-fit after a layout change (expand/collapse) so the new graph is framed.
  // The initial fit is handled by the `fitView` prop.
  useEffect(() => {
    if (laid && rfRef.current) {
      rfRef.current.fitView({ padding: 0.15, maxZoom: 1.4, duration: 250 });
    }
  }, [laid]);

  const onNodeClick = useCallback((_evt, node) => {
    const d = node.data || {};
    if (d.kind === "group") {
      setExpanded((prev) => new Set(prev).add(d.segId)); // expand
    } else if (d.collapsibleSeg) {
      setExpanded((prev) => {
        const next = new Set(prev);
        next.delete(d.segId); // collapse the run this atomic node belongs to
        return next;
      });
    }
  }, []);

  const collapsibleIds = useMemo(() => collapsibleSegmentIds(graph), [graph]);
  const hasCollapsible = collapsibleIds.length > 0;
  const allExpanded = expanded.size === collapsibleIds.length;

  const atomic = useMemo(() => {
    const nodes = graph.nodes || [];
    return {
      count: nodes.length,
      total: nodes.reduce((a, n) => a + (n.flops || 0), 0),
    };
  }, [graph]);

  return (
    <div className="graph-view">
      <div className="meta">
        <span>
          {caption} · {atomic.count} forward passes · total ≈ {fmtFlops(atomic.total)}
          {display.nodes.length !== atomic.count && ` · showing ${display.nodes.length} nodes`}
        </span>
        {hasCollapsible && (
          <span className="toolbar">
            <button
              className="mini"
              onClick={() => setExpanded(new Set(allExpanded ? [] : collapsibleIds))}
            >
              {allExpanded ? "Collapse all" : "Expand all"}
            </button>
          </span>
        )}
      </div>
      <div className="canvas">
        {laid ? (
          <ReactFlow
            nodes={laid.nodes}
            edges={laid.edges}
            nodeTypes={nodeTypes}
            onNodeClick={onNodeClick}
            onInit={(inst) => { rfRef.current = inst; }}
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
