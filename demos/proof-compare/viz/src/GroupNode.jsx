import { Handle, Position } from "@xyflow/react";
import { fmtFlops, inputBadge } from "./graph-model.js";

// A collapsed run of N atomic forward passes (e.g. "write p_less.py" = 1200
// decode nodes). Shows the summed cost; click to expand into the atomic nodes.
// The underlying data is unchanged — this is purely a display node. The
// input-size annotation (total tokens + ctx growth across the run) rides the
// incoming edge (layout.js); a root group floats it above the card instead.
export default function GroupNode({ data }) {
  const tip = `${data.count} × ${data.groupKind} forward passes\n`
    + `${inputBadge(data)}\n`
    + `summed cost: ${fmtFlops(data.flops || 0)}\n`
    + `click to expand`;
  return (
    <>
      {data.isRoot && <div className="task-in-root">{inputBadge(data)}</div>}
      <div className="task-node group-node" style={{ borderColor: data.color }} title={tip}>
        <Handle type="target" position={Position.Top} />
        <div className="task-kind" style={{ color: data.color }}>
          ▸ {data.groupKind} ×{data.count}
        </div>
        <div className="task-label">{(data.label || data.groupKind).slice(0, 28)}</div>
        <div className="task-flops">{fmtFlops(data.flops || 0)}</div>
        <div className="task-bar-track">
          <div
            className="task-bar-fill"
            style={{ width: `${Math.max(2, (data.barFrac || 0) * 100)}%`, background: data.color }}
          />
        </div>
        <Handle type="source" position={Position.Bottom} />
      </div>
    </>
  );
}
