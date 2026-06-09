import { Handle, Position } from "@xyflow/react";
import { fmtFlops } from "./graph-model.js";

// One task = one forward pass. The card mirrors the original renderer: kind line
// (with a ✗ for a rejected draft), label, formatted FLOPs, and a proportional
// FLOPs bar so relative cost is readable at a glance. Hover shows the payload.
export default function TaskNode({ data }) {
  const mark = data.status === "rejected" ? "✗ " : "";
  const tip = data.collapsibleSeg ? buildTip(data) + "\n(click to collapse this run)" : buildTip(data);
  const cls = "task-node" + (data.collapsibleSeg ? " collapsible" : "");
  return (
    <div className={cls} style={{ borderColor: data.color }} title={tip}>
      <Handle type="target" position={Position.Top} />
      <div className="task-kind" style={{ color: data.color }}>
        {mark}
        {data.kind}
      </div>
      <div className="task-label">{(data.label || data.kind).slice(0, 28)}</div>
      <div className="task-flops">{fmtFlops(data.flops || 0)}</div>
      <div className="task-bar-track">
        <div
          className="task-bar-fill"
          style={{ width: `${Math.max(2, (data.barFrac || 0) * 100)}%`, background: data.color }}
        />
      </div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

function buildTip(d) {
  const lines = [
    `${d.kind}${d.status ? " · " + d.status : ""}`,
    `cost: ${fmtFlops(d.flops || 0)} (${d.tokens} tok)`,
  ];
  const p = d.payload || {};
  for (const [k, v] of Object.entries(p)) {
    const val = typeof v === "string" ? v.slice(0, 120) : v;
    lines.push(`${k}: ${val}`);
  }
  return lines.join("\n");
}
