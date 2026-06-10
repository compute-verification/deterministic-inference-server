// Shared presentation helpers for the task-graph viewer.
// Kept framework-free so it is trivially unit-testable.

export const KIND_COLOR = {
  prefill: "#f0883e",      // orange
  decode: "#58a6ff",       // blue
  train_step: "#bc8cff",   // purple
  eval_prefill: "#3fb950", // green
  eval_decode: "#56d4dd",  // cyan
  draft: "#3fb950",        // green
  verify: "#bc8cff",       // purple
};
export const REJECTED = "#f85149"; // red

// A node's fill: rejected drafts go red, everything else by kind. Group nodes
// borrow the colour of the kind they collapse (carried as groupKind).
export function nodeColor(n) {
  const kind = n.kind === "group" ? n.groupKind : n.kind;
  if (kind === "draft" && n.status === "rejected") return REJECTED;
  return KIND_COLOR[kind] || "#8b949e";
}

// Edge appearance, derived from the kinds/status of its endpoints. Mirrors the
// semantics of every scenario: a rejected draft's outgoing edge is dead (red,
// dashed); the K drafts fanning into a verify are faint; the verify->next-draft
// handoff is solid; everything else is a plain dependency edge.
export function edgeStyle(src, dst) {
  // Defensive: a malformed graphs.json could omit an endpoint. Validated graphs
  // never hit this, but a missing node should not blank the whole view.
  if (!src || !dst) return { stroke: "#8b949e", dashed: false, width: 1.3, opacity: 0.7 };
  if (src.kind === "draft" && src.status === "rejected")
    return { stroke: REJECTED, dashed: true, width: 1.2, opacity: 0.6 };
  if (src.kind === "draft" && dst.kind === "verify")
    return { stroke: "#6e7681", dashed: false, width: 1, opacity: 0.32 }; // fan-in
  if (src.kind === "verify" && dst.kind === "draft")
    return { stroke: "#8b949e", dashed: false, width: 1.6, opacity: 0.9 }; // handoff
  return { stroke: "#8b949e", dashed: false, width: 1.3, opacity: 0.7 };
}

// Human-readable FLOPs. Pure function -> unit-tested.
export function fmtFlops(f) {
  if (f >= 1e15) return (f / 1e15).toFixed(2) + " PFLOP";
  if (f >= 1e12) return (f / 1e12).toFixed(2) + " TFLOP";
  if (f >= 1e9) return (f / 1e9).toFixed(2) + " GFLOP";
  if (f >= 1e6) return (f / 1e6).toFixed(2) + " MFLOP";
  return f.toLocaleString() + " FLOP";
}

// Max FLOPs in a graph, for scaling the per-node bar. Never returns 0.
export function maxFlops(nodes) {
  return Math.max(1, ...nodes.map((n) => n.flops || 0));
}

// The four scenarios, in tab order, with the caption shown under each graph.
export const SCENES = [
  { key: "inference", label: "Inference", caption: "Inference (Qwen3-1.7B, real H100) — greedy decode chain" },
  { key: "training", label: "LoRA training", caption: "LoRA fine-tune (Qwen3-1.7B, real H100, toy scale) — train-step spine + real eval branches" },
  { key: "spec", label: "Speculative decoding", caption: "Speculative decoding (real H100) — drafts fan into verify" },
  { key: "coding", label: "Coding agent", caption: "Coding agent (Qwen3-8B, real H100) — parallel reads, plans, codegen; one node = one forward pass (collapsed)" },
];
