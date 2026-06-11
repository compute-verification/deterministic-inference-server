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

// Starting context length of a pass, derived from the trace's exact
// accounting: `attended` is the sum of per-token context lengths, so
// attended = Σ_{j=1..t} (c0 + j)  ⇒  c0 = attended/t − (t+1)/2.
// A prefill (no prior context) yields 0; a decode at position 605 yields 604.
export function ctx0(tokens, attended) {
  const t = tokens || 0;
  if (!t) return 0;
  return Math.max(0, Math.round((attended || 0) / t - (t + 1) / 2));
}

// Human-readable input size for a node: what the pass ingests this step,
// plus the prior context it attends over (when there is one). Bare text —
// used on edge labels, where the arrow already says "into"; inputBadge
// prefixes "in:" for tooltips and root badges.
export function inputText(tokens, attended) {
  const t = tokens || 0;
  if (!t) return "";
  const c = ctx0(t, attended);
  const tok = `${t.toLocaleString()} tok`;
  return c > 0 ? `${tok} + ${c.toLocaleString()} ctx` : tok;
}

// Input text for a collapsed run: total tokens ingested, plus the context
// growth across the run (e.g. a 689-decode stream that started at ctx 604 and
// ended at ctx 1292). A sum of attended over separate passes has no single
// starting context, hence the range. Compact — no separators in the range, or
// "ctx 864→1,212" truncates at label width and loses the end value.
export function groupInputText(d) {
  const t = d.tokens || 0;
  if (!t) return "";
  const tok = `${t.toLocaleString()} tok`;
  const a = d.ctxFirst || 0;
  const b = d.ctxLast || 0;
  if (b > a) return `${tok} · ctx ${a}→${b}`;
  if (a > 0) return `${tok} + ${a} ctx`;
  return tok;
}

// The annotation carried by a node's incoming edge: the input flowing into it.
// A whitelisted node's exact input string is on the graph's whitelist of known
// constants — passing it is free, so no size is shown, just the tag.
export function edgeInputLabel(n) {
  if (!n) return "";
  if (n.whitelisted) return "whitelisted";
  return n.kind === "group" ? groupInputText(n) : inputText(n.tokens, n.attended);
}

// The node-side variant (root badges + tooltips), with the "in:" prefix.
export function inputBadge(n) {
  if (n.whitelisted) return "in: whitelisted (free)";
  const s = n.kind === "group" ? groupInputText(n) : inputText(n.tokens, n.attended);
  return s ? `in: ${s}` : "";
}

// A graphs document may carry its own captions under a non-scene "_meta" key
// (the 4-node tap demo's protocol runs have different provenance than the
// bundled graphs — e.g. its spec rounds are real, not ported). Pure ->
// unit-tested.
export function captionFor(data, scene) {
  return data?._meta?.captions?.[scene.key] || scene.caption;
}

// URL params for embedding. ?scene=<key> picks the initial tab; ?src=<url>
// overrides where graphs.json is fetched from (the 4-node tap demo points
// this at a protocol run's generated graph). Pure function -> unit-tested.
export function viewParams(search, scenes = SCENES) {
  const p = new URLSearchParams(search || "");
  const requested = p.get("scene");
  return {
    scene: scenes.some((s) => s.key === requested) ? requested : scenes[0].key,
    src: p.get("src") || "./graphs.json",
  };
}

// The four scenarios, in tab order, with the caption shown under each graph.
export const SCENES = [
  { key: "inference", label: "Inference", caption: "Inference (Qwen3-1.7B, real H100) — greedy decode chain" },
  { key: "training", label: "LoRA training", caption: "LoRA fine-tune (Qwen3-1.7B, real H100, toy scale) — train-step spine + real eval branches" },
  { key: "spec", label: "Speculative decoding", caption: "Speculative decoding (real H100, ported rounds) — drafts fan into verify" },
  { key: "coding", label: "Coding agent", caption: "Coding agent (Qwen3-8B, real H100) — parallel reads, plans, codegen; one node = one forward pass (collapsed)" },
];
