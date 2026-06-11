import { describe, it, expect } from "vitest";
import { segmentGraph, buildDisplayGraph, collapsibleSegmentIds } from "./collapse.js";

// Build a simple chain of `n` nodes with a given phase, ids starting at `start`.
function chain(n, phase, start = 0, kind = "decode") {
  const nodes = [];
  const edges = [];
  for (let i = 0; i < n; i++) {
    const id = start + i;
    nodes.push({ id, kind, flops: 10, tokens: 1, payload: { phase } });
    if (i > 0) edges.push({ src: id - 1, dst: id });
  }
  return { nodes, edges };
}

describe("segmentGraph", () => {
  it("collapses a long same-phase chain into one collapsible segment", () => {
    const g = chain(20, "plan");
    const segs = segmentGraph(g.nodes, g.edges);
    expect(segs).toHaveLength(1);
    expect(segs[0].nodeIds).toHaveLength(20);
    expect(segs[0].collapsible).toBe(true);
  });

  it("does not collapse a run below the threshold", () => {
    const g = chain(12, "decode"); // inference-sized
    const segs = segmentGraph(g.nodes, g.edges);
    expect(segs[0].collapsible).toBe(false);
  });

  it("splits runs at a phase change", () => {
    const a = chain(20, "plan", 0);
    const b = chain(20, "codegen", 20);
    const nodes = [...a.nodes, ...b.nodes];
    const edges = [...a.edges, ...b.edges, { src: 19, dst: 20 }];
    const segs = segmentGraph(nodes, edges);
    expect(segs).toHaveLength(2);
    expect(segs.map((s) => s.nodeIds.length)).toEqual([20, 20]);
    expect(segs.every((s) => s.collapsible)).toBe(true);
  });

  it("breaks a run at a fan-in node (e.g. spec-decode verify)", () => {
    // 4 drafts fan into a verify; verify must be its own segment.
    const nodes = [
      { id: 0, kind: "draft", flops: 1, payload: {} },
      { id: 1, kind: "draft", flops: 1, payload: {} },
      { id: 2, kind: "draft", flops: 1, payload: {} },
      { id: 3, kind: "draft", flops: 1, payload: {} },
      { id: 4, kind: "verify", flops: 9, payload: {} },
    ];
    const edges = [
      { src: 0, dst: 1 }, { src: 1, dst: 2 }, { src: 2, dst: 3 },
      { src: 0, dst: 4 }, { src: 1, dst: 4 }, { src: 2, dst: 4 }, { src: 3, dst: 4 },
    ];
    const segs = segmentGraph(nodes, edges);
    const verifySeg = segs.find((s) => s.kind === "verify");
    expect(verifySeg.nodeIds).toEqual([4]);
    // none collapsible at default threshold (drafts run is length 4)
    expect(segs.every((s) => !s.collapsible)).toBe(true);
  });

  it("breaks a run after a fan-out node", () => {
    // node 0 fans out to 1 and 2; the run cannot continue through 0.
    const nodes = [0, 1, 2].map((id) => ({ id, kind: "decode", flops: 1, payload: { phase: "p" } }));
    const edges = [{ src: 0, dst: 1 }, { src: 0, dst: 2 }];
    const segs = segmentGraph(nodes, edges, { min: 1 });
    expect(segs.find((s) => s.nodeIds.includes(0)).nodeIds).toEqual([0]);
  });
});

describe("buildDisplayGraph", () => {
  // Realistic shape: a turn's prefill and its decodes share the SAME phase
  // (the turn label), so the test guards that kind-in-key keeps them apart.
  // prefill(plan) -> [20 decode "plan"] -> prefill(codegen) -> [20 decode "codegen"]
  function codingLike() {
    const nodes = [{ id: 0, kind: "prefill", flops: 100, tokens: 40, payload: { phase: "plan" } }];
    const edges = [];
    let prev = 0;
    const addRun = (n, phase, startId) => {
      for (let i = 0; i < n; i++) {
        const id = startId + i;
        nodes.push({ id, kind: "decode", flops: 10, tokens: 1, payload: { phase } });
        edges.push({ src: prev, dst: id });
        prev = id;
      }
    };
    addRun(20, "plan", 1);
    nodes.push({ id: 21, kind: "prefill", flops: 100, tokens: 5, payload: { phase: "codegen" } });
    edges.push({ src: prev, dst: 21 });
    prev = 21;
    addRun(20, "codegen", 22);
    return { nodes, edges };
  }

  it("keeps a turn's prefill separate from its same-phase decode run", () => {
    const g = codingLike();
    const d = buildDisplayGraph(g);
    const planGroup = d.nodes.find((n) => n.label === "plan" && n.kind === "group");
    expect(planGroup.count).toBe(20); // the prefill is NOT absorbed (would be 21)
    expect(planGroup.groupKind).toBe("decode");
    expect(d.nodes.filter((n) => n.kind === "prefill")).toHaveLength(2);
  });

  it("collapses runs but keeps prefills atomic", () => {
    const g = codingLike();
    const d = buildDisplayGraph(g);
    const kinds = d.nodes.map((n) => n.kind);
    expect(kinds.filter((k) => k === "group")).toHaveLength(2); // plan + codegen
    expect(kinds.filter((k) => k === "prefill")).toHaveLength(2);
    expect(d.nodes).toHaveLength(4); // 2 prefills + 2 groups
  });

  it("sums flops/tokens/count on the group node", () => {
    const g = codingLike();
    const d = buildDisplayGraph(g);
    const plan = d.nodes.find((n) => n.label === "plan");
    expect(plan.count).toBe(20);
    expect(plan.flops).toBe(200);
    expect(plan.tokens).toBe(20);
  });

  it("remaps edges to display ids with no self-loops and no duplicates", () => {
    const g = codingLike();
    const d = buildDisplayGraph(g);
    // prefill0 -> planGroup -> prefill21 -> codegenGroup  == 3 edges
    expect(d.edges).toHaveLength(3);
    expect(d.edges.some((e) => e.src === e.dst)).toBe(false);
    const keys = d.edges.map((e) => e.src + "->" + e.dst);
    expect(new Set(keys).size).toBe(keys.length);
    // every endpoint exists as a display node
    const ids = new Set(d.nodes.map((n) => n.id));
    for (const e of d.edges) {
      expect(ids.has(e.src)).toBe(true);
      expect(ids.has(e.dst)).toBe(true);
    }
  });

  it("expands one segment back to atomic nodes", () => {
    const g = codingLike();
    const segs = segmentGraph(g.nodes, g.edges);
    // the collapsible decode run whose first node is id 1 (the "plan" decodes)
    const planSeg = segs.find((s) => s.collapsible && s.nodeIds[0] === 1);
    const d = buildDisplayGraph(g, new Set([planSeg.id]));
    // plan run is now 20 atomic decode nodes; codegen still a group
    expect(d.nodes.filter((n) => n.kind === "decode")).toHaveLength(20);
    expect(d.nodes.filter((n) => n.kind === "group")).toHaveLength(1);
    // expanded atomic nodes carry collapsibleSeg so a click can re-collapse
    expect(d.nodes.filter((n) => n.collapsibleSeg).length).toBe(20);
  });

  it("expand-all then collapse is identical to the default collapsed view", () => {
    const g = codingLike();
    const collapsedDefault = buildDisplayGraph(g, new Set());
    const collapsedAgain = buildDisplayGraph(g, new Set()); // after toggling off
    expect(collapsedAgain.nodes.map((n) => n.id)).toEqual(collapsedDefault.nodes.map((n) => n.id));
    expect(collapsedAgain.edges).toEqual(collapsedDefault.edges);
  });

  it("a single root remains after collapsing", () => {
    const g = codingLike();
    const d = buildDisplayGraph(g);
    const hasIncoming = new Set(d.edges.map((e) => e.dst));
    const roots = d.nodes.filter((n) => !hasIncoming.has(n.id));
    expect(roots).toHaveLength(1);
    expect(roots[0].kind).toBe("prefill");
  });

  it("collapsibleSegmentIds lists exactly the long runs", () => {
    const g = codingLike();
    expect(collapsibleSegmentIds(g)).toHaveLength(2);
  });
});

describe("whitelisted nodes in a collapsed run", () => {
  it("are excluded from the group's displayed input sums (flops still count)", () => {
    const g = chain(20, "plan");
    g.nodes.forEach((n) => { n.attended = 100; });
    g.nodes[0].whitelisted = true; // its input was a known constant — free
    const { nodes } = buildDisplayGraph(g);
    expect(nodes).toHaveLength(1);
    expect(nodes[0].tokens).toBe(19);       // 20 nodes − 1 whitelisted
    expect(nodes[0].attended).toBe(1900);   // 2000 − 100
    expect(nodes[0].flops).toBe(200);       // compute is NOT free
  });
});
