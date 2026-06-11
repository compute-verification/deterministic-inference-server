import { describe, it, expect } from "vitest";
import { layeredPositions, layoutStrategy, layoutGraph, ELK_MAX_NODES } from "./layout.js";

// Which layouter a graph gets. The big-DAG cutoff exists because elk grinds
// for minutes (blocking the tab) on multi-thousand-node graphs.
describe("layoutStrategy", () => {
  it("routes chains to the serpentine grid", () => {
    const ns = [{ _id: "0" }, { _id: "1" }];
    expect(layoutStrategy(ns, [{ src: "0", dst: "1" }])).toBe("serpentine");
  });

  it("routes small branching DAGs to elk", () => {
    const ns = [{ _id: "0" }, { _id: "1" }, { _id: "2" }];
    const es = [{ src: "0", dst: "1" }, { src: "0", dst: "2" }];
    expect(layoutStrategy(ns, es)).toBe("elk");
  });

  it("routes big branching DAGs straight to the iterative layouter", () => {
    const n = ELK_MAX_NODES + 1;
    const ns = Array.from({ length: n }, (_, i) => ({ _id: String(i) }));
    // a fan-out at the root makes it non-chain
    const es = [{ src: "0", dst: "1" }, { src: "0", dst: "2" }];
    for (let i = 2; i < n - 1; i++) es.push({ src: String(i), dst: String(i + 1) });
    expect(layoutStrategy(ns, es)).toBe("layered");
  });
});

// The longest-path fallback used for big branching graphs (where elk overflows).
describe("layeredPositions", () => {
  function nodes(ids) {
    return ids.map((id) => ({ _id: String(id) }));
  }

  it("places a chain in a single column, increasing depth", () => {
    const ns = nodes([0, 1, 2]);
    const es = [{ src: "0", dst: "1" }, { src: "1", dst: "2" }];
    const pos = layeredPositions(ns, es);
    expect(pos.get("0").x).toBe(pos.get("1").x);
    expect(pos.get("2").x).toBe(pos.get("0").x);
    expect(pos.get("1").y).toBeGreaterThan(pos.get("0").y);
    expect(pos.get("2").y).toBeGreaterThan(pos.get("1").y);
  });

  it("places parallel branches side by side at the same depth", () => {
    // 0 -> {1, 2} -> 3  (a diamond)
    const ns = nodes([0, 1, 2, 3]);
    const es = [
      { src: "0", dst: "1" }, { src: "0", dst: "2" },
      { src: "1", dst: "3" }, { src: "2", dst: "3" },
    ];
    const pos = layeredPositions(ns, es);
    // 1 and 2 are parallel: same y, different x
    expect(pos.get("1").y).toBe(pos.get("2").y);
    expect(pos.get("1").x).not.toBe(pos.get("2").x);
    // the merge sits below both
    expect(pos.get("3").y).toBeGreaterThan(pos.get("1").y);
  });

  it("uses the longest path for depth at a fan-in of unequal branch lengths", () => {
    // 0 -> 1 -> 2 -> 4 ; 0 -> 3 -> 4   (one branch length 2, other length 1)
    const ns = nodes([0, 1, 2, 3, 4]);
    const es = [
      { src: "0", dst: "1" }, { src: "1", dst: "2" }, { src: "2", dst: "4" },
      { src: "0", dst: "3" }, { src: "3", dst: "4" },
    ];
    const pos = layeredPositions(ns, es);
    // node 4's depth follows the LONGER branch (0->1->2->4 = depth 3)
    expect(pos.get("4").y).toBeGreaterThan(pos.get("2").y);
    expect(pos.get("3").y).toBeLessThan(pos.get("4").y);
  });

  it("places every node", () => {
    const ns = nodes([0, 1, 2, 3]);
    const es = [{ src: "0", dst: "1" }, { src: "0", dst: "2" }, { src: "1", dst: "3" }, { src: "2", dst: "3" }];
    const pos = layeredPositions(ns, es);
    for (const n of ns) {
      expect(Number.isFinite(pos.get(n._id).x)).toBe(true);
      expect(Number.isFinite(pos.get(n._id).y)).toBe(true);
    }
  });
});

// Input-size annotations ride incoming edges; roots (no incoming edge) are
// flagged so the node renders the badge itself.
describe("layoutGraph input annotations", () => {
  it("labels a node's incoming edge with its input; the root gets isRoot", async () => {
    const graph = {
      nodes: [
        { id: 0, kind: "prefill", tokens: 600, attended: (600 * 601) / 2 },
        { id: 1, kind: "decode", tokens: 1, attended: 605 },
      ],
      edges: [{ src: 0, dst: 1 }],
    };
    const { nodes: ns, edges: es } = await layoutGraph(graph);
    expect(ns[0].data.isRoot).toBe(true);
    expect(ns[1].data.isRoot).toBe(false);
    expect(es[0].label).toBe("1 tok + 604 ctx");
  });

  it("puts a fan-in's total input on exactly one incoming edge", async () => {
    // diamond: 0 -> {1, 2} -> 3; repeating the total on both arrows into 3
    // would read as two separate inputs
    const graph = {
      nodes: [
        { id: 0, kind: "prefill", tokens: 10, attended: 55 },
        { id: 1, kind: "decode", tokens: 1, attended: 11 },
        { id: 2, kind: "decode", tokens: 1, attended: 11 },
        { id: 3, kind: "verify", tokens: 5, attended: 115 },
      ],
      edges: [
        { src: 0, dst: 1 }, { src: 0, dst: 2 },
        { src: 1, dst: 3 }, { src: 2, dst: 3 },
      ],
    };
    const { edges: es } = await layoutGraph(graph);
    const into3 = es.filter((e) => e.target === "3" && e.label);
    expect(into3.length).toBe(1);
    expect(into3[0].label).toBe("5 tok + 20 ctx");
  });
});

describe("whitelisted edge labels", () => {
  it("a whitelisted dst gets the italic tag instead of a size", async () => {
    const graph = {
      whitelist: ["The task statement"],
      nodes: [
        { id: 0, kind: "prefill", tokens: 600, attended: (600 * 601) / 2 },
        { id: 1, kind: "prefill", tokens: 142, attended: 10153, whitelisted: true },
      ],
      edges: [{ src: 0, dst: 1 }],
    };
    const { edges: es } = await layoutGraph(graph);
    expect(es[0].label).toBe("whitelisted");
    expect(es[0].labelStyle.fontStyle).toBe("italic");
  });
});
