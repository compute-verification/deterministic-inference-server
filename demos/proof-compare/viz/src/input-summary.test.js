import { describe, it, expect } from "vitest";
import { ctx0, inputText, inputBadge, groupInputText, edgeInputLabel } from "./graph-model.js";

// The node annotation "how big is this pass's input" is DERIVED from the
// trace's exact attention accounting, not stored: attended = Σ_{j=1..t}(c0+j).
describe("ctx0", () => {
  it("a prefill has no prior context", () => {
    // p=600: attended = 600*601/2
    expect(ctx0(600, (600 * 601) / 2)).toBe(0);
  });

  it("a decode's context is everything before its token", () => {
    // the pass consuming token #605 attends 605 keys (604 prior + itself)
    expect(ctx0(1, 605)).toBe(604);
  });

  it("a spec verify recovers the committed context", () => {
    // t=5 positions on top of ctx 20: attended = Σ_{j=1..5}(20+j) = 115
    expect(ctx0(5, 115)).toBe(20);
  });

  it("a packed train step has no prior context", () => {
    // batch=4, seq=64: tokens=256, attended = 4 * 64*65/2
    expect(ctx0(256, 4 * (64 * 65) / 2)).toBe(0);
  });

  it("never goes negative and tolerates zeros", () => {
    expect(ctx0(0, 0)).toBe(0);
    expect(ctx0(10, 0)).toBe(0);
  });
});

describe("inputBadge", () => {
  it("prefill: tokens only", () => {
    expect(inputBadge({ kind: "prefill", tokens: 600, attended: (600 * 601) / 2 }))
      .toBe("in: 600 tok");
  });

  it("decode: one token plus its context", () => {
    expect(inputBadge({ kind: "decode", tokens: 1, attended: 605 })).toBe("in: 1 tok + 604 ctx");
  });

  it("large counts get thousands separators", () => {
    expect(inputBadge({ kind: "decode", tokens: 1, attended: 4001 }))
      .toBe("in: 1 tok + 4,000 ctx");
  });

  it("empty for a zero-token node", () => {
    expect(inputBadge({ kind: "decode", tokens: 0, attended: 0 })).toBe("");
  });

  it("a group gets the range form", () => {
    expect(inputBadge({ kind: "group", tokens: 349, ctxFirst: 864, ctxLast: 1212 }))
      .toBe("in: 349 tok · ctx 864→1212");
  });

  it("a whitelisted input is free — no size shown", () => {
    expect(inputBadge({ kind: "prefill", tokens: 142, attended: 10153, whitelisted: true }))
      .toBe("in: whitelisted (free)");
  });
});

// The annotation rides each node's incoming edge, where the arrow already
// says "into" — so edge text drops the "in:" prefix.
describe("edge input labels", () => {
  it("inputText is the bare summary", () => {
    expect(inputText(600, (600 * 601) / 2)).toBe("600 tok");
    expect(inputText(1, 605)).toBe("1 tok + 604 ctx");
    expect(inputText(0, 0)).toBe("");
  });

  it("group ranges stay compact (no separators) so the end value survives", () => {
    expect(groupInputText({ tokens: 349, ctxFirst: 864, ctxLast: 1212 })).toBe(
      "349 tok · ctx 864→1212",
    );
  });

  it("a flat-context group reads like an atomic pass", () => {
    expect(groupInputText({ tokens: 20, ctxFirst: 30, ctxLast: 30 })).toBe("20 tok + 30 ctx");
    expect(groupInputText({ tokens: 256, ctxFirst: 0, ctxLast: 0 })).toBe("256 tok");
    expect(groupInputText({ tokens: 0 })).toBe("");
  });

  it("edgeInputLabel dispatches on node kind and tolerates a missing node", () => {
    expect(edgeInputLabel({ kind: "decode", tokens: 1, attended: 605 })).toBe("1 tok + 604 ctx");
    expect(edgeInputLabel({ kind: "group", tokens: 349, ctxFirst: 864, ctxLast: 1212 })).toBe(
      "349 tok · ctx 864→1212",
    );
    expect(edgeInputLabel(undefined)).toBe("");
  });

  it("a whitelisted node's edge carries the tag, not a size", () => {
    expect(edgeInputLabel({ kind: "prefill", tokens: 142, attended: 10153, whitelisted: true }))
      .toBe("whitelisted");
  });
});
