import { describe, it, expect } from "vitest";
import { viewParams, SCENES } from "./graph-model.js";

// Embedding contract: the 4-node tap demo iframes the viz with
// ?scene=<key>&src=<graphs url>. Bad input must never break the app.
describe("viewParams", () => {
  it("defaults to the first scene and the bundled graphs.json", () => {
    expect(viewParams("")).toEqual({ scene: SCENES[0].key, src: "./graphs.json" });
    expect(viewParams(undefined)).toEqual({ scene: SCENES[0].key, src: "./graphs.json" });
  });

  it("picks the requested scene when it exists", () => {
    expect(viewParams("?scene=coding").scene).toBe("coding");
    expect(viewParams("?scene=spec&src=x").scene).toBe("spec");
  });

  it("falls back to the first scene for unknown keys", () => {
    expect(viewParams("?scene=mining").scene).toBe(SCENES[0].key);
  });

  it("passes the src through, urldecoded", () => {
    expect(viewParams("?src=..%2Fdata%2Fprotocol-graphs.json").src)
      .toBe("../data/protocol-graphs.json");
    expect(viewParams("?scene=coding&src=../api/graph%3Fid%3D7").src)
      .toBe("../api/graph?id=7");
  });
});
