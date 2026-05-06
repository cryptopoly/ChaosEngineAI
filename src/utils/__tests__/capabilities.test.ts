import { describe, it, expect } from "vitest";
import { emptyCapabilities, resolveCapabilities } from "../capabilities";

describe("resolveCapabilities", () => {
  it("returns an empty blob for null inputs", () => {
    const caps = resolveCapabilities(null, null);
    expect(caps).toEqual(emptyCapabilities());
  });

  it("maps catalog tags to typed flags", () => {
    const caps = resolveCapabilities("any/model", ["vision", "tool-use", "reasoning"]);
    expect(caps.supportsVision).toBe(true);
    expect(caps.supportsTools).toBe(true);
    expect(caps.supportsReasoning).toBe(true);
    expect(caps.supportsCoding).toBe(false);
    expect(caps.tags).toEqual(["reasoning", "tool-use", "vision"]);
  });

  it("treats catalog tags as authoritative when present", () => {
    // Heuristic would set supportsCoding (ref contains "coder"); catalog overrides.
    const caps = resolveCapabilities("any/coder-7b", ["reasoning"]);
    expect(caps.supportsReasoning).toBe(true);
    expect(caps.supportsCoding).toBe(false);
  });

  it("falls back to ref-name heuristics when no catalog tags", () => {
    const caps = resolveCapabilities("Qwen3-VL-Instruct-7B", null);
    // Heuristic catches vision via "vl" needle, reasoning via "qwen3", and
    // tool-use via "instruct".
    expect(caps.supportsVision).toBe(true);
    expect(caps.supportsReasoning).toBe(true);
    expect(caps.supportsTools).toBe(true);
  });

  it("ignores unknown tags but preserves them", () => {
    const caps = resolveCapabilities("any/model", ["mystery", "vision"]);
    expect(caps.supportsVision).toBe(true);
    expect(caps.tags).toContain("mystery");
    expect(caps.tags).toContain("vision");
  });

  it("normalises and deduplicates tag casing", () => {
    const caps = resolveCapabilities("any/model", ["Vision", "VISION", "vision"]);
    expect(caps.tags).toEqual(["vision"]);
    expect(caps.supportsVision).toBe(true);
  });

  it("returns empty when ref is unknown and no tags", () => {
    const caps = resolveCapabilities("unknown/random-name", null);
    expect(caps.tags).toEqual([]);
    expect(caps.supportsVision).toBe(false);
    expect(caps.supportsTools).toBe(false);
  });
});
