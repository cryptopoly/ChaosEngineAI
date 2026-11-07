import { describe, expect, it } from "vitest";
import { tabKeyTarget } from "../keyboard";

describe("tabKeyTarget", () => {
  it("moves with the arrows and wraps at both ends", () => {
    expect(tabKeyTarget(0, 3, "ArrowRight")).toBe(1);
    expect(tabKeyTarget(2, 3, "ArrowRight")).toBe(0);
    expect(tabKeyTarget(0, 3, "ArrowLeft")).toBe(2);
    expect(tabKeyTarget(1, 3, "ArrowUp")).toBe(0);
  });

  it("jumps with Home and End", () => {
    expect(tabKeyTarget(1, 4, "Home")).toBe(0);
    expect(tabKeyTarget(1, 4, "End")).toBe(3);
  });

  it("ignores other keys and empty lists", () => {
    expect(tabKeyTarget(0, 3, "a")).toBeNull();
    expect(tabKeyTarget(0, 3, "Tab")).toBeNull();
    expect(tabKeyTarget(0, 0, "ArrowRight")).toBeNull();
  });
});
