import { afterEach, beforeAll, beforeEach, describe, expect, it } from "vitest";

beforeAll(() => {
  if (typeof globalThis.window !== "undefined") return;
  const store = new Map<string, string>();
  const localStorage = {
    getItem: (k: string) => (store.has(k) ? store.get(k)! : null),
    setItem: (k: string, v: string) => { store.set(k, String(v)); },
    removeItem: (k: string) => { store.delete(k); },
    clear: () => { store.clear(); },
    get length() { return store.size; },
    key: (i: number) => Array.from(store.keys())[i] ?? null,
  };
  (globalThis as { window?: { localStorage: typeof localStorage } }).window = { localStorage };
});

import { readKvStrategyOverride, writeKvStrategyOverride } from "../kvStrategyOverride";

describe("kvStrategyOverride storage", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });
  afterEach(() => {
    window.localStorage.clear();
  });

  it("returns null when nothing is stored", () => {
    expect(readKvStrategyOverride("s1")).toBeNull();
  });

  it("returns null for null/undefined session id", () => {
    expect(readKvStrategyOverride(null)).toBeNull();
    expect(readKvStrategyOverride(undefined)).toBeNull();
  });

  it("round-trips a typical override", () => {
    writeKvStrategyOverride("s1", { strategy: "turboquant", bits: 4 });
    expect(readKvStrategyOverride("s1")).toEqual({ strategy: "turboquant", bits: 4 });
  });

  it("clears storage when given null", () => {
    writeKvStrategyOverride("s1", { strategy: "chaosengine", bits: 8 });
    writeKvStrategyOverride("s1", null);
    expect(readKvStrategyOverride("s1")).toBeNull();
    expect(window.localStorage.getItem("chat.kvStrategy.s1")).toBeNull();
  });

  it("rejects malformed stored values", () => {
    window.localStorage.setItem("chat.kvStrategy.s1", JSON.stringify({ strategy: 7, bits: 4 }));
    expect(readKvStrategyOverride("s1")).toBeNull();
  });

  it("rejects entries missing required fields", () => {
    window.localStorage.setItem("chat.kvStrategy.s1", JSON.stringify({ strategy: "tq" }));
    expect(readKvStrategyOverride("s1")).toBeNull();
  });

  it("returns null for malformed JSON", () => {
    window.localStorage.setItem("chat.kvStrategy.s1", "{not json");
    expect(readKvStrategyOverride("s1")).toBeNull();
  });

  it("scopes overrides per session", () => {
    writeKvStrategyOverride("s1", { strategy: "chaosengine", bits: 8 });
    writeKvStrategyOverride("s2", { strategy: "turboquant", bits: 4 });
    expect(readKvStrategyOverride("s1")).toEqual({ strategy: "chaosengine", bits: 8 });
    expect(readKvStrategyOverride("s2")).toEqual({ strategy: "turboquant", bits: 4 });
  });
});
