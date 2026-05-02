import { afterEach, beforeAll, beforeEach, describe, expect, it } from "vitest";

// vitest config uses environment: "node" by default — install a minimal
// in-memory localStorage shim on the global object so the storage helpers
// have something to write into. The shim mirrors the contract the helpers
// rely on (getItem / setItem / removeItem / clear).
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

import {
  readSamplerOverrides,
  samplerPayload,
  writeSamplerOverrides,
} from "../samplerOverrides";

describe("samplerOverrides storage", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  afterEach(() => {
    window.localStorage.clear();
  });

  it("returns empty object when nothing is stored", () => {
    expect(readSamplerOverrides("s1")).toEqual({});
  });

  it("round-trips a typical override blob", () => {
    writeSamplerOverrides("s1", {
      topP: 0.9,
      topK: 40,
      minP: 0.05,
      repeatPenalty: 1.1,
      seed: 42,
      mirostatMode: 2,
      mirostatTau: 5.0,
      mirostatEta: 0.1,
    });
    expect(readSamplerOverrides("s1")).toEqual({
      topP: 0.9,
      topK: 40,
      minP: 0.05,
      repeatPenalty: 1.1,
      seed: 42,
      mirostatMode: 2,
      mirostatTau: 5.0,
      mirostatEta: 0.1,
    });
  });

  it("clears storage when given an empty object", () => {
    writeSamplerOverrides("s1", { topP: 0.9 });
    expect(readSamplerOverrides("s1")).toEqual({ topP: 0.9 });
    writeSamplerOverrides("s1", {});
    expect(readSamplerOverrides("s1")).toEqual({});
    expect(window.localStorage.getItem("chat.samplers.s1")).toBeNull();
  });

  it("ignores invalid stored values", () => {
    window.localStorage.setItem(
      "chat.samplers.s1",
      JSON.stringify({ topP: "not a number", topK: NaN, mirostatMode: 9 }),
    );
    expect(readSamplerOverrides("s1")).toEqual({});
  });

  it("returns empty object for malformed JSON", () => {
    window.localStorage.setItem("chat.samplers.s1", "{not json");
    expect(readSamplerOverrides("s1")).toEqual({});
  });

  it("scopes overrides per session", () => {
    writeSamplerOverrides("s1", { topP: 0.5 });
    writeSamplerOverrides("s2", { topP: 0.9 });
    expect(readSamplerOverrides("s1")).toEqual({ topP: 0.5 });
    expect(readSamplerOverrides("s2")).toEqual({ topP: 0.9 });
  });
});

describe("samplerPayload projection", () => {
  it("returns empty object when no overrides set", () => {
    expect(samplerPayload({})).toEqual({});
  });

  it("preserves the GeneratePayload field names", () => {
    expect(
      samplerPayload({
        topP: 0.9,
        topK: 40,
        minP: 0.05,
        repeatPenalty: 1.1,
        seed: 42,
        mirostatMode: 2,
        mirostatTau: 5.0,
        mirostatEta: 0.1,
      }),
    ).toEqual({
      topP: 0.9,
      topK: 40,
      minP: 0.05,
      repeatPenalty: 1.1,
      seed: 42,
      mirostatMode: 2,
      mirostatTau: 5.0,
      mirostatEta: 0.1,
    });
  });

  it("skips null overrides", () => {
    expect(samplerPayload({ topP: 0.9, topK: null, seed: null })).toEqual({ topP: 0.9 });
  });

  it("parses jsonSchemaText into jsonSchema when valid", () => {
    const schemaText = '{"type":"object","properties":{"answer":{"type":"string"}}}';
    expect(samplerPayload({ jsonSchemaText: schemaText })).toEqual({
      jsonSchema: { type: "object", properties: { answer: { type: "string" } } },
    });
  });

  it("drops malformed jsonSchemaText silently", () => {
    expect(samplerPayload({ jsonSchemaText: '{not valid json' })).toEqual({});
  });

  it("rejects jsonSchemaText that parses to an array", () => {
    expect(samplerPayload({ jsonSchemaText: '[1,2,3]' })).toEqual({});
  });

  it("ignores empty jsonSchemaText", () => {
    expect(samplerPayload({ jsonSchemaText: "   " })).toEqual({});
  });
});

describe("samplerOverrides jsonSchemaText round-trip", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("preserves raw schema text across read/write", () => {
    const schemaText = '{\n  "type": "object"\n}';
    writeSamplerOverrides("s1", { jsonSchemaText: schemaText });
    expect(readSamplerOverrides("s1").jsonSchemaText).toBe(schemaText);
  });

  it("preserves mid-type unparseable schema text", () => {
    const schemaText = '{ "type": "obj';
    writeSamplerOverrides("s1", { jsonSchemaText: schemaText });
    expect(readSamplerOverrides("s1").jsonSchemaText).toBe(schemaText);
  });

  it("treats empty schema text as no override", () => {
    writeSamplerOverrides("s1", { jsonSchemaText: "" });
    expect(readSamplerOverrides("s1")).toEqual({});
  });
});
