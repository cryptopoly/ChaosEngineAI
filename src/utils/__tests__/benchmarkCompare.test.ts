import { describe, expect, it } from "vitest";
import { latestRunSpeedComparison } from "../benchmarkCompare";
import type { BenchmarkResult } from "../../types";

function makeRun(overrides: Partial<BenchmarkResult>): BenchmarkResult {
  return {
    id: "bench-1",
    mode: "throughput",
    label: "Run",
    model: "Model",
    modelRef: "org/model",
    backend: "mlx",
    engineLabel: "MLX",
    source: "library",
    measuredAt: "2026-11-18 08:00:00",
    bits: 16,
    fp16Layers: 0,
    cacheStrategy: "native",
    cacheLabel: "Native f16",
    cacheGb: 1,
    baselineCacheGb: 1,
    compression: 1,
    tokS: 100,
    quality: 100,
    responseSeconds: 1,
    loadSeconds: 0,
    totalSeconds: 1,
    promptTokens: 10,
    completionTokens: 100,
    totalTokens: 110,
    contextTokens: 8192,
    maxTokens: 512,
    ...overrides,
  };
}

describe("latestRunSpeedComparison", () => {
  it("compares a speed run with the previous speed run of the same model and the fastest speed run", () => {
    const runs = [
      makeRun({ id: "latest", tokS: 90 }),
      makeRun({ id: "other-model", modelRef: "org/other", tokS: 300 }),
      makeRun({ id: "previous", tokS: 80 }),
    ];
    const { previous, fastest } = latestRunSpeedComparison(runs);
    expect(previous?.id).toBe("previous");
    expect(fastest?.id).toBe("other-model");
  });

  it("gives no speed comparison for a task-accuracy or perplexity run (their tok/s is 0)", () => {
    const runs = [
      makeRun({ id: "mmlu", mode: "task_accuracy", tokS: 0 }),
      makeRun({ id: "speed", tokS: 327.5 }),
    ];
    expect(latestRunSpeedComparison(runs)).toEqual({ previous: null, fastest: null });
    expect(latestRunSpeedComparison([makeRun({ id: "ppl", mode: "perplexity", tokS: 0 }), runs[1]]).previous).toBeNull();
  });

  it("never picks an accuracy run as the previous or the fastest run", () => {
    const runs = [
      makeRun({ id: "latest", tokS: 50 }),
      makeRun({ id: "mmlu", mode: "task_accuracy", tokS: 0 }),
      makeRun({ id: "ppl", mode: "perplexity", tokS: 4000 }),
    ];
    const { previous, fastest } = latestRunSpeedComparison(runs);
    expect(previous).toBeNull();
    expect(fastest?.id).toBe("latest");
  });

  it("treats a run saved without a mode as a speed run, and an empty history as nothing to compare", () => {
    const runs = [makeRun({ id: "latest", tokS: 70 }), makeRun({ id: "old", mode: undefined, tokS: 60 })];
    expect(latestRunSpeedComparison(runs).previous?.id).toBe("old");
    expect(latestRunSpeedComparison([])).toEqual({ previous: null, fastest: null });
  });
});
