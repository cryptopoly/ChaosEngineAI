import { describe, expect, it } from "vitest";
import { accuracyComparable, benchmarkRunsToCsv, latestRunSpeedComparison, taskScoringMethod } from "../benchmarkCompare";
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

describe("task accuracy scoring method", () => {
  const accuracy = (overrides: Partial<Parameters<typeof makeRun>[0]>) =>
    makeRun({ mode: "task_accuracy", tokS: 0, taskName: "mmlu", taskAccuracy: 0.4, ...overrides });

  it("reads the recorded method and takes an older run as generated text", () => {
    expect(taskScoringMethod(accuracy({ taskScoringMethod: "next_token" }))).toBe("next_token");
    expect(taskScoringMethod(accuracy({}))).toBe("generated");
    expect(taskScoringMethod(makeRun({ id: "speed" }))).toBeNull();
  });

  it("compares accuracy only between runs scored the same way", () => {
    const next = accuracy({ id: "new", taskScoringMethod: "next_token" });
    expect(accuracyComparable(next, accuracy({ id: "new-2", taskScoringMethod: "next_token" }))).toBe(true);
    expect(accuracyComparable(next, accuracy({ id: "old" }))).toBe(false);
    expect(accuracyComparable(accuracy({ id: "old-1" }), accuracy({ id: "old-2", taskScoringMethod: "generated" }))).toBe(true);
  });
});

describe("benchmarkRunsToCsv", () => {
  it("writes a header and one line per run, with each run's own measure", () => {
    const csv = benchmarkRunsToCsv([
      makeRun({ id: "speed", tokS: 120.5, label: "Speed run" }),
      makeRun({ id: "ppl", mode: "perplexity", tokS: 5200, perplexity: 21.4, perplexityStdError: 0.6, perplexityDataset: "wikitext-2" }),
      makeRun({ id: "mmlu", mode: "task_accuracy", tokS: 0, taskName: "mmlu", taskAccuracy: 0.41, taskCorrect: 41, taskTotal: 100, taskNumShots: 5 }),
    ]);
    const lines = csv.trimEnd().split("\n");
    expect(lines).toHaveLength(4);
    const header = lines[0].split(",");
    const cell = (line: number, name: string) => lines[line].split(",")[header.indexOf(name)];
    expect(cell(1, "mode")).toBe("throughput");
    expect(cell(1, "tokS")).toBe("120.5");
    expect(cell(2, "tokS")).toBe(""); // eval tokens per second is not a generation speed
    expect(cell(2, "perplexity")).toBe("21.4");
    expect(cell(3, "taskAccuracy")).toBe("0.41");
    expect(cell(3, "taskCorrect")).toBe("41");
    // A run saved before the scoring method was recorded generated its answers as text.
    expect(cell(3, "taskScoringMethod")).toBe("generated");
    expect(cell(1, "taskScoringMethod")).toBe("");
    expect(csv.endsWith("\n")).toBe(true);
  });

  it("quotes commas, quotes and line breaks, and defuses cells that start like a formula", () => {
    const csv = benchmarkRunsToCsv([
      makeRun({ label: 'Model "A", 8K ctx', notes: "line one\nline two", model: "=HYPERLINK(1)" }),
    ]);
    expect(csv).toContain('"Model ""A"", 8K ctx"');
    expect(csv).toContain('"line one\nline two"');
    expect(csv).toContain("'=HYPERLINK(1)");
  });

  it("gives only the header for an empty history", () => {
    expect(benchmarkRunsToCsv([]).trimEnd().split("\n")).toHaveLength(1);
  });
});
