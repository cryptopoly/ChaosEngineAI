import { describe, expect, it } from "vitest";
import { measuredBar } from "../../components/LiveProgress";
import type { GenerationProgressSnapshot } from "../../types";
import {
  benchmarkPhaseIds,
  conversionDialogPhase,
  conversionPhaseIds,
  conversionProgressNumbers,
  countedItems,
} from "../jobProgress";

function snapshot(overrides: Partial<GenerationProgressSnapshot> = {}): GenerationProgressSnapshot {
  return {
    kind: "benchmark",
    active: true,
    phase: "evaluating",
    message: "",
    step: 0,
    totalSteps: 0,
    startedAt: 1,
    updatedAt: 1,
    elapsedSeconds: 0,
    runLabel: null,
    ...overrides,
  };
}

describe("benchmark progress", () => {
  it("lists the phases the mode really goes through", () => {
    expect(benchmarkPhaseIds("throughput")).toEqual(["loading", "generating", "saving"]);
    expect(benchmarkPhaseIds(undefined)).toEqual(["loading", "generating", "saving"]);
    // Perplexity and task accuracy generate no reply: they used to show "Generating N tokens".
    expect(benchmarkPhaseIds("perplexity")).not.toContain("generating");
    expect(benchmarkPhaseIds("task_accuracy")).toEqual(["loading", "dataset", "evaluating", "saving"]);
  });

  it("counts items only when the phase reports a total", () => {
    expect(countedItems(snapshot({ step: 12, totalSteps: 100 }))).toEqual({ done: 12, total: 100 });
    expect(countedItems(snapshot({ step: 12, totalSteps: 0 }))).toBeNull();
    expect(countedItems(snapshot({ active: false, step: 12, totalSteps: 100 }))).toBeNull();
    expect(countedItems(null)).toBeNull();
  });

  it("gives the bar a value only for a counted phase", () => {
    expect(measuredBar(snapshot({ step: 25, totalSteps: 100 }))).toEqual({ fillPct: 25, indeterminate: false });
    // A model load or a throughput generation cannot be counted: no percentage, ever.
    expect(measuredBar(snapshot({ phase: "generating" }))).toEqual({ fillPct: 0, indeterminate: true });
    expect(measuredBar(snapshot({ phase: "loading", step: 3 }))).toEqual({ fillPct: 0, indeterminate: true });
    expect(measuredBar(null)).toEqual({ fillPct: 0, indeterminate: true });
    expect(measuredBar(snapshot({ step: 500, totalSteps: 100 })).fillPct).toBe(100);
  });
});

describe("conversion progress", () => {
  it("has a quantizing phase only when the run quantizes", () => {
    expect(conversionPhaseIds(true)).toEqual(["preparing", "loading", "quantizing", "saving"]);
    expect(conversionPhaseIds(false)).toEqual(["preparing", "loading", "saving"]);
    // The quality comparison is a phase of its own only when it was asked for.
    expect(conversionPhaseIds(true, true)).toEqual(["preparing", "loading", "quantizing", "saving", "measuring"]);
    expect(conversionPhaseIds(false, true)).toEqual(["preparing", "loading", "saving", "measuring"]);
  });

  it("shows a Hub download under the loading phase with its file count", () => {
    const downloading = snapshot({ kind: "conversion", phase: "downloading", step: 5, totalSteps: 11 });
    expect(conversionDialogPhase(downloading)?.phase).toBe("loading");
    expect(conversionDialogPhase(downloading)?.totalSteps).toBe(11);
    expect(conversionProgressNumbers(downloading)).toEqual({ files: { done: 5, total: 11 }, writtenGb: null, expectedGb: null });
    expect(conversionDialogPhase(null)).toBeNull();
  });

  it("reports the bytes on disk and keeps the size estimate apart", () => {
    const saving = snapshot({
      kind: "conversion",
      phase: "saving",
      detail: { bytesWritten: 1024 ** 3, expectedBytes: 2 * 1024 ** 3 },
    });
    expect(conversionProgressNumbers(saving)).toEqual({ files: null, writtenGb: 1, expectedGb: 2 });
    // Bytes written against an estimate is not a measured fraction: the bar stays indeterminate.
    expect(measuredBar(saving).indeterminate).toBe(true);
    expect(conversionProgressNumbers(snapshot({ kind: "conversion", phase: "saving" }))).toEqual({ files: null, writtenGb: null, expectedGb: null });
    expect(conversionProgressNumbers(snapshot({ active: false, detail: { bytesWritten: 5 } })).writtenGb).toBeNull();
  });
});
