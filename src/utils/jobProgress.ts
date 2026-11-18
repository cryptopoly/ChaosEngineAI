/**
 * What the Benchmarks and Conversion progress dialogs show, worked out from the
 * backend's own progress snapshot. Nothing here is timed or estimated, except
 * ``expectedGb``, which is the converter's size estimate and is labelled as one.
 */
import type { BenchmarkMode, GenerationProgressSnapshot } from "../types";

/** Phase ids of a benchmark run, in order, for the mode it runs in. */
export function benchmarkPhaseIds(mode: BenchmarkMode | undefined): string[] {
  if (mode === "perplexity" || mode === "task_accuracy") return ["loading", "dataset", "evaluating", "saving"];
  return ["loading", "generating", "saving"];
}

/** Items the running phase has counted, or null when it does not count. */
export function countedItems(
  progress: GenerationProgressSnapshot | null | undefined,
): { done: number; total: number } | null {
  if (!progress?.active || !(progress.totalSteps > 0)) return null;
  return { done: Math.min(progress.step, progress.totalSteps), total: progress.totalSteps };
}

/** Phase ids of a conversion, in order. A run without quantization has no quantizing phase. */
export function conversionPhaseIds(quantize: boolean): string[] {
  return quantize ? ["preparing", "loading", "quantizing", "saving"] : ["preparing", "loading", "saving"];
}

/** The dialog has no phase of its own for a Hub download: it happens while the source is read. */
export function conversionDialogPhase(progress: GenerationProgressSnapshot | null | undefined): GenerationProgressSnapshot | null {
  if (!progress) return null;
  return progress.phase === "downloading" ? { ...progress, phase: "loading" } : progress;
}

const GB = 1024 ** 3;

export interface ConversionProgressNumbers {
  /** Files of a Hub source downloaded so far (a real count), while the download runs. */
  files: { done: number; total: number } | null;
  /** GB in the output folder, measured on disk. */
  writtenGb: number | null;
  /** The converter's size estimate for the finished model. An estimate, never a measurement. */
  expectedGb: number | null;
}

export function conversionProgressNumbers(
  progress: GenerationProgressSnapshot | null | undefined,
): ConversionProgressNumbers {
  const empty: ConversionProgressNumbers = { files: null, writtenGb: null, expectedGb: null };
  if (!progress?.active) return empty;
  if (progress.phase === "downloading") return { ...empty, files: countedItems(progress) };
  const written = progress.detail?.bytesWritten;
  const expected = progress.detail?.expectedBytes;
  return {
    files: null,
    writtenGb: typeof written === "number" && written > 0 ? written / GB : null,
    expectedGb: typeof expected === "number" && expected > 0 ? expected / GB : null,
  };
}
