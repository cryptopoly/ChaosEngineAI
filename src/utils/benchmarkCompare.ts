import type { BenchmarkResult } from "../types";

/**
 * Runs the latest run's speed may be compared with: the previous speed run of the
 * same model, and the fastest speed run. A perplexity or task-accuracy run has no
 * tok/s (it is stored as 0), so it is never compared and never the comparison.
 */
export function latestRunSpeedComparison(benchmarks: BenchmarkResult[]): {
  previous: BenchmarkResult | null;
  fastest: BenchmarkResult | null;
} {
  const latest = benchmarks[0] ?? null;
  const isSpeedRun = (run: BenchmarkResult) => (run.mode ?? "throughput") === "throughput";
  if (!latest || !isSpeedRun(latest)) return { previous: null, fastest: null };
  const speedRuns = benchmarks.filter(isSpeedRun);
  const previous = speedRuns.find((run) => run.id !== latest.id && run.modelRef === latest.modelRef) ?? null;
  const fastest = [...speedRuns].sort((left, right) => right.tokS - left.tokS)[0] ?? null;
  return { previous, fastest };
}
