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

const CSV_COLUMNS: Array<[string, (run: BenchmarkResult) => string | number | null | undefined]> = [
  ["measuredAt", (run) => run.measuredAt],
  ["label", (run) => run.label],
  ["model", (run) => run.model],
  ["modelRef", (run) => run.modelRef],
  ["backend", (run) => run.backend],
  ["engine", (run) => run.engineLabel],
  ["mode", (run) => run.mode ?? "throughput"],
  ["tokS", (run) => ((run.mode ?? "throughput") === "throughput" ? run.tokS : null)],
  ["perplexity", (run) => run.perplexity],
  ["perplexityStdError", (run) => run.perplexityStdError],
  ["perplexityDataset", (run) => run.perplexityDataset],
  ["taskName", (run) => run.taskName],
  ["taskAccuracy", (run) => run.taskAccuracy],
  ["taskCorrect", (run) => run.taskCorrect],
  ["taskTotal", (run) => run.taskTotal],
  ["taskNumShots", (run) => run.taskNumShots],
  ["responseSeconds", (run) => run.responseSeconds],
  ["loadSeconds", (run) => run.loadSeconds],
  ["promptTokens", (run) => run.promptTokens],
  ["completionTokens", (run) => run.completionTokens],
  ["contextTokens", (run) => run.contextTokens],
  ["maxTokens", (run) => run.maxTokens],
  ["cacheStrategy", (run) => run.cacheStrategy],
  ["cacheLabel", (run) => run.cacheLabel],
  ["cacheGb", (run) => run.cacheGb],
  ["notes", (run) => run.notes],
];

function csvCell(value: string | number | null | undefined): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : "";
  // A text cell that starts like a formula is quoted off, so a spreadsheet shows it and does not run it.
  const text = /^[=+\-@\t\r]/.test(value) ? `'${value}` : value;
  return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

/** The saved runs as CSV (one header line, one line per run, newest first as given). */
export function benchmarkRunsToCsv(benchmarks: BenchmarkResult[]): string {
  const lines = [CSV_COLUMNS.map(([name]) => name).join(",")];
  for (const run of benchmarks) lines.push(CSV_COLUMNS.map(([, read]) => csvCell(read(run))).join(","));
  return `${lines.join("\n")}\n`;
}
