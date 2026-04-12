import { Panel } from "../../components/Panel";
import { StatCard } from "../../components/StatCard";
import { BenchmarkScatter } from "../../components/BenchmarkScatter";
import type { BenchmarkResult } from "../../types";
import { number } from "../../utils";

export interface BenchmarkHistoryTabProps {
  benchmarks: BenchmarkResult[];
  benchmarkModelFilter: string | null;
  benchmarkViewMode: "table" | "chart" | "both";
  selectedBenchmarkId: string;
  compareBenchmarkId: string;
  onBenchmarkModelFilterChange: (filter: string | null) => void;
  onBenchmarkViewModeChange: (mode: "table" | "chart" | "both") => void;
  onSelectedBenchmarkIdChange: (id: string) => void;
  onCompareBenchmarkIdChange: (id: string) => void;
}

export function BenchmarkHistoryTab({
  benchmarks,
  benchmarkModelFilter,
  benchmarkViewMode,
  selectedBenchmarkId,
  compareBenchmarkId,
  onBenchmarkModelFilterChange,
  onBenchmarkViewModeChange,
  onSelectedBenchmarkIdChange,
  onCompareBenchmarkIdChange,
}: BenchmarkHistoryTabProps) {
  const benchmarkModels = [...new Set(benchmarks.map(b => b.model))];
  const filteredBenchmarks = benchmarkModelFilter
    ? benchmarks.filter(b => b.model === benchmarkModelFilter)
    : benchmarks;

  const latestRun = filteredBenchmarks[0] ?? null;
  const fastestRun = [...filteredBenchmarks].sort((left, right) => right.tokS - left.tokS)[0] ?? null;
  const mostEfficientRun = [...filteredBenchmarks].sort(
    (left, right) => left.cacheGb - right.cacheGb || right.quality - left.quality,
  )[0] ?? null;
  const bestQualityRun = [...filteredBenchmarks].sort((a, b) => b.quality - a.quality)[0] ?? null;
  const bestValueRun = [...filteredBenchmarks].sort(
    (a, b) => (b.tokS / Math.max(b.cacheGb, 0.1)) - (a.tokS / Math.max(a.cacheGb, 0.1)),
  )[0] ?? null;

  const selectedInFilter = filteredBenchmarks.find(b => b.id === selectedBenchmarkId);
  const effectiveSelected = selectedInFilter ?? filteredBenchmarks[0] ?? null;
  const effectiveCompare = filteredBenchmarks.find(b => b.id === compareBenchmarkId && b.id !== effectiveSelected?.id)
    ?? filteredBenchmarks.find(b => b.id !== effectiveSelected?.id) ?? null;

  type DeltaRow = { label: string; selected: string; baseline: string; delta: number; pct: number; lowerIsBetter?: boolean; unit?: string };

  function buildComparisonRows(): DeltaRow[] {
    if (!effectiveSelected || !effectiveCompare) return [];
    const s = effectiveSelected;
    const c = effectiveCompare;
    const pct = (a: number, b: number) => b === 0 ? 0 : ((a - b) / b) * 100;
    const mode = s.mode ?? "throughput";

    const rows: DeltaRow[] = [];
    if (mode === "perplexity") {
      rows.push({ label: "Perplexity", selected: number(s.perplexity ?? 0), baseline: number(c.perplexity ?? 0), delta: (s.perplexity ?? 0) - (c.perplexity ?? 0), pct: pct(s.perplexity ?? 0, c.perplexity ?? 0), lowerIsBetter: true });
      rows.push({ label: "Eval speed", selected: `${number(s.evalTokensPerSecond ?? 0)} tok/s`, baseline: `${number(c.evalTokensPerSecond ?? 0)} tok/s`, delta: (s.evalTokensPerSecond ?? 0) - (c.evalTokensPerSecond ?? 0), pct: pct(s.evalTokensPerSecond ?? 0, c.evalTokensPerSecond ?? 0) });
      rows.push({ label: "Eval time", selected: `${number(s.evalSeconds ?? 0)} s`, baseline: `${number(c.evalSeconds ?? 0)} s`, delta: (s.evalSeconds ?? 0) - (c.evalSeconds ?? 0), pct: pct(s.evalSeconds ?? 0, c.evalSeconds ?? 0), lowerIsBetter: true });
    } else if (mode === "task_accuracy") {
      rows.push({ label: "Accuracy", selected: `${((s.taskAccuracy ?? 0) * 100).toFixed(1)}%`, baseline: `${((c.taskAccuracy ?? 0) * 100).toFixed(1)}%`, delta: (s.taskAccuracy ?? 0) - (c.taskAccuracy ?? 0), pct: pct(s.taskAccuracy ?? 0, c.taskAccuracy ?? 0) });
      rows.push({ label: "Correct", selected: `${s.taskCorrect ?? 0}/${s.taskTotal ?? 0}`, baseline: `${c.taskCorrect ?? 0}/${c.taskTotal ?? 0}`, delta: (s.taskCorrect ?? 0) - (c.taskCorrect ?? 0), pct: 0 });
    } else {
      rows.push({ label: "Speed", selected: `${number(s.tokS)} tok/s`, baseline: `${number(c.tokS)} tok/s`, delta: s.tokS - c.tokS, pct: pct(s.tokS, c.tokS) });
      rows.push({ label: "Response", selected: `${number(s.responseSeconds)} s`, baseline: `${number(c.responseSeconds)} s`, delta: s.responseSeconds - c.responseSeconds, pct: pct(s.responseSeconds, c.responseSeconds), lowerIsBetter: true });
    }
    rows.push({ label: "Cache", selected: `${number(s.cacheGb)} GB`, baseline: `${number(c.cacheGb)} GB`, delta: s.cacheGb - c.cacheGb, pct: pct(s.cacheGb, c.cacheGb), lowerIsBetter: true });
    rows.push({ label: "Quality", selected: `${s.quality}%`, baseline: `${c.quality}%`, delta: s.quality - c.quality, pct: pct(s.quality, c.quality) });
    rows.push({ label: "Compression", selected: `${number(s.compression)}x`, baseline: `${number(c.compression)}x`, delta: s.compression - c.compression, pct: pct(s.compression, c.compression) });
    rows.push({ label: "Context", selected: `${Math.round((s.contextTokens ?? 0) / 1024)}K`, baseline: `${Math.round((c.contextTokens ?? 0) / 1024)}K`, delta: 0, pct: 0 });
    rows.push({ label: "Strategy", selected: s.cacheLabel, baseline: c.cacheLabel, delta: 0, pct: 0 });
    return rows;
  }

  function deltaClass(delta: number, lowerIsBetter?: boolean) {
    if (delta === 0) return "";
    const good = lowerIsBetter ? delta < 0 : delta > 0;
    return good ? "bm-delta--good" : "bm-delta--bad";
  }

  function deltaArrow(delta: number, lowerIsBetter?: boolean) {
    if (Math.abs(delta) < 0.001) return "";
    const good = lowerIsBetter ? delta < 0 : delta > 0;
    return good ? " \u25B2" : " \u25BC";
  }

  const comparisonRows = buildComparisonRows();

  return (
    <div className="content-grid">
      <Panel
        title="Benchmark History"
        subtitle={`${filteredBenchmarks.length} run${filteredBenchmarks.length !== 1 ? "s" : ""}${benchmarkModelFilter ? ` for ${benchmarkModelFilter}` : ""}`}
        className="span-2 benchmark-history-page-panel"
        actions={
          <div className="bm-toolbar">
            <select
              className="text-input bm-model-filter"
              value={benchmarkModelFilter ?? ""}
              onChange={(e) => onBenchmarkModelFilterChange(e.target.value || null)}
            >
              <option value="">All models</option>
              {benchmarkModels.map(m => <option key={m} value={m}>{m}</option>)}
            </select>
            <div className="bm-view-toggle">
              {(["table", "chart", "both"] as const).map(mode => (
                <button
                  key={mode}
                  type="button"
                  className={`bm-view-btn${benchmarkViewMode === mode ? " bm-view-btn--active" : ""}`}
                  onClick={() => onBenchmarkViewModeChange(mode)}
                >
                  {mode === "table" ? "Table" : mode === "chart" ? "Chart" : "Both"}
                </button>
              ))}
            </div>
            <button
              className="secondary-button"
              type="button"
              onClick={() => {
                if (!effectiveSelected || !effectiveCompare) return;
                const a = effectiveSelected.id;
                const b = effectiveCompare.id;
                onSelectedBenchmarkIdChange(b);
                onCompareBenchmarkIdChange(a);
              }}
              disabled={!effectiveSelected || !effectiveCompare}
            >
              Swap
            </button>
          </div>
        }
      >
        <div className="benchmark-history-page">
          <div className="benchmark-summary-row">
            <StatCard label="Total runs" value={String(filteredBenchmarks.length)} hint="Persistent history" />
            <StatCard label="Latest" value={latestRun ? `${number(latestRun.tokS)} tok/s` : "None"} hint={latestRun?.cacheLabel ?? "No runs"} onClick={latestRun ? () => onSelectedBenchmarkIdChange(latestRun.id) : undefined} />
            <StatCard label="Fastest" value={fastestRun ? `${number(fastestRun.tokS)} tok/s` : "None"} hint={fastestRun?.label ?? "No runs"} onClick={fastestRun ? () => onSelectedBenchmarkIdChange(fastestRun.id) : undefined} />
            <StatCard label="Leanest cache" value={mostEfficientRun ? `${number(mostEfficientRun.cacheGb)} GB` : "None"} hint={mostEfficientRun?.label ?? "No runs"} onClick={mostEfficientRun ? () => onSelectedBenchmarkIdChange(mostEfficientRun.id) : undefined} />
            <StatCard label="Best quality" value={bestQualityRun ? `${bestQualityRun.quality}%` : "None"} hint={bestQualityRun?.cacheLabel ?? "No runs"} onClick={bestQualityRun ? () => onSelectedBenchmarkIdChange(bestQualityRun.id) : undefined} />
            <StatCard label="Best value" value={bestValueRun ? `${number(bestValueRun.tokS / Math.max(bestValueRun.cacheGb, 0.1))} tok/s/GB` : "None"} hint={bestValueRun?.cacheLabel ?? "No runs"} onClick={bestValueRun ? () => onSelectedBenchmarkIdChange(bestValueRun.id) : undefined} />
          </div>

          {effectiveSelected ? (
            <div className="bm-comparison-section">
              {effectiveCompare ? (
                <div className="bm-comparison-table-wrap">
                  <table className="bm-comparison-table">
                    <thead>
                      <tr>
                        <th>Metric</th>
                        <th>
                          <span className="bm-col-label">Selected</span>
                          <select
                            className="bm-run-select"
                            value={effectiveSelected.id}
                            onChange={(e) => onSelectedBenchmarkIdChange(e.target.value)}
                          >
                            {filteredBenchmarks.map(b => (
                              <option key={b.id} value={b.id}>{b.label} {"\u2014"} {b.model}</option>
                            ))}
                          </select>
                        </th>
                        <th>
                          <span className="bm-col-label">Baseline</span>
                          <select
                            className="bm-run-select"
                            value={effectiveCompare.id}
                            onChange={(e) => onCompareBenchmarkIdChange(e.target.value)}
                          >
                            {filteredBenchmarks.filter(b => b.id !== effectiveSelected.id).map(b => (
                              <option key={b.id} value={b.id}>{b.label} {"\u2014"} {b.model}</option>
                            ))}
                          </select>
                        </th>
                        <th>Delta</th>
                      </tr>
                    </thead>
                    <tbody>
                      {comparisonRows.map((row) => (
                        <tr key={row.label}>
                          <td className="bm-metric-label">{row.label}</td>
                          <td className="bm-metric-val">{row.selected}</td>
                          <td className="bm-metric-val">{row.baseline}</td>
                          <td className={`bm-metric-delta ${deltaClass(row.delta, row.lowerIsBetter)}`}>
                            {row.pct !== 0 ? `${row.pct > 0 ? "+" : ""}${row.pct.toFixed(1)}%${deltaArrow(row.delta, row.lowerIsBetter)}` : row.selected !== row.baseline ? "\u2014" : ""}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="bm-comparison-empty">
                  <div className="bm-selected-summary">
                    <h3>{effectiveSelected.model}</h3>
                    <span className="badge accent">{effectiveSelected.cacheLabel}</span>
                    <div className="bm-selected-headline">
                      <strong>{number(effectiveSelected.tokS)}</strong> <span>tok/s</span>
                    </div>
                  </div>
                  {filteredBenchmarks.length > 1 ? (
                    <div className="bm-pick-baseline">
                      <label className="muted-text">Compare against:</label>
                      <select
                        className="bm-run-select"
                        value=""
                        onChange={(e) => onCompareBenchmarkIdChange(e.target.value)}
                      >
                        <option value="" disabled>Pick a baseline run...</option>
                        {filteredBenchmarks.filter(b => b.id !== effectiveSelected.id).map(b => (
                          <option key={b.id} value={b.id}>{b.label} {"\u2014"} {b.model}</option>
                        ))}
                      </select>
                    </div>
                  ) : (
                    <p className="muted-text">Run another benchmark to enable comparison.</p>
                  )}
                </div>
              )}

              <div className="bm-run-properties">
                <span className="eyebrow">Run properties</span>
                <div className="bm-props-grid">
                  <div><span>Model</span><span>{effectiveSelected.model}</span></div>
                  <div><span>Backend</span><span>{effectiveSelected.backend}</span></div>
                  <div><span>Cache</span><span>{effectiveSelected.cacheLabel}</span></div>
                  <div><span>Strategy</span><span>{effectiveSelected.cacheStrategy}</span></div>
                  <div><span>Bits</span><span>{effectiveSelected.bits}</span></div>
                  <div><span>FP16 layers</span><span>{effectiveSelected.fp16Layers}</span></div>
                  <div><span>Context</span><span>{Math.round((effectiveSelected.contextTokens ?? 0) / 1024)}K</span></div>
                  <div><span>Max tokens</span><span>{effectiveSelected.maxTokens}</span></div>
                  <div><span>Measured</span><span>{effectiveSelected.measuredAt}</span></div>
                  {effectiveSelected.notes ? <div className="bm-prop-full"><span>Notes</span><span>{effectiveSelected.notes}</span></div> : null}
                </div>
              </div>
            </div>
          ) : (
            <div className="empty-state">
              <p>Run a benchmark to populate comparison stats.</p>
            </div>
          )}

          <div className={`benchmark-bottom-row${benchmarkViewMode === "table" ? " bm-bottom--table-only" : benchmarkViewMode === "chart" ? " bm-bottom--chart-only" : ""}`}>
            {benchmarkViewMode !== "chart" ? (
              <div className="benchmark-history-table-wrap">
                <div className="benchmark-history-table">
                  <div className="table-row table-head benchmark-history-row bm-history-row-wide">
                    <span>Run</span>
                    <span>Mode</span>
                    <span>Result</span>
                    <span>Time</span>
                    <span>Cache</span>
                    {benchmarkViewMode === "table" ? (
                      <>
                        <span>Strategy</span>
                        <span>Bits</span>
                        <span>Context</span>
                        <span>Quality</span>
                      </>
                    ) : null}
                  </div>
                  <div className="benchmark-history-list">
                    {filteredBenchmarks.map((result) => {
                      const isSelected = result.id === effectiveSelected?.id;
                      const isCompare = result.id === effectiveCompare?.id;
                      const mode = result.mode ?? "throughput";
                      const resultValue = mode === "perplexity"
                        ? `${number(result.perplexity ?? 0)} ppl`
                        : mode === "task_accuracy"
                        ? `${((result.taskAccuracy ?? 0) * 100).toFixed(1)}%`
                        : `${number(result.tokS)} tok/s`;
                      const modeLabel = mode === "perplexity" ? "PPL" : mode === "task_accuracy" ? (result.taskName ?? "mmlu").toUpperCase() : "Speed";
                      return (
                        <button
                          key={result.id}
                          type="button"
                          className={`table-row table-button-row benchmark-history-row bm-history-row-wide${isSelected ? " active" : ""}${isCompare ? " compare" : ""}`}
                          onClick={() => onSelectedBenchmarkIdChange(result.id)}
                          onDoubleClick={() => onCompareBenchmarkIdChange(result.id)}
                          onContextMenu={(e) => { e.preventDefault(); onCompareBenchmarkIdChange(result.id); }}
                        >
                          <div>
                            <strong>{result.label}</strong>
                            <small>{result.measuredAt} {"\u00B7"} {result.model}</small>
                          </div>
                          <span className="badge muted">{modeLabel}</span>
                          <span>{resultValue}</span>
                          <span>{number(result.responseSeconds)} s</span>
                          <span>{number(result.cacheGb)} GB</span>
                          {benchmarkViewMode === "table" ? (
                            <>
                              <span>{result.cacheStrategy}</span>
                              <span>{result.bits > 0 ? `${result.bits}-bit / ${result.fp16Layers}` : "\u2014"}</span>
                              <span>{Math.round((result.contextTokens ?? 0) / 1024)}K</span>
                              <span>{result.quality}%</span>
                            </>
                          ) : null}
                        </button>
                      );
                    })}
                  </div>
                </div>
              </div>
            ) : null}

            {benchmarkViewMode !== "table" ? (
              <div className="benchmark-scatter-panel">
                <div className="benchmark-scatter-header">
                  <span className="eyebrow">Cache vs Speed</span>
                  <small>Click a dot to select</small>
                </div>
                <BenchmarkScatter runs={filteredBenchmarks} selectedId={effectiveSelected?.id ?? null} compareId={effectiveCompare?.id ?? null} onSelect={(id) => onSelectedBenchmarkIdChange(id)} />
              </div>
            ) : null}
          </div>
        </div>
      </Panel>
    </div>
  );
}
