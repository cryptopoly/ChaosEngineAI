import { useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Panel } from "../../components/Panel";
import { StatCard } from "../../components/StatCard";
import { BenchmarkScatter } from "../../components/BenchmarkScatter";
import type { BenchmarkResult } from "../../types";
import { number } from "../../utils";
import { accuracyComparable, benchmarkRunsToCsv, taskScoringMethod } from "../../utils/benchmarkCompare";

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
  onDeleteBenchmark: (id: string) => void;
  onCopyText: (text: string) => void;
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
  onDeleteBenchmark,
  onCopyText,
}: BenchmarkHistoryTabProps) {
  const { t } = useTranslation("common");
  const beforeClickSelectedRef = useRef<string | null>(null);
  const [csvCopied, setCsvCopied] = useState(false);
  const benchmarkModels = [...new Set(benchmarks.map(b => b.model))];
  const filteredBenchmarks = benchmarkModelFilter
    ? benchmarks.filter(b => b.model === benchmarkModelFilter)
    : benchmarks;

  // What a run measured, in its own unit: an accuracy or perplexity run has no tok/s (stored as 0).
  const runResultLabel = (result: BenchmarkResult): string => {
    const mode = result.mode ?? "throughput";
    if (mode === "perplexity") {
      return t("benchmarkHistory.list.pplValue", { value: number(result.perplexity ?? 0), defaultValue: "{value} ppl" });
    }
    if (mode === "task_accuracy") return `${((result.taskAccuracy ?? 0) * 100).toFixed(1)}%`;
    return `${number(result.tokS)} tok/s`;
  };
  // Speed cards (Fastest, Best value) rank speed runs only.
  const speedRuns = filteredBenchmarks.filter((b) => (b.mode ?? "throughput") === "throughput");

  const latestRun = filteredBenchmarks[0] ?? null;
  const fastestRun = [...speedRuns].sort((left, right) => right.tokS - left.tokS)[0] ?? null;
  const mostEfficientRun = [...filteredBenchmarks].sort(
    (left, right) => left.cacheGb - right.cacheGb || right.quality - left.quality,
  )[0] ?? null;
  const bestQualityRun = [...filteredBenchmarks].sort((a, b) => b.quality - a.quality)[0] ?? null;
  const bestValueRun = [...speedRuns].sort(
    (a, b) => (b.tokS / Math.max(b.cacheGb, 0.1)) - (a.tokS / Math.max(a.cacheGb, 0.1)),
  )[0] ?? null;

  const selectedInFilter = filteredBenchmarks.find(b => b.id === selectedBenchmarkId);
  const effectiveSelected = selectedInFilter ?? filteredBenchmarks[0] ?? null;
  const effectiveCompare = filteredBenchmarks.find(b => b.id === compareBenchmarkId && b.id !== effectiveSelected?.id)
    ?? filteredBenchmarks.find(b => b.id !== effectiveSelected?.id) ?? null;

  type DeltaRow = { label: string; selected: string; baseline: string; delta: number; pct: number; lowerIsBetter?: boolean; unit?: string };

  const scoringLabel = (run: BenchmarkResult) =>
    taskScoringMethod(run) === "next_token"
      ? t("benchmarkTab.scoring.nextToken", { defaultValue: "next-token choice" })
      : t("benchmarkTab.scoring.generated", { defaultValue: "generated text (older method)" });

  function buildComparisonRows(): DeltaRow[] {
    if (!effectiveSelected || !effectiveCompare) return [];
    const s = effectiveSelected;
    const c = effectiveCompare;
    const pct = (a: number, b: number) => b === 0 ? 0 : ((a - b) / b) * 100;
    const mode = s.mode ?? "throughput";

    const rows: DeltaRow[] = [];
    if (mode === "perplexity") {
      rows.push({ label: t("benchmarkHistory.metric.perplexity", { defaultValue: "Perplexity" }), selected: number(s.perplexity ?? 0), baseline: number(c.perplexity ?? 0), delta: (s.perplexity ?? 0) - (c.perplexity ?? 0), pct: pct(s.perplexity ?? 0, c.perplexity ?? 0), lowerIsBetter: true });
      rows.push({ label: t("benchmarkHistory.metric.evalSpeed", { defaultValue: "Eval speed" }), selected: `${number(s.evalTokensPerSecond ?? 0)} tok/s`, baseline: `${number(c.evalTokensPerSecond ?? 0)} tok/s`, delta: (s.evalTokensPerSecond ?? 0) - (c.evalTokensPerSecond ?? 0), pct: pct(s.evalTokensPerSecond ?? 0, c.evalTokensPerSecond ?? 0) });
      rows.push({ label: t("benchmarkHistory.metric.evalTime", { defaultValue: "Eval time" }), selected: `${number(s.evalSeconds ?? 0)} s`, baseline: `${number(c.evalSeconds ?? 0)} s`, delta: (s.evalSeconds ?? 0) - (c.evalSeconds ?? 0), pct: pct(s.evalSeconds ?? 0, c.evalSeconds ?? 0), lowerIsBetter: true });
    } else if (mode === "task_accuracy") {
      rows.push({ label: t("benchmarkHistory.metric.accuracy", { defaultValue: "Accuracy" }), selected: `${((s.taskAccuracy ?? 0) * 100).toFixed(1)}%`, baseline: `${((c.taskAccuracy ?? 0) * 100).toFixed(1)}%`, delta: (s.taskAccuracy ?? 0) - (c.taskAccuracy ?? 0), pct: pct(s.taskAccuracy ?? 0, c.taskAccuracy ?? 0) });
      rows.push({ label: t("benchmarkHistory.metric.correct", { defaultValue: "Correct" }), selected: `${s.taskCorrect ?? 0}/${s.taskTotal ?? 0}`, baseline: `${c.taskCorrect ?? 0}/${c.taskTotal ?? 0}`, delta: (s.taskCorrect ?? 0) - (c.taskCorrect ?? 0), pct: 0 });
    } else {
      rows.push({ label: t("benchmarkHistory.metric.speed", { defaultValue: "Speed" }), selected: `${number(s.tokS)} tok/s`, baseline: `${number(c.tokS)} tok/s`, delta: s.tokS - c.tokS, pct: pct(s.tokS, c.tokS) });
      if (c.mode === "task_accuracy" && !accuracyComparable(s, c)) {
        // One run picked the highest-scoring letter, the other searched generated text: both values
        // are shown, but a difference between them would say nothing about the models.
        for (const row of rows) {
          row.delta = 0;
          row.pct = 0;
        }
      }
      rows.push({ label: t("benchmarkHistory.metric.scoring", { defaultValue: "Scoring" }), selected: scoringLabel(s), baseline: c.mode === "task_accuracy" ? scoringLabel(c) : "\u2014", delta: 0, pct: 0 });
      rows.push({ label: t("benchmarkHistory.metric.response", { defaultValue: "Response" }), selected: `${number(s.responseSeconds)} s`, baseline: `${number(c.responseSeconds)} s`, delta: s.responseSeconds - c.responseSeconds, pct: pct(s.responseSeconds, c.responseSeconds), lowerIsBetter: true });
    }
    if ((c.mode ?? "throughput") !== mode) {
      // The baseline measured something else (accuracy against speed, say): no value and no delta,
      // instead of "0.0 tok/s" and a made-up percentage.
      for (const row of rows) {
        row.baseline = "\u2014";
        row.delta = 0;
        row.pct = 0;
      }
    }
    rows.push({ label: t("benchmarkHistory.metric.cache", { defaultValue: "Cache" }), selected: `${number(s.cacheGb)} GB`, baseline: `${number(c.cacheGb)} GB`, delta: s.cacheGb - c.cacheGb, pct: pct(s.cacheGb, c.cacheGb), lowerIsBetter: true });
    rows.push({ label: t("benchmarkHistory.metric.quality", { defaultValue: "Quality" }), selected: `${s.quality}%`, baseline: `${c.quality}%`, delta: s.quality - c.quality, pct: pct(s.quality, c.quality) });
    rows.push({ label: t("benchmarkHistory.metric.compression", { defaultValue: "Compression" }), selected: `${number(s.compression)}x`, baseline: `${number(c.compression)}x`, delta: s.compression - c.compression, pct: pct(s.compression, c.compression) });
    rows.push({ label: t("benchmarkHistory.metric.context", { defaultValue: "Context" }), selected: `${Math.round((s.contextTokens ?? 0) / 1024)}K`, baseline: `${Math.round((c.contextTokens ?? 0) / 1024)}K`, delta: 0, pct: 0 });
    rows.push({ label: t("benchmarkHistory.metric.strategy", { defaultValue: "Strategy" }), selected: s.cacheLabel, baseline: c.cacheLabel, delta: 0, pct: 0 });
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
        title={t("panels.benchmarkHistory", { defaultValue: "Benchmark History" })}
        subtitle={t("benchmarkHistory.subtitle", {
          count: filteredBenchmarks.length,
          filterSuffix: benchmarkModelFilter ? ` ${t("benchmarkHistory.filterSuffix", { model: benchmarkModelFilter, defaultValue: "for {model}" })}` : "",
          defaultValue: "{count, plural, one {# run} other {# runs}}{filterSuffix}",
        })}
        className="span-2 benchmark-history-page-panel"
        actions={
          <div className="bm-toolbar">
            <select
              className="text-input bm-model-filter"
              value={benchmarkModelFilter ?? ""}
              onChange={(e) => onBenchmarkModelFilterChange(e.target.value || null)}
            >
              <option value="">{t("benchmarkHistory.allModels", { defaultValue: "All models" })}</option>
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
                  {mode === "table"
                    ? t("benchmarkHistory.view.table", { defaultValue: "Table" })
                    : mode === "chart"
                      ? t("benchmarkHistory.view.chart", { defaultValue: "Chart" })
                      : t("benchmarkHistory.view.both", { defaultValue: "Both" })}
                </button>
              ))}
            </div>
            <button
              className="secondary-button"
              type="button"
              disabled={filteredBenchmarks.length === 0}
              onClick={() => {
                // The runs on screen (the model filter applies), newest first.
                onCopyText(benchmarkRunsToCsv(filteredBenchmarks));
                setCsvCopied(true);
                window.setTimeout(() => setCsvCopied(false), 1500);
              }}
            >
              {csvCopied
                ? t("benchmarkHistory.csvCopied", { defaultValue: "Copied" })
                : t("benchmarkHistory.copyCsv", { defaultValue: "Copy CSV" })}
            </button>
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
              {t("benchmarkHistory.swap", { defaultValue: "Swap" })}
            </button>
          </div>
        }
      >
        <div className="benchmark-history-page">
          <div className="benchmark-summary-row">
            <StatCard
              label={t("benchmarkHistory.summary.totalRuns", { defaultValue: "Total runs" })}
              value={String(filteredBenchmarks.length)}
              hint={t("benchmarkHistory.summary.persistentHistory", { defaultValue: "Persistent history" })}
            />
            <StatCard
              label={t("benchmarkHistory.summary.latest", { defaultValue: "Latest" })}
              value={latestRun ? runResultLabel(latestRun) : t("benchmarkHistory.summary.none", { defaultValue: "None" })}
              hint={latestRun?.cacheLabel ?? t("benchmarkHistory.summary.noRuns", { defaultValue: "No runs" })}
              onClick={latestRun ? () => onSelectedBenchmarkIdChange(latestRun.id) : undefined}
            />
            <StatCard
              label={t("benchmarkHistory.summary.fastest", { defaultValue: "Fastest" })}
              value={fastestRun ? `${number(fastestRun.tokS)} tok/s` : t("benchmarkHistory.summary.none", { defaultValue: "None" })}
              hint={fastestRun?.label ?? t("benchmarkHistory.summary.noRuns", { defaultValue: "No runs" })}
              onClick={fastestRun ? () => onSelectedBenchmarkIdChange(fastestRun.id) : undefined}
            />
            <StatCard
              label={t("benchmarkHistory.summary.leanestCache", { defaultValue: "Leanest cache" })}
              value={mostEfficientRun ? `${number(mostEfficientRun.cacheGb)} GB` : t("benchmarkHistory.summary.none", { defaultValue: "None" })}
              hint={mostEfficientRun?.label ?? t("benchmarkHistory.summary.noRuns", { defaultValue: "No runs" })}
              onClick={mostEfficientRun ? () => onSelectedBenchmarkIdChange(mostEfficientRun.id) : undefined}
            />
            <StatCard
              label={t("benchmarkHistory.summary.bestQuality", { defaultValue: "Best quality" })}
              value={bestQualityRun ? `${bestQualityRun.quality}%` : t("benchmarkHistory.summary.none", { defaultValue: "None" })}
              hint={bestQualityRun?.cacheLabel ?? t("benchmarkHistory.summary.noRuns", { defaultValue: "No runs" })}
              onClick={bestQualityRun ? () => onSelectedBenchmarkIdChange(bestQualityRun.id) : undefined}
            />
            <StatCard
              label={t("benchmarkHistory.summary.bestValue", { defaultValue: "Best value" })}
              value={bestValueRun ? `${number(bestValueRun.tokS / Math.max(bestValueRun.cacheGb, 0.1))} tok/s/GB` : t("benchmarkHistory.summary.none", { defaultValue: "None" })}
              hint={bestValueRun?.cacheLabel ?? t("benchmarkHistory.summary.noRuns", { defaultValue: "No runs" })}
              onClick={bestValueRun ? () => onSelectedBenchmarkIdChange(bestValueRun.id) : undefined}
            />
          </div>

          {effectiveSelected ? (
            <div className="bm-comparison-section">
              {effectiveCompare ? (
                <div className="bm-comparison-table-wrap">
                  <table className="bm-comparison-table">
                    <thead>
                      <tr>
                        <th>{t("benchmarkHistory.table.metric", { defaultValue: "Metric" })}</th>
                        <th>
                          <span className="bm-col-label">{t("benchmarkHistory.table.selected", { defaultValue: "Selected" })}</span>
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
                          <span className="bm-col-label">{t("benchmarkHistory.table.baseline", { defaultValue: "Baseline" })}</span>
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
                        <th>{t("benchmarkHistory.table.delta", { defaultValue: "Delta" })}</th>
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
                      <label className="muted-text">{t("benchmarkHistory.compareAgainst", { defaultValue: "Compare against:" })}</label>
                      <select
                        className="bm-run-select"
                        value=""
                        onChange={(e) => onCompareBenchmarkIdChange(e.target.value)}
                      >
                        <option value="" disabled>{t("benchmarkHistory.pickBaseline", { defaultValue: "Pick a baseline run..." })}</option>
                        {filteredBenchmarks.filter(b => b.id !== effectiveSelected.id).map(b => (
                          <option key={b.id} value={b.id}>{b.label} {"\u2014"} {b.model}</option>
                        ))}
                      </select>
                    </div>
                  ) : (
                    <p className="muted-text">{t("benchmarkHistory.runAnother", { defaultValue: "Run another benchmark to enable comparison." })}</p>
                  )}
                </div>
              )}

              <div className="bm-run-properties">
                <span className="eyebrow">{t("benchmarkHistory.properties.heading", { defaultValue: "Run properties" })}</span>
                <div className="bm-props-grid">
                  <div><span>{t("benchmarkHistory.properties.model", { defaultValue: "Model" })}</span><span>{effectiveSelected.model}</span></div>
                  <div><span>{t("benchmarkHistory.properties.backend", { defaultValue: "Backend" })}</span><span>{effectiveSelected.backend}</span></div>
                  <div><span>{t("benchmarkHistory.properties.cache", { defaultValue: "Cache" })}</span><span>{effectiveSelected.cacheLabel}</span></div>
                  <div><span>{t("benchmarkHistory.properties.strategy", { defaultValue: "Strategy" })}</span><span>{effectiveSelected.cacheStrategy}</span></div>
                  <div><span>{t("benchmarkHistory.properties.bits", { defaultValue: "Bits" })}</span><span>{effectiveSelected.bits}</span></div>
                  <div><span>{t("benchmarkHistory.properties.fp16Layers", { defaultValue: "FP16 layers" })}</span><span>{effectiveSelected.fp16Layers}</span></div>
                  <div><span>{t("benchmarkHistory.properties.context", { defaultValue: "Context" })}</span><span>{Math.round((effectiveSelected.contextTokens ?? 0) / 1024)}K</span></div>
                  <div><span>{t("benchmarkHistory.properties.maxTokens", { defaultValue: "Max tokens" })}</span><span>{effectiveSelected.maxTokens}</span></div>
                  <div><span>{t("benchmarkHistory.properties.measured", { defaultValue: "Measured" })}</span><span>{effectiveSelected.measuredAt}</span></div>
                  {effectiveSelected.notes ? <div className="bm-prop-full"><span>{t("benchmarkHistory.properties.notes", { defaultValue: "Notes" })}</span><span>{effectiveSelected.notes}</span></div> : null}
                </div>
                <div className="button-row">
                  <button
                    className="secondary-button danger-button"
                    type="button"
                    onClick={() => {
                      const question = t("benchmarkHistory.deleteRunConfirm", {
                        label: effectiveSelected.label,
                        defaultValue: "Delete the saved run \"{label}\"? This cannot be undone.",
                      });
                      if (window.confirm(question)) onDeleteBenchmark(effectiveSelected.id);
                    }}
                  >
                    {t("benchmarkHistory.deleteRun", { defaultValue: "Delete run" })}
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <div className="empty-state">
              <p>{t("benchmarkHistory.runOneToCompare", { defaultValue: "Run a benchmark to populate comparison stats." })}</p>
            </div>
          )}

          <div className={`benchmark-bottom-row${benchmarkViewMode === "table" ? " bm-bottom--table-only" : benchmarkViewMode === "chart" ? " bm-bottom--chart-only" : ""}`}>
            {benchmarkViewMode !== "chart" ? (
              <div className="benchmark-history-table-wrap">
                <div className="benchmark-history-table">
                  <div className="table-row table-head benchmark-history-row bm-history-row-wide">
                    <span>{t("benchmarkHistory.list.run", { defaultValue: "Run" })}</span>
                    <span>{t("benchmarkHistory.list.mode", { defaultValue: "Mode" })}</span>
                    <span>{t("benchmarkHistory.list.result", { defaultValue: "Result" })}</span>
                    <span>{t("benchmarkHistory.list.time", { defaultValue: "Time" })}</span>
                    <span>{t("benchmarkHistory.list.cache", { defaultValue: "Cache" })}</span>
                    {benchmarkViewMode === "table" ? (
                      <>
                        <span>{t("benchmarkHistory.list.strategy", { defaultValue: "Strategy" })}</span>
                        <span>{t("benchmarkHistory.list.bits", { defaultValue: "Bits" })}</span>
                        <span>{t("benchmarkHistory.list.context", { defaultValue: "Context" })}</span>
                        <span>{t("benchmarkHistory.list.quality", { defaultValue: "Quality" })}</span>
                      </>
                    ) : null}
                  </div>
                  <div className="benchmark-history-list">
                    {filteredBenchmarks.map((result) => {
                      const isSelected = result.id === effectiveSelected?.id;
                      const isCompare = result.id === effectiveCompare?.id;
                      const mode = result.mode ?? "throughput";
                      const resultValue = runResultLabel(result);
                      const modeLabel = mode === "perplexity"
                        ? t("benchmarkHistory.modeLabel.ppl", { defaultValue: "PPL" })
                        : mode === "task_accuracy"
                          ? (result.taskName ?? "mmlu").toUpperCase()
                          : t("benchmarkHistory.modeLabel.speed", { defaultValue: "Speed" });
                      return (
                        <button
                          key={result.id}
                          type="button"
                          className={`table-row table-button-row benchmark-history-row bm-history-row-wide${isSelected ? " active" : ""}${isCompare ? " compare" : ""}`}
                          onClick={(e) => {
                            // A double click sends two clicks first. Only the first one selects, and it
                            // remembers what was selected, so the double click can make this row the
                            // baseline and put the selection back.
                            if (e.detail > 1) return;
                            beforeClickSelectedRef.current = effectiveSelected?.id ?? null;
                            onSelectedBenchmarkIdChange(result.id);
                          }}
                          onDoubleClick={() => {
                            const previous = beforeClickSelectedRef.current;
                            onCompareBenchmarkIdChange(result.id);
                            if (previous && previous !== result.id) onSelectedBenchmarkIdChange(previous);
                          }}
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
                  <span className="eyebrow">{t("benchmarkHistory.scatter.heading", { defaultValue: "Cache vs Speed" })}</span>
                  <small>{t("benchmarkHistory.scatter.hint", { defaultValue: "Click a dot to select" })}</small>
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
