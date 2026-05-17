import { useState } from "react";
import { useTranslation } from "react-i18next";
import { Panel } from "../../components/Panel";
import { PerformancePreview } from "../../components/PerformancePreview";
import { LiveProgress, type LiveProgressPhase } from "../../components/LiveProgress";
import { RuntimeControls } from "../../components/RuntimeControls";
import { ModelLaunchModal } from "../../components/ModelLaunchModal";
import { StatCard } from "../../components/StatCard";
import { BenchmarkGauge } from "../../components/BenchmarkGauge";
import type { BenchmarkResult, BenchmarkRunPayload, LibraryItem, PreviewMetrics, StrategyInstallLog, SystemStats } from "../../types";
import type { ChatModelOption } from "../../types/chat";
import type { MtplxJobState } from "../../api";
import { BENCHMARK_PROMPTS } from "../../constants";
import { number, sizeLabel, signedDelta } from "../../utils";

export interface BenchmarkRunTabProps {
  workspace: {
    benchmarks: BenchmarkResult[];
    library: LibraryItem[];
    system: {
      availableMemoryGb: number;
      totalMemoryGb: number;
      gpuVramTotalGb?: number | null;
      availableCacheStrategies: SystemStats["availableCacheStrategies"];
      llamaServerTurboPath?: string | null;
      dflash?: SystemStats["dflash"];
      mtplx?: SystemStats["mtplx"];
    };
  };
  threadModelOptions: ChatModelOption[];
  benchmarkDraft: BenchmarkRunPayload;
  benchmarkOption: ChatModelOption | null;
  benchmarkPromptId: string;
  preview: PreviewMetrics;
  busy: boolean;
  busyAction: string | null;
  benchmarkStartedAt: number | null;
  benchmarkError: string | null;
  showBenchmarkPicker: boolean;
  showBenchmarkModal: boolean;
  installingPackage: string | null;
  installLogs?: Record<string, StrategyInstallLog>;
  onInstallMtplx?: () => void;
  installingMtplx?: boolean;
  mtplxJob?: MtplxJobState | null;
  /** FU-056 follow-up: forwarded to ``RuntimeControls`` so the MTPLX
   * block hides on non-Apple-Silicon hosts. */
  isAppleSilicon?: boolean;
  onBenchmarkDraftChange: <K extends keyof BenchmarkRunPayload>(key: K, value: BenchmarkRunPayload[K]) => void;
  onBenchmarkPromptIdChange: (id: string) => void;
  onBenchmarkModelKeyChange: (key: string) => void;
  onBenchmarkDraftUpdate: (updater: (current: BenchmarkRunPayload) => BenchmarkRunPayload) => void;
  onRunBenchmark: () => void;
  onCancelBenchmark: () => void;
  onShowBenchmarkPickerChange: (open: boolean) => void;
  onShowBenchmarkModalChange: (open: boolean) => void;
  onSelectedBenchmarkIdChange: (id: string) => void;
  onCompareBenchmarkIdChange: (id: string) => void;
  onActiveTabChange: (tab: string) => void;
  onInstallPackage: (strategyId: string) => Promise<void>;
}

export function BenchmarkRunTab({
  workspace,
  threadModelOptions,
  benchmarkDraft,
  benchmarkOption,
  benchmarkPromptId,
  preview,
  busy,
  busyAction,
  benchmarkStartedAt,
  benchmarkError,
  showBenchmarkPicker,
  showBenchmarkModal,
  installingPackage,
  installLogs,
  onInstallMtplx,
  installingMtplx,
  mtplxJob,
  isAppleSilicon = false,
  onBenchmarkDraftChange,
  onBenchmarkPromptIdChange,
  onBenchmarkModelKeyChange,
  onBenchmarkDraftUpdate,
  onRunBenchmark,
  onCancelBenchmark,
  onShowBenchmarkPickerChange,
  onShowBenchmarkModalChange,
  onSelectedBenchmarkIdChange,
  onCompareBenchmarkIdChange,
  onActiveTabChange,
  onInstallPackage,
}: BenchmarkRunTabProps) {
  const { t } = useTranslation("common");
  const [benchmarkPickerSearch, setBenchmarkPickerSearch] = useState("");
  const latestRun = workspace.benchmarks[0] ?? null;
  const fastestRun = [...workspace.benchmarks].sort((left, right) => right.tokS - left.tokS)[0] ?? null;
  const selectedPrompt = BENCHMARK_PROMPTS.find((p) => p.id === benchmarkPromptId) ?? BENCHMARK_PROMPTS[0];
  // Find the most recent run for the currently selected model, if any
  const prevForModel = benchmarkOption
    ? workspace.benchmarks.find((b) => b.modelRef === benchmarkOption.modelRef && b.id !== latestRun?.id) ?? null
    : null;
  const speedDeltaVsPrev = latestRun && prevForModel ? latestRun.tokS - prevForModel.tokS : null;

  return (
    <div className="content-grid">
      <Panel
        title={t("panels.runBenchmark", { defaultValue: "Run Benchmark" })}
        subtitle={t("panels.runBenchmarkSubtitle", {
          defaultValue: "Launch a consistent benchmark run and see how this profile performs.",
        })}
        className="span-2 benchmark-run-page-panel"
      >
        <div className="benchmark-run-page">
          <div className="benchmark-run-config scrollable-panel-content">
            <div className="benchmark-run-header-grid">
              <label className="field">
                {t("benchmarkTab.modelLabel", { defaultValue: "Benchmark model" })}
                <div className="model-selected-card">
                  <div className="model-selected-info">
                    <strong>{benchmarkOption?.label ?? benchmarkDraft.modelName ?? t("benchmarkTab.selectAModel", { defaultValue: "Select a model" })}</strong>
                    <div className="model-selected-meta">
                      {benchmarkOption?.format ? <span className="badge muted">{benchmarkOption.format}</span> : null}
                      {benchmarkOption?.sizeGb ? <span className="badge muted">{sizeLabel(benchmarkOption.sizeGb)}</span> : null}
                    </div>
                  </div>
                  <button className="secondary-button" type="button" onClick={() => onShowBenchmarkPickerChange(true)}>
                    {t("benchmarkTab.change", { defaultValue: "Change" })}
                  </button>
                </div>
              </label>
              <div className="benchmark-run-mode-stack">
                <label className="field">
                  {t("benchmarkTab.mode", { defaultValue: "Benchmark mode" })}
                  <select
                    className="text-input"
                    value={benchmarkDraft.mode ?? "throughput"}
                    onChange={(event) => onBenchmarkDraftChange("mode", event.target.value as any)}
                  >
                    <option value="throughput">{t("benchmarkTab.modeThroughput", { defaultValue: "Throughput (tok/s)" })}</option>
                    <option value="perplexity">{t("benchmarkTab.modePerplexity", { defaultValue: "Perplexity (quality)" })}</option>
                    <option value="task_accuracy">{t("benchmarkTab.modeTaskAccuracy", { defaultValue: "Task Accuracy (MMLU / HellaSwag)" })}</option>
                  </select>
                </label>
                {(!benchmarkDraft.mode || benchmarkDraft.mode === "throughput") ? (
                  <label className="field">
                    {t("benchmarkTab.promptPreset", { defaultValue: "Prompt preset" })}
                    <select
                      className="text-input"
                      value={benchmarkPromptId}
                      onChange={(event) => onBenchmarkPromptIdChange(event.target.value)}
                    >
                      {BENCHMARK_PROMPTS.map((preset) => (
                        <option key={preset.id} value={preset.id}>
                          {preset.label}
                        </option>
                      ))}
                    </select>
                  </label>
                ) : null}
              </div>
            </div>
            {benchmarkDraft.mode === "perplexity" ? (
              <div className="field-grid">
                <label>
                  {t("benchmarkTab.dataset", { defaultValue: "Dataset" })}
                  <select
                    className="text-input"
                    value={benchmarkDraft.perplexityDataset ?? "wikitext-2"}
                    onChange={(event) => onBenchmarkDraftChange("perplexityDataset", event.target.value as any)}
                  >
                    <option value="wikitext-2">WikiText-2</option>
                  </select>
                </label>
                <label>
                  {t("benchmarkTab.samples", { defaultValue: "Samples" })}
                  <input
                    className="text-input"
                    type="number"
                    min="8"
                    max="1024"
                    step="8"
                    value={benchmarkDraft.perplexityNumSamples ?? 64}
                    onChange={(event) => onBenchmarkDraftChange("perplexityNumSamples", Number(event.target.value) as any)}
                  />
                </label>
              </div>
            ) : null}
            {benchmarkDraft.mode === "task_accuracy" ? (
              <div className="field-grid">
                <label>
                  {t("benchmarkTab.task", { defaultValue: "Task" })}
                  <select
                    className="text-input"
                    value={benchmarkDraft.taskName ?? "mmlu"}
                    onChange={(event) => onBenchmarkDraftChange("taskName", event.target.value as any)}
                  >
                    <option value="mmlu">{t("benchmarkTab.taskMmlu", { defaultValue: "MMLU (multiple choice)" })}</option>
                    <option value="hellaswag">{t("benchmarkTab.taskHellaswag", { defaultValue: "HellaSwag (sentence completion)" })}</option>
                  </select>
                </label>
                <label>
                  {t("benchmarkTab.questions", { defaultValue: "Questions" })}
                  <input
                    className="text-input"
                    type="number"
                    min="10"
                    max="5000"
                    step="10"
                    value={benchmarkDraft.taskLimit ?? 100}
                    onChange={(event) => onBenchmarkDraftChange("taskLimit", Number(event.target.value) as any)}
                  />
                </label>
              </div>
            ) : null}

            {selectedPrompt && (!benchmarkDraft.mode || benchmarkDraft.mode === "throughput") ? (
              <div className="callout quiet benchmark-prompt-preview">
                <span className="eyebrow">{t("benchmarkTab.prompt", { defaultValue: "Prompt" })}</span>
                <p>{selectedPrompt.prompt ?? selectedPrompt.label}</p>
              </div>
            ) : null}
            {benchmarkDraft.mode === "perplexity" ? (
              <div className="callout quiet benchmark-prompt-preview">
                <span className="eyebrow">{t("benchmarkTab.perplexity", { defaultValue: "Perplexity" })}</span>
                <p>{t("benchmarkTab.perplexityDescription", { defaultValue: "Measures how well the model predicts text. Lower is better. Compares real quality loss across quantization levels." })}</p>
              </div>
            ) : null}
            {benchmarkDraft.mode === "task_accuracy" ? (
              <div className="callout quiet benchmark-prompt-preview">
                <span className="eyebrow">{t("benchmarkTab.taskAccuracy", { defaultValue: "Task Accuracy" })}</span>
                <p>{t("benchmarkTab.taskAccuracyDescription", { defaultValue: "Runs multiple-choice questions and scores the model's answers. Higher accuracy is better." })}</p>
              </div>
            ) : null}

            <RuntimeControls
              settings={benchmarkDraft}
              onChange={onBenchmarkDraftChange}
              maxContext={benchmarkOption?.maxContext}
              diskSizeGb={benchmarkOption?.sizeGb}
              preview={preview}
              availableMemoryGb={workspace.system.availableMemoryGb}
              totalMemoryGb={workspace.system.totalMemoryGb}
              gpuVramTotalGb={workspace.system.gpuVramTotalGb}
              availableCacheStrategies={workspace.system.availableCacheStrategies}
              dflashInfo={workspace.system.dflash}
              selectedBackend={benchmarkOption?.backend}
              selectedModelRef={benchmarkOption?.modelRef}
              selectedCanonicalRepo={benchmarkOption?.canonicalRepo}
              selectedModelName={benchmarkOption?.model}
              onInstallPackage={onInstallPackage}
              installingPackage={installingPackage}
              installLogs={installLogs}
              turboInstalled={Boolean(workspace.system.llamaServerTurboPath)}
              showTemperature={false}
              showPreview={false}
            />

            <div className="button-row">
              <button className="primary-button benchmark-run-btn" type="button" onClick={() => void onRunBenchmark()} disabled={busy}>
                {busy
                  ? t("benchmarkTab.running", { defaultValue: "Running..." })
                  : t("benchmarkTab.runBenchmark", { defaultValue: "\u25B6 Run benchmark" })}
              </button>
            </div>

            <div className="callout quiet">
              <h3>{t("benchmarkTab.approachTitle", { defaultValue: "Benchmarking approach" })}</h3>
              <p>
                {t("benchmarkTab.approachBody", { defaultValue: "ChaosEngineAI loads the chosen runtime profile if needed, runs a consistent prompt, captures decode speed and response time, then stores the result so you can compare later runs side by side." })}
              </p>
            </div>
          </div>

          <div className="benchmark-run-preview scrollable-panel-content">
            <PerformancePreview
              preview={preview}
              availableMemoryGb={workspace.system.availableMemoryGb}
              totalMemoryGb={workspace.system.totalMemoryGb}
              gpuVramTotalGb={workspace.system.gpuVramTotalGb}
            />

            {latestRun ? (
              <div className="benchmark-last-run-card">
                <div className="benchmark-last-run-header">
                  <span className="eyebrow">{t("benchmarkTab.latestRun", { defaultValue: "Latest run" })}</span>
                  <span className="badge muted">{latestRun.measuredAt}</span>
                </div>
                <h3>{latestRun.model}</h3>
                <p className="muted-text">{latestRun.cacheLabel} {"\u00B7"} {latestRun.engineLabel}</p>

                {latestRun.mode === "perplexity" ? (
                  <BenchmarkGauge
                    value={latestRun.perplexity ?? 0}
                    max={50}
                    label={t("benchmarkTab.gauge.perplexityLabel", { defaultValue: "perplexity" })}
                    subtitle={t("benchmarkTab.gauge.lowerBetter", { defaultValue: "lower is better" })}
                  />
                ) : latestRun.mode === "task_accuracy" ? (
                  <BenchmarkGauge
                    value={(latestRun.taskAccuracy ?? 0) * 100}
                    max={100}
                    label={t("benchmarkTab.gauge.accuracyLabel", { defaultValue: "% accuracy" })}
                  />
                ) : (
                  <BenchmarkGauge value={latestRun.tokS} max={40} label="tok/s" />
                )}

                <div className="stat-grid compact-grid benchmark-last-run-stats">
                  {latestRun.mode === "perplexity" ? (
                    <>
                      <StatCard
                        label={t("benchmarkTab.statPerplexity", { defaultValue: "Perplexity" })}
                        value={`${number(latestRun.perplexity ?? 0)}`}
                        hint={t("benchmarkTab.perplexityStdErr", { value: number(latestRun.perplexityStdError ?? 0), defaultValue: "\u00B1 {value} SE" })}
                      />
                      <StatCard
                        label={t("benchmarkTab.statEvalSpeed", { defaultValue: "Eval speed" })}
                        value={`${number(latestRun.evalTokensPerSecond ?? 0)} tok/s`}
                        hint={t("benchmarkTab.evalSecondsTotal", { value: number(latestRun.evalSeconds ?? 0), defaultValue: "{value} s total" })}
                      />
                      <StatCard
                        label={t("benchmarkTab.statDataset", { defaultValue: "Dataset" })}
                        value={latestRun.perplexityDataset ?? "wikitext-2"}
                        hint={t("benchmarkTab.samplesCount", { count: latestRun.perplexityNumSamples ?? 0, defaultValue: "{count} samples" })}
                      />
                      <StatCard
                        label={t("benchmarkTab.statCache", { defaultValue: "Cache" })}
                        value={`${number(latestRun.cacheGb)} GB`}
                        hint={t("benchmarkTab.compressionRatio", { ratio: number(latestRun.compression), defaultValue: "{ratio}x compression" })}
                      />
                    </>
                  ) : latestRun.mode === "task_accuracy" ? (
                    <>
                      <StatCard
                        label={t("benchmarkTab.statAccuracy", { defaultValue: "Accuracy" })}
                        value={`${((latestRun.taskAccuracy ?? 0) * 100).toFixed(1)}%`}
                        hint={t("benchmarkTab.correctOutOf", { correct: latestRun.taskCorrect, total: latestRun.taskTotal, defaultValue: "{correct}/{total} correct" })}
                      />
                      <StatCard
                        label={t("benchmarkTab.statTask", { defaultValue: "Task" })}
                        value={(latestRun.taskName ?? "mmlu").toUpperCase()}
                        hint={t("benchmarkTab.shotsHint", { shots: latestRun.taskNumShots ?? 5, defaultValue: "{shots}-shot" })}
                      />
                      <StatCard
                        label={t("benchmarkTab.statEvalTime", { defaultValue: "Eval time" })}
                        value={`${number(latestRun.evalSeconds ?? 0)} s`}
                        hint={t("benchmarkTab.loadSeconds", { value: number(latestRun.loadSeconds), defaultValue: "{value} s load" })}
                      />
                      <StatCard
                        label={t("benchmarkTab.statCache", { defaultValue: "Cache" })}
                        value={`${number(latestRun.cacheGb)} GB`}
                        hint={t("benchmarkTab.compressionRatio", { ratio: number(latestRun.compression), defaultValue: "{ratio}x compression" })}
                      />
                    </>
                  ) : (
                    <>
                      <StatCard
                        label={t("benchmarkTab.statResponseTime", { defaultValue: "Response time" })}
                        value={`${number(latestRun.responseSeconds)} s`}
                        hint={t("benchmarkTab.loadSeconds", { value: number(latestRun.loadSeconds), defaultValue: "{value} s load" })}
                      />
                      <StatCard
                        label={t("benchmarkTab.statCacheFootprint", { defaultValue: "Cache footprint" })}
                        value={`${number(latestRun.cacheGb)} GB`}
                        hint={t("benchmarkTab.compressionRatio", { ratio: number(latestRun.compression), defaultValue: "{ratio}x compression" })}
                      />
                      <StatCard
                        label={t("benchmarkTab.statQuality", { defaultValue: "Quality" })}
                        value={`${latestRun.quality}%`}
                        hint={t("benchmarkTab.tokensGenerated", { count: latestRun.completionTokens, defaultValue: "{count} tokens generated" })}
                      />
                      <StatCard
                        label={t("benchmarkTab.statContext", { defaultValue: "Context" })}
                        value={`${latestRun.contextTokens.toLocaleString()}`}
                        hint={t("benchmarkTab.maxTokensHint", { max: latestRun.maxTokens, defaultValue: "{max} max" })}
                      />
                    </>
                  )}
                </div>

                {speedDeltaVsPrev !== null && prevForModel ? (
                  <div className="callout quiet benchmark-delta-note">
                    <p>
                      {t("benchmarkTab.deltaVsPrev", { delta: signedDelta(speedDeltaVsPrev), cache: prevForModel.cacheLabel, defaultValue: "{delta} tok/s vs your previous {cache} run" })}
                      {fastestRun && fastestRun.id !== latestRun.id ? ` \u00B7 ${t("benchmarkTab.deltaVsFastest", { delta: signedDelta(latestRun.tokS - fastestRun.tokS), cache: fastestRun.cacheLabel, defaultValue: "{delta} tok/s vs fastest ({cache})" })}` : ""}
                    </p>
                  </div>
                ) : null}

                <div className="button-row">
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={() => {
                      onSelectedBenchmarkIdChange(latestRun.id);
                      if (prevForModel) onCompareBenchmarkIdChange(prevForModel.id);
                      onActiveTabChange("benchmark-history");
                    }}
                  >
                    {t("benchmarkTab.viewInHistory", { defaultValue: "View in History" })}
                  </button>
                </div>
              </div>
            ) : (
              <div className="empty-state">
                <p>{t("benchmarkTab.noRunsYet", { defaultValue: "No benchmark runs yet. Configure a profile on the left and click Run benchmark." })}</p>
              </div>
            )}
          </div>
        </div>
      </Panel>

      {showBenchmarkModal ? (
        <div className="modal-overlay benchmark-result-modal">
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>
                {busyAction === "Running benchmark..."
                  ? t("benchmarkTab.modal.titleRunning", { defaultValue: "Running benchmark" })
                  : benchmarkError
                    ? t("benchmarkTab.modal.titleFailed", { defaultValue: "Benchmark failed" })
                    : t("benchmarkTab.modal.titleComplete", { defaultValue: "Benchmark complete" })}
              </h3>
            </div>
            <div className="modal-body">
              {busyAction === "Running benchmark..." && benchmarkStartedAt ? (
                <>
                  <LiveProgress
                    title={t("benchmarkTab.progress.title", { defaultValue: "Running benchmark" })}
                    subtitle={benchmarkOption?.model ?? undefined}
                    startedAt={benchmarkStartedAt}
                    accent="benchmark"
                    phases={[
                      { id: "load", label: t("benchmarkTab.progress.phases.load", { defaultValue: "Loading model into memory" }), estimatedSeconds: 12 },
                      { id: "warm", label: t("benchmarkTab.progress.phases.warm", { defaultValue: "Warming up KV cache" }), estimatedSeconds: 4 },
                      { id: "prompt", label: t("benchmarkTab.progress.phases.prompt", { defaultValue: "Processing prompt" }), estimatedSeconds: 3 },
                      { id: "generate", label: t("benchmarkTab.progress.phases.generate", { tokens: benchmarkDraft.maxTokens, defaultValue: `Generating ${benchmarkDraft.maxTokens} tokens` }), estimatedSeconds: Math.max(8, benchmarkDraft.maxTokens / 25) },
                      { id: "measure", label: t("benchmarkTab.progress.phases.measure", { defaultValue: "Measuring stats" }), estimatedSeconds: 2 },
                    ] as LiveProgressPhase[]}
                  />
                  <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 12 }}>
                    <button
                      type="button"
                      className="secondary-button danger-button"
                      onClick={onCancelBenchmark}
                    >
                      {t("benchmarkTab.modal.cancel", { defaultValue: "Cancel benchmark" })}
                    </button>
                  </div>
                </>
              ) : benchmarkError ? (
                <div className="callout error">
                  <h3>{t("benchmarkTab.modal.titleFailed", { defaultValue: "Benchmark failed" })}</h3>
                  <p>{benchmarkError}</p>
                  <details className="debug-details">
                    <summary>{t("benchmarkTab.modal.debugDetails", { defaultValue: "Debug details" })}</summary>
                    <dl className="debug-grid">
                      <dt>{t("benchmarkTab.modal.model", { defaultValue: "Model" })}</dt>
                      <dd><code>{benchmarkDraft.modelRef || "\u2014"}</code></dd>
                      <dt>{t("benchmarkTab.modal.source", { defaultValue: "Source" })}</dt>
                      <dd><code>{benchmarkDraft.source || "\u2014"}</code></dd>
                      <dt>{t("benchmarkTab.modal.backend", { defaultValue: "Backend" })}</dt>
                      <dd>{benchmarkDraft.backend || "auto"}</dd>
                      <dt>{t("benchmarkTab.modal.path", { defaultValue: "Path" })}</dt>
                      <dd><code>{benchmarkDraft.path || "\u2014"}</code></dd>
                      <dt>{t("benchmarkTab.modal.profile", { defaultValue: "Profile" })}</dt>
                      <dd>{benchmarkDraft.cacheStrategy} {benchmarkDraft.cacheBits}-bit {"\u00B7"} fp16{"\u00D7"}{benchmarkDraft.fp16Layers} {"\u00B7"} ctx {benchmarkDraft.contextTokens} {"\u00B7"} {benchmarkDraft.maxTokens} tok</dd>
                      <dt>{t("benchmarkTab.promptPreset", { defaultValue: "Prompt preset" })}</dt>
                      <dd>{benchmarkPromptId}</dd>
                    </dl>
                    <p className="muted-text debug-hint">
                      {t("benchmarkTab.modal.tailHint", { defaultValue: "Run" })} <code>tail -100 $(ls -t $TMPDIR/chaosengine-backend-*.log | head -1)</code> {t("benchmarkTab.modal.terminalHint", { defaultValue: "in Terminal for full stderr." })}
                    </p>
                  </details>
                </div>
              ) : latestRun ? (
                <div className="benchmark-last-run-card">
                  <div className="benchmark-last-run-header">
                    <span className="eyebrow">{t("benchmarkTab.latestRun", { defaultValue: "Latest run" })}</span>
                    <span className="badge muted">{latestRun.measuredAt}</span>
                  </div>
                  <h3>{latestRun.model}</h3>
                  <p className="muted-text">{latestRun.cacheLabel} {"\u00B7"} {latestRun.engineLabel}</p>

                  <BenchmarkGauge value={latestRun.tokS} max={40} label="tok/s" />

                  <div className="stat-grid compact-grid benchmark-last-run-stats">
                    <StatCard
                      label={t("benchmarkTab.statResponseTime", { defaultValue: "Response time" })}
                      value={`${number(latestRun.responseSeconds)} s`}
                      hint={t("benchmarkTab.loadSeconds", { value: number(latestRun.loadSeconds), defaultValue: "{value} s load" })}
                    />
                    <StatCard
                      label={t("benchmarkTab.statCacheFootprint", { defaultValue: "Cache footprint" })}
                      value={`${number(latestRun.cacheGb)} GB`}
                      hint={t("benchmarkTab.compressionRatio", { ratio: number(latestRun.compression), defaultValue: "{ratio}x compression" })}
                    />
                    <StatCard
                      label={t("benchmarkTab.statQuality", { defaultValue: "Quality" })}
                      value={`${latestRun.quality}%`}
                      hint={t("benchmarkTab.tokensGenerated", { count: latestRun.completionTokens, defaultValue: "{count} tokens generated" })}
                    />
                    <StatCard
                      label={t("benchmarkTab.statContext", { defaultValue: "Context" })}
                      value={`${latestRun.contextTokens.toLocaleString()}`}
                      hint={t("benchmarkTab.maxTokensHint", { max: latestRun.maxTokens, defaultValue: "{max} max" })}
                    />
                  </div>

                  {speedDeltaVsPrev !== null && prevForModel ? (
                    <div className="callout quiet benchmark-delta-note">
                      <p>
                        {t("benchmarkTab.deltaVsPrev", { delta: signedDelta(speedDeltaVsPrev), cache: prevForModel.cacheLabel, defaultValue: "{delta} tok/s vs your previous {cache} run" })}
                        {fastestRun && fastestRun.id !== latestRun.id ? ` \u00B7 ${t("benchmarkTab.deltaVsFastest", { delta: signedDelta(latestRun.tokS - fastestRun.tokS), cache: fastestRun.cacheLabel, defaultValue: "{delta} tok/s vs fastest ({cache})" })}` : ""}
                      </p>
                    </div>
                  ) : null}
                </div>
              ) : null}
            </div>
            {busyAction !== "Running benchmark..." ? (
              <div className="modal-footer">
                <button
                  className="primary-button"
                  type="button"
                  onClick={() => onShowBenchmarkModalChange(false)}
                >
                  {benchmarkError
                    ? t("benchmarkTab.modal.close", { defaultValue: "Close" })
                    : t("benchmarkTab.modal.ok", { defaultValue: "OK" })}
                </button>
              </div>
            ) : null}
          </div>
        </div>
      ) : null}
      <ModelLaunchModal
        open={showBenchmarkPicker}
        title={t("panels.selectBenchmarkModel", { defaultValue: "Select Benchmark Model" })}
        confirmLabel={t("benchmarkTab.select", { defaultValue: "Select" })}
        selectedKey={benchmarkOption?.key ?? threadModelOptions[0]?.key ?? ""}
        collapseOnOpen={Boolean(benchmarkOption?.key)}
        search={benchmarkPickerSearch}
        options={threadModelOptions}
        settings={{
          contextTokens: benchmarkDraft.contextTokens,
          maxTokens: benchmarkDraft.maxTokens,
          temperature: benchmarkDraft.temperature,
          cacheBits: benchmarkDraft.cacheBits,
          fp16Layers: benchmarkDraft.fp16Layers,
          fusedAttention: benchmarkDraft.fusedAttention,
          cacheStrategy: benchmarkDraft.cacheStrategy,
          fitModelInMemory: benchmarkDraft.fitModelInMemory,
          speculativeDecoding: benchmarkDraft.speculativeDecoding,
          treeBudget: benchmarkDraft.treeBudget,
          kvBudget: benchmarkDraft.kvBudget,
        }}
        preview={preview}
        availableMemoryGb={workspace.system.availableMemoryGb}
        totalMemoryGb={workspace.system.totalMemoryGb}
        gpuVramTotalGb={workspace.system.gpuVramTotalGb}
        availableCacheStrategies={workspace.system.availableCacheStrategies}
        dflashInfo={workspace.system.dflash}
        installingPackage={installingPackage}
        installLogs={installLogs}
        turboInstalled={Boolean(workspace.system.llamaServerTurboPath)}
        mtplxSystemInfo={workspace.system.mtplx}
        onInstallMtplx={onInstallMtplx}
        installingMtplx={installingMtplx}
        mtplxJob={mtplxJob}
        isAppleSilicon={isAppleSilicon}
        onSelectedKeyChange={(key) => {
          onBenchmarkModelKeyChange(key);
        }}
        onSearchChange={setBenchmarkPickerSearch}
        onSettingChange={(key, value) => {
          onBenchmarkDraftChange(key as keyof BenchmarkRunPayload, value as BenchmarkRunPayload[typeof key]);
        }}
        onConfirm={(selectedKey) => {
          onBenchmarkModelKeyChange(selectedKey);
          const option = threadModelOptions.find((o) => o.key === selectedKey);
          if (option) {
            onBenchmarkDraftUpdate((current) => ({
              ...current,
              modelRef: option.modelRef ?? option.model,
              modelName: option.label,
              source: option.source,
              backend: option.backend,
              path: option.path ?? undefined,
            }));
          }
          setBenchmarkPickerSearch("");
          onShowBenchmarkPickerChange(false);
        }}
        onClose={() => {
          setBenchmarkPickerSearch("");
          onShowBenchmarkPickerChange(false);
        }}
        onInstallPackage={(strategyId) => void onInstallPackage(strategyId)}
      />
    </div>
  );
}
