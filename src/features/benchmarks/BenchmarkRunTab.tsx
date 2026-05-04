import { useState } from "react";
import { Panel } from "../../components/Panel";
import { PerformancePreview } from "../../components/PerformancePreview";
import { LiveProgress, type LiveProgressPhase } from "../../components/LiveProgress";
import { RuntimeControls } from "../../components/RuntimeControls";
import { ModelLaunchModal } from "../../components/ModelLaunchModal";
import { StatCard } from "../../components/StatCard";
import { BenchmarkGauge } from "../../components/BenchmarkGauge";
import type { BenchmarkResult, BenchmarkRunPayload, LibraryItem, PreviewMetrics, StrategyInstallLog, SystemStats } from "../../types";
import type { ChatModelOption } from "../../types/chat";
import { BENCHMARK_PROMPTS } from "../../constants";
import { number, sizeLabel, signedDelta } from "../../utils";

export interface BenchmarkRunTabProps {
  workspace: {
    benchmarks: BenchmarkResult[];
    library: LibraryItem[];
    system: {
      availableMemoryGb: number;
      totalMemoryGb: number;
      availableCacheStrategies: SystemStats["availableCacheStrategies"];
      llamaServerTurboPath?: string | null;
      dflash?: SystemStats["dflash"];
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
  onBenchmarkDraftChange: <K extends keyof BenchmarkRunPayload>(key: K, value: BenchmarkRunPayload[K]) => void;
  onBenchmarkPromptIdChange: (id: string) => void;
  onBenchmarkModelKeyChange: (key: string) => void;
  onBenchmarkDraftUpdate: (updater: (current: BenchmarkRunPayload) => BenchmarkRunPayload) => void;
  onRunBenchmark: () => void;
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
  onBenchmarkDraftChange,
  onBenchmarkPromptIdChange,
  onBenchmarkModelKeyChange,
  onBenchmarkDraftUpdate,
  onRunBenchmark,
  onShowBenchmarkPickerChange,
  onShowBenchmarkModalChange,
  onSelectedBenchmarkIdChange,
  onCompareBenchmarkIdChange,
  onActiveTabChange,
  onInstallPackage,
}: BenchmarkRunTabProps) {
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
        title="Run Benchmark"
        subtitle="Launch a consistent benchmark run and see how this profile performs."
        className="span-2 benchmark-run-page-panel"
      >
        <div className="benchmark-run-page">
          <div className="benchmark-run-config scrollable-panel-content">
            <div className="benchmark-run-header-grid">
              <label className="field">
                Benchmark model
                <div className="model-selected-card">
                  <div className="model-selected-info">
                    <strong>{benchmarkOption?.label ?? benchmarkDraft.modelName ?? "Select a model"}</strong>
                    <div className="model-selected-meta">
                      {benchmarkOption?.format ? <span className="badge muted">{benchmarkOption.format}</span> : null}
                      {benchmarkOption?.sizeGb ? <span className="badge muted">{sizeLabel(benchmarkOption.sizeGb)}</span> : null}
                    </div>
                  </div>
                  <button className="secondary-button" type="button" onClick={() => onShowBenchmarkPickerChange(true)}>
                    Change
                  </button>
                </div>
              </label>
              <div className="benchmark-run-mode-stack">
                <label className="field">
                  Benchmark mode
                  <select
                    className="text-input"
                    value={benchmarkDraft.mode ?? "throughput"}
                    onChange={(event) => onBenchmarkDraftChange("mode", event.target.value as any)}
                  >
                    <option value="throughput">Throughput (tok/s)</option>
                    <option value="perplexity">Perplexity (quality)</option>
                    <option value="task_accuracy">Task Accuracy (MMLU / HellaSwag)</option>
                  </select>
                </label>
                {(!benchmarkDraft.mode || benchmarkDraft.mode === "throughput") ? (
                  <label className="field">
                    Prompt preset
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
                  Dataset
                  <select
                    className="text-input"
                    value={benchmarkDraft.perplexityDataset ?? "wikitext-2"}
                    onChange={(event) => onBenchmarkDraftChange("perplexityDataset", event.target.value as any)}
                  >
                    <option value="wikitext-2">WikiText-2</option>
                  </select>
                </label>
                <label>
                  Samples
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
                  Task
                  <select
                    className="text-input"
                    value={benchmarkDraft.taskName ?? "mmlu"}
                    onChange={(event) => onBenchmarkDraftChange("taskName", event.target.value as any)}
                  >
                    <option value="mmlu">MMLU (multiple choice)</option>
                    <option value="hellaswag">HellaSwag (sentence completion)</option>
                  </select>
                </label>
                <label>
                  Questions
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
                <span className="eyebrow">Prompt</span>
                <p>{selectedPrompt.prompt ?? selectedPrompt.label}</p>
              </div>
            ) : null}
            {benchmarkDraft.mode === "perplexity" ? (
              <div className="callout quiet benchmark-prompt-preview">
                <span className="eyebrow">Perplexity</span>
                <p>Measures how well the model predicts text. Lower is better. Compares real quality loss across quantization levels.</p>
              </div>
            ) : null}
            {benchmarkDraft.mode === "task_accuracy" ? (
              <div className="callout quiet benchmark-prompt-preview">
                <span className="eyebrow">Task Accuracy</span>
                <p>Runs multiple-choice questions and scores the model's answers. Higher accuracy is better.</p>
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
                {busy ? "Running..." : "\u25B6 Run benchmark"}
              </button>
            </div>

            <div className="callout quiet">
              <h3>Benchmarking approach</h3>
              <p>
                ChaosEngineAI loads the chosen runtime profile if needed, runs a consistent prompt, captures decode speed and response time, then stores the result so you can compare later runs side by side.
              </p>
            </div>
          </div>

          <div className="benchmark-run-preview scrollable-panel-content">
            <PerformancePreview
              preview={preview}
              availableMemoryGb={workspace.system.availableMemoryGb}
              totalMemoryGb={workspace.system.totalMemoryGb}
            />

            {latestRun ? (
              <div className="benchmark-last-run-card">
                <div className="benchmark-last-run-header">
                  <span className="eyebrow">Latest run</span>
                  <span className="badge muted">{latestRun.measuredAt}</span>
                </div>
                <h3>{latestRun.model}</h3>
                <p className="muted-text">{latestRun.cacheLabel} {"\u00B7"} {latestRun.engineLabel}</p>

                {latestRun.mode === "perplexity" ? (
                  <BenchmarkGauge value={latestRun.perplexity ?? 0} max={50} label="perplexity" subtitle="lower is better" />
                ) : latestRun.mode === "task_accuracy" ? (
                  <BenchmarkGauge value={(latestRun.taskAccuracy ?? 0) * 100} max={100} label="% accuracy" />
                ) : (
                  <BenchmarkGauge value={latestRun.tokS} max={40} label="tok/s" />
                )}

                <div className="stat-grid compact-grid benchmark-last-run-stats">
                  {latestRun.mode === "perplexity" ? (
                    <>
                      <StatCard label="Perplexity" value={`${number(latestRun.perplexity ?? 0)}`} hint={`\u00B1 ${number(latestRun.perplexityStdError ?? 0)} SE`} />
                      <StatCard label="Eval speed" value={`${number(latestRun.evalTokensPerSecond ?? 0)} tok/s`} hint={`${number(latestRun.evalSeconds ?? 0)} s total`} />
                      <StatCard label="Dataset" value={latestRun.perplexityDataset ?? "wikitext-2"} hint={`${latestRun.perplexityNumSamples ?? 0} samples`} />
                      <StatCard label="Cache" value={`${number(latestRun.cacheGb)} GB`} hint={`${number(latestRun.compression)}x compression`} />
                    </>
                  ) : latestRun.mode === "task_accuracy" ? (
                    <>
                      <StatCard label="Accuracy" value={`${((latestRun.taskAccuracy ?? 0) * 100).toFixed(1)}%`} hint={`${latestRun.taskCorrect}/${latestRun.taskTotal} correct`} />
                      <StatCard label="Task" value={(latestRun.taskName ?? "mmlu").toUpperCase()} hint={`${latestRun.taskNumShots ?? 5}-shot`} />
                      <StatCard label="Eval time" value={`${number(latestRun.evalSeconds ?? 0)} s`} hint={`${number(latestRun.loadSeconds)} s load`} />
                      <StatCard label="Cache" value={`${number(latestRun.cacheGb)} GB`} hint={`${number(latestRun.compression)}x compression`} />
                    </>
                  ) : (
                    <>
                      <StatCard label="Response time" value={`${number(latestRun.responseSeconds)} s`} hint={`${number(latestRun.loadSeconds)} s load`} />
                      <StatCard label="Cache footprint" value={`${number(latestRun.cacheGb)} GB`} hint={`${number(latestRun.compression)}x compression`} />
                      <StatCard label="Quality" value={`${latestRun.quality}%`} hint={`${latestRun.completionTokens} tokens generated`} />
                      <StatCard label="Context" value={`${latestRun.contextTokens.toLocaleString()}`} hint={`${latestRun.maxTokens} max`} />
                    </>
                  )}
                </div>

                {speedDeltaVsPrev !== null && prevForModel ? (
                  <div className="callout quiet benchmark-delta-note">
                    <p>
                      {signedDelta(speedDeltaVsPrev)} tok/s vs your previous {prevForModel.cacheLabel} run
                      {fastestRun && fastestRun.id !== latestRun.id ? ` \u00B7 ${signedDelta(latestRun.tokS - fastestRun.tokS)} tok/s vs fastest (${fastestRun.cacheLabel})` : ""}
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
                    View in History
                  </button>
                </div>
              </div>
            ) : (
              <div className="empty-state">
                <p>No benchmark runs yet. Configure a profile on the left and click Run benchmark.</p>
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
                  ? "Running benchmark"
                  : benchmarkError
                    ? "Benchmark failed"
                    : "Benchmark complete"}
              </h3>
            </div>
            <div className="modal-body">
              {busyAction === "Running benchmark..." && benchmarkStartedAt ? (
                <LiveProgress
                  title="Running benchmark"
                  subtitle={benchmarkOption?.model ?? undefined}
                  startedAt={benchmarkStartedAt}
                  accent="benchmark"
                  phases={[
                    { id: "load", label: "Loading model into memory", estimatedSeconds: 12 },
                    { id: "warm", label: "Warming up KV cache", estimatedSeconds: 4 },
                    { id: "prompt", label: "Processing prompt", estimatedSeconds: 3 },
                    { id: "generate", label: `Generating ${benchmarkDraft.maxTokens} tokens`, estimatedSeconds: Math.max(8, benchmarkDraft.maxTokens / 25) },
                    { id: "measure", label: "Measuring stats", estimatedSeconds: 2 },
                  ] as LiveProgressPhase[]}
                />
              ) : benchmarkError ? (
                <div className="callout error">
                  <h3>Benchmark failed</h3>
                  <p>{benchmarkError}</p>
                  <details className="debug-details">
                    <summary>Debug details</summary>
                    <dl className="debug-grid">
                      <dt>Model</dt>
                      <dd><code>{benchmarkDraft.modelRef || "\u2014"}</code></dd>
                      <dt>Source</dt>
                      <dd><code>{benchmarkDraft.source || "\u2014"}</code></dd>
                      <dt>Backend</dt>
                      <dd>{benchmarkDraft.backend || "auto"}</dd>
                      <dt>Path</dt>
                      <dd><code>{benchmarkDraft.path || "\u2014"}</code></dd>
                      <dt>Profile</dt>
                      <dd>{benchmarkDraft.cacheStrategy} {benchmarkDraft.cacheBits}-bit {"\u00B7"} fp16{"\u00D7"}{benchmarkDraft.fp16Layers} {"\u00B7"} ctx {benchmarkDraft.contextTokens} {"\u00B7"} {benchmarkDraft.maxTokens} tok</dd>
                      <dt>Prompt preset</dt>
                      <dd>{benchmarkPromptId}</dd>
                    </dl>
                    <p className="muted-text debug-hint">
                      Run <code>tail -100 $(ls -t $TMPDIR/chaosengine-backend-*.log | head -1)</code> in Terminal for full stderr.
                    </p>
                  </details>
                </div>
              ) : latestRun ? (
                <div className="benchmark-last-run-card">
                  <div className="benchmark-last-run-header">
                    <span className="eyebrow">Latest run</span>
                    <span className="badge muted">{latestRun.measuredAt}</span>
                  </div>
                  <h3>{latestRun.model}</h3>
                  <p className="muted-text">{latestRun.cacheLabel} {"\u00B7"} {latestRun.engineLabel}</p>

                  <BenchmarkGauge value={latestRun.tokS} max={40} label="tok/s" />

                  <div className="stat-grid compact-grid benchmark-last-run-stats">
                    <StatCard
                      label="Response time"
                      value={`${number(latestRun.responseSeconds)} s`}
                      hint={`${number(latestRun.loadSeconds)} s load`}
                    />
                    <StatCard
                      label="Cache footprint"
                      value={`${number(latestRun.cacheGb)} GB`}
                      hint={`${number(latestRun.compression)}x compression`}
                    />
                    <StatCard
                      label="Quality"
                      value={`${latestRun.quality}%`}
                      hint={`${latestRun.completionTokens} tokens generated`}
                    />
                    <StatCard
                      label="Context"
                      value={`${latestRun.contextTokens.toLocaleString()}`}
                      hint={`${latestRun.maxTokens} max`}
                    />
                  </div>

                  {speedDeltaVsPrev !== null && prevForModel ? (
                    <div className="callout quiet benchmark-delta-note">
                      <p>
                        {signedDelta(speedDeltaVsPrev)} tok/s vs your previous {prevForModel.cacheLabel} run
                        {fastestRun && fastestRun.id !== latestRun.id ? ` \u00B7 ${signedDelta(latestRun.tokS - fastestRun.tokS)} tok/s vs fastest (${fastestRun.cacheLabel})` : ""}
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
                  {benchmarkError ? "Close" : "OK"}
                </button>
              </div>
            ) : null}
          </div>
        </div>
      ) : null}
      <ModelLaunchModal
        open={showBenchmarkPicker}
        title="Select Benchmark Model"
        confirmLabel="Select"
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
        availableCacheStrategies={workspace.system.availableCacheStrategies}
        dflashInfo={workspace.system.dflash}
        installingPackage={installingPackage}
        installLogs={installLogs}
        turboInstalled={Boolean(workspace.system.llamaServerTurboPath)}
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
