import { useTranslation } from "react-i18next";
import { Panel } from "../../components/Panel";
import { PerformancePreview } from "../../components/PerformancePreview";
import { LiveProgress, type LiveProgressPhase } from "../../components/LiveProgress";
import { StatCard } from "../../components/StatCard";
import { ModelPicker } from "../../components/ModelPicker";
import { SliderField } from "../../components/SliderField";
import type { ConversionResult, LibraryItem, ModelVariant, NativeBackendStatus, PreviewMetrics } from "../../types";
import { number, sizeLabel, libraryItemFormat } from "../../utils";

export interface ConversionDraft {
  modelRef: string;
  path: string;
  hfRepo: string;
  outputPath: string;
  quantize: boolean;
  qBits: number;
  qGroupSize: number;
  dtype: string;
}

export interface ConversionTabProps {
  conversionSource: LibraryItem | null;
  conversionVariant: ModelVariant | null;
  conversionDraft: ConversionDraft;
  lastConversion: ConversionResult | null;
  conversionReady: boolean;
  convertibleLibrary: LibraryItem[];
  nativeBackends: NativeBackendStatus | undefined;
  preview: PreviewMetrics;
  workspace: {
    system: { availableMemoryGb: number; totalMemoryGb: number; gpuVramTotalGb?: number | null };
    library: LibraryItem[];
  };
  launchCacheLabel: string;
  busy: boolean;
  busyAction: string | null;
  conversionStartedAt: number | null;
  conversionError: string | null;
  showConversionPicker: boolean;
  showConversionModal: boolean;
  fileRevealLabel: string;
  onConversionDraftChange: <K extends keyof ConversionDraft>(key: K, value: ConversionDraft[K]) => void;
  onConversionDraftReset: () => void;
  onConvertModel: () => void;
  onPickConversionOutputDir: () => void;
  onShowConversionPickerChange: (open: boolean) => void;
  onShowConversionModalChange: (open: boolean) => void;
  onPrepareLibraryConversion: (item: LibraryItem, resolvedPath?: string) => void;
  onRevealPath: (path: string) => void;
}

export function ConversionTab({
  conversionSource,
  conversionVariant,
  conversionDraft,
  lastConversion,
  conversionReady,
  convertibleLibrary,
  nativeBackends,
  preview,
  workspace,
  launchCacheLabel,
  busy,
  busyAction,
  conversionStartedAt,
  conversionError,
  showConversionPicker,
  showConversionModal,
  fileRevealLabel,
  onConversionDraftChange,
  onConversionDraftReset,
  onConvertModel,
  onPickConversionOutputDir,
  onShowConversionPickerChange,
  onShowConversionModalChange,
  onPrepareLibraryConversion,
  onRevealPath,
}: ConversionTabProps) {
  const { t } = useTranslation("common");
  const beforeSize = conversionSource?.sizeGb ?? lastConversion?.sourceSizeGb ?? null;
  const estimatedContext = lastConversion?.contextWindow ?? conversionVariant?.contextWindow ?? t("conversionTab.varies", { defaultValue: "Varies" });

  // --- Live projection from the conversion draft (responds immediately to slider changes) ---
  // Detect the source's effective bits-per-weight from name/quantization metadata
  const detectSourceBpw = (): number => {
    const haystack = `${conversionSource?.name ?? ""} ${conversionSource?.format ?? ""} ${conversionVariant?.quantization ?? ""} ${conversionVariant?.format ?? ""}`.toLowerCase();
    const match = haystack.match(/(\d)[\s-]?bit|q(\d)/);
    if (match) {
      const bits = Number(match[1] ?? match[2]);
      if (bits >= 2 && bits <= 8) return bits + 0.5; // +0.5 for group overhead
    }
    if (/bf16|fp16|float16|f16/.test(haystack)) return 16;
    if (/fp32|float32|f32/.test(haystack)) return 32;
    return 16; // safe default — assume bf16
  };
  const sourceBpw = detectSourceBpw();
  const isReQuantizing = sourceBpw < 12; // source is already quantized

  const dtypeBytes = conversionDraft.dtype === "float32" ? 4 : 2;
  // Group quantization adds ~16 bits scale/zero per group, amortized per weight
  const groupOverheadBitsPerWeight = conversionDraft.quantize ? 16 / Math.max(8, conversionDraft.qGroupSize) : 0;
  const effectiveBitsPerWeight = conversionDraft.quantize
    ? conversionDraft.qBits + groupOverheadBitsPerWeight
    : dtypeBytes * 8;

  // Project disk-after by scaling source disk size by the bits ratio (much more accurate than paramsB-based math)
  const projectedDiskGb = beforeSize ? beforeSize * (effectiveBitsPerWeight / sourceBpw) : null;
  const afterSize = lastConversion?.outputSizeGb ?? projectedDiskGb;

  // Quality model: anchored to MLX-LM published recovery numbers (vs FP16 reference)
  const qualityByBits: Record<number, number> = { 2: 78, 3: 90, 4: 96.5, 5: 98.5, 6: 99.3, 8: 99.85 };
  const baseQuality = conversionDraft.quantize ? (qualityByBits[conversionDraft.qBits] ?? 95) : 100;
  // Smaller groups = higher quality (up to +1.5); larger groups = up to -1.5
  const groupQualityShift = conversionDraft.quantize ? Math.max(-1.5, Math.min(1.5, ((64 - conversionDraft.qGroupSize) / 32) * 0.75)) : 0;
  // Re-quantization penalty: requantizing an already-quantized source loses additional quality
  let reQuantPenalty = 0;
  if (isReQuantizing && conversionDraft.quantize) {
    if (conversionDraft.qBits >= sourceBpw - 0.5) {
      reQuantPenalty = 0.5; // round-trip noise
    } else {
      // Going lower than source — losses compound
      const drop = sourceBpw - conversionDraft.qBits;
      reQuantPenalty = Math.min(20, drop * 6);
    }
  }
  const projectedQualityPercent = Math.min(100, Math.max(0, baseQuality + groupQualityShift - reQuantPenalty));

  // Speed projection: memory-bandwidth bound, scales inversely with effective bytes/weight vs source
  const speedupVsSource = sourceBpw / effectiveBitsPerWeight;
  const baseTokS = preview.estimatedTokS > 0 ? preview.estimatedTokS : 35;
  const projectedTokS = baseTokS * speedupVsSource;
  const estimatedTokS = lastConversion?.estimatedTokS ?? projectedTokS;

  const cacheBefore = lastConversion?.baselineCacheGb ?? preview.baselineCacheGb;
  const cacheAfter = lastConversion?.optimizedCacheGb ?? preview.optimizedCacheGb;

  const conversionCompression =
    beforeSize && afterSize && afterSize > 0
      ? t("conversionTab.smallerOnDisk", { ratio: number(beforeSize / afterSize), defaultValue: "{ratio}x smaller on disk" })
      : projectedDiskGb && beforeSize
        ? t("conversionTab.projectedRatio", { ratio: number(beforeSize / projectedDiskGb), defaultValue: "≈ {ratio}x projected" })
        : t("conversionTab.pickSourceHint", { defaultValue: "Pick a source and bits to project disk footprint" });

  return (
    <div className="content-grid">
      <Panel
        title={t("panels.mlxConversion", { defaultValue: "MLX Conversion" })}
        subtitle={t("panels.mlxConversionSubtitle", {
          defaultValue: "Prepare a local source, compare before and after stats, then convert into an MLX-ready output.",
        })}
        className="span-2"
        actions={
          <span className={`badge ${conversionReady ? "success" : "warning"}`}>
            {conversionReady
              ? t("conversionTab.converterReady", { defaultValue: "Converter ready" })
              : t("conversionTab.converterUnavailable", { defaultValue: "Converter unavailable" })}
          </span>
        }
      >
        <div className="conversion-layout">
          <div className="conversion-builder">
            {convertibleLibrary.length ? (
              <>
                <div className="conversion-source-picker">
                  <span className="eyebrow">{t("conversionTab.sourceModel", { defaultValue: "Source model" })}</span>
                  {conversionSource ? (
                    <div className="model-selected-card">
                      <div className="model-selected-info">
                        <strong>{conversionSource.name}</strong>
                        <div className="model-selected-meta">
                          <span className="badge muted">{conversionSource.format}</span>
                          <span className="badge muted">{sizeLabel(conversionSource.sizeGb)}</span>
                          {conversionSource.directoryLabel ? <span className="badge muted">{conversionSource.directoryLabel}</span> : null}
                        </div>
                      </div>
                      <button className="secondary-button" type="button" onClick={() => onShowConversionPickerChange(true)}>
                        {t("conversionTab.change", { defaultValue: "Change" })}
                      </button>
                    </div>
                  ) : (
                    <button className="secondary-button" type="button" onClick={() => onShowConversionPickerChange(true)} style={{ width: "100%" }}>
                      {t("conversionTab.selectModelToConvert", { defaultValue: "Select a model to convert..." })}
                    </button>
                  )}
                </div>

                <div className="field-grid">
                  <label>
                    {t("conversionTab.outputPath", { defaultValue: "Output path" })}
                    <div className="input-with-button">
                      <input
                        className="text-input"
                        type="text"
                        placeholder={t("conversionTab.outputPathPlaceholder", { defaultValue: "Leave blank to use ~/Models/<name>-mlx" })}
                        value={conversionDraft.outputPath}
                        onChange={(event) => onConversionDraftChange("outputPath", event.target.value)}
                      />
                      <button
                        className="secondary-button"
                        type="button"
                        onClick={() => void onPickConversionOutputDir()}
                        title={t("conversionTab.chooseOutputFolder", { defaultValue: "Choose output folder" })}
                      >
                        {t("conversionTab.browse", { defaultValue: "Browse..." })}
                      </button>
                    </div>
                  </label>
                  {conversionSource?.format?.toUpperCase() === "GGUF" ? (
                    <label>
                      {t("conversionTab.baseHfRepo", { defaultValue: "Base HF repo (required for GGUF)" })}
                      <input
                        className="text-input"
                        type="text"
                        placeholder={t("conversionTab.baseHfRepoPlaceholder", { defaultValue: "e.g. Qwen/Qwen2.5-7B-Instruct" })}
                        value={conversionDraft.hfRepo}
                        onChange={(event) => onConversionDraftChange("hfRepo", event.target.value)}
                      />
                    </label>
                  ) : null}
                </div>

                <SliderField
                  label={t("conversionTab.quantizationBits", { defaultValue: "Quantization bits" })}
                  value={conversionDraft.qBits}
                  min={2} max={8} step={1}
                  ticks={[{ value: 2, label: "2" }, { value: 3, label: "3" }, { value: 4, label: "4" }, { value: 5, label: "5" }, { value: 6, label: "6" }, { value: 7, label: "7" }, { value: 8, label: "8" }]}
                  formatValue={(v) => t("conversionTab.bitFormat", { value: v, defaultValue: "{value}-bit" })}
                  onChange={(v) => onConversionDraftChange("qBits", v)}
                />

                <SliderField
                  label={t("conversionTab.groupSize", { defaultValue: "Group size" })}
                  value={conversionDraft.qGroupSize}
                  min={32} max={128} step={32}
                  ticks={[{ value: 32, label: "32" }, { value: 64, label: "64" }, { value: 96, label: "96" }, { value: 128, label: "128" }]}
                  formatValue={(v) => t("conversionTab.weightsPerGroup", { value: v, defaultValue: "{value} weights/group" })}
                  onChange={(v) => onConversionDraftChange("qGroupSize", v)}
                />

                <div className="field-grid">
                  <label>
                    {t("conversionTab.dtype", { defaultValue: "Dtype" })}
                    <select
                      className="text-input"
                      value={conversionDraft.dtype}
                      onChange={(event) => onConversionDraftChange("dtype", event.target.value)}
                    >
                      <option value="float16">float16</option>
                      <option value="bfloat16">bfloat16</option>
                      <option value="float32">float32</option>
                    </select>
                  </label>
                  <label className="check-row">
                    <input
                      type="checkbox"
                      checked={conversionDraft.quantize}
                      onChange={(event) => onConversionDraftChange("quantize", event.target.checked)}
                    />
                    {t("conversionTab.quantizeConvertedWeights", { defaultValue: "Quantize converted weights" })}
                  </label>
                  <div className={`callout ${isReQuantizing ? "warning" : "quiet"} compact-callout`}>
                    <h3>{isReQuantizing
                      ? t("conversionTab.reQuantizingTitle", { defaultValue: "Re-quantizing an already quantized source" })
                      : t("conversionTab.backendNote", { defaultValue: "Backend note" })}</h3>
                    <p>
                      {isReQuantizing
                        ? t("conversionTab.reQuantizingDetail", { bits: Math.round(sourceBpw), defaultValue: "Source is already ~{bits}-bit. Going lower compounds quality loss — for best results convert from the original FP16/BF16 weights." })
                        : conversionReady
                          ? t("conversionTab.mlxLmAvailable", { defaultValue: "mlx-lm conversion is available in the active backend." })
                          : nativeBackends?.mlxMessage ?? t("conversionTab.startNativeSidecar", { defaultValue: "Start the native sidecar to enable conversion." })}
                    </p>
                  </div>
                </div>

                <div className="button-row">
                  <button
                    className="primary-button"
                    type="button"
                    onClick={() => void onConvertModel()}
                    disabled={!conversionReady || !conversionDraft.path || busy}
                  >
                    {busy
                      ? t("conversionTab.converting", { defaultValue: "Converting..." })
                      : t("conversionTab.convertToMlx", { defaultValue: "Convert to MLX" })}
                  </button>
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={onConversionDraftReset}
                  >
                    {t("conversionTab.clear", { defaultValue: "Clear" })}
                  </button>
                </div>
              </>
            ) : (
              <div className="empty-state">
                <p>{t("conversionTab.addDirsHint", { defaultValue: "Add model directories in Settings first, then conversion sources found there will appear here." })}</p>
              </div>
            )}
          </div>

          <div className="conversion-visuals">
            <div className="stat-grid compact-grid">
              <StatCard
                label={t("conversionTab.params", { defaultValue: "Params" })}
                value={conversionVariant ? `${number(conversionVariant.paramsB)}B` : (lastConversion?.paramsB ? `${number(lastConversion.paramsB)}B` : t("conversionTab.unknown", { defaultValue: "Unknown" }))}
                hint={estimatedContext}
              />
              <StatCard
                label={t("conversionTab.diskBefore", { defaultValue: "Disk before" })}
                value={beforeSize ? sizeLabel(beforeSize) : t("conversionTab.unknown", { defaultValue: "Unknown" })}
                hint={conversionSource?.format ?? lastConversion?.sourceFormat ?? t("conversionTab.sourceFallback", { defaultValue: "Source" })}
              />
              <StatCard
                label={t("conversionTab.diskAfter", { defaultValue: "Disk after" })}
                value={afterSize ? sizeLabel(afterSize) : t("conversionTab.pending", { defaultValue: "Pending" })}
                hint={conversionCompression}
              />
              <StatCard
                label={t("conversionTab.estTokS", { defaultValue: "Est. tok/s" })}
                value={`${number(estimatedTokS)} tok/s`}
                hint={t("conversionTab.usingCache", { cache: launchCacheLabel, defaultValue: "Using {cache}" })}
              />
            </div>

            <div className="conversion-compare">
              <div className="conversion-card">
                <span className="eyebrow">{t("conversionTab.before", { defaultValue: "Before" })}</span>
                <h3>{conversionSource?.name ?? lastConversion?.sourceLabel ?? t("conversionTab.chooseSource", { defaultValue: "Choose a source" })}</h3>
                <p>{conversionSource?.path ?? lastConversion?.sourcePath ?? t("conversionTab.sourceFootprintHint", { defaultValue: "Select a local GGUF or HF-cache source to inspect its current footprint." })}</p>
                <div className="metric-list">
                  <div className="metric-row">
                    <span>{t("conversionTab.format", { defaultValue: "Format" })}</span>
                    <strong>{conversionSource?.format ?? lastConversion?.sourceFormat ?? t("conversionTab.unknown", { defaultValue: "Unknown" })}{isReQuantizing ? ` · ~${Math.round(sourceBpw)}-bit` : ""}</strong>
                  </div>
                  <div className="metric-row">
                    <span>{t("conversionTab.onDiskSize", { defaultValue: "On-disk size" })}</span>
                    <strong>{beforeSize ? sizeLabel(beforeSize) : t("conversionTab.unknown", { defaultValue: "Unknown" })}</strong>
                  </div>
                  <div className="metric-row">
                    <span>{t("conversionTab.context", { defaultValue: "Context" })}</span>
                    <strong>{estimatedContext}</strong>
                  </div>
                  <div className="metric-row">
                    <span>{t("conversionTab.cacheFootprint", { defaultValue: "Cache footprint" })}</span>
                    <strong>{number(cacheBefore)} GB</strong>
                  </div>
                </div>
              </div>

              <div className="conversion-arrow" aria-hidden="true">
                <span>MLX</span>
              </div>

              <div className="conversion-card accent-card">
                <span className="eyebrow">{t("conversionTab.after", { defaultValue: "After" })}</span>
                <h3>{lastConversion
                  ? t("conversionTab.mlxReadyOutput", { defaultValue: "MLX-ready output" })
                  : t("conversionTab.targetPreview", { defaultValue: "Target preview" })}</h3>
                <p>{lastConversion?.outputPath ?? t("conversionTab.outputHint", { defaultValue: "Converted output will appear here together with derived stats and metadata." })}</p>
                <div className="metric-list">
                  <div className="metric-row">
                    <span>{t("conversionTab.targetProfile", { defaultValue: "Target profile" })}</span>
                    <strong>{conversionDraft.quantize
                      ? t("conversionTab.bitProfile", { bits: conversionDraft.qBits, group: conversionDraft.qGroupSize, defaultValue: "{bits}-bit g{group}" })
                      : t("conversionTab.unquantized", { defaultValue: "Unquantized" })} / {conversionDraft.dtype}</strong>
                  </div>
                  <div className="metric-row">
                    <span>{t("conversionTab.onDiskSize", { defaultValue: "On-disk size" })}</span>
                    <strong>{afterSize ? sizeLabel(afterSize) : t("conversionTab.pending", { defaultValue: "Pending" })}</strong>
                  </div>
                  <div className="metric-row">
                    <span>{t("conversionTab.cacheFootprint", { defaultValue: "Cache footprint" })}</span>
                    <strong>{number(cacheAfter)} GB</strong>
                  </div>
                  <div className="metric-row">
                    <span>{t("conversionTab.qualityEstimate", { defaultValue: "Quality estimate" })}</span>
                    <strong>{number(lastConversion?.qualityPercent ?? projectedQualityPercent, 1)}%</strong>
                  </div>
                </div>
              </div>
            </div>

            <PerformancePreview
              preview={preview}
              availableMemoryGb={workspace.system.availableMemoryGb}
              totalMemoryGb={workspace.system.totalMemoryGb}
              gpuVramTotalGb={workspace.system.gpuVramTotalGb}
            />

            {lastConversion && !busy ? (
              <div className="callout">
                <span className="badge success">{t("conversionTab.lastConversion", { defaultValue: "Last conversion" })}</span>
                <h3>{lastConversion.sourceLabel}</h3>
                <p>{lastConversion.outputPath}</p>
                <div className="field-grid detail-grid">
                  <div>
                    <span className="eyebrow">{t("conversionTab.baseRepo", { defaultValue: "Base repo" })}</span>
                    <p>{lastConversion.hfRepo}</p>
                  </div>
                  <div>
                    <span className="eyebrow">{t("conversionTab.architecture", { defaultValue: "Architecture" })}</span>
                    <p>{lastConversion.architecture ?? t("conversionTab.unknown", { defaultValue: "Unknown" })}</p>
                  </div>
                  <div>
                    <span className="eyebrow">{t("conversionTab.context", { defaultValue: "Context" })}</span>
                    <p>{lastConversion.contextWindow ?? estimatedContext}</p>
                  </div>
                  <div>
                    <span className="eyebrow">{t("conversionTab.compression", { defaultValue: "Compression" })}</span>
                    <p>{lastConversion.compressionRatio
                      ? t("conversionTab.cacheReduction", { ratio: number(lastConversion.compressionRatio), defaultValue: "{ratio}x cache reduction" })
                      : conversionCompression}</p>
                  </div>
                </div>
              </div>
            ) : null}
          </div>
        </div>
      </Panel>

      <ModelPicker
        open={showConversionPicker}
        title={t("panels.selectSourceModel", { defaultValue: "Select Source Model" })}
        library={workspace.library}
        filter={(item) => libraryItemFormat(item) !== "MLX"}
        selectedPath={conversionDraft.path || null}
        onSelect={(item, resolvedPath) => {
          onPrepareLibraryConversion(item, resolvedPath);
        }}
        onClose={() => onShowConversionPickerChange(false)}
      />

      {showConversionModal ? (
        <div className="modal-overlay conversion-result-modal">
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>
                {busyAction === "Converting model..."
                  ? t("conversionTab.modal.titleConverting", { defaultValue: "Converting model" })
                  : conversionError
                    ? t("conversionTab.modal.titleFailed", { defaultValue: "Conversion failed" })
                    : t("conversionTab.modal.titleComplete", { defaultValue: "Conversion complete" })}
              </h3>
            </div>
            <div className="modal-body">
              {busyAction === "Converting model..." && conversionStartedAt ? (
                <LiveProgress
                  title={t("conversionTab.progress.title", { defaultValue: "Converting model" })}
                  subtitle={conversionSource?.name ?? conversionDraft.modelRef ?? undefined}
                  startedAt={conversionStartedAt}
                  accent="convert"
                  phases={[
                    { id: "resolve", label: t("conversionTab.progress.phases.resolve", { defaultValue: "Resolving source" }), estimatedSeconds: 3 },
                    { id: "download", label: t("conversionTab.progress.phases.download", { defaultValue: "Fetching weights" }), estimatedSeconds: 60 },
                    { id: "load", label: t("conversionTab.progress.phases.load", { defaultValue: "Loading into MLX" }), estimatedSeconds: 15 },
                    { id: "quantize", label: t("conversionTab.progress.phases.quantize", { bits: conversionDraft.qBits, group: conversionDraft.qGroupSize, defaultValue: `Quantizing to ${conversionDraft.qBits}-bit g${conversionDraft.qGroupSize}` }), estimatedSeconds: 45 },
                    { id: "shard", label: t("conversionTab.progress.phases.shard", { defaultValue: "Sharding & writing safetensors" }), estimatedSeconds: 10 },
                    { id: "verify", label: t("conversionTab.progress.phases.verify", { defaultValue: "Verifying output" }), estimatedSeconds: 5 },
                  ] as LiveProgressPhase[]}
                />
              ) : conversionError ? (
                <div className="callout error">
                  <h3>{t("conversionTab.modal.titleFailed", { defaultValue: "Conversion failed" })}</h3>
                  <p>{conversionError}</p>
                  <details className="debug-details">
                    <summary>{t("conversionTab.modal.debugDetails", { defaultValue: "Debug details" })}</summary>
                    <dl className="debug-grid">
                      <dt>{t("conversionTab.modal.modelRef", { defaultValue: "Model ref" })}</dt>
                      <dd><code>{conversionDraft.modelRef || "\u2014"}</code></dd>
                      <dt>{t("conversionTab.modal.sourcePath", { defaultValue: "Source path" })}</dt>
                      <dd><code>{conversionDraft.path || "\u2014"}</code></dd>
                      <dt>{t("conversionTab.modal.hfRepoOverride", { defaultValue: "HF repo override" })}</dt>
                      <dd><code>{conversionDraft.hfRepo || "\u2014"}</code></dd>
                      <dt>{t("conversionTab.outputPath", { defaultValue: "Output path" })}</dt>
                      <dd>
                        <code>{conversionDraft.outputPath || t("conversionTab.modal.default", { defaultValue: "(default)" })}</code>
                        {conversionDraft.outputPath && !conversionDraft.outputPath.startsWith("/") && !conversionDraft.outputPath.startsWith("~") ? (
                          <small className="muted-text"> {"\u2192"} {t("conversionTab.modal.resolvedUnder", { defaultValue: "resolved under" })} <code>~/Models/</code></small>
                        ) : null}
                      </dd>
                      <dt>{t("conversionTab.modal.quantize", { defaultValue: "Quantize" })}</dt>
                      <dd>{conversionDraft.quantize
                        ? t("conversionTab.modal.quantizeYes", { bits: conversionDraft.qBits, group: conversionDraft.qGroupSize, defaultValue: "yes \u00B7 q{bits} g{group}" })
                        : t("conversionTab.modal.quantizeNo", { defaultValue: "no" })}</dd>
                      <dt>{t("conversionTab.dtype", { defaultValue: "Dtype" })}</dt>
                      <dd>{conversionDraft.dtype}</dd>
                    </dl>
                    <p className="muted-text debug-hint">
                      {t("conversionTab.modal.backendLogHint", { defaultValue: "Backend log:" })} <code>~/Library/.../chaosengine-backend-8876.log</code>. {t("conversionTab.modal.tailHint", { defaultValue: "Run" })} <code>tail -100 $(ls -t $TMPDIR/chaosengine-backend-*.log | head -1)</code> {t("conversionTab.modal.terminalHint", { defaultValue: "in Terminal for full stderr." })}
                    </p>
                  </details>
                </div>
              ) : lastConversion ? (
                <div className="callout">
                  <span className="badge success">{"\u2713"} {t("conversionTab.modal.titleComplete", { defaultValue: "Conversion complete" })}</span>
                  <h3>{lastConversion.sourceLabel}</h3>
                  <div className="conversion-output-row">
                    <p className="mono-text">{lastConversion.outputPath}</p>
                    <button
                      className="secondary-button icon-button"
                      type="button"
                      title={fileRevealLabel}
                      onClick={() => void onRevealPath(lastConversion.outputPath)}
                    >
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
                        <polyline points="15 3 21 3 21 9" />
                        <line x1="10" y1="14" x2="21" y2="3" />
                      </svg>
                    </button>
                  </div>
                  <div className="field-grid detail-grid">
                    <div>
                      <span className="eyebrow">{t("conversionTab.baseRepo", { defaultValue: "Base repo" })}</span>
                      <p>{lastConversion.hfRepo}</p>
                    </div>
                    <div>
                      <span className="eyebrow">{t("conversionTab.architecture", { defaultValue: "Architecture" })}</span>
                      <p>{lastConversion.architecture ?? t("conversionTab.unknown", { defaultValue: "Unknown" })}</p>
                    </div>
                    <div>
                      <span className="eyebrow">{t("conversionTab.context", { defaultValue: "Context" })}</span>
                      <p>{lastConversion.contextWindow ?? t("conversionTab.varies", { defaultValue: "Varies" })}</p>
                    </div>
                    <div>
                      <span className="eyebrow">{t("conversionTab.compression", { defaultValue: "Compression" })}</span>
                      <p>{lastConversion.compressionRatio
                        ? t("conversionTab.cacheReduction", { ratio: number(lastConversion.compressionRatio), defaultValue: "{ratio}x cache reduction" })
                        : "\u2014"}</p>
                    </div>
                    <div>
                      <span className="eyebrow">{t("conversionTab.quality", { defaultValue: "Quality" })}</span>
                      <p>{number(lastConversion.qualityPercent ?? 0, 1)}%</p>
                    </div>
                    <div>
                      <span className="eyebrow">{t("conversionTab.diskBeforeAfter", { defaultValue: "Disk before \u2192 after" })}</span>
                      <p>
                        {lastConversion.sourceSizeGb ? sizeLabel(lastConversion.sourceSizeGb) : "\u2014"}
                        {" \u2192 "}
                        {lastConversion.outputSizeGb ? sizeLabel(lastConversion.outputSizeGb) : "\u2014"}
                      </p>
                    </div>
                  </div>
                </div>
              ) : null}
            </div>
            {busyAction !== "Converting model..." ? (
              <div className="modal-footer">
                <button
                  className="primary-button"
                  type="button"
                  onClick={() => onShowConversionModalChange(false)}
                >
                  {conversionError
                    ? t("conversionTab.modal.close", { defaultValue: "Close" })
                    : t("conversionTab.modal.ok", { defaultValue: "OK" })}
                </button>
              </div>
            ) : null}
          </div>
        </div>
      ) : null}
    </div>
  );
}
