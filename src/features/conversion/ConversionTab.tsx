import { useTranslation } from "react-i18next";
import { Panel } from "../../components/Panel";
import { PerformancePreview } from "../../components/PerformancePreview";
import { LiveProgress, type LiveProgressPhase } from "../../components/LiveProgress";
import { StatCard } from "../../components/StatCard";
import { ModelPicker } from "../../components/ModelPicker";
import { SliderField } from "../../components/SliderField";
import type { ConversionResult, LibraryItem, ModelVariant, NativeBackendStatus, PreviewMetrics } from "../../types";
import { number, sizeLabel, libraryItemFormat } from "../../utils";
import { conversionDialogPhase, conversionPhaseIds, conversionProgressNumbers } from "../../utils/jobProgress";
import { useGenerationProgress } from "../../hooks/useGenerationProgress";

export interface ConversionDraft {
  modelRef: string;
  path: string;
  hfRepo: string;
  outputPath: string;
  quantize: boolean;
  qBits: number;
  qGroupSize: number;
  dtype: string;
  /** Compare the converted model with the source after converting. */
  measureQuality: boolean;
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
  onCancelConversion: () => void;
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
  onCancelConversion,
  onPickConversionOutputDir,
  onShowConversionPickerChange,
  onShowConversionModalChange,
  onPrepareLibraryConversion,
  onRevealPath,
}: ConversionTabProps) {
  const { t } = useTranslation("common");
  // The converter's own progress: its phase, files downloaded, bytes written. Nothing in the dialog is timed.
  const conversionProgress = useGenerationProgress("conversion", busyAction === "Converting model..." && Boolean(conversionStartedAt));
  const conversionPhaseLabels: Record<string, string> = {
    preparing: t("conversionTab.progress.phases.resolve", { defaultValue: "Resolving source" }),
    loading: t("conversionTab.progress.phases.load", { defaultValue: "Loading into MLX" }),
    quantizing: t("conversionTab.progress.phases.quantize", { bits: conversionDraft.qBits, group: conversionDraft.qGroupSize, defaultValue: "Quantizing to {bits}-bit g{group}" }),
    saving: t("conversionTab.progress.phases.shard", { defaultValue: "Sharding & writing safetensors" }),
    measuring: t("conversionTab.progress.phases.measuring", { defaultValue: "Measuring quality on a short text sample" }),
  };
  const measureQuality = conversionDraft.measureQuality;
  const conversionNumbers = conversionProgressNumbers(conversionProgress);
  const conversionProgressDetail = conversionProgress?.phase === "measuring" && conversionProgress.totalSteps > 0
    ? t("conversionTab.progress.detail.modelsScored", { done: conversionProgress.step, total: conversionProgress.totalSteps, defaultValue: "{done} of {total} models scored" })
    : conversionNumbers.files
    ? t("conversionTab.progress.detail.files", { done: conversionNumbers.files.done, total: conversionNumbers.files.total, defaultValue: "downloading the source: {done} of {total} files" })
    : conversionNumbers.writtenGb !== null
      ? conversionNumbers.expectedGb !== null
        ? t("conversionTab.progress.detail.writtenOfAbout", { written: number(conversionNumbers.writtenGb, 2), expected: number(conversionNumbers.expectedGb, 1), defaultValue: "{written} GB written, about {expected} GB expected (estimate)" })
        : t("conversionTab.progress.detail.written", { written: number(conversionNumbers.writtenGb, 2), defaultValue: "{written} GB written" })
      : null;
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
  // A GGUF source is not re-quantized: the backend downloads the original model named in the
  // file and converts that, so the "going lower compounds quality loss" note does not apply.
  const isGgufSource = (conversionSource?.format ?? "").toUpperCase() === "GGUF" || conversionDraft.path.toLowerCase().endsWith(".gguf");
  const isReQuantizing = !isGgufSource && sourceBpw < 12; // source is already quantized

  const dtypeBytes = conversionDraft.dtype === "float32" ? 4 : 2;
  // Group quantization adds ~16 bits scale/zero per group, amortized per weight
  const groupOverheadBitsPerWeight = conversionDraft.quantize ? 16 / Math.max(8, conversionDraft.qGroupSize) : 0;
  const effectiveBitsPerWeight = conversionDraft.quantize
    ? conversionDraft.qBits + groupOverheadBitsPerWeight
    : dtypeBytes * 8;

  // Project disk-after by scaling source disk size by the bits ratio (much more accurate than paramsB-based math)
  const projectedDiskGb = beforeSize ? beforeSize * (effectiveBitsPerWeight / sourceBpw) : null;
  const afterSize = lastConversion?.outputSizeGb ?? projectedDiskGb;

  // Quality is measured after the conversion (perplexity against the source), never projected here:
  // the table of "recovery" percentages this page used to show was not a measurement of any model.
  const measuredQuality = lastConversion?.quality ?? null;
  const signedPercent = (value: number) => `${value > 0 ? "+" : ""}${number(value, 1)}`;
  const qualityHeadline = measuredQuality?.status === "measured" && typeof measuredQuality.changePercent === "number"
    ? t("conversionTab.qualityChange", { change: signedPercent(measuredQuality.changePercent), defaultValue: "{change}% perplexity vs the source" })
    : measuredQuality?.status === "skipped" || (lastConversion && !measuredQuality)
      ? t("conversionTab.qualityNotMeasured", { defaultValue: "Not measured" })
      : measureQuality
        ? t("conversionTab.qualityAfterConverting", { defaultValue: "Measured after converting" })
        : t("conversionTab.qualityOff", { defaultValue: "Not measured (switched off)" });
  const qualityDetail = measuredQuality?.status === "measured"
    ? t("conversionTab.qualityDetail", {
        samples: measuredQuality.samples ?? 0,
        dataset: measuredQuality.dataset ?? "",
        source: number(measuredQuality.sourcePerplexity ?? 0, 2),
        converted: number(measuredQuality.convertedPerplexity ?? 0, 2),
        defaultValue: "Perplexity on {samples} samples of {dataset}: source {source}, converted {converted}. Lower is better.",
      })
    : measuredQuality?.status === "skipped"
      ? measuredQuality.reason ?? null
      : null;
  const measuredDiskRatio = lastConversion?.diskRatio
    ? t("conversionTab.smallerOnDisk", { ratio: number(lastConversion.diskRatio), defaultValue: "{ratio}x smaller on disk" })
    : "\u2014";
  // Nothing is projected until there is something to project from.
  const hasConversionSubject = Boolean(conversionSource || lastConversion);

  // Speed projection: memory-bandwidth bound, scales inversely with effective bytes/weight vs source
  const speedupVsSource = sourceBpw / effectiveBitsPerWeight;
  // No estimate from the preview means no figure: this used to fall back to a fixed 35 tok/s.
  const projectedTokS = preview.estimatedTokS > 0 ? preview.estimatedTokS * speedupVsSource : null;
  const estimatedTokS = hasConversionSubject ? lastConversion?.estimatedTokS ?? projectedTokS : null;

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
                  <label className="check-row" title={t("conversionTab.measureQualityHint", { defaultValue: "Loads the converted model and the source once each and compares their perplexity on a short WikiText-2 sample. Skipped when either does not fit in free memory." })}>
                    <input
                      type="checkbox"
                      checked={measureQuality}
                      onChange={(event) => onConversionDraftChange("measureQuality", event.target.checked)}
                    />
                    {t("conversionTab.measureQuality", { defaultValue: "Measure quality after converting" })}
                  </label>
                  <div className={`callout ${isReQuantizing || isGgufSource ? "warning" : "quiet"} compact-callout`}>
                    <h3>{isGgufSource
                      ? t("conversionTab.ggufSourceTitle", { defaultValue: "A GGUF source is downloaded again in full size" })
                      : isReQuantizing
                      ? t("conversionTab.reQuantizingTitle", { defaultValue: "Re-quantizing an already quantized source" })
                      : t("conversionTab.backendNote", { defaultValue: "Backend note" })}</h3>
                    <p>
                      {isGgufSource
                        ? t("conversionTab.ggufSourceDetail", { defaultValue: "A GGUF file cannot be turned into MLX directly. The original model named in the file is downloaded from Hugging Face and converted, so this needs a network connection and disk space for the full-size weights. The download shows no progress; Cancel stops it." })
                        : isReQuantizing
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
                value={estimatedTokS !== null ? `${number(estimatedTokS)} tok/s` : "\u2014"}
                hint={hasConversionSubject
                  ? t("conversionTab.usingCache", { cache: launchCacheLabel, defaultValue: "Using {cache}" })
                  : t("conversionTab.pickSourceFirst", { defaultValue: "Pick a source first" })}
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
                  {hasConversionSubject ? (
                    <div className="metric-row">
                      <span>{t("conversionTab.cacheFootprint", { defaultValue: "Cache footprint" })}</span>
                      <strong>{number(cacheBefore)} GB</strong>
                    </div>
                  ) : null}
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
                  {hasConversionSubject ? (
                    <div className="metric-row">
                      <span>{t("conversionTab.cacheFootprint", { defaultValue: "Cache footprint" })}</span>
                      <strong>{number(cacheAfter)} GB</strong>
                    </div>
                  ) : null}
                  <div className="metric-row" data-testid="conversion-quality-row">
                    <span>{t("conversionTab.qualityMeasured", { defaultValue: "Quality (measured)" })}</span>
                    <strong>{qualityHeadline}</strong>
                  </div>
                  {lastConversion?.bitsPerWeight ? (
                    <div className="metric-row">
                      <span>{t("conversionTab.bitsPerWeight", { defaultValue: "Bits per weight" })}</span>
                      <strong>{t("conversionTab.bitsPerWeightValue", { bits: number(lastConversion.bitsPerWeight, 3), defaultValue: "{bits} (measured by the converter)" })}</strong>
                    </div>
                  ) : null}
                </div>
              </div>
            </div>

            {hasConversionSubject ? (
              <PerformancePreview
                preview={preview}
                availableMemoryGb={workspace.system.availableMemoryGb}
                totalMemoryGb={workspace.system.totalMemoryGb}
                gpuVramTotalGb={workspace.system.gpuVramTotalGb}
              />
            ) : null}

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
                    <span className="eyebrow">{t("conversionTab.sizeOnDisk", { defaultValue: "Size on disk" })}</span>
                    <p>{measuredDiskRatio}</p>
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
                <>
                <LiveProgress
                  title={t("conversionTab.progress.title", { defaultValue: "Converting model" })}
                  subtitle={conversionSource?.name ?? conversionDraft.modelRef ?? undefined}
                  startedAt={conversionStartedAt}
                  accent="convert"
                  measuredOnly
                  realProgress={conversionDialogPhase(conversionProgress)}
                  activeDetail={conversionProgressDetail}
                  phases={[
                    ...conversionPhaseIds(conversionDraft.quantize, measureQuality).map((id) => ({ id, label: conversionPhaseLabels[id] ?? id, estimatedSeconds: 0 })),
                  ] as LiveProgressPhase[]}
                />
                <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 12 }}>
                  <button type="button" className="secondary-button danger-button" onClick={onCancelConversion}>
                    {t("conversionTab.modal.cancel", { defaultValue: "Cancel conversion" })}
                  </button>
                </div>
                </>
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
                      <span className="eyebrow">{t("conversionTab.sizeOnDisk", { defaultValue: "Size on disk" })}</span>
                      <p>{measuredDiskRatio}</p>
                    </div>
                    {lastConversion.bitsPerWeight ? (
                      <div>
                        <span className="eyebrow">{t("conversionTab.bitsPerWeight", { defaultValue: "Bits per weight" })}</span>
                        <p>{t("conversionTab.bitsPerWeightValue", { bits: number(lastConversion.bitsPerWeight, 3), defaultValue: "{bits} (measured by the converter)" })}</p>
                      </div>
                    ) : null}
                    <div data-testid="conversion-quality-result">
                      <span className="eyebrow">{t("conversionTab.qualityMeasured", { defaultValue: "Quality (measured)" })}</span>
                      <p>{qualityHeadline}</p>
                      {qualityDetail ? <p className="muted-text">{qualityDetail}</p> : null}
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
