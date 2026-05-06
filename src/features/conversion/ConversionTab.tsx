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
    system: { availableMemoryGb: number; totalMemoryGb: number };
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
  const beforeSize = conversionSource?.sizeGb ?? lastConversion?.sourceSizeGb ?? null;
  const estimatedContext = lastConversion?.contextWindow ?? conversionVariant?.contextWindow ?? "Varies";

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
      ? `${number(beforeSize / afterSize)}x smaller on disk`
      : projectedDiskGb && beforeSize
        ? `≈ ${number(beforeSize / projectedDiskGb)}x projected`
        : "Pick a source and bits to project disk footprint";

  return (
    <div className="content-grid">
      <Panel
        title="MLX Conversion"
        subtitle="Prepare a local source, compare before and after stats, then convert into an MLX-ready output."
        className="span-2"
        actions={
          <span className={`badge ${conversionReady ? "success" : "warning"}`}>
            {conversionReady ? "Converter ready" : "Converter unavailable"}
          </span>
        }
      >
        <div className="conversion-layout">
          <div className="conversion-builder">
            {convertibleLibrary.length ? (
              <>
                <div className="conversion-source-picker">
                  <span className="eyebrow">Source model</span>
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
                        Change
                      </button>
                    </div>
                  ) : (
                    <button className="secondary-button" type="button" onClick={() => onShowConversionPickerChange(true)} style={{ width: "100%" }}>
                      Select a model to convert...
                    </button>
                  )}
                </div>

                <div className="field-grid">
                  <label>
                    Output path
                    <div className="input-with-button">
                      <input
                        className="text-input"
                        type="text"
                        placeholder="Leave blank to use ~/Models/<name>-mlx"
                        value={conversionDraft.outputPath}
                        onChange={(event) => onConversionDraftChange("outputPath", event.target.value)}
                      />
                      <button
                        className="secondary-button"
                        type="button"
                        onClick={() => void onPickConversionOutputDir()}
                        title="Choose output folder"
                      >
                        Browse...
                      </button>
                    </div>
                  </label>
                  {conversionSource?.format?.toUpperCase() === "GGUF" ? (
                    <label>
                      Base HF repo (required for GGUF)
                      <input
                        className="text-input"
                        type="text"
                        placeholder="e.g. Qwen/Qwen2.5-7B-Instruct"
                        value={conversionDraft.hfRepo}
                        onChange={(event) => onConversionDraftChange("hfRepo", event.target.value)}
                      />
                    </label>
                  ) : null}
                </div>

                <SliderField
                  label="Quantization bits"
                  value={conversionDraft.qBits}
                  min={2} max={8} step={1}
                  ticks={[{ value: 2, label: "2" }, { value: 3, label: "3" }, { value: 4, label: "4" }, { value: 5, label: "5" }, { value: 6, label: "6" }, { value: 7, label: "7" }, { value: 8, label: "8" }]}
                  formatValue={(v) => `${v}-bit`}
                  onChange={(v) => onConversionDraftChange("qBits", v)}
                />

                <SliderField
                  label="Group size"
                  value={conversionDraft.qGroupSize}
                  min={32} max={128} step={32}
                  ticks={[{ value: 32, label: "32" }, { value: 64, label: "64" }, { value: 96, label: "96" }, { value: 128, label: "128" }]}
                  formatValue={(v) => `${v} weights/group`}
                  onChange={(v) => onConversionDraftChange("qGroupSize", v)}
                />

                <div className="field-grid">
                  <label>
                    Dtype
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
                    Quantize converted weights
                  </label>
                  <div className={`callout ${isReQuantizing ? "warning" : "quiet"} compact-callout`}>
                    <h3>{isReQuantizing ? "Re-quantizing an already quantized source" : "Backend note"}</h3>
                    <p>
                      {isReQuantizing
                        ? `Source is already ~${Math.round(sourceBpw)}-bit. Going lower compounds quality loss — for best results convert from the original FP16/BF16 weights.`
                        : conversionReady
                          ? "mlx-lm conversion is available in the active backend."
                          : nativeBackends?.mlxMessage ?? "Start the native sidecar to enable conversion."}
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
                    {busy ? "Converting..." : "Convert to MLX"}
                  </button>
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={onConversionDraftReset}
                  >
                    Clear
                  </button>
                </div>
              </>
            ) : (
              <div className="empty-state">
                <p>Add model directories in Settings first, then conversion sources found there will appear here.</p>
              </div>
            )}
          </div>

          <div className="conversion-visuals">
            <div className="stat-grid compact-grid">
              <StatCard label="Params" value={conversionVariant ? `${number(conversionVariant.paramsB)}B` : (lastConversion?.paramsB ? `${number(lastConversion.paramsB)}B` : "Unknown")} hint={estimatedContext} />
              <StatCard
                label="Disk before"
                value={beforeSize ? sizeLabel(beforeSize) : "Unknown"}
                hint={conversionSource?.format ?? lastConversion?.sourceFormat ?? "Source"}
              />
              <StatCard
                label="Disk after"
                value={afterSize ? sizeLabel(afterSize) : "Pending"}
                hint={conversionCompression}
              />
              <StatCard
                label="Est. tok/s"
                value={`${number(estimatedTokS)} tok/s`}
                hint={`Using ${launchCacheLabel}`}
              />
            </div>

            <div className="conversion-compare">
              <div className="conversion-card">
                <span className="eyebrow">Before</span>
                <h3>{conversionSource?.name ?? lastConversion?.sourceLabel ?? "Choose a source"}</h3>
                <p>{conversionSource?.path ?? lastConversion?.sourcePath ?? "Select a local GGUF or HF-cache source to inspect its current footprint."}</p>
                <div className="metric-list">
                  <div className="metric-row">
                    <span>Format</span>
                    <strong>{conversionSource?.format ?? lastConversion?.sourceFormat ?? "Unknown"}{isReQuantizing ? ` · ~${Math.round(sourceBpw)}-bit` : ""}</strong>
                  </div>
                  <div className="metric-row">
                    <span>On-disk size</span>
                    <strong>{beforeSize ? sizeLabel(beforeSize) : "Unknown"}</strong>
                  </div>
                  <div className="metric-row">
                    <span>Context</span>
                    <strong>{estimatedContext}</strong>
                  </div>
                  <div className="metric-row">
                    <span>Cache footprint</span>
                    <strong>{number(cacheBefore)} GB</strong>
                  </div>
                </div>
              </div>

              <div className="conversion-arrow" aria-hidden="true">
                <span>MLX</span>
              </div>

              <div className="conversion-card accent-card">
                <span className="eyebrow">After</span>
                <h3>{lastConversion ? "MLX-ready output" : "Target preview"}</h3>
                <p>{lastConversion?.outputPath ?? "Converted output will appear here together with derived stats and metadata."}</p>
                <div className="metric-list">
                  <div className="metric-row">
                    <span>Target profile</span>
                    <strong>{conversionDraft.quantize ? `${conversionDraft.qBits}-bit g${conversionDraft.qGroupSize}` : "Unquantized"} / {conversionDraft.dtype}</strong>
                  </div>
                  <div className="metric-row">
                    <span>On-disk size</span>
                    <strong>{afterSize ? sizeLabel(afterSize) : "Pending"}</strong>
                  </div>
                  <div className="metric-row">
                    <span>Cache footprint</span>
                    <strong>{number(cacheAfter)} GB</strong>
                  </div>
                  <div className="metric-row">
                    <span>Quality estimate</span>
                    <strong>{number(lastConversion?.qualityPercent ?? projectedQualityPercent, 1)}%</strong>
                  </div>
                </div>
              </div>
            </div>

            <PerformancePreview
              preview={preview}
              availableMemoryGb={workspace.system.availableMemoryGb}
              totalMemoryGb={workspace.system.totalMemoryGb}
            />

            {lastConversion && !busy ? (
              <div className="callout">
                <span className="badge success">Last conversion</span>
                <h3>{lastConversion.sourceLabel}</h3>
                <p>{lastConversion.outputPath}</p>
                <div className="field-grid detail-grid">
                  <div>
                    <span className="eyebrow">Base repo</span>
                    <p>{lastConversion.hfRepo}</p>
                  </div>
                  <div>
                    <span className="eyebrow">Architecture</span>
                    <p>{lastConversion.architecture ?? "Unknown"}</p>
                  </div>
                  <div>
                    <span className="eyebrow">Context</span>
                    <p>{lastConversion.contextWindow ?? estimatedContext}</p>
                  </div>
                  <div>
                    <span className="eyebrow">Compression</span>
                    <p>{lastConversion.compressionRatio ? `${number(lastConversion.compressionRatio)}x cache reduction` : conversionCompression}</p>
                  </div>
                </div>
              </div>
            ) : null}
          </div>
        </div>
      </Panel>

      <ModelPicker
        open={showConversionPicker}
        title="Select Source Model"
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
                  ? "Converting model"
                  : conversionError
                    ? "Conversion failed"
                    : "Conversion complete"}
              </h3>
            </div>
            <div className="modal-body">
              {busyAction === "Converting model..." && conversionStartedAt ? (
                <LiveProgress
                  title="Converting model"
                  subtitle={conversionSource?.name ?? conversionDraft.modelRef ?? undefined}
                  startedAt={conversionStartedAt}
                  accent="convert"
                  phases={[
                    { id: "resolve", label: "Resolving source", estimatedSeconds: 3 },
                    { id: "download", label: "Fetching weights", estimatedSeconds: 60 },
                    { id: "load", label: "Loading into MLX", estimatedSeconds: 15 },
                    { id: "quantize", label: `Quantizing to ${conversionDraft.qBits}-bit g${conversionDraft.qGroupSize}`, estimatedSeconds: 45 },
                    { id: "shard", label: "Sharding & writing safetensors", estimatedSeconds: 10 },
                    { id: "verify", label: "Verifying output", estimatedSeconds: 5 },
                  ] as LiveProgressPhase[]}
                />
              ) : conversionError ? (
                <div className="callout error">
                  <h3>Conversion failed</h3>
                  <p>{conversionError}</p>
                  <details className="debug-details">
                    <summary>Debug details</summary>
                    <dl className="debug-grid">
                      <dt>Model ref</dt>
                      <dd><code>{conversionDraft.modelRef || "\u2014"}</code></dd>
                      <dt>Source path</dt>
                      <dd><code>{conversionDraft.path || "\u2014"}</code></dd>
                      <dt>HF repo override</dt>
                      <dd><code>{conversionDraft.hfRepo || "\u2014"}</code></dd>
                      <dt>Output path</dt>
                      <dd>
                        <code>{conversionDraft.outputPath || "(default)"}</code>
                        {conversionDraft.outputPath && !conversionDraft.outputPath.startsWith("/") && !conversionDraft.outputPath.startsWith("~") ? (
                          <small className="muted-text"> {"\u2192"} resolved under <code>~/Models/</code></small>
                        ) : null}
                      </dd>
                      <dt>Quantize</dt>
                      <dd>{conversionDraft.quantize ? `yes \u00B7 q${conversionDraft.qBits} g${conversionDraft.qGroupSize}` : "no"}</dd>
                      <dt>Dtype</dt>
                      <dd>{conversionDraft.dtype}</dd>
                    </dl>
                    <p className="muted-text debug-hint">
                      Backend log: <code>~/Library/.../chaosengine-backend-8876.log</code>. Run <code>tail -100 $(ls -t $TMPDIR/chaosengine-backend-*.log | head -1)</code> in Terminal for full stderr.
                    </p>
                  </details>
                </div>
              ) : lastConversion ? (
                <div className="callout">
                  <span className="badge success">{"\u2713"} Conversion complete</span>
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
                      <span className="eyebrow">Base repo</span>
                      <p>{lastConversion.hfRepo}</p>
                    </div>
                    <div>
                      <span className="eyebrow">Architecture</span>
                      <p>{lastConversion.architecture ?? "Unknown"}</p>
                    </div>
                    <div>
                      <span className="eyebrow">Context</span>
                      <p>{lastConversion.contextWindow ?? "Varies"}</p>
                    </div>
                    <div>
                      <span className="eyebrow">Compression</span>
                      <p>{lastConversion.compressionRatio ? `${number(lastConversion.compressionRatio)}x cache reduction` : "\u2014"}</p>
                    </div>
                    <div>
                      <span className="eyebrow">Quality</span>
                      <p>{number(lastConversion.qualityPercent ?? 0, 1)}%</p>
                    </div>
                    <div>
                      <span className="eyebrow">Disk before {"\u2192"} after</span>
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
                  {conversionError ? "Close" : "OK"}
                </button>
              </div>
            ) : null}
          </div>
        </div>
      ) : null}
    </div>
  );
}
