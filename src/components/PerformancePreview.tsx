import type { PreviewMetrics } from "../types";
import { ProgressRow } from "./ProgressRow";

interface PerformancePreviewProps {
  preview: PreviewMetrics;
  availableMemoryGb: number;
  totalMemoryGb: number;
  compact?: boolean;
  actualDiskSizeGb?: number;
}

function fmt(value: number, digits = 1): string {
  return value.toFixed(digits);
}

interface FitStatus {
  label: string;
  className: string;
  /** Human-readable explanation of the dominant lever when things don't
   * fit. Only populated for the "May not fit" tier — the other tiers are
   * self-explanatory. */
  advice: string | null;
}

function getFitStatus(
  optimizedCacheGb: number,
  diskSizeGb: number,
  totalGb: number,
  bits: number,
): FitStatus {
  // Use total system memory since loading a new model unloads the previous one.
  const totalNeeded = optimizedCacheGb + diskSizeGb;
  // Reserve ~20% for OS and other apps
  const usable = totalGb * 0.80;
  const ratio = usable > 0 ? totalNeeded / usable : 1;
  if (ratio < 0.7) return { label: "Fits easily", className: "success", advice: null };
  if (ratio < 0.95) return { label: "Tight fit", className: "warning", advice: null };

  // "May not fit" — pick the most useful lever to show the user. When the
  // cache pool dwarfs the weights (classic "256K context on a 26B model"
  // situation), the right fix is context + strategy, not model size. When
  // the weights themselves are the problem, no context lever will help.
  const cacheDominates = optimizedCacheGb > diskSizeGb * 1.5;
  let advice: string;
  if (!cacheDominates) {
    advice =
      "Model weights alone exceed available RAM. Pick a smaller model or a more aggressive quantisation.";
  } else if (bits <= 0) {
    advice =
      "Native f16 cache grows with context — at this setting it's bigger than RAM. Lower the context slider, or pick a compressed strategy (RotorQuant / TriAttention).";
  } else {
    advice =
      "Compressed cache still exceeds RAM at this context. Lower the context slider or reduce FP16 layers.";
  }
  return { label: "May not fit", className: "warning", advice };
}

function getSpeedLabel(tokS: number): { label: string; className: string } | null {
  if (tokS < 5) return { label: "Slow", className: "perf-preview__speed-label--slow" };
  if (tokS < 15) return { label: "Good", className: "perf-preview__speed-label--good" };
  if (tokS < 30) return { label: "Fast", className: "perf-preview__speed-label--fast" };
  return { label: "Very fast", className: "perf-preview__speed-label--fast" };
}

export function PerformancePreview({ preview, availableMemoryGb, totalMemoryGb, compact, actualDiskSizeGb }: PerformancePreviewProps) {
  const diskGb = actualDiskSizeGb ?? preview.diskSizeGb;
  const fitStatus = getFitStatus(preview.optimizedCacheGb, diskGb, totalMemoryGb, preview.bits);
  const cacheDelta = preview.baselineCacheGb - preview.optimizedCacheGb;
  const qualityDelta = preview.qualityPercent - 100;
  const cacheMax = Math.max(preview.baselineCacheGb, totalMemoryGb * 0.6, 1);
  const ramUsedPercent = totalMemoryGb > 0
    ? Math.min(100, ((preview.optimizedCacheGb + diskGb) / totalMemoryGb) * 100)
    : 0;
  const ramColor = ramUsedPercent > 90 ? "var(--warning, #e4be75)" : "var(--accent)";
  const speedLabel = getSpeedLabel(preview.estimatedTokS);

  return (
    <div className={`perf-preview${compact ? " perf-preview--compact" : ""}`}>
      <div className="perf-preview__header">
        <span className="eyebrow">Performance preview</span>
        <span className={`badge ${fitStatus.className}`}>{fitStatus.label}</span>
      </div>

      {fitStatus.advice ? (
        <p className="perf-preview__advice" role="note">
          {fitStatus.advice}
        </p>
      ) : null}

      {cacheDelta > 0.1 ? (
        <div className="perf-preview__headline">
          <span className="perf-preview__headline-delta">
            {fmt(cacheDelta)} GB
          </span>
          <span className="perf-preview__headline-label">
            cache savings ({fmt(preview.compressionRatio)}x compression)
          </span>
        </div>
      ) : null}

      <div className="perf-preview__compare">
        {/* Baseline column */}
        <div className="perf-preview__col">
          <span className="eyebrow">Baseline (FP16)</span>
          <div className="metric-list">
            <ProgressRow
              label="Cache"
              value={preview.baselineCacheGb}
              max={cacheMax}
              valueLabel={`${fmt(preview.baselineCacheGb)} GB`}
            />
            <div className="metric-row">
              <span>Speed</span>
              <strong className="muted-text">baseline</strong>
            </div>
            <div className="metric-row">
              <span>Quality</span>
              <strong>100%</strong>
            </div>
          </div>
        </div>

        {/* Arrow divider */}
        <div className="perf-preview__arrow" aria-hidden="true">
          <span>{preview.bits > 0 ? `${preview.bits}-bit` : "f16"}</span>
        </div>

        {/* Optimized column */}
        <div className="perf-preview__col perf-preview__col--accent">
          <span className="eyebrow">Optimized</span>
          <div className="metric-list">
            <ProgressRow
              label="Cache"
              value={preview.optimizedCacheGb}
              max={cacheMax}
              valueLabel={`${fmt(preview.optimizedCacheGb)} GB`}
              baseline={preview.baselineCacheGb}
              delta={cacheDelta > 0.05 ? `-${fmt(cacheDelta)} GB` : undefined}
              deltaPositive={cacheDelta > 0}
            />
            <div className="metric-row">
              <span>Speed</span>
              <span className="metric-row-right">
                <strong>{fmt(preview.estimatedTokS)} tok/s</strong>
                {speedLabel ? (
                  <span className={`perf-preview__speed-label ${speedLabel.className}`}>
                    {speedLabel.label}
                  </span>
                ) : null}
              </span>
            </div>
            <div className="metric-row">
              <span>Quality</span>
              <span className="metric-row-right">
                <strong>{fmt(preview.qualityPercent, 1)}%</strong>
                {qualityDelta < -0.1 ? (
                  <span className="delta-badge delta-badge--negative">
                    {fmt(qualityDelta, 1)}%
                  </span>
                ) : null}
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="perf-preview__footer">
        <span className="perf-preview__stat">Disk: {fmt(diskGb)} GB</span>
        <span className="perf-preview__stat">{fmt(preview.compressionRatio)}x compression</span>
        <div className="perf-preview__ram-group">
          <span className="perf-preview__ram-label">RAM</span>
          <div className="perf-preview__ram-bar">
            <div
              className="perf-preview__ram-fill"
              style={{ width: `${ramUsedPercent}%`, background: ramColor }}
            />
          </div>
          <span className="perf-preview__ram-label">
            {fmt(preview.optimizedCacheGb + diskGb)}/{fmt(totalMemoryGb)} GB
          </span>
        </div>
      </div>
    </div>
  );
}
