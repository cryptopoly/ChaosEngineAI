import { useTranslation } from "react-i18next";
import type { PreviewMetrics } from "../types";
import { ProgressRow } from "./ProgressRow";
import { getCacheFitStatus } from "../utils/cache";
import type { TFunction } from "i18next";

interface PerformancePreviewProps {
  preview: PreviewMetrics;
  availableMemoryGb: number;
  totalMemoryGb: number;
  /** Discrete GPU VRAM in GB (CUDA card on Windows / Linux). When set,
   * the cache-fit check uses this as the binding constraint -- llama.cpp
   * places the KV cache on GPU with full offload, so a 60 GB cache on a
   * 24 GB 4090 fails on VRAM long before it would have failed on system
   * RAM. Null on Apple Silicon (unified memory already in
   * totalMemoryGb) or hosts with no detected discrete GPU. */
  gpuVramTotalGb?: number | null;
  compact?: boolean;
  actualDiskSizeGb?: number;
}

function fmt(value: number, digits = 1): string {
  return value.toFixed(digits);
}

function getSpeedLabel(tokS: number, t: TFunction): { label: string; className: string } | null {
  if (tokS < 5) return { label: t("performancePreview.speedSlow", { defaultValue: "Slow" }), className: "perf-preview__speed-label--slow" };
  if (tokS < 15) return { label: t("performancePreview.speedGood", { defaultValue: "Good" }), className: "perf-preview__speed-label--good" };
  if (tokS < 30) return { label: t("performancePreview.speedFast", { defaultValue: "Fast" }), className: "perf-preview__speed-label--fast" };
  return { label: t("performancePreview.speedVeryFast", { defaultValue: "Very fast" }), className: "perf-preview__speed-label--fast" };
}

export function PerformancePreview({ preview, availableMemoryGb, totalMemoryGb, gpuVramTotalGb, compact, actualDiskSizeGb }: PerformancePreviewProps) {
  const { t } = useTranslation("common");
  const diskGb = actualDiskSizeGb ?? preview.diskSizeGb;
  const fitStatus = getCacheFitStatus(preview.optimizedCacheGb, diskGb, totalMemoryGb, preview.bits, gpuVramTotalGb);
  const cacheDelta = preview.baselineCacheGb - preview.optimizedCacheGb;
  const qualityDelta = preview.qualityPercent - 100;
  const cacheMax = Math.max(preview.baselineCacheGb, totalMemoryGb * 0.6, 1);
  const ramUsedPercent = totalMemoryGb > 0
    ? Math.min(100, ((preview.optimizedCacheGb + diskGb) / totalMemoryGb) * 100)
    : 0;
  const ramColor = ramUsedPercent > 90 ? "var(--warning, #e4be75)" : "var(--accent)";
  const speedLabel = getSpeedLabel(preview.estimatedTokS, t);
  const cacheLabel = t("performancePreview.cacheLabel", { defaultValue: "Cache" });

  return (
    <div className={`perf-preview${compact ? " perf-preview--compact" : ""}`}>
      <div className="perf-preview__header">
        <span className="eyebrow">{t("performancePreview.heading", { defaultValue: "Performance preview" })}</span>
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
            {t("performancePreview.headlineDelta", {
              defaultValue: "{value} GB",
              value: fmt(cacheDelta),
            })}
          </span>
          <span className="perf-preview__headline-label">
            {t("performancePreview.headlineLabel", {
              defaultValue: "cache savings ({ratio}x compression)",
              ratio: fmt(preview.compressionRatio),
            })}
          </span>
        </div>
      ) : null}

      <div className="perf-preview__compare">
        {/* Baseline column */}
        <div className="perf-preview__col">
          <span className="eyebrow">{t("performancePreview.baselineHeading", { defaultValue: "Baseline (FP16)" })}</span>
          <div className="metric-list">
            <ProgressRow
              label={cacheLabel}
              value={preview.baselineCacheGb}
              max={cacheMax}
              valueLabel={t("performancePreview.gbValue", {
                defaultValue: "{value} GB",
                value: fmt(preview.baselineCacheGb),
              })}
            />
            <div className="metric-row">
              <span>{t("performancePreview.speed", { defaultValue: "Speed" })}</span>
              <strong className="muted-text">{t("performancePreview.baselineSpeedValue", { defaultValue: "baseline" })}</strong>
            </div>
            <div className="metric-row">
              <span>{t("performancePreview.quality", { defaultValue: "Quality" })}</span>
              <strong>{t("performancePreview.baselineQualityValue", { defaultValue: "100%" })}</strong>
            </div>
          </div>
        </div>

        {/* Arrow divider */}
        <div className="perf-preview__arrow" aria-hidden="true">
          <span>{preview.bits > 0
            ? t("performancePreview.bitsLabel", { defaultValue: "{bits}-bit", bits: preview.bits })
            : t("performancePreview.f16Label", { defaultValue: "f16" })
          }</span>
        </div>

        {/* Optimized column */}
        <div className="perf-preview__col perf-preview__col--accent">
          <span className="eyebrow">{t("performancePreview.optimizedHeading", { defaultValue: "Optimized" })}</span>
          <div className="metric-list">
            <ProgressRow
              label={cacheLabel}
              value={preview.optimizedCacheGb}
              max={cacheMax}
              valueLabel={t("performancePreview.gbValue", {
                defaultValue: "{value} GB",
                value: fmt(preview.optimizedCacheGb),
              })}
              baseline={preview.baselineCacheGb}
              delta={cacheDelta > 0.05
                ? t("performancePreview.cacheDelta", {
                    defaultValue: "-{value} GB",
                    value: fmt(cacheDelta),
                  })
                : undefined}
              deltaPositive={cacheDelta > 0}
            />
            <div className="metric-row">
              <span>{t("performancePreview.speed", { defaultValue: "Speed" })}</span>
              <span className="metric-row-right">
                <strong>{t("performancePreview.tokensPerSec", {
                  defaultValue: "{value} tok/s",
                  value: fmt(preview.estimatedTokS),
                })}</strong>
                {speedLabel ? (
                  <span className={`perf-preview__speed-label ${speedLabel.className}`}>
                    {speedLabel.label}
                  </span>
                ) : null}
              </span>
            </div>
            <div className="metric-row">
              <span>{t("performancePreview.quality", { defaultValue: "Quality" })}</span>
              <span className="metric-row-right">
                <strong>{t("performancePreview.percentValue", {
                  defaultValue: "{value}%",
                  value: fmt(preview.qualityPercent, 1),
                })}</strong>
                {qualityDelta < -0.1 ? (
                  <span className="delta-badge delta-badge--negative">
                    {t("performancePreview.percentValue", {
                      defaultValue: "{value}%",
                      value: fmt(qualityDelta, 1),
                    })}
                  </span>
                ) : null}
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="perf-preview__footer">
        <span className="perf-preview__stat">{t("performancePreview.diskStat", {
          defaultValue: "Disk: {value} GB",
          value: fmt(diskGb),
        })}</span>
        <span className="perf-preview__stat">{t("performancePreview.compressionStat", {
          defaultValue: "{ratio}x compression",
          ratio: fmt(preview.compressionRatio),
        })}</span>
        <div className="perf-preview__ram-group">
          <span className="perf-preview__ram-label">{t("performancePreview.ramLabel", { defaultValue: "RAM" })}</span>
          <div className="perf-preview__ram-bar">
            <div
              className="perf-preview__ram-fill"
              style={{ width: `${ramUsedPercent}%`, background: ramColor }}
            />
          </div>
          <span className="perf-preview__ram-label">
            {t("performancePreview.ramValue", {
              defaultValue: "{used}/{total} GB",
              used: fmt(preview.optimizedCacheGb + diskGb),
              total: fmt(totalMemoryGb),
            })}
          </span>
        </div>
      </div>
    </div>
  );
}
