import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import type { TFunction } from "i18next";
import type { RecommendedSettings } from "../types/recommendations";
import {
  applyRecommendationRows,
  recommendationRows,
  type RecommendationAppliers,
  type RecommendationCurrent,
  type RecommendationRow,
} from "./recommendedSettings";

export interface RecommendedSettingsPanelProps {
  recommendation: RecommendedSettings | null | undefined;
  /** Studio setters; a setting without one is not shown (the model cannot use it). */
  appliers: RecommendationAppliers;
  /** The Studio's current values, to mark rows already in use. */
  current: RecommendationCurrent;
  /** Display names for samplers / cache strategies (ids otherwise). */
  labelFor?: (kind: "sampler" | "cacheStrategy" | "qualityPreset", id: string) => string;
  disabled?: boolean;
}

function formatNumber(value: number): string {
  return Number.isInteger(value) ? String(value) : String(Number(value.toFixed(3)));
}

export function formatRecommendationRow(
  row: RecommendationRow,
  t: TFunction,
  labelFor?: RecommendedSettingsPanelProps["labelFor"],
): { label: string; value: string } {
  const v = row.values;
  const on = t("recommended.on", { defaultValue: "On" });
  const off = t("recommended.off", { defaultValue: "Off" });
  switch (row.id) {
    case "qualityPreset":
      return {
        label: t("recommended.labels.qualityPreset", { defaultValue: "Quality preset" }),
        value: labelFor?.("qualityPreset", String(v.qualityPreset)) ?? String(v.qualityPreset),
      };
    case "steps":
      return {
        label: t("recommended.labels.steps", { defaultValue: "Steps" }),
        value: String(v.steps),
      };
    case "guidance":
      return {
        label: t("recommended.labels.guidance", { defaultValue: "Guidance" }),
        value: formatNumber(Number(v.guidance)),
      };
    case "sampler":
      return {
        label: t("recommended.labels.sampler", { defaultValue: "Sampler" }),
        value: labelFor?.("sampler", String(v.sampler)) ?? String(v.sampler),
      };
    case "size":
      return {
        label: t("recommended.labels.size", { defaultValue: "Size" }),
        value: `${v.width} x ${v.height}`,
      };
    case "numFrames":
      return {
        label: t("recommended.labels.numFrames", { defaultValue: "Frames" }),
        value: String(v.numFrames),
      };
    case "fps":
      return {
        label: t("recommended.labels.fps", { defaultValue: "FPS" }),
        value: String(v.fps),
      };
    case "cache": {
      const name = labelFor?.("cacheStrategy", String(v.cacheStrategy)) ?? String(v.cacheStrategy);
      return {
        label: t("recommended.labels.cache", { defaultValue: "Diffusion cache" }),
        value:
          v.cacheThreshold !== undefined
            ? t("recommended.cacheValue", {
                defaultValue: "{name}, threshold {threshold}",
                name,
                threshold: formatNumber(Number(v.cacheThreshold)),
              })
            : name,
      };
    }
    case "cfgDecay":
      return {
        label: t("recommended.labels.cfgDecay", { defaultValue: "CFG decay" }),
        value: v.cfgDecay ? on : off,
      };
    case "previewVae":
      return {
        label: t("recommended.labels.previewVae", { defaultValue: "Preview VAE" }),
        value: v.previewVae ? on : off,
      };
    case "fp8LayerwiseCasting":
      return {
        label: t("recommended.labels.fp8LayerwiseCasting", { defaultValue: "FP8 layerwise" }),
        value: v.fp8LayerwiseCasting ? on : off,
      };
  }
}

/** "Recommended for this machine": the settings that work best for the selected model on
 * this hardware, each with its reason and a measured / estimated label. Hidden when the
 * backend sent no recommendation the Studio can apply. */
export function RecommendedSettingsPanel({
  recommendation,
  appliers,
  current,
  labelFor,
  disabled = false,
}: RecommendedSettingsPanelProps) {
  const { t } = useTranslation("studio");
  const rows = useMemo(
    () => recommendationRows(recommendation, appliers, current),
    [recommendation, appliers, current],
  );
  if (!recommendation || rows.length === 0) return null;
  const pending = rows.filter((row) => !row.inUse);
  const anyMeasured = rows.some((row) => row.basis === "measured");

  const basisChip = (basis: RecommendationRow["basis"]) =>
    basis === "measured" ? (
      <span
        className="badge success"
        title={t("recommended.measuredTitle", {
          defaultValue: "Timed on {machine}.",
          machine: recommendation.measuredOn ?? recommendation.hardwareLabel ?? recommendation.hardwareBucket,
        })}
      >
        {t("recommended.measured", { defaultValue: "Measured" })}
      </span>
    ) : (
      <span
        className="badge muted"
        title={t("recommended.estimatedTitle", {
          defaultValue: "From the model card, upstream documentation or a similar machine; not timed on this hardware.",
        })}
      >
        {t("recommended.estimated", { defaultValue: "Estimated" })}
      </span>
    );

  return (
    <section
      className="callout quiet recommended-settings-panel"
      aria-label={t("recommended.title", { defaultValue: "Recommended for this machine" })}
    >
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, flexWrap: "wrap" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          <span className="eyebrow">{t("recommended.title", { defaultValue: "Recommended for this machine" })}</span>
          <span className="muted-text" style={{ fontSize: 12 }}>
            {recommendation.hardwareLabel ?? recommendation.hardwareBucket}
            {anyMeasured && recommendation.measuredOn
              ? ` · ${t("recommended.measuredOn", { defaultValue: "timed on {machine}", machine: recommendation.measuredOn })}`
              : ""}
          </span>
        </div>
        <button
          className="secondary-button small-button"
          type="button"
          disabled={disabled || pending.length === 0}
          onClick={() => applyRecommendationRows(pending, appliers)}
        >
          {pending.length === 0
            ? t("recommended.allApplied", { defaultValue: "All in use" })
            : t("recommended.applyAll", { defaultValue: "Apply recommended" })}
        </button>
      </div>
      <ul style={{ listStyle: "none", margin: 0, padding: 0, display: "flex", flexDirection: "column", gap: 8 }}>
        {rows.map((row) => {
          const { label, value } = formatRecommendationRow(row, t, labelFor);
          return (
            <li key={row.id} style={{ display: "flex", alignItems: "flex-start", gap: 8 }}>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
                  <strong>{label}</strong>
                  <span>{value}</span>
                  {basisChip(row.basis)}
                </div>
                {row.reason ? (
                  <p className="muted-text" style={{ margin: "2px 0 0", fontSize: 12 }}>{row.reason}</p>
                ) : null}
              </div>
              {row.inUse ? (
                <span className="badge muted">{t("recommended.inUse", { defaultValue: "In use" })}</span>
              ) : (
                <button
                  className="secondary-button small-button"
                  type="button"
                  disabled={disabled}
                  onClick={() => applyRecommendationRows([row], appliers)}
                >
                  {t("recommended.use", { defaultValue: "Use" })}
                </button>
              )}
            </li>
          );
        })}
      </ul>
    </section>
  );
}
