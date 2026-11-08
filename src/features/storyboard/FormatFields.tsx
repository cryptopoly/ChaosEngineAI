/** Aspect ratio, Resolution and Frame rate pickers with a fit hint per choice (D-64). Shared by the New project dialog and Project defaults. */
import { useTranslation } from "react-i18next";
import type { StoryboardExportPreset, StoryboardResolution } from "../../types";
import { formatCapacityGb } from "../video/videoCapacity";
import {
  ASPECTS,
  FPS_CHOICES,
  HINT_CLIP_SECONDS,
  RESOLUTIONS,
  RESOLUTION_ALIAS,
  fpsNeedsConversion,
  resolutionFit,
  sizeFor,
  type FitLevel,
  type VideoFitContext,
} from "./formatLogic";

export interface FormatFieldsProps {
  aspect: StoryboardExportPreset;
  resolution: StoryboardResolution;
  fps: number;
  disabled?: boolean;
  fit: VideoFitContext;
  onAspect: (aspect: StoryboardExportPreset) => void;
  onResolution: (resolution: StoryboardResolution) => void;
  onFps: (fps: number) => void;
}

export function FormatFields({ aspect, resolution, fps, disabled, fit, onAspect, onResolution, onFps }: FormatFieldsProps) {
  const { t } = useTranslation("studio");
  const levelLabel: Record<FitLevel, string | null> = {
    safe: t("storyProjects.format.fit.safe", { defaultValue: "Fits" }),
    tight: t("storyProjects.format.fit.tight", { defaultValue: "Tight" }),
    wontfit: t("storyProjects.format.fit.wontfit", { defaultValue: "Will not fit" }),
    unknown: null,
  };
  const selected = resolutionFit(fit, aspect, resolution, fps);
  const modelName = fit.model?.name ?? "";
  const size = sizeFor(aspect, resolution);

  let fitNote: string;
  if (!fit.model) {
    fitNote = t("storyProjects.format.note.noModel", { defaultValue: "Install a video model to see whether each size fits this computer." });
  } else if (selected.peakGb == null) {
    fitNote = "";
  } else {
    const args = { model: modelName, peak: formatCapacityGb(selected.peakGb), seconds: HINT_CLIP_SECONDS };
    fitNote =
      selected.level === "safe"
        ? t("storyProjects.format.note.safe", { defaultValue: "{model}: about {peak} GB at peak for a {seconds}-second clip. This fits.", ...args })
        : selected.level === "tight"
          ? t("storyProjects.format.note.tight", {
              defaultValue: "{model}: about {peak} GB at peak for a {seconds}-second clip. This is close to the limit of this computer.",
              ...args,
            })
          : t("storyProjects.format.note.wontfit", {
              defaultValue: "{model}: about {peak} GB at peak for a {seconds}-second clip. This will not fit on this computer.",
              ...args,
            });
  }
  const cappedNote = selected.modelSize
    ? t("storyProjects.format.note.capped", {
        defaultValue: "{model} makes up to {size} px, so its clips are scaled up to {resolution} when the film is joined.",
        model: modelName,
        size: `${selected.modelSize.width} × ${selected.modelSize.height}`,
        resolution: `${size.width} × ${size.height}`,
      })
    : null;
  const fpsNote = fpsNeedsConversion(fit.model, fps)
    ? t("storyProjects.format.note.fpsConvert", {
        defaultValue:
          "{model} makes {native} fps clips. They are converted to {fps} fps when the film is joined: frames are repeated or dropped, not interpolated.",
        model: modelName,
        native: fit.model?.nativeFps,
        fps,
      })
    : null;

  return (
    <>
      <div className="control-stack">
        <span className="eyebrow">{t("storyProjects.dialog.aspect", { defaultValue: "Aspect ratio" })}</span>
        <div className="preset-row" role="radiogroup" aria-label={t("storyProjects.dialog.aspect", { defaultValue: "Aspect ratio" })}>
          {ASPECTS.map((key) => (
            <button
              key={key}
              type="button"
              role="radio"
              aria-checked={aspect === key}
              disabled={disabled}
              className={`preset-pill${aspect === key ? " active" : ""}`}
              onClick={() => aspect !== key && onAspect(key)}
            >
              <span className="preset-pill-label">{key}</span>
            </button>
          ))}
        </div>
      </div>
      <div className="control-stack">
        <span className="eyebrow">{t("storyProjects.dialog.resolution", { defaultValue: "Resolution" })}</span>
        <div className="preset-row" role="radiogroup" aria-label={t("storyProjects.dialog.resolution", { defaultValue: "Resolution" })}>
          {RESOLUTIONS.map((key) => {
            const pixels = sizeFor(aspect, key);
            const label = levelLabel[resolutionFit(fit, aspect, key, fps).level];
            return (
              <button
                key={key}
                type="button"
                role="radio"
                aria-checked={resolution === key}
                disabled={disabled}
                className={`preset-pill${resolution === key ? " active" : ""}`}
                onClick={() => resolution !== key && onResolution(key)}
              >
                <span className="preset-pill-label">{RESOLUTION_ALIAS[key] ? `${key} (${RESOLUTION_ALIAS[key]})` : key}</span>
                <span className="preset-pill-sub">{pixels.width} × {pixels.height}</span>
                {label ? <span className="preset-pill-sub">{label}</span> : null}
              </button>
            );
          })}
        </div>
        <p className="muted-text">
          {t("storyProjects.format.note.cost", { defaultValue: "Higher resolutions render slower and use more memory." })}
          {fitNote ? ` ${fitNote}` : ""}
          {fit.model ? ` ${t("storyProjects.format.note.estimated", { defaultValue: "Fits, Tight and Will not fit are estimates; every size stays selectable." })}` : ""}
        </p>
        {cappedNote ? <p className="muted-text">{cappedNote}</p> : null}
      </div>
      <div className="control-stack">
        <span className="eyebrow">{t("storyProjects.dialog.fps", { defaultValue: "Frame rate" })}</span>
        <div className="preset-row" role="radiogroup" aria-label={t("storyProjects.dialog.fps", { defaultValue: "Frame rate" })}>
          {FPS_CHOICES.map((value) => (
            <button
              key={value}
              type="button"
              role="radio"
              aria-checked={fps === value}
              disabled={disabled}
              className={`preset-pill${fps === value ? " active" : ""}`}
              onClick={() => fps !== value && onFps(value)}
            >
              <span className="preset-pill-label">{value} fps</span>
            </button>
          ))}
        </div>
        {fpsNote ? <p className="muted-text">{fpsNote}</p> : null}
      </div>
    </>
  );
}
