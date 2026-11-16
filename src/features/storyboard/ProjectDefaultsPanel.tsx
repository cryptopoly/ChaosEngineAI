/**
 * Project settings on Overview: aspect ratio, resolution, frame rate, the models that make stills and clips, and the
 * voice. Each still model shows its time per still at the project size; Automatic picks the fastest for Make and
 * drafts and the best quality for the final pass (S2, S6). A slower saved still model gets "Use the fast model".
 */
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import type { StoryboardStillModels } from "../../api/storyboard";
import { formatDuration } from "../../components/RunEstimateStrip";
import { Disclosure } from "../../components/Disclosure";
import type { StoryboardProject } from "../../types";
import { FormatFields } from "./FormatFields";
import { RESOLUTION_ALIAS, fitContext, formatPatch, type VideoFitSource } from "./formatLogic";
import { stillModelPick } from "./storyboardLogic";
import { UseFastModelButton } from "./UseFastModelButton";
import { useStillModels } from "./useStillModels";
import type { RunOp } from "./useStoryboard";
import { VoicePicker, voiceDisplayName } from "./VoicePicker";

export interface ModelChoice {
  id: string;
  name: string;
}

export interface ProjectDefaultsPanelProps {
  project: StoryboardProject;
  busy: boolean;
  op: RunOp;
  imageModels: ModelChoice[];
  videoModels: ModelChoice[];
  /** Installed video models and the machine, for the fit hint per resolution. */
  videoFit?: VideoFitSource;
  /** Open while the project is new; collapsed once it has shots. */
  defaultOpen?: boolean;
  /** A new value opens the settings and moves the focus there (the header's format chips). */
  openRequest?: number;
}

type TFn = (key: string, options: Record<string, unknown>) => string;

function ModelSelect({
  label,
  value,
  choices,
  disabled,
  onChange,
  automaticLabel,
  secondsOf,
  hint,
}: {
  label: string;
  value: string | null;
  choices: ModelChoice[];
  disabled: boolean;
  onChange: (id: string | null) => void;
  automaticLabel?: string;
  /** Seconds per still of a choice, shown after its name. */
  secondsOf?: (id: string) => number | null;
  hint?: string | null;
}) {
  const { t } = useTranslation("studio");
  const known = value == null || choices.some((choice) => choice.id === value);
  const timed = (choice: ModelChoice) => {
    const seconds = secondsOf?.(choice.id);
    return seconds != null
      ? t("board.defaults.modelTimed", { defaultValue: "{name} · about {time} per still", name: choice.name, time: formatDuration(t as unknown as TFn, seconds) })
      : choice.name;
  };
  return (
    <label className="control-stack">
      <span className="eyebrow">{label}</span>
      <select value={value ?? ""} disabled={disabled} onChange={(event) => onChange(event.target.value || null)}>
        <option value="">{automaticLabel ?? t("board.defaults.automatic", { defaultValue: "Automatic (an installed model)" })}</option>
        {!known && value ? <option value={value}>{value}</option> : null}
        {choices.map((choice) => (
          <option key={choice.id} value={choice.id}>{timed(choice)}</option>
        ))}
      </select>
      {hint ? <span className="muted-text">{hint}</span> : null}
    </label>
  );
}

/** The Automatic entry and the hint under a still model select: which model it is and its time per still. */
function stillLabels(t: TFn, models: StoryboardStillModels | null, saved: string | null | undefined, which: "fast" | "quality") {
  const pick = stillModelPick(models?.models ?? [], saved, models?.[which] ?? null);
  const automaticLabel = models?.[which]
    ? which === "fast"
      ? t("board.defaults.automaticFast", { defaultValue: "Automatic: fastest ({name})", name: stillModelPick(models.models, null, models.fast).name })
      : t("board.defaults.automaticQuality", { defaultValue: "Automatic: best quality ({name})", name: stillModelPick(models.models, null, models.quality).name })
    : undefined;
  const hint =
    pick.seconds != null
      ? t("board.defaults.stillTime", {
          defaultValue: "{name}: about {time} per still at {width} x {height}",
          name: pick.name,
          time: formatDuration(t, pick.seconds),
          width: models?.width,
          height: models?.height,
        })
      : null;
  return { automaticLabel, hint };
}

export function ProjectDefaultsPanel({ project, busy, op, imageModels, videoModels, videoFit, defaultOpen, openRequest }: ProjectDefaultsPanelProps) {
  const { t } = useTranslation("studio");
  const [open, setOpen] = useState(Boolean(defaultOpen) || Boolean(openRequest));
  useEffect(() => {
    if (!openRequest) return;
    setOpen(true);
    requestAnimationFrame(() => {
      const summary = document.querySelector<HTMLElement>("#disclosure-story-settings > summary");
      summary?.focus();
      summary?.scrollIntoView?.({ block: "nearest" });
    });
  }, [openRequest]);
  const d = project.defaults;
  const stills = useStillModels(d.width, d.height, open);
  const tf = t as unknown as TFn;
  const secondsOf = (id: string) => stills?.models.find((m) => m.id === id)?.secondsPerStill ?? null;
  const fastLabels = stillLabels(tf, stills, d.imageModel, "fast");
  const qualityLabels = stillLabels(tf, stills, d.finalImageModel, "quality");
  const nameOf = (choices: ModelChoice[], id: string | null) =>
    id ? (choices.find((c) => c.id === id)?.name ?? id) : t("board.defaults.automaticName", { defaultValue: "Automatic" });

  return (
    <Disclosure
      id="story-settings"
      title={t("overview.settingsTitle", { defaultValue: "Project settings" })}
      open={open}
      onToggle={setOpen}
      summary={t("board.defaultsSummary", {
        defaultValue: "{aspect} · {resolution} · {fps} fps · {imageModel} · {videoModel} · voice {voice}",
        aspect: d.exportPreset,
        resolution: RESOLUTION_ALIAS[d.resolution] ?? d.resolution,
        fps: d.fps,
        imageModel: nameOf(imageModels, d.imageModel),
        videoModel: nameOf(videoModels, d.videoModel),
        voice: d.voice ? voiceDisplayName(d.voice) : t("board.defaults.noVoice", { defaultValue: "none" }),
      })}
    >
      <div className="project-defaults">
        <FormatFields
          aspect={d.exportPreset}
          resolution={d.resolution}
          fps={d.fps}
          disabled={busy}
          fit={fitContext(videoFit, d.videoModel)}
          onAspect={(aspect) => void op("edit_project", formatPatch(d, { aspect }))}
          onResolution={(resolution) => void op("edit_project", formatPatch(d, { resolution }))}
          onFps={(fps) => void op("edit_project", { defaults: { fps } })}
        />
        <ModelSelect
          label={t("storyProjects.dialog.imageModel", { defaultValue: "Image model" })}
          value={d.imageModel}
          choices={imageModels}
          disabled={busy}
          onChange={(id) => void op("edit_project", { defaults: { imageModel: id } })}
          automaticLabel={fastLabels.automaticLabel}
          secondsOf={secondsOf}
          hint={fastLabels.hint}
        />
        <UseFastModelButton
          models={stills}
          saved={d.imageModel}
          disabled={busy}
          onUse={(id) => void op("edit_project", { defaults: { imageModel: id } })}
        />
        <ModelSelect
          label={t("board.defaults.finalImageModel", { defaultValue: "Final pass image model" })}
          value={d.finalImageModel ?? null}
          choices={imageModels}
          disabled={busy}
          onChange={(id) => void op("edit_project", { defaults: { finalImageModel: id } })}
          automaticLabel={qualityLabels.automaticLabel}
          secondsOf={secondsOf}
          hint={qualityLabels.hint}
        />
        <ModelSelect
          label={t("storyProjects.dialog.videoModel", { defaultValue: "Video model" })}
          value={d.videoModel}
          choices={videoModels}
          disabled={busy}
          onChange={(id) => void op("edit_project", { defaults: { videoModel: id } })}
        />
        <VoicePicker value={d.voice} disabled={busy} onChange={(voice) => void op("edit_project", { defaults: { voice } })} />
      </div>
    </Disclosure>
  );
}
