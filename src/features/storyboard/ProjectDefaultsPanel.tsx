/** Project settings on Overview: aspect ratio, resolution, frame rate, the models that make stills and clips, and the voice. */
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { Disclosure } from "../../components/Disclosure";
import type { StoryboardProject } from "../../types";
import { FormatFields } from "./FormatFields";
import { RESOLUTION_ALIAS, fitContext, formatPatch, type VideoFitSource } from "./formatLogic";
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

function ModelSelect({
  label,
  value,
  choices,
  disabled,
  onChange,
}: {
  label: string;
  value: string | null;
  choices: ModelChoice[];
  disabled: boolean;
  onChange: (id: string | null) => void;
}) {
  const { t } = useTranslation("studio");
  const known = value == null || choices.some((choice) => choice.id === value);
  return (
    <label className="control-stack">
      <span className="eyebrow">{label}</span>
      <select value={value ?? ""} disabled={disabled} onChange={(event) => onChange(event.target.value || null)}>
        <option value="">{t("board.defaults.automatic", { defaultValue: "Automatic (an installed model)" })}</option>
        {!known && value ? <option value={value}>{value}</option> : null}
        {choices.map((choice) => (
          <option key={choice.id} value={choice.id}>{choice.name}</option>
        ))}
      </select>
    </label>
  );
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
