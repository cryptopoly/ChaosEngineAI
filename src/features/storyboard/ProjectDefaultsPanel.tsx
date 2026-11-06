/** Project defaults: shape, frame rate, and which models make the stills and clips. */
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { Disclosure } from "../../components/Disclosure";
import type { StoryboardExportPreset, StoryboardProject } from "../../types";
import type { RunOp } from "./ReferencesPanel";
import { EXPORT_PRESETS, FPS_CHOICES } from "./storyboardLogic";

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

export function ProjectDefaultsPanel({ project, busy, op, imageModels, videoModels }: ProjectDefaultsPanelProps) {
  const { t } = useTranslation("studio");
  const d = project.defaults;
  const [voice, setVoice] = useState(d.voice ?? "");
  useEffect(() => setVoice(d.voice ?? ""), [d.voice]);
  const nameOf = (choices: ModelChoice[], id: string | null) =>
    id ? (choices.find((c) => c.id === id)?.name ?? id) : t("board.defaults.automaticShort", { defaultValue: "auto" });

  return (
    <Disclosure
      id="story-defaults"
      title={t("board.defaultsTitle", { defaultValue: "Project defaults" })}
      summary={t("board.defaultsSummary", {
        defaultValue: "{aspect} · {fps} fps · {imageModel} · {videoModel} · voice {voice}",
        aspect: d.exportPreset,
        fps: d.fps,
        imageModel: nameOf(imageModels, d.imageModel),
        videoModel: nameOf(videoModels, d.videoModel),
        voice: d.voice ?? t("board.defaults.automaticShort", { defaultValue: "auto" }),
      })}
    >
      <div className="project-defaults">
        <div className="control-stack">
          <span className="eyebrow">{t("storyProjects.dialog.aspect", { defaultValue: "Shape" })}</span>
          <div className="preset-row" role="radiogroup" aria-label={t("storyProjects.dialog.aspect", { defaultValue: "Shape" })}>
            {(Object.keys(EXPORT_PRESETS) as StoryboardExportPreset[]).map((key) => (
              <button
                key={key}
                type="button"
                role="radio"
                aria-checked={d.exportPreset === key}
                disabled={busy}
                className={`preset-pill${d.exportPreset === key ? " active" : ""}`}
                onClick={() => d.exportPreset !== key && void op("edit_project", { defaults: { ...EXPORT_PRESETS[key], exportPreset: key } })}
              >
                <span className="preset-pill-label">{key}</span>
                <span className="preset-pill-sub">{EXPORT_PRESETS[key].width} × {EXPORT_PRESETS[key].height}</span>
              </button>
            ))}
          </div>
        </div>
        <div className="control-stack">
          <span className="eyebrow">{t("storyProjects.dialog.fps", { defaultValue: "Frame rate" })}</span>
          <div className="preset-row" role="radiogroup" aria-label={t("storyProjects.dialog.fps", { defaultValue: "Frame rate" })}>
            {FPS_CHOICES.map((value) => (
              <button
                key={value}
                type="button"
                role="radio"
                aria-checked={d.fps === value}
                disabled={busy}
                className={`preset-pill${d.fps === value ? " active" : ""}`}
                onClick={() => d.fps !== value && void op("edit_project", { defaults: { fps: value } })}
              >
                <span className="preset-pill-label">{value} fps</span>
              </button>
            ))}
          </div>
        </div>
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
        <label className="control-stack">
          <span className="eyebrow">{t("storyProjects.dialog.voice", { defaultValue: "Voice" })}</span>
          <input
            type="text"
            value={voice}
            maxLength={200}
            disabled={busy}
            onChange={(event) => setVoice(event.target.value)}
            onBlur={() => voice.trim() !== (d.voice ?? "") && void op("edit_project", { defaults: { voice: voice.trim() || null } })}
          />
        </label>
      </div>
    </Disclosure>
  );
}
