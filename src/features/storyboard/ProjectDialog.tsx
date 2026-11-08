/**
 * New project (name, start-from template, aspect ratio, resolution, frame rate with fit hints, models) and
 * Rename (name only), on the shared modal classes. Every size and rate stays selectable; the hint is advice.
 */
import { useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Disclosure } from "../../components/Disclosure";
import { useDialogFocus } from "../../hooks/useDialogFocus";
import type { StoryboardExportPreset, StoryboardResolution } from "../../types";
import { FormatFields } from "./FormatFields";
import { DEFAULT_FPS, DEFAULT_RESOLUTION, RESOLUTION_ALIAS, fitContext, type VideoFitSource } from "./formatLogic";
import type { ModelChoice } from "./ProjectDefaultsPanel";
import { PROJECT_TEMPLATES, templateById, type ProjectTemplate } from "./projectTemplates";

export interface ProjectDialogResult {
  name: string;
  aspect: StoryboardExportPreset;
  resolution: StoryboardResolution;
  fps: number;
  template: ProjectTemplate["id"];
  imageModel: string | null;
  videoModel: string | null;
}

export interface ProjectDialogProps {
  mode: "create" | "rename";
  initialName?: string;
  busy?: boolean;
  error?: string | null;
  /** Installed video models and the machine, for the fit hint per resolution. */
  fitSource?: VideoFitSource;
  imageModels?: ModelChoice[];
  videoModels?: ModelChoice[];
  onSubmit: (result: ProjectDialogResult) => void;
  onCancel: () => void;
}

export function ProjectDialog({
  mode, initialName = "", busy, error, fitSource, imageModels = [], videoModels = [], onSubmit, onCancel,
}: ProjectDialogProps) {
  const { t } = useTranslation("studio");
  const [name, setName] = useState(initialName);
  const [template, setTemplate] = useState<ProjectTemplate["id"]>("blank");
  const [aspect, setAspect] = useState<StoryboardExportPreset>("16:9");
  const [resolution, setResolution] = useState<StoryboardResolution>(DEFAULT_RESOLUTION);
  const [fps, setFps] = useState<number>(DEFAULT_FPS);
  const [imageModel, setImageModel] = useState<string | null>(null);
  const [videoModel, setVideoModel] = useState<string | null>(null);
  const trimmed = name.trim();

  const dialog = useRef<HTMLFormElement>(null);
  useDialogFocus(dialog, { onEscape: onCancel });

  const title =
    mode === "create"
      ? t("storyProjects.new", { defaultValue: "New project" })
      : t("storyProjects.menu.rename", { defaultValue: "Rename" });

  function chooseTemplate(id: ProjectTemplate["id"]) {
    const picked = templateById(id);
    setTemplate(id);
    setAspect(picked.aspect);
    setResolution(picked.resolution);
    setFps(picked.fps);
  }

  const templateLabel = (item: ProjectTemplate) => t(`templates.${item.id}`, { defaultValue: item.label });
  const templateMeta = (item: ProjectTemplate) =>
    item.shots.length === 0
      ? t("templates.empty", { defaultValue: "No shots" })
      : t("templates.metaFormat", {
          defaultValue: "{aspect} · {resolution} · {fps} fps · {shots, plural, one {# shot} other {# shots}}",
          aspect: item.aspect,
          resolution: RESOLUTION_ALIAS[item.resolution] ?? item.resolution,
          fps: item.fps,
          shots: item.shots.length,
        });
  const automatic = t("board.defaults.automaticName", { defaultValue: "Automatic" });
  const nameOf = (choices: ModelChoice[], id: string | null) => (id ? (choices.find((c) => c.id === id)?.name ?? id) : automatic);

  return (
    <div className="modal-overlay" onMouseDown={(event) => event.target === event.currentTarget && onCancel()}>
      <form
        ref={dialog}
        className="modal-content project-dialog"
        role="dialog"
        aria-modal="true"
        aria-label={title}
        onSubmit={(event) => {
          event.preventDefault();
          if (trimmed) onSubmit({ name: trimmed, aspect, resolution, fps, template, imageModel, videoModel });
        }}
      >
        <div className="modal-header">
          <h3>{title}</h3>
        </div>
        <div className="modal-body">
          <label className="control-stack">
            <span className="eyebrow">{t("storyProjects.dialog.name", { defaultValue: "Name" })}</span>
            <input
              type="text"
              value={name}
              maxLength={200}
              data-autofocus=""
              onChange={(event) => setName(event.target.value)}
            />
          </label>
          {mode === "create" ? (
            <>
              <div className="control-stack">
                <span className="eyebrow">{t("templates.startFrom", { defaultValue: "Start from" })}</span>
                <div className="preset-row" role="radiogroup" aria-label={t("templates.startFrom", { defaultValue: "Start from" })}>
                  {PROJECT_TEMPLATES.map((item) => (
                    <button
                      key={item.id}
                      type="button"
                      role="radio"
                      aria-checked={template === item.id}
                      className={`preset-pill${template === item.id ? " active" : ""}`}
                      onClick={() => chooseTemplate(item.id)}
                    >
                      <span className="preset-pill-label">{templateLabel(item)}</span>
                      <span className="preset-pill-sub">{templateMeta(item)}</span>
                    </button>
                  ))}
                </div>
                {template !== "blank" ? (
                  <p className="muted-text" role="status">
                    {t("templates.sets", {
                      defaultValue: "{name} set the format to {aspect}, {resolution}, {fps} fps. Change any of them below.",
                      name: templateLabel(templateById(template)),
                      aspect,
                      resolution: RESOLUTION_ALIAS[resolution] ?? resolution,
                      fps,
                    })}
                  </p>
                ) : null}
              </div>
              <FormatFields
                aspect={aspect}
                resolution={resolution}
                fps={fps}
                fit={fitContext(fitSource, videoModel)}
                onAspect={setAspect}
                onResolution={setResolution}
                onFps={setFps}
              />
              <Disclosure
                id="new-project-models"
                title={t("storyProjects.dialog.models", { defaultValue: "Models" })}
                summary={`${nameOf(imageModels, imageModel)} · ${nameOf(videoModels, videoModel)}`}
              >
                <label className="control-stack">
                  <span className="eyebrow">{t("storyProjects.dialog.imageModel", { defaultValue: "Image model" })}</span>
                  <select value={imageModel ?? ""} onChange={(event) => setImageModel(event.target.value || null)}>
                    <option value="">{t("board.defaults.automatic", { defaultValue: "Automatic (an installed model)" })}</option>
                    {imageModels.map((choice) => <option key={choice.id} value={choice.id}>{choice.name}</option>)}
                  </select>
                </label>
                <label className="control-stack">
                  <span className="eyebrow">{t("storyProjects.dialog.videoModel", { defaultValue: "Video model" })}</span>
                  <select value={videoModel ?? ""} onChange={(event) => setVideoModel(event.target.value || null)}>
                    <option value="">{t("board.defaults.automatic", { defaultValue: "Automatic (an installed model)" })}</option>
                    {videoModels.map((choice) => <option key={choice.id} value={choice.id}>{choice.name}</option>)}
                  </select>
                </label>
              </Disclosure>
            </>
          ) : null}
          {error ? <p className="error-text" role="alert">{error}</p> : null}
        </div>
        <div className="modal-footer">
          <button type="button" className="secondary-button" onClick={onCancel}>
            {t("storyboard.cancel", { defaultValue: "Cancel" })}
          </button>
          <button type="submit" className="primary-button" disabled={!trimmed || busy}>
            {mode === "create"
              ? t("storyProjects.dialog.create", { defaultValue: "Create" })
              : t("storyProjects.dialog.save", { defaultValue: "Save" })}
          </button>
        </div>
      </form>
    </div>
  );
}
