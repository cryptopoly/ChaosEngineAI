/** New project (name, shape, frame rate) and Rename (name only), on the shared modal classes. */
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import type { StoryboardExportPreset } from "../../types";
import { EXPORT_PRESETS, FPS_CHOICES } from "./storyboardLogic";

export interface ProjectDialogResult {
  name: string;
  preset: StoryboardExportPreset;
  fps: number;
}

export interface ProjectDialogProps {
  mode: "create" | "rename";
  initialName?: string;
  busy?: boolean;
  error?: string | null;
  onSubmit: (result: ProjectDialogResult) => void;
  onCancel: () => void;
}

export function ProjectDialog({ mode, initialName = "", busy, error, onSubmit, onCancel }: ProjectDialogProps) {
  const { t } = useTranslation("studio");
  const [name, setName] = useState(initialName);
  const [preset, setPreset] = useState<StoryboardExportPreset>("16:9");
  const [fps, setFps] = useState<number>(24);
  const trimmed = name.trim();

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onCancel();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onCancel]);

  const title =
    mode === "create"
      ? t("storyProjects.new", { defaultValue: "New project" })
      : t("storyProjects.menu.rename", { defaultValue: "Rename" });

  return (
    <div className="modal-overlay" onMouseDown={(event) => event.target === event.currentTarget && onCancel()}>
      <form
        className="modal-content project-dialog"
        role="dialog"
        aria-modal="true"
        aria-label={title}
        onSubmit={(event) => {
          event.preventDefault();
          if (trimmed) onSubmit({ name: trimmed, preset, fps });
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
              autoFocus
              onChange={(event) => setName(event.target.value)}
            />
          </label>
          {mode === "create" ? (
            <>
              <div className="control-stack">
                <span className="eyebrow">{t("storyProjects.dialog.aspect", { defaultValue: "Shape" })}</span>
                <div className="preset-row" role="radiogroup" aria-label={t("storyProjects.dialog.aspect", { defaultValue: "Shape" })}>
                  {(Object.keys(EXPORT_PRESETS) as StoryboardExportPreset[]).map((key) => (
                    <button
                      key={key}
                      type="button"
                      role="radio"
                      aria-checked={preset === key}
                      className={`preset-pill${preset === key ? " active" : ""}`}
                      onClick={() => setPreset(key)}
                    >
                      <span className="preset-pill-label">{key}</span>
                      <span className="preset-pill-sub">
                        {EXPORT_PRESETS[key].width} × {EXPORT_PRESETS[key].height}
                      </span>
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
                      aria-checked={fps === value}
                      className={`preset-pill${fps === value ? " active" : ""}`}
                      onClick={() => setFps(value)}
                    >
                      <span className="preset-pill-label">{value} fps</span>
                    </button>
                  ))}
                </div>
              </div>
            </>
          ) : null}
          {error ? <p className="error-text">{error}</p> : null}
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
