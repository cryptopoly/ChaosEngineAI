/**
 * The Mixer's captions disclosure (closed by default): make captions from the voice lane with the
 * speech model, fix the wording, add one by hand, write an .srt. Burning them in is a switch in the
 * Export panel. A caption made from a voice clip moves with that clip.
 */
import { useTranslation } from "react-i18next";
import type { StoryboardProject, StoryboardTimeline } from "../../../types";
import { captionRows } from "./mixerLogic";
import { fileNameOf } from "./mixerLogic";
import { formatTimecode } from "./timelineGeometry";

export interface CaptionsState {
  busy: boolean;
  notes: string[];
  error: string | null;
  /** The last .srt written, project-relative. */
  file: string | null;
}

export interface CaptionsPanelProps {
  project: StoryboardProject;
  timeline: StoryboardTimeline;
  state: CaptionsState;
  disabled: boolean;
  playheadS: number;
  revealLabel: string;
  onGenerate: () => void;
  onAdd: (startS: number) => void;
  onEditText: (id: string, text: string) => void;
  onRemove: (id: string) => void;
  onExportSrt: () => void;
  onReveal: (file: string) => void;
}

export function CaptionsPanel(props: CaptionsPanelProps) {
  const { project, timeline, state, disabled } = props;
  const { t } = useTranslation("studio");
  const rows = captionRows(project, timeline);
  const hasVoice = project.audio.some((clip) => clip.kind === "voice" && clip.status !== "missing");
  const working = disabled || state.busy;

  return (
    <details className="captions-panel">
      <summary>
        {t("mixer.captions.title", { defaultValue: "Captions ({count})", count: rows.length })}
      </summary>
      <div className="captions-panel__body">
        <div className="captions-panel__actions">
          <button type="button" className="secondary-button" disabled={working || !hasVoice} onClick={props.onGenerate}>
            {state.busy
              ? t("mixer.captions.generating", { defaultValue: "Listening…" })
              : t("mixer.captions.generate", { defaultValue: "Make from voice" })}
          </button>
          <button type="button" className="secondary-button" disabled={working} onClick={() => props.onAdd(props.playheadS)}>
            {t("mixer.captions.add", { defaultValue: "Add at playhead" })}
          </button>
          <button type="button" className="secondary-button" disabled={working || rows.length === 0} onClick={props.onExportSrt}>
            {t("mixer.captions.exportSrt", { defaultValue: "Write .srt" })}
          </button>
        </div>
        {!hasVoice ? (
          <p className="muted-text">{t("mixer.captions.needVoice", { defaultValue: "Add speech to the voice lane to make captions from it." })}</p>
        ) : null}
        {state.file ? (
          <p className="muted-text" role="status">
            {fileNameOf(state.file)}{" "}
            <button type="button" className="secondary-button" onClick={() => props.onReveal(state.file as string)}>
              {props.revealLabel}
            </button>
          </p>
        ) : null}
        {state.notes.map((note) => (
          <p key={note} className="muted-text" role="status">{note}</p>
        ))}
        {state.error ? <p className="callout error" role="alert">{state.error}</p> : null}
        <ul className="captions-panel__list">
          {rows.map((row) => (
            <li key={`${row.id}:${row.text}`} className="captions-panel__row">
              <span className="muted-text captions-panel__time">
                {row.start === null
                  ? t("mixer.captions.trimmedAway", { defaultValue: "Trimmed away" })
                  : `${formatTimecode(row.start)} – ${formatTimecode(row.end ?? row.start)}`}
              </span>
              <input
                type="text"
                defaultValue={row.text}
                maxLength={500}
                disabled={working}
                aria-label={t("mixer.captions.textLabel", { defaultValue: "Caption text" })}
                onBlur={(event) => {
                  const text = event.target.value.trim();
                  if (text && text !== row.text) props.onEditText(row.id, text);
                }}
              />
              <button type="button" className="secondary-button" disabled={working} onClick={() => props.onRemove(row.id)}>
                {t("mixer.captions.remove", { defaultValue: "Remove" })}
              </button>
            </li>
          ))}
        </ul>
      </div>
    </details>
  );
}
