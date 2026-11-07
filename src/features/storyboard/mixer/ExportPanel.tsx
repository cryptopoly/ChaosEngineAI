/**
 * The Mixer's export strip (UX spec D.6): shape presets with the pixels they give, the AI-label options
 * (D-34), the loudness target, captions (not built yet, shown disabled), the run estimate with the
 * Export button, render progress, and the result card of the last export.
 */
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { LiveProgress, type LiveProgressPhase } from "../../../components/LiveProgress";
import { RunEstimateStrip } from "../../../components/RunEstimateStrip";
import type { GenerationProgressSnapshot, StoryboardExportPreset, StoryboardProject, StoryboardRenderJob } from "../../../types";
import {
  EXPORT_PRESET_ORDER,
  LOUDNESS_SUGGESTED,
  RENDER_PHASES,
  aiLabelState,
  exportSize,
  fileNameOf,
  formatBytes,
  parseLoudness,
  renderPercent,
  renderPhase,
  type RenderPhaseId,
} from "./mixerLogic";
import { formatTimecode } from "./timelineGeometry";

// ---- AI label ----
export interface AiLabelNoticeProps {
  metadata: boolean;
  /** A cloned voice is in the project: the tag cannot be turned off. */
  forced: boolean;
  disabled?: boolean;
  onChange: (change: { metadata?: boolean }) => void;
}

export function AiLabelNotice({ metadata, forced, disabled, onChange }: AiLabelNoticeProps) {
  const { t } = useTranslation("studio");
  return (
    <fieldset className="ai-label-notice">
      <legend className="eyebrow">{t("aiLabel.title", { defaultValue: "AI label" })}</legend>
      <label
        className="check-row"
        title={t("aiLabel.metadataTitle", {
          defaultValue: "A machine-readable tag in the MP4 metadata. It holds no prompts or folder names.",
        })}
      >
        <input
          type="checkbox"
          role="switch"
          checked={metadata}
          disabled={disabled || forced}
          onChange={(event) => onChange({ metadata: event.target.checked })}
        />
        {t("aiLabel.metadata", { defaultValue: "Add a machine-readable AI tag" })}
      </label>
      {forced ? (
        <p className="muted-text ai-label-notice__reason">
          {t("aiLabel.forced", { defaultValue: "This project uses a cloned voice, so the AI tag is always written." })}
        </p>
      ) : null}
    </fieldset>
  );
}

// ---- render progress ----
const PHASE_SHARE: Record<RenderPhaseId, number> = { normalise: 0.4, join: 0.2, mix: 0.15, encode: 0.25 };

export interface RenderProgressProps {
  job: StoryboardRenderJob;
  startedAt: number;
  /** Length of the film, to size the phase estimates. */
  filmSeconds: number;
  title: string;
}

export function RenderProgress({ job, startedAt, filmSeconds, title }: RenderProgressProps) {
  const { t } = useTranslation("studio");
  const labels: Record<RenderPhaseId, string> = {
    normalise: t("mixer.export.phase.normalise", { defaultValue: "Normalise shots" }),
    join: t("mixer.export.phase.join", { defaultValue: "Join" }),
    mix: t("mixer.export.phase.mix", { defaultValue: "Mix sound" }),
    encode: t("mixer.export.phase.encode", { defaultValue: "Encode" }),
  };
  const total = Math.max(8, filmSeconds * (job.preview ? 0.6 : 1.5));
  const phases: LiveProgressPhase[] = RENDER_PHASES.map((id) => ({ id, label: labels[id], estimatedSeconds: total * PHASE_SHARE[id] }));
  const phase = renderPhase(job.message);
  // The assembler reports its own stages, not the Studios' ones: the snapshot carries our phase ids.
  const snapshot = {
    kind: "video",
    active: true,
    phase: phase as unknown as GenerationProgressSnapshot["phase"],
    message: `${renderPercent(job)} %`,
    step: 0,
    totalSteps: 0,
    startedAt: startedAt / 1000,
    updatedAt: Date.now() / 1000,
    elapsedSeconds: (Date.now() - startedAt) / 1000,
    runLabel: null,
  } satisfies GenerationProgressSnapshot;
  return <LiveProgress title={title} accent="storyboard" phases={phases} startedAt={startedAt} realProgress={snapshot} />;
}

// ---- loudness ----
function LoudnessField({ value, disabled, onCommit }: { value: number | null; disabled: boolean; onCommit: (lufs: number | null) => void }) {
  const { t } = useTranslation("studio");
  const [text, setText] = useState(value == null ? "" : String(value));
  useEffect(() => setText(value == null ? "" : String(value)), [value]);
  const commit = () => {
    const parsed = parseLoudness(text);
    if (parsed === undefined) {
      setText(value == null ? "" : String(value));
      return;
    }
    setText(parsed == null ? "" : String(parsed));
    if (parsed !== value) onCommit(parsed);
  };
  return (
    <label className="export-panel__loudness">
      <span className="eyebrow">{t("mixer.export.loudness", { defaultValue: "Loudness target" })}</span>
      <span className="clip-inspector__number">
        <input
          type="number"
          min={-40}
          max={-5}
          step={0.5}
          value={text}
          disabled={disabled}
          placeholder={t("mixer.export.loudnessOff", { defaultValue: "Off" })}
          onChange={(event) => setText(event.target.value)}
          onBlur={commit}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              commit();
            }
          }}
        />
        <span aria-hidden="true">LUFS</span>
      </span>
      <span className="muted-text export-panel__hint">
        {t("mixer.export.loudnessHint", {
          defaultValue: "{target} LUFS suits web video. Empty leaves the mix as it is.",
          target: LOUDNESS_SUGGESTED,
        })}
      </span>
    </label>
  );
}

// ---- the panel ----
export interface ExportResult {
  job: StoryboardRenderJob;
  bytes: number | null;
}

export interface ExportPanelProps {
  project: StoryboardProject;
  filmSeconds: number;
  /** Nothing to export: no shot has a playable clip. */
  empty: boolean;
  /** A render of either kind is running. */
  rendering: boolean;
  /** The running job, when it is an export. */
  exportJob: StoryboardRenderJob | null;
  startedAt: number;
  result: ExportResult | null;
  disabled: boolean;
  revealLabel: string;
  /** The path was just copied. */
  copied: boolean;
  onPreset: (preset: StoryboardExportPreset) => void;
  onAiLabel: (change: { metadata?: boolean }) => void;
  onLoudness: (lufs: number | null) => void;
  onExport: () => void;
  onCancel: () => void;
  onReveal: (file: string) => void;
  onCopyPath: (file: string) => void;
}

export function ExportPanel(props: ExportPanelProps) {
  const { project, filmSeconds, empty, rendering, exportJob, startedAt, result, disabled } = props;
  const { t } = useTranslation("studio");
  const preset = project.defaults.exportPreset;
  const size = exportSize(project.defaults, preset);
  const ai = aiLabelState(project);
  const names: Record<StoryboardExportPreset, string> = {
    "16:9": t("mixer.export.preset.landscape", { defaultValue: "16:9" }),
    "9:16": t("mixer.export.preset.portrait", { defaultValue: "9:16" }),
    "1:1": t("mixer.export.preset.square", { defaultValue: "1:1" }),
  };
  const running = exportJob?.state === "running";

  return (
    <section className="export-panel" aria-label={t("mixer.export.title", { defaultValue: "Export" })}>
      <div className="export-panel__row">
        <div className="export-panel__shapes">
          <span className="eyebrow">{t("mixer.export.shape", { defaultValue: "Shape" })}</span>
          <div className="preset-row" role="radiogroup" aria-label={t("mixer.export.shape", { defaultValue: "Shape" })}>
            {EXPORT_PRESET_ORDER.map((id) => (
              <button
                key={id}
                type="button"
                role="radio"
                aria-checked={preset === id}
                className={`preset-pill${preset === id ? " active" : ""}`}
                disabled={disabled || rendering}
                onClick={() => preset !== id && props.onPreset(id)}
              >
                {names[id]}
              </button>
            ))}
          </div>
          <p className="muted-text export-panel__hint">
            {t("mixer.export.pixels", { defaultValue: "{width} × {height} px.", width: size.width, height: size.height })}{" "}
            {t("mixer.export.reframeNote", { defaultValue: "Reframed at export; shots are not regenerated." })}
          </p>
        </div>
        <LoudnessField value={project.export.loudnessLufs} disabled={disabled || rendering} onCommit={props.onLoudness} />
      </div>

      <div className="export-panel__options">
        <AiLabelNotice
          metadata={ai.metadata}
          forced={ai.forced}
          disabled={disabled || rendering}
          onChange={props.onAiLabel}
        />
        <fieldset className="export-panel__captions" disabled>
          <legend className="eyebrow">{t("mixer.captions.label", { defaultValue: "Captions" })}</legend>
          <label className="check-row">
            <input type="checkbox" disabled /> {t("mixer.captions.srt", { defaultValue: "SRT file" })}
          </label>
          <label className="check-row">
            <input type="checkbox" disabled /> {t("mixer.captions.burnIn", { defaultValue: "Burn in" })}
          </label>
          <p className="muted-text">{t("mixer.captions.soon", { defaultValue: "Captions are coming" })}</p>
        </fieldset>
      </div>

      <RunEstimateStrip
        seconds={null}
        peakGb={null}
        totalGb={null}
        diskMb={null}
        basis={null}
        risk="safe"
        action={
          running ? (
            <button type="button" className="secondary-button" onClick={props.onCancel}>
              {t("mixer.export.cancel", { defaultValue: "Cancel export" })}
            </button>
          ) : (
            <button type="button" className="primary-button" disabled={disabled || rendering || empty} onClick={props.onExport}>
              {t("mixer.export.button", { defaultValue: "Export MP4" })}
            </button>
          )
        }
      />

      {exportJob && running ? (
        <RenderProgress
          job={exportJob}
          startedAt={startedAt}
          filmSeconds={filmSeconds}
          title={t("mixer.export.exporting", { defaultValue: "Exporting…" })}
        />
      ) : null}

      {exportJob?.state === "failed" ? (
        <div className="callout error export-panel__failed" role="alert">
          <p>{t("mixer.export.failedShort", { defaultValue: "Export failed. Nothing was written." })}</p>
          {exportJob.error ? (
            <details>
              <summary>{t("mixer.export.details", { defaultValue: "Technical details" })}</summary>
              <p className="muted-text">{exportJob.error}</p>
            </details>
          ) : null}
          <button type="button" className="secondary-button" disabled={disabled || rendering || empty} onClick={props.onExport}>
            {t("mixer.export.retry", { defaultValue: "Retry" })}
          </button>
        </div>
      ) : null}
      {exportJob?.state === "cancelled" ? (
        <p className="muted-text">{t("mixer.export.cancelled", { defaultValue: "Export cancelled. Nothing was written." })}</p>
      ) : null}

      {result?.job.file ? (
        <div className="export-panel__result" role="status">
          <p className="export-panel__done">
            {t("mixer.export.done", {
              defaultValue: "Exported · {size} · {duration}",
              size: result.bytes != null ? formatBytes(result.bytes) : "…",
              duration: formatTimecode(filmSeconds, false),
            })}
          </p>
          <p className="muted-text export-panel__file" title={result.job.file}>{fileNameOf(result.job.file)}</p>
          <div className="export-panel__result-actions">
            <button type="button" className="secondary-button" onClick={() => props.onReveal(result.job.file as string)}>
              {props.revealLabel}
            </button>
            <button type="button" className="secondary-button" onClick={() => props.onCopyPath(result.job.file as string)}>
              {props.copied ? t("mixer.export.copiedShort", { defaultValue: "Copied" }) : t("mixer.export.copyPath", { defaultValue: "Copy path" })}
            </button>
          </div>
        </div>
      ) : null}
    </section>
  );
}
