/**
 * One upscale job: its state in words, frames done of total with the time left from the measured speed, and the
 * buttons its state allows (``jobButtons``). Done, it shows where the film went and the input's first frame
 * beside the result.
 */
import { useTranslation } from "react-i18next";
import type { TFunction } from "i18next";
import type { VideoUpscaleJob } from "../../api/videoUpscale";
import { etaParts, etaSeconds, fileNameOf, jobButtons, percentDone } from "./upscaleLogic";

export interface UpscaleJobCardProps {
  job: VideoUpscaleJob;
  /** The quality row's label ("Fast"), when known. */
  modelLabel: string | null;
  /** A playable URL of the input (a Video Gallery clip), for the before/after. */
  beforeUrl: string | null;
  /** A request (cancel, resume) is in flight. */
  busy: boolean;
  actionError: string | null;
  onCancel: () => void;
  onResume: () => void;
  onReveal: () => void;
  onOpen: () => void;
  onDismiss: () => void;
}

export function etaText(seconds: number | null, t: TFunction<"studio">): string | null {
  if (seconds == null) return null;
  const parts = etaParts(seconds);
  if (parts.unit === "seconds") return t("videoUpscale.job.etaSeconds", { defaultValue: "about {count} s left", count: parts.value });
  if (parts.unit === "minutes") return t("videoUpscale.job.etaMinutes", { defaultValue: "about {count} min left", count: parts.value });
  return t("videoUpscale.job.etaHours", { defaultValue: "about {hours} h {minutes} min left", hours: parts.value, minutes: parts.minutes ?? 0 });
}

/** The job's state in one line. */
export function stateText(job: VideoUpscaleJob, t: TFunction<"studio">): string {
  switch (job.state) {
    case "queued":
      return t("videoUpscale.job.queued", { defaultValue: "Waiting for another job on the GPU to finish." });
    case "running":
      return t("videoUpscale.job.running", { defaultValue: "Upscaling…" });
    case "done":
      return t("videoUpscale.job.done", { defaultValue: "Done." });
    case "failed":
      return t("videoUpscale.job.failed", { defaultValue: "Stopped with an error: {error}", error: job.error || "?" });
    case "cancelled":
      return t("videoUpscale.job.cancelled", { defaultValue: "Cancelled. The finished parts are kept, so Resume carries on from there." });
    case "interrupted":
    default:
      return t("videoUpscale.job.interrupted", { defaultValue: "Stopped when the app closed. Resume carries on from where it stopped." });
  }
}

/** "120 of 2,400 frames" with the time left when there is a measured speed. */
export function progressText(job: VideoUpscaleJob, t: TFunction<"studio">): string {
  const frames = t("videoUpscale.job.frames", {
    defaultValue: "{done, number} of {total, number} frames",
    done: job.framesDone,
    total: job.framesTotal,
  });
  const eta = etaText(etaSeconds(job), t);
  const speed = job.state === "running" && job.framesPerSecond
    ? t("videoUpscale.job.speed", { defaultValue: "{fps} frames a second", fps: job.framesPerSecond })
    : null;
  return [frames, speed, eta].filter(Boolean).join(" · ");
}

export function UpscaleJobCard({
  job, modelLabel, beforeUrl, busy, actionError, onCancel, onResume, onReveal, onOpen, onDismiss,
}: UpscaleJobCardProps) {
  const { t } = useTranslation("studio");
  const buttons = jobButtons(job);
  const percent = percentDone(job);
  const done = job.state === "done";
  const failed = job.state === "failed";

  return (
    <div className={`callout video-upscale-job${failed ? " error" : ""}`} data-state={job.state}>
      <div className="video-upscale-job__head">
        <strong title={job.input}>{fileNameOf(job.input)}</strong>
        <span className="badge muted">
          {t("videoUpscale.job.target", { defaultValue: "to {width} × {height}", width: job.width, height: job.height })}
        </span>
        {modelLabel ? <span className="badge subtle">{modelLabel}</span> : null}
      </div>
      <p className={failed ? "error-text" : undefined}>{stateText(job, t)}</p>
      {!done ? (
        <div className="video-upscale-job__progress">
          <progress max={100} value={percent} aria-label={t("videoUpscale.job.progressLabel", { defaultValue: "Upscale progress" })} />
          <span className="muted-text">{percent}% · {progressText(job, t)}</span>
        </div>
      ) : null}
      {job.segments > 1 && !done ? (
        <p className="muted-text">
          {t("videoUpscale.job.parts", {
            defaultValue: "The film is made in {segments} parts; each finished part is kept if the job stops.",
            segments: job.segments,
          })}
        </p>
      ) : null}
      {done ? (
        <>
          <p className="muted-text" title={job.output}>
            {t("videoUpscale.job.savedAs", { defaultValue: "Saved as {path}", path: job.output })}
          </p>
          <div className="video-upscale-compare">
            <figure>
              <figcaption>{t("videoUpscale.compare.before", { defaultValue: "Before" })}</figcaption>
              {beforeUrl ? (
                <video src={beforeUrl} preload="metadata" muted playsInline controls />
              ) : (
                <div className="video-upscale-compare__empty muted-text">
                  {t("videoUpscale.compare.beforeMissing", { defaultValue: "No preview of this file in the app." })}
                </div>
              )}
            </figure>
            <figure>
              <figcaption>
                {t("videoUpscale.compare.after", { defaultValue: "After ({width} × {height})", width: job.width, height: job.height })}
              </figcaption>
              <div className="video-upscale-compare__empty muted-text">
                {t("videoUpscale.compare.afterOpen", { defaultValue: "Open the result to compare it at full size." })}
              </div>
            </figure>
          </div>
        </>
      ) : null}
      {job.notes.length > 0 ? (
        <ul className="muted-text video-upscale-job__notes">
          {job.notes.map((note) => <li key={note}>{note}</li>)}
        </ul>
      ) : null}
      <div className="button-row">
        {buttons.cancel ? (
          <button type="button" className="secondary-button" disabled={busy} onClick={onCancel}>
            {t("videoUpscale.job.cancel", { defaultValue: "Cancel" })}
          </button>
        ) : null}
        {buttons.resume ? (
          <button type="button" className="primary-button" disabled={busy} onClick={onResume}>
            {t("videoUpscale.job.resume", { defaultValue: "Resume" })}
          </button>
        ) : null}
        {buttons.open ? (
          <button type="button" className="primary-button" onClick={onOpen}>
            {t("videoUpscale.job.open", { defaultValue: "Open the result" })}
          </button>
        ) : null}
        {buttons.reveal ? (
          <button type="button" className="secondary-button" onClick={onReveal}>
            {t("videoUpscale.job.reveal", { defaultValue: "Show in folder" })}
          </button>
        ) : null}
        {buttons.dismiss ? (
          <button type="button" className="secondary-button" disabled={busy} onClick={onDismiss}>
            {t("videoUpscale.job.dismiss", { defaultValue: "Close" })}
          </button>
        ) : null}
      </div>
      {actionError ? <p className="error-text">{actionError}</p> : null}
    </div>
  );
}
