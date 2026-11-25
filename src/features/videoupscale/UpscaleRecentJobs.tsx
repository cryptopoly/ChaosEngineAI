/**
 * The newest upscale jobs (the backend keeps 20, newest first): click one to look at it in the job card above,
 * and Resume one that stopped part-way.
 */
import { useTranslation } from "react-i18next";
import type { TFunction } from "i18next";
import type { VideoUpscaleJob, VideoUpscaleState } from "../../api/videoUpscale";
import { fileNameOf, isLive, jobButtons, percentDone } from "./upscaleLogic";

export interface UpscaleRecentJobsProps {
  jobs: VideoUpscaleJob[];
  selectedId: string | null;
  /** A request is in flight, or a job is running (one at a time): Resume waits. */
  resumeDisabled: boolean;
  onSelect: (job: VideoUpscaleJob) => void;
  onResume: (job: VideoUpscaleJob) => void;
}

/** A state in one or two words, for a list row. */
export function stateLabel(state: VideoUpscaleState, t: TFunction<"studio">): string {
  switch (state) {
    case "queued":
      return t("videoUpscale.recent.queued", { defaultValue: "Waiting" });
    case "running":
      return t("videoUpscale.recent.running", { defaultValue: "Upscaling" });
    case "done":
      return t("videoUpscale.recent.done", { defaultValue: "Done" });
    case "failed":
      return t("videoUpscale.recent.failed", { defaultValue: "Failed" });
    case "cancelled":
      return t("videoUpscale.recent.cancelled", { defaultValue: "Cancelled" });
    case "interrupted":
    default:
      return t("videoUpscale.recent.interrupted", { defaultValue: "Stopped" });
  }
}

export function UpscaleRecentJobs({ jobs, selectedId, resumeDisabled, onSelect, onResume }: UpscaleRecentJobsProps) {
  const { t } = useTranslation("studio");
  if (jobs.length === 0) return null;
  return (
    <div className="control-stack video-upscale-recent">
      <span className="eyebrow">{t("videoUpscale.recent.label", { defaultValue: "Recent upscales" })}</span>
      <ul className="video-upscale-recent__list">
        {jobs.map((item) => (
          <li key={item.id} data-state={item.state} aria-current={item.id === selectedId ? "true" : undefined}>
            <button type="button" className="secondary-button" title={item.input} onClick={() => onSelect(item)}>
              <strong>{fileNameOf(item.input)}</strong>
              <span>{t("videoUpscale.job.target", { defaultValue: "to {width} × {height}", width: item.width, height: item.height })}</span>
              <span>
                {stateLabel(item.state, t)}
                {isLive(item.state) ? ` · ${percentDone(item)}%` : ""}
              </span>
            </button>
            {jobButtons(item).resume ? (
              <button type="button" className="primary-button" disabled={resumeDisabled} onClick={() => onResume(item)}>
                {t("videoUpscale.job.resume", { defaultValue: "Resume" })}
              </button>
            ) : null}
          </li>
        ))}
      </ul>
    </div>
  );
}
