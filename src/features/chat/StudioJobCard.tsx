import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { cancelStudioJob, fetchStudioJob, fetchStudioJobMedia } from "../../api";
import type { StudioJob, ToolCallInfo } from "../../types";
import { openProjectAt } from "../storyboard/projectRoute";
import { isFinished, jobPercent, libraryTab, mediaKind, openAppTab, studioJobId } from "./studioChat";

const POLL_MS = 1500;

/** A declined call: nothing ran. */
function DeclinedCard({ toolCall }: { toolCall: ToolCallInfo }) {
  const { t } = useTranslation("chat");
  return (
    <div className="studio-job-card studio-job-card--declined">
      <span className="studio-job-card__tool">{toolCall.name}</span>
      <span>{t("studioChat.job.declined", { defaultValue: "Declined. Nothing was made." })}</span>
    </div>
  );
}

/**
 * D-60: a job a studio tool started in Chat. Polls the job (queued, running with its own progress),
 * offers Cancel, and when it is done shows the result: image preview, audio player or video, with
 * "Open in Library".
 */
export function StudioJobCard({ toolCall }: { toolCall: ToolCallInfo }) {
  const { t } = useTranslation("chat");
  const jobId = studioJobId(toolCall);
  const [job, setJob] = useState<StudioJob | null>(null);
  const [mediaUrl, setMediaUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!jobId) return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const poll = async () => {
      try {
        const next = await fetchStudioJob(jobId);
        if (stopped) return;
        setJob(next);
        setError(null);
        if (!isFinished(next)) timer = setTimeout(() => void poll(), POLL_MS);
      } catch (err) {
        if (stopped) return;
        setError(err instanceof Error ? err.message : String(err));
      }
    };
    void poll();
    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
    };
  }, [jobId]);

  const kind = job ? mediaKind(job) : "none";
  useEffect(() => {
    if (!jobId || job?.state !== "done" || kind === "none") return;
    let url: string | null = null;
    let stopped = false;
    fetchStudioJobMedia(jobId)
      .then((blob) => {
        if (stopped) return;
        url = URL.createObjectURL(blob);
        setMediaUrl(url);
      })
      .catch(() => undefined);
    return () => {
      stopped = true;
      if (url) URL.revokeObjectURL(url);
    };
  }, [jobId, job?.state, kind]);

  if (toolCall.data?.declined) return <DeclinedCard toolCall={toolCall} />;
  if (!jobId) return null;

  const percent = job ? jobPercent(job) : null;
  const state = job?.state ?? "queued";
  const tab = job ? libraryTab(job, toolCall.name) : null;
  const stateText =
    state === "queued"
      ? t("studioChat.job.queued", { position: job?.queuePosition ?? 0, defaultValue: "Queued (position {position})" })
      : state === "running"
      ? job?.progress?.phase
        ? t("studioChat.job.runningPhase", { phase: job.progress.phase, defaultValue: "Running: {phase}" })
        : t("studioChat.job.running", { defaultValue: "Running" })
      : state === "done"
      ? t("studioChat.job.done", { defaultValue: "Done" })
      : state === "cancelled"
      ? t("studioChat.job.cancelled", { defaultValue: "Cancelled" })
      : state === "interrupted"
      ? t("studioChat.job.interrupted", { defaultValue: "Interrupted (the app restarted)" })
      : t("studioChat.job.failed", { defaultValue: "Failed" });

  return (
    <div className={`studio-job-card studio-job-card--${state}`} data-job-id={jobId}>
      <div className="studio-job-card__head">
        <span className="studio-job-card__tool">{toolCall.name}</span>
        <span className="studio-job-card__label">{job?.label ?? String(toolCall.data?.label ?? "")}</span>
        <span className="studio-job-card__state" role="status">{stateText}</span>
        {job && !isFinished(job) ? (
          <button type="button" className="secondary-button studio-job-card__cancel"
            onClick={() => void cancelStudioJob(jobId).then(setJob).catch((err) => setError(String(err)))}>
            {t("studioChat.job.cancel", { defaultValue: "Cancel" })}
          </button>
        ) : null}
      </div>
      {state === "running" || state === "queued" ? (
        <div className="studio-job-card__bar" role="progressbar" aria-valuemin={0} aria-valuemax={100}
          aria-valuenow={percent ?? undefined}>
          <span style={{ width: `${percent ?? (state === "running" ? 30 : 0)}%` }} data-indeterminate={percent === null} />
        </div>
      ) : null}
      {job?.error ? <p className="studio-job-card__error">{job.error}</p> : null}
      {error ? <p className="studio-job-card__error">{error}</p> : null}
      {state === "done" && mediaUrl ? (
        <div className="studio-job-card__media">
          {kind === "image" ? <img src={mediaUrl} alt={job?.label ?? toolCall.name} /> : null}
          {kind === "audio" ? <audio controls src={mediaUrl} /> : null}
          {kind === "video" ? <video controls preload="metadata" src={mediaUrl} /> : null}
        </div>
      ) : null}
      {state === "done" && tab ? (
        <button type="button" className="secondary-button studio-job-card__open" onClick={() => {
          // A storyboard job opens its own project on the Storyboard tab.
          const projectId = toolCall.arguments?.project_id;
          if (tab === "storyboard" && typeof projectId === "string") openProjectAt(projectId, "storyboard");
          openAppTab(tab);
        }}>
          {tab === "storyboard"
            ? t("studioChat.job.openProject", { defaultValue: "Open in Project" })
            : t("studioChat.job.openLibrary", { defaultValue: "Open in Library" })}
        </button>
      ) : null}
    </div>
  );
}
