/**
 * The batch bar of a paused batch (S-M2, on the S-M1 durable queue): it stopped between two steps, by Pause or
 * because the app closed while it ran, and keeps its remaining steps. It says why it paused, the steps done and
 * left and the time left when there is a basis, with Resume (primary) and Cancel.
 */
import { useTranslation } from "react-i18next";
import { formatDuration } from "../../components/RunEstimateStrip";
import type { JobProgress, StoryboardBatchRecord, StoryboardPausedProgress } from "../../types";
import "./pause.css";

type TFn = (key: string, options: Record<string, unknown>) => string;

/** The bar data of a paused batch, when the progress the backend sent is one. */
export function pausedProgressOf(progress: JobProgress | StoryboardPausedProgress | null | undefined): StoryboardPausedProgress | null {
  return progress && (progress as StoryboardPausedProgress).paused === true ? (progress as StoryboardPausedProgress) : null;
}

/** Steps done and left from the record alone, for the moment before the bar data arrives. */
export function pausedCounts(batch: StoryboardBatchRecord, progress: StoryboardPausedProgress | null): { done: number; left: number; total: number } {
  if (progress) return { done: progress.step, left: progress.remainingSteps, total: progress.totalSteps };
  const counted = batch.results.filter((r) => r.state !== "skipped");
  const left = counted.filter((r) => r.state === "queued" || r.state === "running").length;
  const done = counted.filter((r) => r.state === "done" || r.state === "failed").length;
  return { done, left, total: done + left };
}

/** The reason line: a restart says Resume carries on; a Pause says where it stopped. Pure so the words are tested. */
export function pausedReasonText(t: TFn, reason: StoryboardBatchRecord["pauseReason"]): string {
  return reason === "restart"
    ? t("board.pause.reasonRestart", { defaultValue: "Paused when the app closed; Resume carries on." })
    : t("board.pause.reasonUser", { defaultValue: "Paused after a finished step; Resume carries on with the next one." });
}

export function PausedBatchBar({
  batch,
  progress: raw = null,
  busy = false,
  onResume,
  onCancel,
}: {
  batch: StoryboardBatchRecord;
  progress?: JobProgress | StoryboardPausedProgress | null;
  busy?: boolean;
  onResume: () => void;
  onCancel: () => void;
}) {
  const { t } = useTranslation("studio");
  const tf = t as unknown as TFn;
  const progress = pausedProgressOf(raw);
  const { done, left, total } = pausedCounts(batch, progress);
  const eta = progress?.etaSeconds ?? null;
  const reason = progress?.pauseReason ?? batch.pauseReason ?? null;
  return (
    <div className="batch-bar batch-bar--paused" role="status" aria-live="polite">
      <div className="batch-bar__text">
        <strong>{t("board.pause.title", { defaultValue: "Paused" })}</strong>
        <span>{pausedReasonText(tf, reason)}</span>
        <span className="batch-bar__count">
          {t("board.pause.steps", {
            defaultValue: "{done} of {total} steps done, {left, plural, one {# left} other {# left}}",
            done,
            total,
            left,
          })}
          {eta != null
            ? ` · ${t("board.pause.timeLeft", { defaultValue: "about {time} left", time: formatDuration(tf, eta) })}`
            : ""}
        </span>
      </div>
      <progress className="batch-bar__meter" max={Math.max(1, total)} value={done} aria-hidden="true" />
      <div className="batch-bar__actions">
        <button type="button" className="primary-button" disabled={busy || left === 0} onClick={onResume}>
          {t("board.pause.resume", { defaultValue: "Resume" })}
        </button>
        <button type="button" className="secondary-button" disabled={busy} onClick={onCancel}>
          {t("board.batch.cancel", { defaultValue: "Cancel batch" })}
        </button>
      </div>
    </div>
  );
}
