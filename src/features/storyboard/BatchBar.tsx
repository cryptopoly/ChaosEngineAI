/**
 * The sticky bar while a batch runs: where it is, Pause (stops after the step in progress and keeps the rest, S-M2)
 * and one Cancel for the whole batch.
 */
import { useTranslation } from "react-i18next";
import { JobProgressLine } from "../../components/JobProgressLine";
import type { JobProgress, StoryboardBatchRecord } from "../../types";
import { batchProgress, type BatchProgress } from "./storyboardLogic";
import "./pause.css";
import "./quality.css";

type TFn = (key: string, options: Record<string, unknown>) => string;

/** "Shot 3 of 8 · stills, then clips". Pure so the wording is tested. */
export function batchProgressText(t: TFn, progress: BatchProgress): string {
  const kind =
    progress.kind === "stills"
      ? t("board.batch.kind.stills", { defaultValue: "stills" })
      : t("board.batch.kind.clips", { defaultValue: "clips" });
  const head = t("board.batch.shotOf", {
    defaultValue: "Shot {n} of {total}",
    n: progress.shotNumber,
    total: progress.shotCount,
  });
  const phase =
    progress.kind === "stills" && progress.thenClips
      ? t("board.batch.stillsThenClips", { defaultValue: "{kind}, then clips", kind })
      : kind;
  return `${head} · ${phase}`;
}

export function BatchBar({
  batch,
  progress: live = null,
  onCancel,
  onPause,
  pausing = false,
}: {
  batch: StoryboardBatchRecord;
  /** Live progress from the backend: "clip 2 of 4", elapsed, time left timed on this machine. */
  progress?: JobProgress | null;
  onCancel: () => void;
  /** Left out, there is no Pause button. */
  onPause?: () => void;
  /** Pause was pressed: the step in progress finishes first. */
  pausing?: boolean;
}) {
  const { t } = useTranslation("studio");
  const progress = batchProgress(batch);
  const total = progress?.total ?? 0;
  const done = progress?.done ?? 0;
  // A shot that failed while the run goes on: say so here, the card has the reason and the one button that helps.
  const failedShots = new Set(batch.results.filter((r) => r.state === "failed").map((r) => r.shotId)).size;
  return (
    <div className="batch-bar" role="status" aria-live="polite">
      <div className="batch-bar__text">
        <strong>
          {batch.draft
            ? t("board.batch.titleDraft", { defaultValue: "Making drafts" })
            : batch.what === "final"
              ? t("board.batch.titleFinal", { defaultValue: "Final pass" })
              : t("board.batch.title", { defaultValue: "Making shots" })}
        </strong>
        <span>{progress ? batchProgressText(t as unknown as TFn, progress) : t("board.batch.starting", { defaultValue: "Starting…" })}</span>
        {progress ? (
          <span className="batch-bar__count">
            {t("board.batch.count", { defaultValue: "{done} of {total} done", done, total })}
          </span>
        ) : null}
      </div>
      <JobProgressLine progress={live} running startedAt={batch.createdAt ? batch.createdAt * 1000 : null} />
      <progress className="batch-bar__meter" max={Math.max(1, total)} value={done} aria-hidden="true" />
      {failedShots > 0 ? (
        <p className="batch-bar__failed">
          {t("board.batch.failedSoFar", {
            defaultValue: "{n, plural, one {# shot did not render. Its card says why and what to do.} other {# shots did not render. Their cards say why and what to do.}}",
            n: failedShots,
          })}
        </p>
      ) : null}
      <div className="batch-bar__actions">
        {onPause ? (
          <button
            type="button"
            className="secondary-button"
            disabled={pausing}
            title={t("board.pause.pauseTitle", { defaultValue: "Stops after the step in progress. The rest waits, also across a restart, until you resume." })}
            onClick={onPause}
          >
            {pausing ? t("board.pause.pausing", { defaultValue: "Pausing after this step…" }) : t("board.pause.pause", { defaultValue: "Pause" })}
          </button>
        ) : null}
        <button type="button" className="secondary-button" onClick={onCancel}>
          {t("board.batch.cancel", { defaultValue: "Cancel batch" })}
        </button>
      </div>
    </div>
  );
}
