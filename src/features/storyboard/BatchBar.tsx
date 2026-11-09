/** The sticky bar while a batch runs: where it is, and one Cancel for the whole batch. */
import { useTranslation } from "react-i18next";
import { JobProgressLine } from "../../components/JobProgressLine";
import type { JobProgress, StoryboardBatchRecord } from "../../types";
import { batchProgress, type BatchProgress } from "./storyboardLogic";

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
}: {
  batch: StoryboardBatchRecord;
  /** Live progress from the backend: "clip 2 of 4", elapsed, time left from the clips made so far. */
  progress?: JobProgress | null;
  onCancel: () => void;
}) {
  const { t } = useTranslation("studio");
  const progress = batchProgress(batch);
  const total = progress?.total ?? 0;
  const done = progress?.done ?? 0;
  return (
    <div className="batch-bar" role="status" aria-live="polite">
      <div className="batch-bar__text">
        <strong>{t("board.batch.title", { defaultValue: "Making shots" })}</strong>
        <span>{progress ? batchProgressText(t as unknown as TFn, progress) : t("board.batch.starting", { defaultValue: "Starting…" })}</span>
        {progress ? (
          <span className="batch-bar__count">
            {t("board.batch.count", { defaultValue: "{done} of {total} done", done, total })}
          </span>
        ) : null}
      </div>
      <JobProgressLine progress={live} running startedAt={batch.createdAt ? batch.createdAt * 1000 : null} />
      <progress className="batch-bar__meter" max={Math.max(1, total)} value={done} aria-hidden="true" />
      <button type="button" className="secondary-button" onClick={onCancel}>
        {t("board.batch.cancel", { defaultValue: "Cancel batch" })}
      </button>
    </div>
  );
}
