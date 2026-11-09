import { useMemo } from "react";
import { useTranslation } from "react-i18next";
import type { JobProgress } from "../types";
import { formatClock, formatDuration, isValidTime, progressFacts, useNow } from "../utils/chatTime";

/**
 * One line for a long job: where it is ("clip 2 of 4"), how long it has run (ticking between updates), and
 * the time left only when the backend had a basis for it. No basis, no estimate: elapsed only.
 */
export function JobProgressLine({
  progress,
  running,
  startedAt,
}: {
  progress: JobProgress | null | undefined;
  running: boolean;
  /** Epoch ms the work began, shown as "started 07:31". */
  startedAt?: number | null;
}) {
  const { t, i18n } = useTranslation("chat");
  const locale = i18n.language;
  // Each update restarts the local clock, so the numbers tick between updates instead of jumping.
  const anchor = useMemo(() => ({ at: Date.now(), progress }), [progress]);
  const now = useNow(running);
  const facts = progressFacts(progress);
  const sinceUpdate = running ? Math.max(0, (now - anchor.at) / 1000) : 0;
  const base = typeof progress?.elapsedSeconds === "number" ? progress.elapsedSeconds : null;
  const elapsed = base === null ? null : base + sinceUpdate;
  const eta = facts.etaSeconds === null ? null : Math.max(0, facts.etaSeconds - sinceUpdate);

  const parts: string[] = [];
  if (isValidTime(startedAt)) parts.push(t("studioChat.job.startedAt", { time: formatClock(startedAt, locale), defaultValue: "started {time}" }));
  if (facts.phase) parts.push(facts.phase);
  // A phase like "clip 2 of 4" already says the step. A film render reports a percent as step of 100: no step text.
  const phaseSaysStep = facts.phase !== null && /\d/.test(facts.phase);
  if (!phaseSaysStep && facts.step !== null && facts.totalSteps !== null && facts.totalSteps > 0 && facts.totalSteps !== 100) {
    parts.push(t("studioChat.job.stepOf", { step: facts.step, total: facts.totalSteps, defaultValue: "step {step} of {total}" }));
  }
  if (elapsed !== null) parts.push(t("studioChat.job.elapsed", { elapsed: formatDuration(elapsed, locale), defaultValue: "{elapsed} elapsed" }));
  if (eta !== null && running) {
    const key = facts.etaBasis === "clips" ? "etaClips" : facts.etaBasis === "catalog" ? "etaCatalog" : "eta";
    const fallback =
      key === "etaClips" ? "about {eta} left (from the clips made so far)"
      : key === "etaCatalog" ? "about {eta} left (estimated for this model)"
      : "about {eta} left";
    parts.push(t(`studioChat.job.${key}`, { eta: formatDuration(eta, locale), defaultValue: fallback }));
  }
  if (parts.length === 0) return null;
  return (
    <div className="job-progress-line" data-testid="job-progress-line">
      {parts.join(" · ")}
    </div>
  );
}
