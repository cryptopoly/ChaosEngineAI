import { useTranslation } from "react-i18next";
import { useGenerationProgress } from "../hooks/useGenerationProgress";

const WATCH_INTERVAL_MS = 2000;

/**
 * A run this screen did not start (a Storyboard batch, an agent, another window) still holds the model and the
 * GPU. Say so, with the step it is on, instead of showing an idle studio. Hidden while this screen runs its own job.
 */
export function RunningElsewhere({ kind, ownRunActive, className = "" }: { kind: "image" | "video"; ownRunActive: boolean; className?: string }) {
  const { t } = useTranslation("studio");
  const snapshot = useGenerationProgress(kind, !ownRunActive, WATCH_INTERVAL_MS);
  if (ownRunActive || !snapshot?.active) return null;
  const hasSteps = snapshot.totalSteps > 0 && (snapshot.phase === "diffusing" || snapshot.phase === "compose");
  const what = snapshot.runLabel ?? (kind === "image" ? t("runningElsewhere.image", { defaultValue: "An image" }) : t("runningElsewhere.video", { defaultValue: "A video" }));
  const minutes = Math.floor(snapshot.elapsedSeconds / 60);
  const seconds = Math.floor(snapshot.elapsedSeconds % 60);
  return (
    <div className={`running-elsewhere ${className}`.trim()} role="status" aria-live="polite">
      <strong>{t("runningElsewhere.title", { defaultValue: "Generating now" })}</strong>
      <span>
        {what}
        {" · "}
        {snapshot.message || snapshot.phase}
        {hasSteps ? ` · ${t("runningElsewhere.step", { defaultValue: "step {step} of {total}", step: snapshot.step, total: snapshot.totalSteps })}` : ""}
        {" · "}
        {t("runningElsewhere.elapsed", { defaultValue: "{minutes} min {seconds} s elapsed", minutes, seconds })}
      </span>
      {hasSteps ? <progress max={snapshot.totalSteps} value={snapshot.step} aria-hidden="true" /> : null}
      <span className="running-elsewhere__hint">
        {t("runningElsewhere.hint", { defaultValue: "Started from another screen, such as a Storyboard batch. Wait for it to finish before starting another." })}
      </span>
    </div>
  );
}
