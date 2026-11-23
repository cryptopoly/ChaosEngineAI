/**
 * Sync scoring for lip-sync takes (D-132), shown in the Video Studio's lip-sync panel when a run makes more than one
 * take: whether the takes will be scored, and an install button (MediaPipe, OpenCV and the face model, in the
 * add-on's own folder) when they will not. `LipSyncScoreNote` is the view; `LipSyncScoreAddon` fetches and installs.
 */
import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { getLipSyncScoreStatus, startLipSyncScoreInstall, type LipSyncScoreStatus } from "../../api";

export interface LipSyncScoreNoteProps {
  status: LipSyncScoreStatus | null;
  takes: number;
  onInstall: () => void;
  disabled?: boolean;
}

export function LipSyncScoreNote({ status, takes, onInstall, disabled }: LipSyncScoreNoteProps) {
  const { t } = useTranslation("studio");
  if (!status) return null;
  const job = status.job;
  if (status.installed && !job.running) {
    return (
      <p className="muted-text" role="note">
        {t("videoStudio.lipSync.score.installed", {
          defaultValue: "Sync scoring is installed: the best of the {takes} takes is kept.",
          takes,
        })}
      </p>
    );
  }
  const phaseText =
    job.phase === "model"
      ? t("videoStudio.lipSync.score.phaseModel", { defaultValue: "Downloading the face model…" })
      : job.phase === "check"
        ? t("videoStudio.lipSync.score.phaseCheck", { defaultValue: "Checking the install…" })
        : t("videoStudio.lipSync.score.phasePackages", { defaultValue: "Installing MediaPipe and OpenCV…" });
  return (
    <div className="lip-sync-picker__score" role="note">
      <p className="caution-text">
        {t("videoStudio.lipSync.score.missing", {
          defaultValue:
            "Takes are not scored: sync scoring is not installed, so all {takes} takes are kept in the Video Gallery and take 1 is used.",
          takes,
        })}
      </p>
      {job.running ? (
        <p className="muted-text">{phaseText}</p>
      ) : (
        <button type="button" className="secondary-button" disabled={disabled} onClick={onInstall}>
          {t("videoStudio.lipSync.score.install", {
            defaultValue: "Install sync scoring (about {size} MB)",
            size: status.downloadMb,
          })}
        </button>
      )}
      {job.phase === "error" && job.error ? (
        <p className="caution-text" title={job.error}>
          {t("videoStudio.lipSync.score.failed", { defaultValue: "Install failed: {error}", error: job.error })}
        </p>
      ) : null}
    </div>
  );
}

export function LipSyncScoreAddon({ takes, disabled }: { takes: number; disabled?: boolean }) {
  const [status, setStatus] = useState<LipSyncScoreStatus | null>(null);
  const poll = useRef<number | null>(null);

  const stopPolling = () => {
    if (poll.current != null) window.clearInterval(poll.current);
    poll.current = null;
  };

  const follow = () => {
    stopPolling();
    poll.current = window.setInterval(() => {
      getLipSyncScoreStatus()
        .then((next) => {
          setStatus(next);
          if (!next.job.running) stopPolling();
        })
        .catch(stopPolling);
    }, 1500);
  };

  useEffect(() => {
    let cancelled = false;
    getLipSyncScoreStatus()
      .then((next) => {
        if (cancelled) return;
        setStatus(next);
        if (next.job.running) follow();
      })
      .catch(() => {
        /* best-effort: without a status the note stays hidden; the run's note still says the takes were not scored */
      });
    return () => {
      cancelled = true;
      stopPolling();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const install = () => {
    startLipSyncScoreInstall()
      .then((next) => {
        setStatus(next);
        follow();
      })
      .catch(() => undefined);
  };

  return <LipSyncScoreNote status={status} takes={takes} onInstall={install} disabled={disabled} />;
}
