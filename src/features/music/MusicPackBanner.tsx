/**
 * The music pack install banner: shown only while the pack is missing, installing or failed, and gone once
 * it is installed. The log uses the same terminal panel as the other install jobs.
 */
import { useTranslation } from "react-i18next";
import type { UseMusicState } from "../../hooks/useMusicState";
import { MUSIC_PACK_GB } from "./musicLogic";

export function MusicPackBanner({ music }: { music: Pick<UseMusicState, "runtime" | "installJob" | "startInstall" | "cancelInstall"> }) {
  const { t } = useTranslation("studio");
  const { runtime, installJob } = music;
  const installed = Boolean(runtime?.installed);
  const running = Boolean(installJob?.running);
  const failed = Boolean(installJob?.error) && !running;
  if (!runtime || (installed && !running && !failed)) return null;
  const percent = Math.max(0, Math.min(100, Math.round(installJob?.percent ?? 0)));

  return (
    <div className={`callout music-pack-banner${failed ? " error" : ""}`}>
      <div className="music-pack-banner__head">
        <div>
          <strong>
            {failed
              ? t("musicModels.pack.failedTitle", { defaultValue: "The music pack could not be installed." })
              : t("musicModels.pack.title", { defaultValue: "Install the music pack" })}
          </strong>
          <p className="muted-text">
            {t("musicModels.pack.body", {
              defaultValue: "The pack runs the music models. About {size} GB, installed once.",
              size: MUSIC_PACK_GB,
            })}
          </p>
        </div>
        <div className="button-row">
          {running ? (
            <button type="button" className="secondary-button" onClick={() => void music.cancelInstall()}>
              {t("musicModels.pack.cancel", { defaultValue: "Cancel" })}
            </button>
          ) : (
            <button type="button" className="primary-button" onClick={() => void music.startInstall()}>
              {failed
                ? t("error.retry", { defaultValue: "Retry" })
                : t("musicModels.pack.install", { defaultValue: "Install the music pack" })}
            </button>
          )}
        </div>
      </div>
      {running ? (
        <div className="music-pack-banner__progress">
          <progress className="music-pack-banner__bar" max={100} value={percent} />
          <span className="muted-text">
            {installJob?.phaseLabel || installJob?.message || t("musicModels.pack.working", { defaultValue: "Working…" })} · {percent}%
          </span>
        </div>
      ) : null}
      {failed ? (
        <p className="muted-text">
          {t("musicModels.pack.failedBody", { defaultValue: "The details are below. You can try again." })}
        </p>
      ) : null}
      {installJob && installJob.log.length > 0 && (running || failed) ? (
        <details className="install-log-panel debug-details" open={failed}>
          <summary className="install-log-summary">
            {failed
              ? t("error.technicalDetails", { defaultValue: "Technical details" })
              : t("musicModels.pack.log", { defaultValue: "Install log" })}
          </summary>
          <pre className="install-log-terminal">{[installJob.error, ...installJob.log].filter(Boolean).join("\n")}</pre>
        </details>
      ) : null}
    </div>
  );
}
