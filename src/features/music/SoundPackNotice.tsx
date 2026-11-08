/**
 * The sounds pack install notice under the Stable Audio Open row on Music and Audio Models: shown only while the pack
 * is missing, installing or failed, and gone once it is installed. The sounds run in a pack of their own
 * (about 1 GB of Python packages), separate from the music pack, so a person who only wants songs never
 * installs it. Same terminal log panel as the music pack banner.
 */
import { useTranslation } from "react-i18next";
import type { UseMusicState } from "../../hooks/useMusicState";
import { SOUNDS_PACK_GB } from "./soundsLogic";

export function SoundPackNotice({ music }: { music: Pick<UseMusicState, "runtime" | "soundInstallJob" | "startSoundPackInstall" | "cancelSoundPackInstall"> }) {
  const { t } = useTranslation("studio");
  const { runtime, soundInstallJob: job } = music;
  const installed = Boolean(runtime?.sound?.installed);
  const running = Boolean(job?.running);
  const failed = Boolean(job?.error) && !running;
  if (!runtime?.sound || runtime.sound.unavailableReason || (installed && !running && !failed)) return null;
  const percent = Math.max(0, Math.min(100, Math.round(job?.percent ?? 0)));

  return (
    <div className={`callout music-pack-banner${failed ? " error" : " quiet"}`}>
      <div className="music-pack-banner__head">
        <div>
          <strong>
            {failed
              ? t("musicModels.sounds.pack.failedTitle", { defaultValue: "The sounds pack could not be installed." })
              : t("musicModels.sounds.pack.title", { defaultValue: "Install the sounds pack" })}
          </strong>
          <p className="muted-text">
            {t("musicModels.sounds.pack.body", {
              defaultValue: "Sound effects run in a small pack of their own. About {size} GB, installed once.",
              size: SOUNDS_PACK_GB,
            })}
          </p>
        </div>
        <div className="button-row">
          {running ? (
            <button type="button" className="secondary-button" onClick={() => void music.cancelSoundPackInstall()}>
              {t("musicModels.pack.cancel", { defaultValue: "Cancel" })}
            </button>
          ) : (
            <button type="button" className="primary-button" onClick={() => void music.startSoundPackInstall()}>
              {failed ? t("error.retry", { defaultValue: "Retry" }) : t("musicModels.sounds.pack.install", { defaultValue: "Install the sounds pack" })}
            </button>
          )}
        </div>
      </div>
      {running ? (
        <div className="music-pack-banner__progress">
          <progress className="music-pack-banner__bar" max={100} value={percent} />
          <span className="muted-text">
            {job?.phaseLabel || job?.message || t("musicModels.pack.working", { defaultValue: "Working…" })} · {percent}%
          </span>
        </div>
      ) : null}
      {failed ? <p className="muted-text">{t("musicModels.pack.failedBody", { defaultValue: "The details are below. You can try again." })}</p> : null}
      {job && job.log.length > 0 && (running || failed) ? (
        <details className="install-log-panel debug-details" open={failed}>
          <summary className="install-log-summary">
            {failed ? t("error.technicalDetails", { defaultValue: "Technical details" }) : t("musicModels.pack.log", { defaultValue: "Install log" })}
          </summary>
          <pre className="install-log-terminal">{[job.error, ...job.log].filter(Boolean).join("\n")}</pre>
        </details>
      ) : null}
    </div>
  );
}
