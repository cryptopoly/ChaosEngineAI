/**
 * The Upscale tab's first-visit card: what the video upscaler add-on installs, its size and the models' licence,
 * an Install button, the install's step while it runs, and the reason when it fails.
 */
import { useTranslation } from "react-i18next";
import type { VideoUpscaleAddonStatus } from "../../api/videoUpscale";
import { installSizeMb, installStep } from "./upscaleLogic";

export interface UpscaleInstallCardProps {
  status: VideoUpscaleAddonStatus;
  /** The catalogue's licence line ("BSD-3-Clause") and its note, when the models route answered. */
  licence: string | null;
  licenceNote: string | null;
  /** The install request itself failed (no job started). */
  requestError: string | null;
  onInstall: () => void;
}

export function UpscaleInstallCard({ status, licence, licenceNote, requestError, onInstall }: UpscaleInstallCardProps) {
  const { t } = useTranslation("studio");
  const job = status.job;
  const running = job.running;
  const error = job.phase === "error" ? job.error || t("videoUpscale.install.unknownError", { defaultValue: "No reason was given." }) : requestError;
  const step = installStep(job.phase);
  const phaseText =
    job.phase === "weights"
      ? t("videoUpscale.install.phaseWeights", { defaultValue: "Downloading the upscaling models…" })
      : job.phase === "check"
        ? t("videoUpscale.install.phaseCheck", { defaultValue: "Checking the install…" })
        : t("videoUpscale.install.phasePackages", { defaultValue: "Installing the library that runs the models…" });

  return (
    <div className={`callout video-upscale-install${error && !running ? " error" : ""}`} role="region"
      aria-label={t("videoUpscale.install.title", { defaultValue: "Install the video upscaler" })}>
      <strong>{t("videoUpscale.install.title", { defaultValue: "Install the video upscaler" })}</strong>
      <p>
        {t("videoUpscale.install.body", {
          defaultValue:
            "This adds two upscaling models (Real-ESRGAN, by xinntao) and the small library that runs them, in their own folder. About {size} MB, downloaded once. Nothing else on your computer changes.",
          size: installSizeMb(status),
        })}
      </p>
      {licence ? (
        <p className="muted-text" title={licenceNote ?? undefined}>
          {t("videoUpscale.install.licence", { defaultValue: "Model licence: {licence}.", licence })}
        </p>
      ) : null}
      {running ? (
        <div className="video-upscale-install__progress">
          <progress max={3} value={step} aria-label={phaseText} />
          <span className="muted-text">
            {t("videoUpscale.install.step", { defaultValue: "Step {step} of 3: {phase}", step: Math.max(1, step), phase: phaseText })}
          </span>
        </div>
      ) : (
        <div className="button-row">
          <button type="button" className="primary-button" onClick={onInstall}>
            {error
              ? t("videoUpscale.install.retry", { defaultValue: "Try the install again" })
              : t("videoUpscale.install.button", { defaultValue: "Install (about {size} MB)", size: installSizeMb(status) })}
          </button>
        </div>
      )}
      {error && !running ? (
        <p className="error-text">{t("videoUpscale.install.failed", { defaultValue: "The install failed: {error}", error })}</p>
      ) : null}
    </div>
  );
}
