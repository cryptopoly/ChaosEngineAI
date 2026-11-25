/**
 * The Upscale tab's tier choice (D0145): Fast (the default, faithful) or Quality (FlashVSR+ Turbo, NVIDIA only).
 * Shown only where the quality tier can run. Choosing Quality shows what it does to a picture (it invents texture,
 * so faces and changing text such as a timecode can come out wrong), the measured speed and memory, and the install
 * card until its add-on is in. The backend checks the card and the size again at start.
 */
import { useTranslation } from "react-i18next";
import type { VideoUpscaleQualityModel, VideoUpscaleQualityStatus } from "../../api/videoUpscale";
import { qualityInstallPercent, showQualityInstall, type UpscaleTier } from "./upscaleLogic";

export interface UpscaleQualityTierProps {
  row: VideoUpscaleQualityModel;
  status: VideoUpscaleQualityStatus;
  tier: UpscaleTier;
  onTier: (tier: UpscaleTier) => void;
  /** The install request itself failed (no job started). */
  requestError: string | null;
  onInstall: () => void;
  onCancelInstall: () => void;
}

export function UpscaleQualityTier({ row, status, tier, onTier, requestError, onInstall, onCancelInstall }: UpscaleQualityTierProps) {
  const { t } = useTranslation("studio");
  const fit = row.qualityFit;
  const job = status.job;
  const running = job.running;
  const installError = job.phase === "error"
    ? job.error || t("videoUpscale.install.unknownError", { defaultValue: "No reason was given." })
    : requestError;

  const notOffered = !fit.offered
    ? fit.code === "smallCard"
      ? t("videoUpscale.qualityTier.smallCard", {
          defaultValue: "Not offered on this computer: the quality upscaler needs a graphics card with at least {need} GB, and this one has {card} GB.",
          need: row.minCardGb, card: Math.round(fit.cardGb ?? 0),
        })
      : t("videoUpscale.qualityTier.notCuda", {
          defaultValue: "Not offered on this computer: the quality upscaler needs an NVIDIA graphics card with its own memory.",
        })
    : null;

  const phaseText = (() => {
    switch (job.phase) {
      case "venv":
        return t("videoUpscale.qualityTier.phaseVenv", { defaultValue: "Making its own Python environment…" });
      case "packages":
        return t("videoUpscale.qualityTier.phasePackages", { defaultValue: "Downloading PyTorch and the other packages (about 3.4 GB)…" });
      case "code":
        return t("videoUpscale.qualityTier.phaseCode", { defaultValue: "Downloading the FlashVSR+ code…" });
      case "weights":
        return t("videoUpscale.qualityTier.phaseWeights", { defaultValue: "Downloading and checking the model (7 GB)…" });
      case "check":
        return t("videoUpscale.qualityTier.phaseCheck", { defaultValue: "Checking that it runs on the graphics card…" });
      default:
        return t("videoUpscale.qualityTier.phasePreflight", { defaultValue: "Checking free space…" });
    }
  })();

  return (
    <div className="control-stack video-upscale-tier">
      <span className="eyebrow">{t("videoUpscale.qualityTier.label", { defaultValue: "Upscaler" })}</span>
      <div className="image-pill-row" role="radiogroup" aria-label={t("videoUpscale.qualityTier.label", { defaultValue: "Upscaler" })}>
        <button
          type="button"
          role="radio"
          aria-checked={tier === "fast"}
          className={`pill-button${tier === "fast" ? " active" : ""}`}
          onClick={() => onTier("fast")}
        >
          <strong>{t("videoUpscale.qualityTier.fast", { defaultValue: "Fast (default)" })}</strong>
          <span>{t("videoUpscale.qualityTier.fastHint", { defaultValue: "Faithful to the original. Many times faster." })}</span>
        </button>
        <button
          type="button"
          role="radio"
          aria-checked={tier === "quality"}
          className={`pill-button${tier === "quality" ? " active" : ""}`}
          disabled={!fit.offered}
          title={notOffered ?? undefined}
          onClick={() => onTier("quality")}
        >
          <strong>{t("videoUpscale.qualityTier.quality", { defaultValue: "Quality" })}</strong>
          <span>{row.name}</span>
          <span>{t("videoUpscale.qualityTier.qualityHint", { defaultValue: "Sharper, with new detail. Slow; NVIDIA only." })}</span>
        </button>
      </div>
      {notOffered ? <p className="muted-text">{notOffered}</p> : null}

      {tier === "quality" && fit.offered ? (
        <>
          <div className="callout video-upscale-quality-warning" role="note">
            <strong>{t("videoUpscale.qualityTier.warningTitle", { defaultValue: "This upscaler invents detail" })}</strong>
            <p>
              {t("videoUpscale.qualityTier.warningBody", {
                defaultValue:
                  "It adds texture the original never had: hair, skin, fabric, water. That suits AI footage. It can change faces and gets changing text wrong, such as a timecode or a caption. For family tapes, the fast upscaler is the faithful choice.",
              })}
            </p>
          </div>
          <p className="muted-text">
            {t("videoUpscale.qualityTier.speed", {
              defaultValue:
                "Measured on an RTX 4090: about 5 frames a second to 1280 × 1024, 3 to 1080p and 1.4 to 1440p; the fast upscaler does 11 to 27. An hour of tape takes about 5 hours.",
            })}
          </p>
          {fit.maxOutput ? (
            <p className="muted-text">
              {t("videoUpscale.qualityTier.maxOutput", {
                defaultValue: "On this {card} GB card it makes up to {width} × {height}. 4K is not offered.",
                card: Math.round(fit.cardGb ?? 0), width: fit.maxOutput[0], height: fit.maxOutput[1],
              })}
            </p>
          ) : null}

          {showQualityInstall(status) ? (
            <div className={`callout video-upscale-install${installError && !running ? " error" : ""}`} role="region"
              aria-label={t("videoUpscale.qualityTier.installTitle", { defaultValue: "Install the quality upscaler" })}>
              <strong>{t("videoUpscale.qualityTier.installTitle", { defaultValue: "Install the quality upscaler" })}</strong>
              <p>
                {t("videoUpscale.qualityTier.installBody", {
                  defaultValue:
                    "This adds FlashVSR+ Turbo (by OpenImagingLab) with its own copy of Python and PyTorch, in its own folder. About {size} GB to download and {free} GB of free disk space. Nothing else on your computer changes.",
                  size: Math.round(status.downloadGb), free: Math.round(status.minFreeGb),
                })}
              </p>
              <p className="muted-text" title={row.licenceNote}>
                {t("videoUpscale.install.licence", { defaultValue: "Model licence: {licence}.", licence: row.licence.name })}
              </p>
              {running ? (
                <div className="video-upscale-install__progress">
                  <progress max={100} value={qualityInstallPercent(job)} aria-label={phaseText} />
                  <span className="muted-text">
                    {t("videoUpscale.qualityTier.step", {
                      defaultValue: "Step {step} of {total}: {phase}", step: Math.max(1, job.phaseIndex), total: job.phaseTotal, phase: phaseText,
                    })}
                  </span>
                  <div className="button-row">
                    <button type="button" className="secondary-button" onClick={onCancelInstall}>
                      {t("videoUpscale.qualityTier.cancelInstall", { defaultValue: "Cancel the install" })}
                    </button>
                  </div>
                </div>
              ) : (
                <div className="button-row">
                  <button type="button" className="primary-button" onClick={onInstall}>
                    {installError || job.phase === "cancelled"
                      ? t("videoUpscale.install.retry", { defaultValue: "Try the install again" })
                      : t("videoUpscale.qualityTier.installButton", {
                          defaultValue: "Install the quality upscaler (about {size} GB)", size: Math.round(status.downloadGb),
                        })}
                  </button>
                </div>
              )}
              {job.phase === "cancelled" && !running ? (
                <p className="muted-text">
                  {t("videoUpscale.qualityTier.cancelled", {
                    defaultValue: "The install was cancelled. Installing again carries on where it stopped.",
                  })}
                </p>
              ) : null}
              {installError && !running ? (
                <p className="error-text">{t("videoUpscale.install.failed", { defaultValue: "The install failed: {error}", error: installError })}</p>
              ) : null}
            </div>
          ) : null}
        </>
      ) : null}
    </div>
  );
}
