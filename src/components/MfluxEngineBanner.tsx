import { useEffect } from "react";
import { useTranslation } from "react-i18next";
import { useMfluxInstall } from "../hooks/useMfluxInstall";
import { InstallLogPanel } from "./InstallLogPanel";

interface MfluxEngineBannerProps {
  /** Only models that run on the MLX engine show the banner. */
  modelNeedsMflux: boolean;
  /** Called once an install finishes, so the caller can refresh its model list. */
  onInstalled?: () => void | Promise<void>;
}

/**
 * Tells the user when the selected image model runs on the mflux MLX engine
 * and that engine is not installed yet, with a one-click install into its own
 * environment. Renders nothing when the model does not need it, the engine is
 * already there, or this machine cannot run it.
 */
export function MfluxEngineBanner({ modelNeedsMflux, onInstalled }: MfluxEngineBannerProps) {
  const { t } = useTranslation("studio");
  const { mfluxJob, mfluxStatus, installingMflux, handleInstallMflux, refreshMfluxStatus } =
    useMfluxInstall(onInstalled);

  useEffect(() => {
    if (modelNeedsMflux) void refreshMfluxStatus();
  }, [modelNeedsMflux, refreshMfluxStatus]);

  if (!modelNeedsMflux || !mfluxStatus || !mfluxStatus.supported || mfluxStatus.installed) return null;

  const failed = mfluxJob?.phase === "error";
  return (
    <div className="callout image-callout">
      <p>
        {t("image.mflux.notInstalled", {
          defaultValue:
            "This model runs on the mflux MLX engine, which is not installed yet. It installs into its own " +
            "environment (about 1.3 GB) and does not change the app's packages.",
        })}
      </p>
      {failed && mfluxJob?.error ? <p className="error-text">{mfluxJob.error}</p> : null}
      <div className="button-row">
        <button
          className="secondary-button"
          type="button"
          onClick={() => void handleInstallMflux().catch(() => undefined)}
          disabled={installingMflux}
        >
          {installingMflux
            ? t("image.mflux.installing", { defaultValue: "Installing…" })
            : failed
              ? t("image.mflux.retry", { defaultValue: "Retry install" })
              : t("image.mflux.install", { defaultValue: "Install mflux engine" })}
        </button>
      </div>
      {mfluxJob && mfluxJob.phase !== "idle" && mfluxJob.phase !== "done" ? (
        <InstallLogPanel job={mfluxJob} variant="mflux" />
      ) : null}
    </div>
  );
}
