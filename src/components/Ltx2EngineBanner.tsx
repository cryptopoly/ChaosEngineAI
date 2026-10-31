import { useEffect } from "react";
import { useTranslation } from "react-i18next";
import { useLtx2Install } from "../hooks/useLtx2Install";
import { InstallLogPanel } from "./InstallLogPanel";

interface Ltx2EngineBannerProps {
  /** Only models that run on the LTX-2.3 MLX engine show the banner. */
  modelNeedsLtx2: boolean;
  /** Called once an install finishes, so the caller can refresh its model list. */
  onInstalled?: () => void | Promise<void>;
}

/**
 * Tells the user when the selected video model runs on the LTX-2.3 MLX engine
 * and that engine is not installed yet, with a one-click install into its own
 * environment. Renders nothing when the model does not need it, the engine is
 * already there, or this machine cannot run it.
 */
export function Ltx2EngineBanner({ modelNeedsLtx2, onInstalled }: Ltx2EngineBannerProps) {
  const { t } = useTranslation("studio");
  const { ltx2Job, ltx2Status, installingLtx2, handleInstallLtx2, refreshLtx2Status } = useLtx2Install(onInstalled);

  useEffect(() => {
    if (modelNeedsLtx2) void refreshLtx2Status();
  }, [modelNeedsLtx2, refreshLtx2Status]);

  if (!modelNeedsLtx2 || !ltx2Status || !ltx2Status.supported || ltx2Status.installed) return null;

  const failed = ltx2Job?.phase === "error";
  return (
    <div className="callout image-callout">
      <p>
        {t("video.ltx2.notInstalled", {
          defaultValue:
            "This model runs on the LTX-2.3 MLX engine, which is not installed yet. It installs into its own " +
            "environment (about 1 GB) and does not change the app's packages.",
        })}
      </p>
      {failed && ltx2Job?.error ? <p className="error-text">{ltx2Job.error}</p> : null}
      <div className="button-row">
        <button
          className="secondary-button"
          type="button"
          onClick={() => void handleInstallLtx2().catch(() => undefined)}
          disabled={installingLtx2}
        >
          {installingLtx2
            ? t("video.ltx2.installing", { defaultValue: "Installing…" })
            : failed
              ? t("video.ltx2.retry", { defaultValue: "Retry install" })
              : t("video.ltx2.install", { defaultValue: "Install LTX-2.3 engine" })}
        </button>
      </div>
      {ltx2Job && ltx2Job.phase !== "idle" && ltx2Job.phase !== "done" ? (
        <InstallLogPanel job={ltx2Job} variant="ltx2" />
      ) : null}
    </div>
  );
}
