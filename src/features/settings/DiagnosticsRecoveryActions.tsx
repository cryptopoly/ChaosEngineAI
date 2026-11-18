import { useState } from "react";
import { useTranslation } from "react-i18next";
import {
  installCudaTorch,
  installPipPackage,
  reextractRuntime,
  type CudaTorchInstallResult,
  type InstallResult,
} from "../../api";

// The recovery buttons at the foot of Settings > Diagnostics: reinstall
// mlx-video (Apple Silicon), reinstall CUDA torch (Windows / Linux) and
// re-extract the embedded runtime. Showing both installs to everyone is
// noise; showing neither hides the way out when a prior install broke.

export interface DiagnosticsRecoveryActionsProps {
  /** ``snapshot.os``: decides which install button this host gets. */
  os: Record<string, unknown>;
  backendOnline: boolean;
  onRestartServer: () => void;
  busyAction: string | null;
  /** Re-probe after an install so the snapshot shows the new state. */
  onRefresh: () => Promise<void>;
}

export function DiagnosticsRecoveryActions({
  os,
  backendOnline,
  onRestartServer,
  busyAction,
  onRefresh,
}: DiagnosticsRecoveryActionsProps) {
  const { t } = useTranslation("diagnostics");
  const [reextractBusy, setReextractBusy] = useState(false);
  const [reextractMessage, setReextractMessage] = useState<string | null>(null);
  const [cudaBusy, setCudaBusy] = useState(false);
  const [cudaResult, setCudaResult] = useState<CudaTorchInstallResult | null>(null);
  const [mlxVideoBusy, setMlxVideoBusy] = useState(false);
  const [mlxVideoResult, setMlxVideoResult] = useState<InstallResult | null>(null);

  async function handleInstallCuda() {
    if (cudaBusy) return;
    const confirmed = window.confirm(
      t("panel.cuda.confirm", {
        defaultValue:
          "Reinstall CUDA torch into the extras directory? This downloads ~2 GB, takes 1-5 minutes, and replaces any stale or CPU-only torch currently installed. Use this when the GPU is detected but torch imports stuck as CPU.",
      }),
    );
    if (!confirmed) return;
    setCudaBusy(true);
    setCudaResult(null);
    try {
      const result = await installCudaTorch();
      setCudaResult(result);
      if (result.ok) await onRefresh();
    } catch (err) {
      setCudaResult({
        ok: false,
        output: err instanceof Error ? err.message : String(err),
        indexUrl: null,
        attempts: [],
        requiresRestart: false,
        pythonExecutable: "",
        pythonVersion: null,
        noWheelForPython: false,
        capabilities: {},
      });
    } finally {
      setCudaBusy(false);
    }
  }

  async function handleInstallMlxVideo() {
    if (mlxVideoBusy) return;
    const confirmed = window.confirm(
      t("panel.mlxVideo.confirm", {
        defaultValue:
          "Reinstall mlx-video from the Blaizzy/mlx-video git source? This pulls the package fresh from GitHub (replaces any stale or PyPI-shadow install) and takes 1-2 minutes. Use this on Apple Silicon to enable LTX-2 native generation, or after a failed install left the package partially staged.",
      }),
    );
    if (!confirmed) return;
    setMlxVideoBusy(true);
    setMlxVideoResult(null);
    try {
      const result = await installPipPackage("mlx-video");
      setMlxVideoResult(result);
      if (result.ok) await onRefresh();
    } catch (err) {
      setMlxVideoResult({
        ok: false,
        output: err instanceof Error ? err.message : String(err),
        capabilities: {},
      });
    } finally {
      setMlxVideoBusy(false);
    }
  }

  async function handleReextract() {
    if (reextractBusy) return;
    const confirmed = window.confirm(
      t("panel.reextract.confirm", {
        defaultValue:
          "Delete the cached embedded runtime extraction? The backend will re-extract from scratch on the next restart. Use this when the runtime tree looks corrupted (rare — usually after a crashed install).",
      }),
    );
    if (!confirmed) return;
    setReextractBusy(true);
    setReextractMessage(null);
    const failed = (error: string) => t("panel.reextract.failed", { defaultValue: "Re-extract failed: {error}", error });
    try {
      const result = await reextractRuntime();
      if (result.error) {
        setReextractMessage(failed(result.error));
      } else if (result.deleted) {
        setReextractMessage(
          t("panel.reextract.deleted", {
            defaultValue: "Runtime cache deleted at {path}. Click Restart Backend to re-extract.",
            path: result.path ?? "",
          }),
        );
      } else {
        setReextractMessage(
          t("panel.reextract.nothing", {
            defaultValue: "No cached runtime to delete at {path} — next restart will extract fresh.",
            path: result.path ?? t("panel.reextract.unknownPath", { defaultValue: "(unknown path)" }),
          }),
        );
      }
    } catch (err) {
      setReextractMessage(failed(err instanceof Error ? err.message : String(err)));
    } finally {
      setReextractBusy(false);
    }
  }

  const osSystem = String(os.system ?? "").toLowerCase();
  const osMachine = String(os.machine ?? "").toLowerCase();
  const isAppleSilicon = osSystem === "darwin" && (osMachine === "arm64" || osMachine === "aarch64");
  const isCudaHost = osSystem === "windows" || osSystem === "linux";
  const pipOutput = t("panel.pipOutput", { defaultValue: "pip output" });

  return (
    <>
      {isAppleSilicon ? (
        <div className="diagnostics-actions" style={{ marginTop: 18 }}>
          <p className="muted-text" style={{ margin: "0 0 8px" }}>
            {t("panel.mlxVideo.hint", {
              defaultValue:
                "Reinstall mlx-video from the Blaizzy/mlx-video git source. Use this on Apple Silicon when LTX-2 generation fails with ``ModuleNotFoundError: No module named 'mlx_video.ltx_2'`` (the unrelated PyPI 0.1.0 package shadowed the real one), or when a prior install left the package partially staged.",
            })}
          </p>
          <div className="button-row">
            <button
              className="secondary-button"
              type="button"
              onClick={() => void handleInstallMlxVideo()}
              disabled={!backendOnline || mlxVideoBusy}
            >
              {mlxVideoBusy
                ? t("panel.mlxVideo.installing", { defaultValue: "Installing mlx-video..." })
                : t("panel.mlxVideo.install", { defaultValue: "Install / reinstall mlx-video" })}
            </button>
          </div>
          {mlxVideoResult ? (
            <div style={{ marginTop: 8 }}>
              <p className="muted-text" style={{ margin: 0 }}>
                {mlxVideoResult.ok
                  ? t("panel.mlxVideo.installed", { defaultValue: "mlx-video installed. Restart the backend to activate." })
                  : t("panel.mlxVideo.failed", { defaultValue: "Install failed — see the pip output below." })}
              </p>
              {mlxVideoResult.output ? (
                <details style={{ marginTop: 4 }}>
                  <summary className="muted-text" style={{ cursor: "pointer" }}>{pipOutput}</summary>
                  <pre className="install-log-output" style={{ maxHeight: 240 }}>{mlxVideoResult.output}</pre>
                </details>
              ) : null}
            </div>
          ) : null}
        </div>
      ) : null}

      {isCudaHost ? (
        <div className="diagnostics-actions" style={{ marginTop: 18 }}>
          <p className="muted-text" style={{ margin: "0 0 8px" }}>
            {t("panel.cuda.hint", {
              defaultValue:
                "Reinstall CUDA torch into the extras directory. Use this on Windows/Linux when an NVIDIA GPU is detected but torch imports as CPU-only, or when a prior install left stale / mismatched torch files in extras. The main-page banner offers the same action but can be dismissed — this button is always available.",
            })}
          </p>
          <div className="button-row">
            <button
              className="secondary-button"
              type="button"
              onClick={() => void handleInstallCuda()}
              disabled={!backendOnline || cudaBusy}
            >
              {cudaBusy
                ? t("panel.cuda.installing", { defaultValue: "Installing CUDA torch..." })
                : t("panel.cuda.install", { defaultValue: "Install / reinstall CUDA torch" })}
            </button>
          </div>
          {cudaResult ? (
            <div style={{ marginTop: 8 }}>
              <p className="muted-text" style={{ margin: 0 }}>
                {cudaResult.ok
                  ? t("panel.cuda.installedFrom", {
                    defaultValue: "Installed from {index}. Restart the backend to activate.",
                    index: cudaResult.indexUrl ?? t("panel.cuda.unknownIndex", { defaultValue: "(unknown index)" }),
                  })
                  : cudaResult.noWheelForPython
                    ? t("panel.cuda.noWheel", {
                      defaultValue:
                        "No CUDA torch wheel exists for Python {version}. Ship ChaosEngineAI against a supported Python (3.13 recommended).",
                      version: cudaResult.pythonVersion ?? t("panel.cuda.unknownPython", { defaultValue: "(unknown)" }),
                    })
                    : t("panel.cuda.failed", { defaultValue: "Install failed — see the log tail below for the full pip output." })}
              </p>
              {cudaResult.output ? (
                <details style={{ marginTop: 4 }}>
                  <summary className="muted-text" style={{ cursor: "pointer" }}>{pipOutput}</summary>
                  <pre className="install-log-output" style={{ maxHeight: 240 }}>{cudaResult.output}</pre>
                </details>
              ) : null}
            </div>
          ) : null}
        </div>
      ) : null}

      <div className="diagnostics-actions" style={{ marginTop: 18 }}>
        <p className="muted-text" style={{ margin: "0 0 8px" }}>
          {t("panel.reextract.hint", {
            defaultValue:
              "Advanced: delete the cached embedded runtime and re-extract on next restart. Use this only when the runtime tree looks corrupted (rare — usually after a crashed install).",
          })}
        </p>
        <div className="button-row">
          <button
            className="secondary-button"
            type="button"
            onClick={() => void handleReextract()}
            disabled={!backendOnline || reextractBusy}
          >
            {reextractBusy
              ? t("panel.reextract.deleting", { defaultValue: "Deleting..." })
              : t("panel.reextract.button", { defaultValue: "Re-extract runtime" })}
          </button>
          <button
            className="secondary-button"
            type="button"
            onClick={() => onRestartServer()}
            disabled={Boolean(busyAction)}
          >
            {busyAction === "Restarting server..."
              ? t("panel.restarting", { defaultValue: "Restarting..." })
              : t("panel.restart", { defaultValue: "Restart Backend" })}
          </button>
        </div>
        {reextractMessage ? (
          <p className="muted-text" style={{ marginTop: 8 }}>{reextractMessage}</p>
        ) : null}
      </div>
    </>
  );
}
