import { useTranslation } from "react-i18next";
import { GpuBanner } from "./GpuBanner";
import type { useCudaTorchInstall, useGpuStatus, useWorkspace } from "../hooks";

type CudaTorch = ReturnType<typeof useCudaTorchInstall>;
type WorkspaceHook = ReturnType<typeof useWorkspace>;

/** GPU notice, backend-offline notice, error banner (with llama.cpp rebuild action) and rebuild output. */
export function StatusBanners({
  gpuStatus,
  cudaTorchResult,
  installingCudaTorch,
  handleInstallCudaTorch,
  error,
  rebuildingLlama,
  handleRebuildLlamaCpp,
  rebuildOutput,
  setRebuildOutput,
  backendOffline = false,
}: {
  gpuStatus: ReturnType<typeof useGpuStatus>;
  cudaTorchResult: CudaTorch["cudaTorchResult"];
  installingCudaTorch: CudaTorch["installingCudaTorch"];
  handleInstallCudaTorch: CudaTorch["handleInstallCudaTorch"];
  error: WorkspaceHook["error"];
  rebuildingLlama: WorkspaceHook["rebuildingLlama"];
  handleRebuildLlamaCpp: WorkspaceHook["handleRebuildLlamaCpp"];
  rebuildOutput: WorkspaceHook["rebuildOutput"];
  setRebuildOutput: WorkspaceHook["setRebuildOutput"];
  /** The backend stopped answering after the app had loaded. */
  backendOffline?: boolean;
}) {
  const { t: tCommon } = useTranslation("common");
  return (
    <div className="workspace-status-stack">
      {gpuStatus.showBanner && gpuStatus.status ? (
        <GpuBanner
          status={gpuStatus.status}
          onDismiss={gpuStatus.dismiss}
          cudaTorchResult={cudaTorchResult}
          installingCudaTorch={installingCudaTorch}
          handleInstallCudaTorch={handleInstallCudaTorch}
        />
      ) : null}
      {/* Without this the only sign of a stopped backend was the small badge at the foot of the sidebar,
          which sits inside the closed drawer below 1180 px; most pages kept showing their last data. */}
      {backendOffline ? (
        <div className="notice-banner warn-banner backend-offline-banner" role="status">
          <strong>{tCommon("statusBanners.backendOffline.title", { defaultValue: "The backend is not answering." })}</strong>{" "}
          <span>
            {tCommon("statusBanners.backendOffline.detail", {
              defaultValue: "What you see may be out of date and actions will fail until it is back. The app reconnects by itself.",
            })}
          </span>
        </div>
      ) : null}
      {error ? (
        <div className="notice-banner error-banner">
          <span>{error}</span>
          {error.includes("update-llama-cpp.sh") ? (
            <button className="primary-button" type="button" disabled={rebuildingLlama} onClick={() => void handleRebuildLlamaCpp()}>
              {rebuildingLlama
                ? tCommon("llamaRebuild.rebuilding", { defaultValue: "Rebuilding llama.cpp..." })
                : tCommon("llamaRebuild.rebuild", { defaultValue: "Rebuild llama.cpp" })}
            </button>
          ) : null}
        </div>
      ) : null}
      {rebuildOutput ? (
        <div className="notice-banner">
          <pre className="mono-text rebuild-output">{rebuildOutput.slice(-2000)}</pre>
          <button className="secondary-button" type="button" onClick={() => setRebuildOutput(null)}>{tCommon("actions.dismiss", { defaultValue: "Dismiss" })}</button>
        </div>
      ) : null}
    </div>
  );
}
