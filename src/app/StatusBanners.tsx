import { useTranslation } from "react-i18next";
import { GpuBanner } from "./GpuBanner";
import type { useCudaTorchInstall, useGpuStatus, useWorkspace } from "../hooks";

type CudaTorch = ReturnType<typeof useCudaTorchInstall>;
type WorkspaceHook = ReturnType<typeof useWorkspace>;

/** GPU notice, error banner (with llama.cpp rebuild action) and rebuild output. */
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
