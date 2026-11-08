import { useTranslation } from "react-i18next";
import type { useCudaTorchInstall } from "../hooks";
import type { UseGpuStatus } from "../hooks/useGpuStatus";

type CudaTorch = ReturnType<typeof useCudaTorchInstall>;

/** "Running on CPU" notice with the CUDA torch install button. */
export function GpuBanner({
  status,
  onDismiss,
  cudaTorchResult,
  installingCudaTorch,
  handleInstallCudaTorch,
}: {
  status: NonNullable<UseGpuStatus["status"]>;
  onDismiss: () => void;
  cudaTorchResult: CudaTorch["cudaTorchResult"];
  installingCudaTorch: CudaTorch["installingCudaTorch"];
  handleInstallCudaTorch: CudaTorch["handleInstallCudaTorch"];
}) {
  const { t: tCommon } = useTranslation("common");
  return (
    <div className="notice-banner warn-banner">
      <span>
        <strong>{tCommon("gpuBanner.runningOnCpu", { defaultValue: "Running on CPU." })}</strong>{" "}
        {status.recommendation ??
          tCommon("gpuBanner.recommendationFallback", {
            defaultValue: "An NVIDIA GPU is visible but torch can't reach CUDA — image and video generation will be very slow.",
          })}
        {cudaTorchResult?.ok ? (
          <>
            {" "}
            <strong>
              {tCommon("gpuBanner.cudaInstalled", {
                defaultValue: "CUDA torch installed{source}{python}. Restart the app to use the GPU.",
                source: cudaTorchResult.indexUrl
                  ? tCommon("gpuBanner.cudaInstalledSource", {
                    defaultValue: " from {index}",
                    index: cudaTorchResult.indexUrl.replace("https://download.pytorch.org/whl/", ""),
                  })
                  : "",
                python: cudaTorchResult.pythonVersion
                  ? tCommon("gpuBanner.cudaInstalledPython", {
                    defaultValue: " into bundled Python {version}",
                    version: cudaTorchResult.pythonVersion,
                  })
                  : "",
              })}
            </strong>
          </>
        ) : cudaTorchResult && !cudaTorchResult.ok ? (
          <>
            {" "}
            {cudaTorchResult.noWheelForPython ? (
              <>
                <strong>
                  {tCommon("gpuBanner.noWheel", {
                    defaultValue: "No CUDA torch wheel for Python{version}.",
                    version: cudaTorchResult.pythonVersion
                      ? ` ${cudaTorchResult.pythonVersion}`
                      : tCommon("gpuBanner.noWheelBundledVenv", { defaultValue: " (bundled venv)" }),
                  })}
                </strong>{" "}
                {tCommon("gpuBanner.noWheelDetail", {
                  defaultValue: "PyTorch currently ships CUDA wheels for Python 3.9–3.13. Reinstall the app with a supported Python on PATH (e.g. 3.12 or 3.13) and try again.",
                })}
              </>
            ) : (
              <>
                <strong>{tCommon("gpuBanner.installFailed", { defaultValue: "Install failed:" })}</strong>{" "}
                <span className="mono-text">{cudaTorchResult.message}</span>
              </>
            )}
          </>
        ) : installingCudaTorch ? (
          <>
            {" "}
            <em>{tCommon("gpuBanner.downloading", {
              defaultValue: "Downloading ~2.5 GB CUDA wheel into the app's bundled Python — this can take several minutes.",
            })}</em>
          </>
        ) : null}
      </span>
      <button
        className="primary-button"
        type="button"
        disabled={installingCudaTorch || cudaTorchResult?.ok === true}
        onClick={() => void handleInstallCudaTorch()}
      >
        {installingCudaTorch
          ? tCommon("gpuBanner.installing", { defaultValue: "Installing CUDA torch..." })
          : cudaTorchResult?.ok
            ? tCommon("status.installed", { defaultValue: "Installed" })
            : tCommon("gpuBanner.install", { defaultValue: "Install CUDA torch" })}
      </button>
      <button className="secondary-button" type="button" onClick={onDismiss}>
        {tCommon("actions.dismiss", { defaultValue: "Dismiss" })}
      </button>
    </div>
  );
}
