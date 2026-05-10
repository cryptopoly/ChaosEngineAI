import { useState } from "react";

import { installCudaTorch, type CudaTorchInstallResult } from "../api";


export type CudaTorchSummary =
  | { ok: true; indexUrl: string | null; pythonVersion: string | null }
  | { ok: false; message: string; pythonVersion: string | null; noWheelForPython: boolean };

interface UseCudaTorchInstallOptions {
  /**
   * Optional refresh callbacks fired after each install attempt (success or
   * failure). The image / video runtime probes cache the
   * ``torchInstallWarning`` field, so without re-probing the warning banner
   * keeps reading the stale pre-install value and the user thinks the
   * button did nothing.
   */
  onAfterInstall?: () => Promise<void> | void;
}

/**
 * Self-contained CUDA torch install flow. Owns the three state slots
 * (busy flag, reduced summary, raw result for the install log panel) and
 * the install handler that walks the Python error shapes into a tidy
 * ``ok | not-ok`` summary suitable for the inline banner.
 *
 * Extracted from ``src/App.tsx`` as part of the v0.8.0 refactor.
 */
export function useCudaTorchInstall(options: UseCudaTorchInstallOptions = {}) {
  const { onAfterInstall } = options;
  const [installingCudaTorch, setInstallingCudaTorch] = useState(false);
  const [cudaTorchResult, setCudaTorchResult] = useState<CudaTorchSummary | null>(null);
  // Raw install result, kept alongside the reduced ``cudaTorchResult``
  // shape above so the Studio's CudaTorchLogPanel can render the full
  // per-attempt pip output (the reduced shape drops ``attempts`` to
  // keep the in-line success/failure summary terse). One more state
  // slot is cheaper than reshaping every existing call site.
  const [cudaTorchRawResult, setCudaTorchRawResult] = useState<CudaTorchInstallResult | null>(null);

  const handleInstallCudaTorch = async () => {
    if (installingCudaTorch) return;
    setInstallingCudaTorch(true);
    setCudaTorchResult(null);
    setCudaTorchRawResult(null);
    try {
      const result = await installCudaTorch();
      setCudaTorchRawResult(result);
      if (result.ok) {
        setCudaTorchResult({
          ok: true,
          indexUrl: result.indexUrl,
          pythonVersion: result.pythonVersion,
        });
      } else {
        const last = result.attempts[result.attempts.length - 1];
        const tail = (last?.output ?? result.output ?? "").split("\n").slice(-3).join("\n");
        setCudaTorchResult({
          ok: false,
          message: tail || "pip install failed — see backend logs for details.",
          pythonVersion: result.pythonVersion,
          noWheelForPython: result.noWheelForPython,
        });
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setCudaTorchResult({
        ok: false,
        message,
        pythonVersion: null,
        noWheelForPython: false,
      });
      // Always synthesize a raw result on exception so the
      // CudaTorchLogPanel renders the failure instead of silently
      // hiding -- previously any network error / 5xx / timeout left
      // the panel showing nothing and the user couldn't tell whether
      // the install was running, finished, or never reached the
      // backend at all. The synthesized "attempt" carries the
      // exception text so the panel surfaces it as a [FAIL] entry.
      setCudaTorchRawResult({
        ok: false,
        output: message,
        indexUrl: null,
        attempts: [
          { indexUrl: "(request never returned)", ok: false, output: message },
        ],
        requiresRestart: false,
        pythonExecutable: "",
        pythonVersion: null,
        noWheelForPython: false,
        capabilities: {},
      });
    } finally {
      setInstallingCudaTorch(false);
    }
    // Refresh runtime status after install completes (success or
    // failure). Without this, the warning banner keeps reading the
    // pre-install torchInstallWarning value and the user thinks the
    // button did nothing -- the cache is bound to whatever the probe
    // last returned. Caller passes refresh hooks via onAfterInstall.
    if (onAfterInstall) {
      try {
        await onAfterInstall();
      } catch {
        /* refresh is best-effort */
      }
    }
  };

  return {
    installingCudaTorch,
    cudaTorchResult,
    cudaTorchRawResult,
    handleInstallCudaTorch,
  };
}
