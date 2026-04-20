import { useCallback, useEffect, useState } from "react";
import { getGpuStatus, type GpuStatus } from "../api";
import { isTransientNetworkError } from "../utils/runtime";

const DISMISSED_KEY = "chaosengine.gpuStatus.dismissed.v1";

// Poll the /api/system/gpu-status endpoint at startup until torch is imported
// (the backend warms torch lazily, so the first few polls may come back with
// torchImported=false even on boxes that have a working CUDA setup). Once
// torch has been imported the answer won't change without a restart, so we
// stop polling to avoid burning cycles.
const POLL_INTERVAL_MS = 10_000;
const MAX_POLLS = 30; // ~5 minutes — plenty for torch cold-start on Windows

function readDismissed(): boolean {
  if (typeof window === "undefined") return false;
  try {
    return window.localStorage.getItem(DISMISSED_KEY) === "1";
  } catch {
    return false;
  }
}

function writeDismissed(value: boolean) {
  if (typeof window === "undefined") return;
  try {
    if (value) {
      window.localStorage.setItem(DISMISSED_KEY, "1");
    } else {
      window.localStorage.removeItem(DISMISSED_KEY);
    }
  } catch {
    /* ignore */
  }
}

export interface UseGpuStatus {
  status: GpuStatus | null;
  dismissed: boolean;
  dismiss: () => void;
  showBanner: boolean;
}

export function useGpuStatus(backendOnline: boolean): UseGpuStatus {
  const [status, setStatus] = useState<GpuStatus | null>(null);
  const [dismissed, setDismissedState] = useState<boolean>(() => readDismissed());

  useEffect(() => {
    if (!backendOnline) return;

    let cancelled = false;
    let polls = 0;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const tick = async () => {
      if (cancelled) return;
      polls += 1;
      try {
        const next = await getGpuStatus();
        if (cancelled) return;
        setStatus(next);
        // Once torch has been imported, further polls won't change the
        // answer — stop polling.
        if (next.torchImported) return;
      } catch (err) {
        if (!isTransientNetworkError(err)) {
          // Give up on non-transient errors (e.g. 401 if auth wiring drifts)
          // rather than hammering a known-broken endpoint.
          return;
        }
      }
      if (polls >= MAX_POLLS) return;
      timer = setTimeout(tick, POLL_INTERVAL_MS);
    };

    void tick();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [backendOnline]);

  const dismiss = useCallback(() => {
    setDismissedState(true);
    writeDismissed(true);
  }, []);

  const showBanner = Boolean(status?.cpuFallbackWarning) && !dismissed;

  return { status, dismissed, dismiss, showBanner };
}
