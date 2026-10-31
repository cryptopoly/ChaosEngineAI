import { useState, useCallback, useRef, useEffect } from "react";
import {
  getMfluxInstallStatus,
  getMfluxStatus,
  startMfluxInstall,
  type MfluxJobState,
  type MfluxStatus,
} from "../api";

const POLL_INTERVAL_MS = 1500;

export interface UseMfluxInstallReturn {
  mfluxJob: MfluxJobState | null;
  mfluxStatus: MfluxStatus | null;
  installingMflux: boolean;
  handleInstallMflux: () => Promise<void>;
  refreshMfluxStatus: () => Promise<void>;
}

/**
 * Install flow for the mflux MLX image engine: POST starts the background
 * job, the hook polls it, and `onInstalled` runs once it finishes.
 */
export function useMfluxInstall(onInstalled?: () => void | Promise<void>): UseMfluxInstallReturn {
  const [mfluxJob, setMfluxJob] = useState<MfluxJobState | null>(null);
  const [mfluxStatus, setMfluxStatus] = useState<MfluxStatus | null>(null);
  const [installingMflux, setInstallingMflux] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const onInstalledRef = useRef(onInstalled);
  onInstalledRef.current = onInstalled;

  const stopPoll = useCallback(() => {
    if (pollRef.current !== null) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const startPoll = useCallback(() => {
    stopPoll();
    pollRef.current = setInterval(async () => {
      try {
        const state = await getMfluxInstallStatus();
        setMfluxJob(state);
        if (state.done) {
          stopPoll();
          setInstallingMflux(false);
          try {
            setMfluxStatus(await getMfluxStatus());
          } catch {
            // best-effort
          }
          if (state.phase === "done") {
            try {
              await onInstalledRef.current?.();
            } catch {
              // best-effort: the next refresh picks the install up
            }
          }
        }
      } catch {
        stopPoll();
        setInstallingMflux(false);
      }
    }, POLL_INTERVAL_MS);
  }, [stopPoll]);

  const handleInstallMflux = useCallback(async () => {
    setInstallingMflux(true);
    try {
      setMfluxJob(await startMfluxInstall());
      startPoll();
    } catch (err) {
      setInstallingMflux(false);
      throw err;
    }
  }, [startPoll]);

  const refreshMfluxStatus = useCallback(async () => {
    try {
      setMfluxStatus(await getMfluxStatus());
    } catch {
      // best-effort
    }
  }, []);

  useEffect(() => {
    return () => {
      stopPoll();
    };
  }, [stopPoll]);

  return { mfluxJob, mfluxStatus, installingMflux, handleInstallMflux, refreshMfluxStatus };
}
