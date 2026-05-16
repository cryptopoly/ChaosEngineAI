import { useState, useCallback, useRef, useEffect } from "react";
import {
  getMtplxInstallStatus,
  getMtplxStatus,
  startMtplxInstall,
  type MtplxJobState,
  type MtplxStatus,
} from "../api";

const POLL_INTERVAL_MS = 1500;

export interface UseMtplxInstallReturn {
  mtplxJob: MtplxJobState | null;
  mtplxStatus: MtplxStatus | null;
  installingMtplx: boolean;
  handleInstallMtplx: () => Promise<void>;
  refreshMtplxStatus: () => Promise<void>;
}

export function useMtplxInstall(): UseMtplxInstallReturn {
  const [mtplxJob, setMtplxJob] = useState<MtplxJobState | null>(null);
  const [mtplxStatus, setMtplxStatus] = useState<MtplxStatus | null>(null);
  const [installingMtplx, setInstallingMtplx] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

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
        const state = await getMtplxInstallStatus();
        setMtplxJob(state);
        if (state.done) {
          stopPoll();
          setInstallingMtplx(false);
          // Refresh installed status after job completes.
          try {
            const status = await getMtplxStatus();
            setMtplxStatus(status);
          } catch {
            // best-effort
          }
        }
      } catch {
        stopPoll();
        setInstallingMtplx(false);
      }
    }, POLL_INTERVAL_MS);
  }, [stopPoll]);

  const handleInstallMtplx = useCallback(async () => {
    setInstallingMtplx(true);
    try {
      const initialState = await startMtplxInstall();
      setMtplxJob(initialState);
      startPoll();
    } catch (err) {
      setInstallingMtplx(false);
      throw err;
    }
  }, [startPoll]);

  const refreshMtplxStatus = useCallback(async () => {
    try {
      const status = await getMtplxStatus();
      setMtplxStatus(status);
    } catch {
      // best-effort
    }
  }, []);

  useEffect(() => {
    return () => {
      stopPoll();
    };
  }, [stopPoll]);

  return { mtplxJob, mtplxStatus, installingMtplx, handleInstallMtplx, refreshMtplxStatus };
}
