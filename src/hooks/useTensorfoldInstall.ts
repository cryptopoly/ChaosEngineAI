import { useState, useCallback, useRef, useEffect } from "react";
import {
  getTensorfoldInstallStatus,
  getTensorfoldStatus,
  startTensorfoldInstall,
  type TensorfoldJobState,
  type TensorfoldStatus,
} from "../api";

const POLL_INTERVAL_MS = 1500;

export interface UseTensorfoldInstallReturn {
  tensorfoldJob: TensorfoldJobState | null;
  tensorfoldStatus: TensorfoldStatus | null;
  installingTensorfold: boolean;
  handleInstallTensorfold: () => Promise<void>;
  refreshTensorfoldStatus: () => Promise<void>;
}

/**
 * Install flow for the TensorFold engine: POST starts the background job,
 * the hook polls it, and `onInstalled` runs once it finishes so the caller
 * can re-read the capability flags the launch settings gate on.
 */
export function useTensorfoldInstall(onInstalled?: () => void | Promise<void>): UseTensorfoldInstallReturn {
  const [tensorfoldJob, setTensorfoldJob] = useState<TensorfoldJobState | null>(null);
  const [tensorfoldStatus, setTensorfoldStatus] = useState<TensorfoldStatus | null>(null);
  const [installingTensorfold, setInstallingTensorfold] = useState(false);
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
        const state = await getTensorfoldInstallStatus();
        setTensorfoldJob(state);
        if (state.done) {
          stopPoll();
          setInstallingTensorfold(false);
          // Refresh installed status after the job completes.
          try {
            setTensorfoldStatus(await getTensorfoldStatus());
          } catch {
            // best-effort
          }
          if (state.phase === "done") {
            try {
              await onInstalledRef.current?.();
            } catch {
              // best-effort: the next workspace refresh picks the install up
            }
          }
        }
      } catch {
        stopPoll();
        setInstallingTensorfold(false);
      }
    }, POLL_INTERVAL_MS);
  }, [stopPoll]);

  const handleInstallTensorfold = useCallback(async () => {
    setInstallingTensorfold(true);
    try {
      const initialState = await startTensorfoldInstall();
      setTensorfoldJob(initialState);
      startPoll();
    } catch (err) {
      setInstallingTensorfold(false);
      throw err;
    }
  }, [startPoll]);

  const refreshTensorfoldStatus = useCallback(async () => {
    try {
      setTensorfoldStatus(await getTensorfoldStatus());
    } catch {
      // best-effort
    }
  }, []);

  useEffect(() => {
    return () => {
      stopPoll();
    };
  }, [stopPoll]);

  return {
    tensorfoldJob,
    tensorfoldStatus,
    installingTensorfold,
    handleInstallTensorfold,
    refreshTensorfoldStatus,
  };
}
