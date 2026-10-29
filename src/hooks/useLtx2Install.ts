import { useState, useCallback, useRef, useEffect } from "react";
import {
  getLtx2InstallStatus,
  getLtx2Status,
  startLtx2Install,
  type Ltx2JobState,
  type Ltx2Status,
} from "../api";

const POLL_INTERVAL_MS = 1500;

export interface UseLtx2InstallReturn {
  ltx2Job: Ltx2JobState | null;
  ltx2Status: Ltx2Status | null;
  installingLtx2: boolean;
  handleInstallLtx2: () => Promise<void>;
  refreshLtx2Status: () => Promise<void>;
}

/**
 * Install flow for the LTX-2.3 MLX video engine (ltx-2-mlx): POST starts the background
 * job, the hook polls it, and `onInstalled` runs once it finishes.
 */
export function useLtx2Install(onInstalled?: () => void | Promise<void>): UseLtx2InstallReturn {
  const [ltx2Job, setLtx2Job] = useState<Ltx2JobState | null>(null);
  const [ltx2Status, setLtx2Status] = useState<Ltx2Status | null>(null);
  const [installingLtx2, setInstallingLtx2] = useState(false);
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
        const state = await getLtx2InstallStatus();
        setLtx2Job(state);
        if (state.done) {
          stopPoll();
          setInstallingLtx2(false);
          try {
            setLtx2Status(await getLtx2Status());
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
        setInstallingLtx2(false);
      }
    }, POLL_INTERVAL_MS);
  }, [stopPoll]);

  const handleInstallLtx2 = useCallback(async () => {
    setInstallingLtx2(true);
    try {
      setLtx2Job(await startLtx2Install());
      startPoll();
    } catch (err) {
      setInstallingLtx2(false);
      throw err;
    }
  }, [startPoll]);

  const refreshLtx2Status = useCallback(async () => {
    try {
      setLtx2Status(await getLtx2Status());
    } catch {
      // best-effort
    }
  }, []);

  useEffect(() => {
    return () => {
      stopPoll();
    };
  }, [stopPoll]);

  return { ltx2Job, ltx2Status, installingLtx2, handleInstallLtx2, refreshLtx2Status };
}
