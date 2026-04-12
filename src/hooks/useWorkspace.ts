import { useEffect, useRef, useState } from "react";
import {
  checkBackend,
  getWorkspace,
  getTauriBackendInfo,
} from "../api";
import { checkForUpdates } from "../updater";
import { mockWorkspace } from "../mockData";
import { settingsDraftFromWorkspace } from "../utils";
import type { TauriBackendInfo, WorkspaceData } from "../types";

export function useWorkspace() {
  const [workspace, setWorkspace] = useState<WorkspaceData>(mockWorkspace);
  const [loading, setLoading] = useState(true);
  const [backendOnline, setBackendOnline] = useState(false);
  const [tauriBackend, setTauriBackend] = useState<TauriBackendInfo | null>(null);
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const busy = busyAction !== null;

  const [error, setErrorState] = useState<string | null>(null);
  const errorTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  function setError(msg: string | null) {
    if (errorTimerRef.current) clearTimeout(errorTimerRef.current);
    setErrorState(msg);
    const isSticky = !!msg && msg.includes("update-llama-cpp.sh");
    if (msg && !isSticky) {
      errorTimerRef.current = setTimeout(() => setErrorState(null), 8000);
    }
  }

  const [rebuildingLlama, setRebuildingLlama] = useState(false);
  const [rebuildOutput, setRebuildOutput] = useState<string | null>(null);

  async function handleRebuildLlamaCpp() {
    setRebuildingLlama(true);
    setRebuildOutput(null);
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      const result = await invoke<{ ok: boolean; exitCode: number | null; output: string }>(
        "rebuild_llama_cpp",
      );
      setRebuildOutput(result.output);
      if (result.ok) {
        setError(null);
        try {
          await invoke("restart_backend_sidecar");
        } catch {
          /* best-effort restart */
        }
      } else {
        setError(`Rebuild failed (exit ${result.exitCode ?? "?"}). See output below.`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Rebuild failed to start.");
    } finally {
      setRebuildingLlama(false);
    }
  }

  async function refreshWorkspace(preferredChatId?: string) {
    const [online, payload] = await Promise.all([checkBackend(), getWorkspace()]);
    setBackendOnline(online);
    setWorkspace(payload);
    return { online, payload, preferredChatId };
  }

  // Background update check on startup
  useEffect(() => {
    const timer = setTimeout(() => {
      void checkForUpdates({ silent: true });
    }, 4000);
    return () => clearTimeout(timer);
  }, []);

  // Initial workspace load with retry
  useEffect(() => {
    let cancelled = false;

    async function loadInitial(): Promise<void> {
      const delays = [0, 400, 800, 1500, 2500, 4000, 6000];
      for (const delay of delays) {
        if (cancelled) return;
        if (delay) await new Promise((r) => setTimeout(r, delay));
        try {
          const [online, payload, runtimeInfo] = await Promise.all([
            checkBackend(),
            getWorkspace(),
            getTauriBackendInfo(),
          ]);
          if (cancelled) return;
          setBackendOnline(online);
          setTauriBackend(runtimeInfo);
          setWorkspace(payload);
          setLoading(false);
          return;
        } catch {
          if (!cancelled) setBackendOnline(false);
        }
      }
      if (!cancelled) setLoading(false);
    }

    void loadInitial();
    return () => {
      cancelled = true;
    };
  }, []);

  return {
    workspace,
    setWorkspace,
    loading,
    setLoading,
    backendOnline,
    setBackendOnline,
    tauriBackend,
    setTauriBackend,
    error,
    setError,
    busyAction,
    setBusyAction,
    busy,
    rebuildingLlama,
    rebuildOutput,
    setRebuildOutput,
    handleRebuildLlamaCpp,
    refreshWorkspace,
  };
}
