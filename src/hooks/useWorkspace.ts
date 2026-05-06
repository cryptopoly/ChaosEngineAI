import { useEffect, useRef, useState } from "react";
import {
  checkBackend,
  getWorkspace,
  getTauriBackendInfo,
} from "../api";
import { checkForUpdates } from "../updater";
import { emptyWorkspace } from "../defaults";
import { settingsDraftFromWorkspace } from "../utils";
import type { TauriBackendInfo, WorkspaceData } from "../types";

export function useWorkspace() {
  const [workspace, setWorkspace] = useState<WorkspaceData>(emptyWorkspace);
  const [loading, setLoading] = useState(true);
  // Elapsed seconds since the initial load started. Drives a phased
  // startup message in App.tsx so users on first-launch cold starts
  // (extract 280 MB runtime, spawn Python, import FastAPI/MLX) can
  // see *what* is taking the time instead of staring at a static
  // "Loading workspace state..." for 30 s.
  const [loadingElapsedSeconds, setLoadingElapsedSeconds] = useState(0);
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
    const online = await checkBackend();
    setBackendOnline(online);
    if (!online) {
      return { online, payload: null, preferredChatId };
    }
    const payload = await getWorkspace();
    // Merge chat sessions rather than replacing wholesale — this prevents
    // in-flight streaming messages from vanishing when a background poll
    // returns stale session data from the backend.
    setWorkspace((current) => {
      const currentSessionMap = new Map(current.chatSessions.map((s) => [s.id, s]));
      const mergedSessions = payload.chatSessions.map((backendSession) => {
        const local = currentSessionMap.get(backendSession.id);
        // Keep the local version if it has MORE messages (streaming in progress)
        if (local && local.messages.length > backendSession.messages.length) {
          return local;
        }
        return backendSession;
      });
      // Also keep any local-only sessions (created offline, not yet on backend)
      const backendIds = new Set(payload.chatSessions.map((s) => s.id));
      for (const local of current.chatSessions) {
        if (!backendIds.has(local.id)) {
          mergedSessions.push(local);
        }
      }
      return { ...payload, chatSessions: mergedSessions };
    });
    return { online, payload, preferredChatId };
  }

  // Background update check on startup
  useEffect(() => {
    const timer = setTimeout(() => {
      void checkForUpdates({ silent: true });
    }, 4000);
    return () => clearTimeout(timer);
  }, []);

  // Initial workspace load — retry until the backend is ready.
  //
  // The Tauri sidecar now bootstraps on a background thread so the window
  // paints instantly, which means the backend may not be ready for anywhere
  // from ~2s (warm start) to ~30s (cold first launch: extract embedded
  // runtime, start Python, import FastAPI/MLX). A fixed retry budget gave
  // up before cold starts finished and left the workspace stuck empty with
  // nothing to retrigger the load. Instead, back off exponentially (capped
  // at 10s) and keep going until we succeed or the component unmounts.
  useEffect(() => {
    let cancelled = false;
    const startedAt = Date.now();
    // Tick once per second so App.tsx can render a phased startup
    // message ("Extracting runtime", "Starting Python", ...) driven
    // off elapsed wall time rather than a silent spinner.
    const elapsedTimer = setInterval(() => {
      if (!cancelled) {
        setLoadingElapsedSeconds(Math.floor((Date.now() - startedAt) / 1000));
      }
    }, 1000);

    async function loadInitial(): Promise<void> {
      let attempt = 0;
      while (!cancelled) {
        if (attempt > 0) {
          const delay = Math.min(10_000, 400 * Math.pow(1.6, attempt - 1));
          await new Promise((r) => setTimeout(r, delay));
        }
        attempt++;
        if (cancelled) return;
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
          if (!cancelled) {
            setBackendOnline(false);
            // Poll tauri backend info even while the FastAPI sidecar
            // is still booting — it's served by the Rust shell so it
            // comes online ~immediately and exposes startupError when
            // the sidecar fails to spawn at all.
            try {
              const runtimeInfo = await getTauriBackendInfo();
              if (!cancelled) setTauriBackend(runtimeInfo);
            } catch {
              /* tauri command not ready yet */
            }
          }
        }
      }
    }

    void loadInitial();
    return () => {
      cancelled = true;
      clearInterval(elapsedTimer);
    };
  }, []);

  return {
    workspace,
    setWorkspace,
    loading,
    loadingElapsedSeconds,
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
