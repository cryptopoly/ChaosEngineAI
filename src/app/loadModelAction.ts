import { getWorkspace, loadModel } from "../api";
import type { LaunchPreferences, LoadModelActionResult, WorkspaceData } from "../types";
import type { TabRequest } from "../constants";
import { syncRuntime } from "../utils";
import type { LoadModelPayload } from "./context";

type Dispatch<T> = (value: T | ((current: T) => T)) => void;

export interface LoadModelDeps {
  setError: (msg: string | null) => void;
  setBusyAction: (label: string | null) => void;
  setWorkspace: Dispatch<WorkspaceData>;
  refreshWorkspace: (preferredChatId?: string) => Promise<unknown>;
  setActiveTab: (request: TabRequest) => void;
  launchSettings: LaunchPreferences;
  sanitizeSpeculativeForModel: (params: {
    backend: string;
    modelRef: string;
    canonicalRepo?: string | null;
    modelName: string;
    speculativeDecoding: boolean;
    treeBudget: number;
  }) => { speculativeDecoding: boolean; treeBudget: number };
  activeChatId: string;
}

/** Load a model into the runtime; polls the workspace when the request times out. */
export async function performLoadModel(
  payload: LoadModelPayload,
  deps: LoadModelDeps,
): Promise<LoadModelActionResult> {
  const {
    setError, setBusyAction, setWorkspace, refreshWorkspace, setActiveTab,
    launchSettings, sanitizeSpeculativeForModel, activeChatId,
  } = deps;
  setError(null);
  setBusyAction(payload.busyLabel ?? "Loading model...");
  try {
    const sanitizedSpeculative = sanitizeSpeculativeForModel({
      backend: payload.backend ?? "auto",
      modelRef: payload.modelRef,
      canonicalRepo: payload.canonicalRepo ?? undefined,
      modelName: payload.modelName ?? payload.modelRef,
      speculativeDecoding: payload.speculativeDecoding ?? launchSettings.speculativeDecoding,
      treeBudget: payload.treeBudget ?? launchSettings.treeBudget,
    });
    const loadPayload = {
      modelRef: payload.modelRef,
      modelName: payload.modelName,
      canonicalRepo: payload.canonicalRepo ?? undefined,
      source: payload.source ?? "catalog",
      backend: payload.backend ?? "auto",
      path: payload.path,
      cacheBits: payload.cacheBits ?? launchSettings.cacheBits,
      fp16Layers: payload.fp16Layers ?? launchSettings.fp16Layers,
      fusedAttention: payload.fusedAttention ?? launchSettings.fusedAttention,
      cacheStrategy: payload.cacheStrategy ?? launchSettings.cacheStrategy,
      fitModelInMemory: payload.fitModelInMemory ?? launchSettings.fitModelInMemory,
      contextTokens: payload.contextTokens ?? launchSettings.contextTokens,
      speculativeDecoding: sanitizedSpeculative.speculativeDecoding,
      treeBudget: sanitizedSpeculative.treeBudget,
      kvBudget: payload.kvBudget ?? launchSettings.kvBudget,
    };

    let loadSucceeded = false;
    let loadErrorMessage: string | null = null;
    let loadedRuntime: LoadModelActionResult["runtime"];
    try {
      const runtime = await loadModel(loadPayload);
      setWorkspace((current) => syncRuntime(current, runtime));
      loadSucceeded = true;
      loadedRuntime = runtime;
    } catch (loadErr) {
      loadErrorMessage = loadErr instanceof Error ? loadErr.message : String(loadErr);
      // If the error is a definitive failure (server returned 500, backend
      // unavailable, etc.), show it immediately — don't poll for 15 minutes.
      // Only poll when the error looks like a timeout (the load may still be
      // progressing in the background).
      const isTimeout = /timed?\s*out|timeout|abort/i.test(loadErrorMessage);
      if (isTimeout) {
        for (let attempt = 0; attempt < 450; attempt++) {
          await new Promise((r) => setTimeout(r, 2000));
          try {
            const ws = await getWorkspace();
            if (ws.runtime.loadedModel?.ref === payload.modelRef || ws.runtime.loadedModel?.runtimeTarget === payload.path) {
              setWorkspace(ws);
              loadSucceeded = true;
              loadErrorMessage = null;
              loadedRuntime = ws.runtime;
              break;
            }
            if (ws.server.loading) { setWorkspace(ws); continue; }
            break;
          } catch { /* backend unreachable */ }
        }
        if (loadSucceeded) {
          // Let the shared success path below refresh the workspace and tab state.
        }
      }
    }

    if (loadSucceeded) {
      await refreshWorkspace(activeChatId || undefined);
      if (payload.nextTab) setActiveTab(payload.nextTab);
      return { ok: true, runtime: loadedRuntime };
    } else {
      const detail = loadErrorMessage || "The model could not be loaded. Check the server logs for details.";
      setError(`Failed to load ${payload.modelName ?? payload.modelRef}: ${detail}`);
      return { ok: false, error: detail };
    }
  } catch (actionError) {
    const detail = actionError instanceof Error ? actionError.message : "Unknown error";
    console.error("[handleLoadModel] Load failed for", payload.modelRef, "—", detail, actionError);
    setError(`Failed to load ${payload.modelName ?? payload.modelRef}: ${detail}`);
    return { ok: false, error: detail };
  } finally {
    setBusyAction(null);
  }
}
