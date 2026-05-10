/**
 * Model lifecycle actions used by the App composition root.
 *
 * Pulled out of ``App.tsx`` so the root file stays focused on
 * top-level state wiring + render. Each action takes its
 * dependencies as a typed deps object so the App keeps thin
 * wrappers that close over the live setters.
 *
 * Extracted as part of the v0.8.0 Phase 2c-8 refactor.
 */

import { deleteModelPath, unloadModel } from "../../api";
import { syncRuntime } from "../../utils";
import type { LibraryItem, WorkspaceData } from "../../types";


type SetWorkspace = (
  updater: (current: WorkspaceData) => WorkspaceData,
) => void;


export async function performUnloadModel(
  ref: string | undefined,
  deps: {
    setBusyAction: (label: string | null) => void;
    setError: (msg: string | null) => void;
    setWorkspace: SetWorkspace;
    refreshWorkspace: (preferredChatId?: string) => Promise<unknown>;
    activeChatId: string;
  },
): Promise<void> {
  const { setBusyAction, setError, setWorkspace, refreshWorkspace, activeChatId } = deps;
  setBusyAction("Unloading model...");
  try {
    const runtime = await unloadModel(ref);
    setWorkspace((current) => syncRuntime(current, runtime));
    await refreshWorkspace(activeChatId || undefined);
  } catch (actionError) {
    setError(actionError instanceof Error ? actionError.message : "Failed to unload model.");
  } finally {
    setBusyAction(null);
  }
}


export async function performDeleteModel(
  item: LibraryItem,
  deps: {
    setBusyAction: (label: string | null) => void;
    setError: (msg: string | null) => void;
    setWorkspace: (updater: (current: WorkspaceData) => WorkspaceData) => void;
    refreshWorkspace: (preferredChatId?: string) => Promise<unknown>;
    activeChatId: string;
  },
): Promise<void> {
  const { setBusyAction, setError, setWorkspace, refreshWorkspace, activeChatId } = deps;
  const confirmed = window.confirm(
    `Delete "${item.name}"?\n\nThis will permanently remove the files at:\n${item.path}\n\nThis action cannot be undone.`,
  );
  if (!confirmed) return;
  setBusyAction("Deleting model...");
  try {
    const result = await deleteModelPath(item.path);
    setWorkspace((current) => ({ ...current, library: result.library }));
    await refreshWorkspace(activeChatId || undefined);
  } catch (actionError) {
    setError(actionError instanceof Error ? actionError.message : "Failed to delete model.");
  } finally {
    setBusyAction(null);
  }
}
