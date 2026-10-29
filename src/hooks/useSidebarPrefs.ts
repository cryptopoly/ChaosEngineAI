import { useCallback, useEffect, useState } from "react";
import type { SidebarGroupId, SidebarMode } from "../types";
import { migrateLastChildByGroup } from "../constants/tabRedirects";

const STORAGE_KEY = "chaosengine.sidebar.prefs.v1";

interface StoredPrefs {
  mode?: SidebarMode;
  collapsedGroups?: SidebarGroupId[];
  lastChildByGroup?: Partial<Record<SidebarGroupId, string>>;
}

interface SidebarPrefs {
  mode: SidebarMode;
  collapsedGroups: Set<SidebarGroupId>;
  lastChildByGroup: Partial<Record<SidebarGroupId, string>>;
  toggleGroupCollapsed: (group: SidebarGroupId) => void;
  setMode: (mode: SidebarMode) => void;
  rememberLastChild: (group: SidebarGroupId, tabId: string) => void;
}

export function readStoredSidebarPrefs(): StoredPrefs {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as StoredPrefs;
    if (!parsed) return {};
    // Ids of the old Discover / My Models tabs map to their Models page.
    return { ...parsed, lastChildByGroup: migrateLastChildByGroup(parsed.lastChildByGroup) };
  } catch {
    return {};
  }
}

function writeStored(prefs: StoredPrefs) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(prefs));
  } catch {
    /* ignore quota / privacy mode errors */
  }
}

export function useSidebarPrefs(): SidebarPrefs {
  const [mode, setModeState] = useState<SidebarMode>(() => readStoredSidebarPrefs().mode ?? "collapsible");
  const [collapsedGroups, setCollapsedGroups] = useState<Set<SidebarGroupId>>(
    () => new Set(readStoredSidebarPrefs().collapsedGroups ?? []),
  );
  const [lastChildByGroup, setLastChildByGroup] = useState<Partial<Record<SidebarGroupId, string>>>(
    () => readStoredSidebarPrefs().lastChildByGroup ?? {},
  );

  useEffect(() => {
    writeStored({
      mode,
      collapsedGroups: Array.from(collapsedGroups),
      lastChildByGroup,
    });
  }, [mode, collapsedGroups, lastChildByGroup]);

  const toggleGroupCollapsed = useCallback((group: SidebarGroupId) => {
    setCollapsedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(group)) next.delete(group);
      else next.add(group);
      return next;
    });
  }, []);

  const setMode = useCallback((next: SidebarMode) => {
    setModeState(next);
  }, []);

  const rememberLastChild = useCallback((group: SidebarGroupId, tabId: string) => {
    setLastChildByGroup((prev) => ({ ...prev, [group]: tabId }));
  }, []);

  return {
    mode,
    collapsedGroups,
    lastChildByGroup,
    toggleGroupCollapsed,
    setMode,
    rememberLastChild,
  };
}
