import { useCallback, useEffect, useState } from "react";
import type { SidebarGroupId, SidebarMode } from "../types";

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

function readStored(): StoredPrefs {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as StoredPrefs;
    return parsed ?? {};
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
  const [mode, setModeState] = useState<SidebarMode>(() => readStored().mode ?? "collapsible");
  const [collapsedGroups, setCollapsedGroups] = useState<Set<SidebarGroupId>>(
    () => new Set(readStored().collapsedGroups ?? []),
  );
  const [lastChildByGroup, setLastChildByGroup] = useState<Partial<Record<SidebarGroupId, string>>>(
    () => readStored().lastChildByGroup ?? {},
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
