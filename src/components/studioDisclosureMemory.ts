/**
 * Which disclosures of a Studio are open, remembered per `studioId` in
 * localStorage under `chaos.studio.<id>.disclosures`. Every storage access
 * is wrapped: storage can be missing or throw (private mode, previews).
 */
import { createContext, useCallback, useMemo, useRef } from "react";

export type DisclosureState = Record<string, boolean>;

export interface DisclosureMemory {
  isOpen: (id: string) => boolean | undefined;
  setOpen: (id: string, open: boolean) => void;
}

export const DisclosureMemoryContext = createContext<DisclosureMemory | null>(null);

export function disclosureStorageKey(studioId: string): string {
  return `chaos.studio.${studioId}.disclosures`;
}

export function readDisclosures(studioId: string): DisclosureState {
  try {
    const raw = window.localStorage.getItem(disclosureStorageKey(studioId));
    if (!raw) return {};
    const parsed: unknown = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return {};
    const result: DisclosureState = {};
    for (const [key, value] of Object.entries(parsed)) {
      if (typeof value === "boolean") result[key] = value;
    }
    return result;
  } catch {
    return {};
  }
}

export function writeDisclosures(studioId: string, state: DisclosureState): void {
  try {
    window.localStorage.setItem(disclosureStorageKey(studioId), JSON.stringify(state));
  } catch {
    /* quota or privacy mode: the screen still works, it just forgets */
  }
}

/** The memory object for one Studio: reads storage once per `studioId`, writes on every change. */
export function useDisclosureMemory(studioId: string): DisclosureMemory {
  const stored = useRef<{ id: string; state: DisclosureState } | null>(null);
  const read = useCallback((): DisclosureState => {
    if (!stored.current || stored.current.id !== studioId) {
      stored.current = { id: studioId, state: readDisclosures(studioId) };
    }
    return stored.current.state;
  }, [studioId]);

  return useMemo<DisclosureMemory>(
    () => ({
      isOpen: (id) => read()[id],
      setOpen: (id, open) => {
        const next = { ...read(), [id]: open };
        stored.current = { id: studioId, state: next };
        writeDisclosures(studioId, next);
      },
    }),
    [read, studioId],
  );
}
