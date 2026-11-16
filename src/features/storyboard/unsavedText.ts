/**
 * Typed text whose save did not land (S-O, defect 14 of S-N). A field that saves on blur hands its text here; when
 * the save is refused (another window changed the project first, or the backend did not answer) the text is kept
 * per field, so the field can show the latest and still offer the user's own words back. The store lives with the
 * open project (`useStoryboard`), so a field that unmounts (the tab hides while the backend is away) finds its text
 * again when it comes back. Nothing typed is replaced without a way back.
 */
import { createContext } from "react";

export type UnsavedKind = "conflict" | "failed";

export interface UnsavedEntry {
  text: string;
  kind: UnsavedKind;
}

export interface UnsavedTextStore {
  get: (key: string) => UnsavedEntry | null;
  set: (key: string, entry: UnsavedEntry | null) => void;
  subscribe: (listener: () => void) => () => void;
  /** Why the last change of the project failed; set by `useStoryboard` before it resolves the change with false. */
  lastFailure: UnsavedKind | null;
}

export function createUnsavedTextStore(): UnsavedTextStore {
  const entries = new Map<string, UnsavedEntry>();
  const listeners = new Set<() => void>();
  const store: UnsavedTextStore = {
    get: (key) => entries.get(key) ?? null,
    set: (key, entry) => {
      const before = entries.get(key) ?? null;
      if (!entry && !before) return;
      if (entry && before && entry.text === before.text && entry.kind === before.kind) return;
      if (entry) entries.set(key, entry);
      else entries.delete(key);
      for (const listener of Array.from(listeners)) listener();
    },
    subscribe: (listener) => {
      listeners.add(listener);
      return () => void listeners.delete(listener);
    },
    lastFailure: null,
  };
  return store;
}

/** The open project's store; a field outside a project shell (tests, a lone panel) keeps its own. */
export const UnsavedTextContext = createContext<UnsavedTextStore | null>(null);

/** A save a field started: a promise of whether it landed, or nothing (the caller did not say). */
export type SaveResult = Promise<boolean> | void | undefined;

/**
 * Follows one save of `text` for `key`: clears the kept text when it lands, keeps it with the reason when it does
 * not. A caller that does not hand back a promise is taken as saved.
 */
export function trackSave(store: UnsavedTextStore, key: string, text: string, result: SaveResult): void {
  if (!result || typeof result.then !== "function") {
    store.set(key, null);
    return;
  }
  result.then(
    (ok) => store.set(key, ok ? null : { text, kind: store.lastFailure ?? "failed" }),
    () => store.set(key, { text, kind: "failed" }),
  );
}
