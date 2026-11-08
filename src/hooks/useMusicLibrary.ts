/**
 * The Music Library list: load, rename, tag and delete. The list is loaded whole (newest first) and the
 * Library screen filters and sorts it in the browser (`filterLibrary`), so typing in the search box never
 * waits for the backend.
 */
import { useCallback, useRef, useState } from "react";
import { deleteMusicLibraryItems, getMusicLibrary, updateMusicLibraryItem } from "../api";
import type { MusicLibraryItem } from "../types";

function message(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

export function useMusicLibrary(backendOnline: boolean, setErrorProp: (message: string | null) => void) {
  const errorRef = useRef(setErrorProp);
  errorRef.current = setErrorProp;
  const setError = useCallback((text: string | null) => errorRef.current(text), []);
  const [items, setItems] = useState<MusicLibraryItem[]>([]);
  const [diskBytes, setDiskBytes] = useState(0);
  const [loaded, setLoaded] = useState(false);

  const refreshLibrary = useCallback(async () => {
    try {
      const result = await getMusicLibrary();
      setItems(result.items);
      setDiskBytes(result.diskBytes);
      setLoaded(true);
    } catch (err) {
      setError(message(err));
    }
  }, [setError]);

  // No fetch on mount: the Library screen reads the list when it opens, and a run or a delete refreshes it.

  const patchItem = useCallback(
    async (id: string, patch: { title?: string; tags?: string[] }) => {
      try {
        const next = await updateMusicLibraryItem(id, patch);
        setItems((previous) => previous.map((item) => (item.id === id ? next : item)));
        return next;
      } catch (err) {
        setError(message(err));
        return null;
      }
    },
    [setError],
  );

  const deleteItems = useCallback(
    async (ids: string[]) => {
      try {
        const deleted = await deleteMusicLibraryItems(ids);
        setItems((previous) => previous.filter((item) => !deleted.includes(item.id)));
        await refreshLibrary();
        return deleted;
      } catch (err) {
        setError(message(err));
        return [];
      }
    },
    [setError, refreshLibrary],
  );

  return { items, diskBytes, loaded, refreshLibrary, renameItem: (id: string, title: string) => patchItem(id, { title }), setTags: (id: string, tags: string[]) => patchItem(id, { tags }), deleteItems };
}

export type UseMusicLibrary = ReturnType<typeof useMusicLibrary>;
