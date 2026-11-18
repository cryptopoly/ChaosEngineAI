import type { LibraryItem } from "../types";

/**
 * The library item a conversion draft points at. A GGUF source is picked as one
 * weights file inside the item's folder, so the draft path is the file while the
 * library lists the folder; matching on equal paths only left the source card
 * empty ("Select a model to convert...") with Convert enabled.
 */
export function findConversionSource(library: LibraryItem[], draftPath: string): LibraryItem | null {
  if (!draftPath) return null;
  const exact = library.find((item) => item.path === draftPath);
  if (exact) return exact;
  return (
    library.find((item) => {
      if (!item.path) return false;
      const folder = item.path.replace(/[\\/]+$/, "");
      return draftPath.startsWith(`${folder}/`) || draftPath.startsWith(`${folder}\\`);
    }) ?? null
  );
}
