/**
 * Two read-only lookups the Export tab needs: the project's guide block (for "Ready to export?") when the payload the
 * shell holds does not carry it, and whether the Upscale add-on is installed (for the sharper upscale method).
 * Both only read; a failed read leaves the answer unknown and the tab works without it.
 */
import { useEffect, useState } from "react";
import { getVideoUpscaleAddonStatus } from "../../api";
// Not in the api barrel yet (`src/api/index.ts` lists its storyboard exports by name), so read it from its file.
import { getStoryboardGuide } from "../../api/storyboard";
import type { StoryboardGuide } from "../../types";

/** The guide for this project at this revision. `given` (the payload's own block) wins and is never fetched again. */
export function useExportGuide(projectId: string, revision: number, given: StoryboardGuide | null | undefined): StoryboardGuide | null {
  const [fetched, setFetched] = useState<{ revision: number; guide: StoryboardGuide } | null>(null);
  const skip = Boolean(given);
  useEffect(() => {
    if (skip) return;
    let cancelled = false;
    getStoryboardGuide(projectId)
      .then((answer) => !cancelled && setFetched({ revision: answer.revision, guide: answer.guide }))
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [projectId, revision, skip]);
  return given ?? fetched?.guide ?? null;
}

/** true / false once the add-on's status is read; null while it is unknown (not asked yet, or the read failed). */
export function useUpscaleAddon(enabled: boolean): boolean | null {
  const [installed, setInstalled] = useState<boolean | null>(null);
  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    getVideoUpscaleAddonStatus()
      .then((status) => !cancelled && setInstalled(Boolean(status.installed)))
      .catch(() => !cancelled && setInstalled(null));
    return () => {
      cancelled = true;
    };
  }, [enabled]);
  return installed;
}
