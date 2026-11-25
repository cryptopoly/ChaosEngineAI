/**
 * The project's lip-sync coverage (`guide.coverage`), read from the backend whenever the project's revision changes,
 * so the table outlives a tab switch or a reload. A run's own answer can be shown at once with `show`.
 */
import { useEffect, useState } from "react";
import { getStoryboardGuide } from "../../../api/storyboard";
import type { StoryboardLipSyncCoverage } from "../../../types";

export function useSongCoverage(projectId: string, revision: number, hasSong: boolean) {
  const [coverage, setCoverage] = useState<{ revision: number; value: StoryboardLipSyncCoverage | null } | null>(null);

  useEffect(() => {
    if (!hasSong) return;
    let live = true;
    getStoryboardGuide(projectId)
      .then((out) => {
        if (live) setCoverage({ revision: out.revision, value: out.guide.coverage });
      })
      .catch(() => undefined); // keep what is shown; the next revision asks again
    return () => {
      live = false;
    };
  }, [projectId, revision, hasSong]);

  return {
    coverage: hasSong ? (coverage?.value ?? null) : null,
    show: (value: StoryboardLipSyncCoverage | null | undefined, at: number) => {
      if (value !== undefined) setCoverage((prev) => (prev && prev.revision > at ? prev : { revision: at, value }));
    },
  };
}
