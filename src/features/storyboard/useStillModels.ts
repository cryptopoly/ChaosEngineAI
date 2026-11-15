/**
 * The installed still models with the catalogue time per still at the project size, and what Automatic picks:
 * the fastest for Make and drafts, the best quality for the final pass (S2, the same rule as the agent's default).
 */
import { useEffect, useState } from "react";
import { getStoryboardStillModels, type StoryboardStillModels } from "../../api/storyboard";

export function useStillModels(width: number, height: number, enabled = true): StoryboardStillModels | null {
  const [models, setModels] = useState<StoryboardStillModels | null>(null);
  useEffect(() => {
    if (!enabled) return;
    let live = true;
    getStoryboardStillModels(width, height)
      .then((result) => live && setModels(result))
      .catch(() => live && setModels(null));
    return () => {
      live = false;
    };
  }, [width, height, enabled]);
  return models;
}
