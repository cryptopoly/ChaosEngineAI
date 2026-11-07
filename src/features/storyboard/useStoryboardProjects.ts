/** The list of storyboard projects with what a card shows, loaded once and refreshed on demand. */
import { useCallback, useEffect, useState } from "react";
import { getStoryboardProject, listStoryboardProjects } from "../../api";
import type { StoryboardProject, StoryboardTimeline } from "../../types";
import { projectRecency, sortByRecent } from "./storyboardLogic";

export interface ProjectCardData {
  id: string;
  name: string;
  project: StoryboardProject | null;
  timeline: StoryboardTimeline | null;
  /** Set when the project could not be read (its folder moved or its file is damaged). */
  error: string | null;
  recency: number;
}

export function useStoryboardProjects(enabled: boolean) {
  const [cards, setCards] = useState<ProjectCardData[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const rows = await listStoryboardProjects();
      const loaded = await Promise.all(
        rows.map(async (row): Promise<ProjectCardData> => {
          if (row.error !== undefined) {
            return { id: row.id, name: row.id, project: null, timeline: null, error: row.error, recency: 0 };
          }
          try {
            const payload = await getStoryboardProject(row.id);
            return {
              id: row.id,
              name: payload.project.name,
              project: payload.project,
              timeline: payload.timeline,
              error: null,
              recency: projectRecency(payload.project),
            };
          } catch (err) {
            return {
              id: row.id,
              name: row.name,
              project: null,
              timeline: null,
              error: err instanceof Error ? err.message : String(err),
              recency: row.createdAt,
            };
          }
        }),
      );
      setCards(sortByRecent(loaded));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (enabled) void refresh();
  }, [enabled, refresh]);

  return { cards, loading, error, refresh };
}
