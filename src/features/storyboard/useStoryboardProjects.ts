/** The list of storyboard projects with what a card shows, loaded once and refreshed on demand. */
import { useCallback, useEffect, useState } from "react";
import { getStoryboardProject, listStoryboardProjects } from "../../api";
import type { StoryboardProject, StoryboardTimeline } from "../../types";
import { mapWithLimit, projectRecency, sortByRecent } from "./storyboardLogic";

/** How many project files are read at once when the list opens. */
export const PROJECT_READ_LIMIT = 3;

/** Cards already read this session, kept while the project is still at the revision the list reports. */
const readCards = new Map<string, { revision: number; card: ProjectCardData }>();

export function clearProjectCardCache(): void {
  readCards.clear();
}

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
      const loaded = await mapWithLimit(
        rows,
        PROJECT_READ_LIMIT,
        async (row): Promise<ProjectCardData> => {
          if (row.error !== undefined) {
            return { id: row.id, name: row.id, project: null, timeline: null, error: row.error, recency: 0 };
          }
          const known = readCards.get(row.id);
          if (known && known.revision === row.revision) return known.card;
          try {
            const payload = await getStoryboardProject(row.id);
            const card: ProjectCardData = {
              id: row.id,
              name: payload.project.name,
              project: payload.project,
              timeline: payload.timeline,
              error: null,
              recency: projectRecency(payload.project),
            };
            readCards.set(row.id, { revision: payload.project.revision, card });
            return card;
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
        },
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
