/**
 * The list of storyboard projects with what a card shows. Read on mount, when the window becomes visible again,
 * and every 10 s while it is visible, so a project an agent just made shows up without a reload; unchanged
 * projects are not read again (the list reports their revision).
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { getStoryboardProject, listStoryboardProjects } from "../../api";
import type { StoryboardProject, StoryboardTimeline } from "../../types";
import { mapWithLimit, projectRecency, sortByRecent } from "./storyboardLogic";

/** How many project files are read at once when the list opens. */
export const PROJECT_READ_LIMIT = 3;
/** How often the visible list looks for changes. */
export const LIST_REFRESH_MS = 10_000;

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
  createdAt: number;
  /** Length of the film in seconds. */
  seconds: number;
}

export function useStoryboardProjects(enabled: boolean) {
  const [cards, setCards] = useState<ProjectCardData[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const loadedOnce = useRef(false);

  const refresh = useCallback(async () => {
    if (!loadedOnce.current) setLoading(true);
    try {
      const rows = await listStoryboardProjects();
      const loaded = await mapWithLimit(
        rows,
        PROJECT_READ_LIMIT,
        async (row): Promise<ProjectCardData> => {
          if (row.error !== undefined) {
            return { id: row.id, name: row.id, project: null, timeline: null, error: row.error, recency: 0, createdAt: 0, seconds: 0 };
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
              createdAt: payload.project.createdAt,
              seconds: payload.timeline?.total ?? 0,
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
              createdAt: row.createdAt,
              seconds: 0,
            };
          }
        },
      );
      setCards(sortByRecent(loaded));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      loadedOnce.current = true;
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!enabled) return;
    void refresh();
    let timer: ReturnType<typeof setInterval> | undefined;
    const start = () => {
      if (!timer) timer = setInterval(() => void refresh(), LIST_REFRESH_MS);
    };
    const stop = () => {
      if (timer) clearInterval(timer);
      timer = undefined;
    };
    const onVisibility = () => {
      if (document.visibilityState === "visible") {
        void refresh();
        start();
      } else {
        stop();
      }
    };
    if (document.visibilityState !== "hidden") start();
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      stop();
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [enabled, refresh]);

  return { cards, loading, error, refresh };
}
