/**
 * The project's finished films (the files in its exports/ folder), newest first, each with Reveal.
 * Read-only: films stay in the project and are not copied to the Video gallery.
 */
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { fetchJson } from "../../api";

export interface FilmRow {
  file: string;
  name: string;
  bytes: number;
  modifiedAt: number;
}

export async function listFilms(projectId: string): Promise<FilmRow[]> {
  const result = await fetchJson<{ films: FilmRow[] }>(`/api/storyboard/projects/${encodeURIComponent(projectId)}/films`);
  return result.films ?? [];
}

export function formatFilmSize(bytes: number): string {
  if (bytes >= 1024 * 1024 * 1024) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

export interface FilmListViewProps {
  films: FilmRow[];
  revealLabel: string;
  onReveal: (file: string) => void;
}

export function FilmListView({ films, revealLabel, onReveal, showEmpty }: FilmListViewProps & { showEmpty?: boolean }) {
  const { t } = useTranslation("studio");
  if (films.length === 0 && !showEmpty) return null;
  return (
    <div className="film-list">
      <h4>{t("storyboard.films.title", { defaultValue: "Film" })}</h4>
      {films.length === 0 ? <p className="muted-text">{t("storyboard.films.none", { defaultValue: "No films yet. An export writes one here." })}</p> : null}
      <ul>
        {films.map((film) => (
          <li key={film.file}>
            <span className="muted-text" title={film.file}>{film.name}</span>{" "}
            <span className="muted-text">{formatFilmSize(film.bytes)}</span>{" "}
            <button type="button" className="secondary-button" onClick={() => onReveal(film.file)}>
              {revealLabel}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Loads the list when the project or its `version` (bump after an export) changes. */
export function FilmList({ projectId, version, revealLabel, onReveal }: { projectId: string; version?: string | number | null } & Omit<FilmListViewProps, "films">) {
  const [films, setFilms] = useState<FilmRow[]>([]);
  useEffect(() => {
    let live = true;
    listFilms(projectId)
      .then((rows) => {
        if (live) setFilms(rows);
      })
      .catch(() => {
        if (live) setFilms([]);
      });
    return () => {
      live = false;
    };
  }, [projectId, version]);
  return <FilmListView films={films} revealLabel={revealLabel} onReveal={onReveal} showEmpty />;
}
