/** One project on the Projects screen: up to three stills, the numbers, and its actions. */
import { useTranslation } from "react-i18next";
import { ActionMenu } from "./ActionMenu";
import { ProjectImage } from "./ProjectMedia";
import { filmSeconds, relativeTime, staleCount, thumbnailOf } from "./storyboardLogic";
import type { ProjectCardData } from "./useStoryboardProjects";

export interface ProjectCardProps {
  card: ProjectCardData;
  now: number;
  onOpen: () => void;
  onExport: () => void;
  onRename: () => void;
  onDelete: () => void;
  exporting?: boolean;
}

export function ProjectCard({ card, now, onOpen, onExport, onRename, onDelete, exporting }: ProjectCardProps) {
  const { t, i18n } = useTranslation("studio");
  const { project, timeline } = card;

  if (!project) {
    return (
      <article className="project-card project-card--error" aria-label={card.name}>
        <h3 className="project-card__title">{card.name}</h3>
        <p className="project-card__note">{t("storyProjects.missing", { defaultValue: "This project's folder was moved or deleted." })}</p>
        <details className="debug-details">
          <summary>{t("error.technicalDetails", { defaultValue: "Technical details" })}</summary>
          <pre>{card.error}</pre>
        </details>
      </article>
    );
  }

  const thumbs = project.shots
    .map((shot) => ({ shot, thumb: thumbnailOf(shot) }))
    .filter((entry) => entry.thumb?.kind === "still")
    .slice(0, 3);
  const stale = staleCount(project);
  const shots = project.shots.length;

  return (
    <article className="project-card" aria-label={card.name}>
      <div className="project-card__thumbs" aria-hidden="true">
        {thumbs.length === 0 ? (
          <span className="project-card__thumb-empty">{t("storyProjects.card.noShots", { defaultValue: "no shots yet" })}</span>
        ) : (
          thumbs.map(({ shot, thumb }) => (
            <ProjectImage
              key={shot.id}
              projectId={project.id}
              rel={thumb?.path ?? null}
              version={shot.activeTake}
              className="project-card__thumb"
            />
          ))
        )}
      </div>
      <h3 className="project-card__title" title={card.name}>{card.name}</h3>
      <p className="project-card__meta">
        {shots === 0
          ? t("storyProjects.card.zeroShots", { defaultValue: "0 shots" })
          : t("storyProjects.card.meta", {
              defaultValue: "{shots, plural, one {# shot} other {# shots}} · {seconds} s · {aspect}",
              shots,
              seconds: filmSeconds(timeline),
              aspect: project.defaults.exportPreset,
            })}
        {stale > 0 ? (
          <span className="badge warning project-card__stale">
            {t("storyProjects.card.stale", { defaultValue: "{n} stale", n: stale })}
          </span>
        ) : null}
      </p>
      <p className="project-card__meta project-card__edited">
        {t("storyProjects.card.edited", {
          defaultValue: "edited {when}",
          when: relativeTime(card.recency, now, i18n.language),
        })}
      </p>
      <div className="project-card__actions">
        <button type="button" className="primary-button" onClick={onOpen}>
          {t("storyProjects.open", { defaultValue: "Open" })}
        </button>
        <button type="button" className="secondary-button" onClick={onExport} disabled={exporting}>
          {exporting
            ? t("storyProjects.exporting", { defaultValue: "Exporting…" })
            : t("storyProjects.exportBundle", { defaultValue: "Export bundle" })}
        </button>
        <ActionMenu
          label={t("storyProjects.moreActions", { defaultValue: "More actions" })}
          items={[
            { id: "rename", label: t("storyProjects.menu.rename", { defaultValue: "Rename" }), onSelect: onRename },
            { id: "delete", label: t("storyProjects.menu.delete", { defaultValue: "Delete" }), onSelect: onDelete, danger: true },
          ]}
        />
      </div>
    </article>
  );
}
