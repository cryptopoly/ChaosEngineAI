/** One project on the Projects list: up to three stills, the numbers, open on click or Enter, and a menu. */
import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";
import { ActionMenu } from "../../components/ActionMenu";
import { RESOLUTION_ALIAS } from "./formatLogic";
import { ProjectImage, ProjectVideo } from "./ProjectMedia";
import { emptyThumbState, filmSeconds, relativeTime, staleCount, thumbnailOf } from "./storyboardLogic";
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

function emptyThumbText(t: TFunction<"studio">, state: ReturnType<typeof emptyThumbState>): string {
  if (state === "noShots") return t("storyProjects.card.noShots", { defaultValue: "no shots yet" });
  if (state === "noTakes") return t("storyProjects.card.noTakes", { defaultValue: "no pictures yet" });
  return t("storyProjects.card.noPreview", { defaultValue: "no preview" });
}

export function ProjectCard({ card, now, onOpen, onExport, onRename, onDelete, exporting }: ProjectCardProps) {
  const { t, i18n } = useTranslation("studio");
  const { project, timeline } = card;
  const menuLabel = t("storyProjects.moreActionsFor", { defaultValue: "More actions for {name}", name: card.name });

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
    .filter((entry) => entry.thumb !== null)
    .slice(0, 3);
  const stale = staleCount(project);
  const shots = project.shots.length;
  const d = project.defaults;
  const format = `${d.exportPreset} · ${RESOLUTION_ALIAS[d.resolution] ?? d.resolution} · ${d.fps} fps`;

  return (
    <article className="project-card" aria-label={card.name}>
      <button
        type="button"
        className="project-card__open"
        aria-label={t("storyProjects.openNamed", { defaultValue: "Open {name}", name: card.name })}
        onClick={onOpen}
      >
        <span className="project-card__thumbs" aria-hidden="true">
          {thumbs.length === 0 ? (
            <span className="project-card__thumb-empty">{emptyThumbText(t, emptyThumbState(project))}</span>
          ) : (
            thumbs.map(({ shot, thumb }) =>
              thumb?.kind === "clip" ? (
                <ProjectVideo key={shot.id} projectId={project.id} rel={thumb.path} version={shot.activeTake} className="project-card__thumb" />
              ) : (
                <ProjectImage key={shot.id} projectId={project.id} rel={thumb?.path ?? null} version={shot.activeTake} className="project-card__thumb" thumb={256} />
              ),
            )
          )}
        </span>
      </button>
      <div className="project-card__head">
        <h3 className="project-card__title" title={card.name}>{card.name}</h3>
        <ActionMenu
          label={menuLabel}
          items={[
            { id: "rename", label: t("storyProjects.menu.rename", { defaultValue: "Rename" }), onSelect: onRename },
            {
              id: "export",
              label: exporting ? t("storyProjects.exporting", { defaultValue: "Exporting…" }) : t("storyProjects.exportBundle", { defaultValue: "Export bundle" }),
              onSelect: onExport,
              disabled: exporting,
            },
            { id: "delete", label: t("storyProjects.menu.delete", { defaultValue: "Delete" }), onSelect: onDelete, danger: true },
          ]}
        />
      </div>
      <p className="project-card__meta">
        {shots === 0
          ? t("storyProjects.card.zeroShotsFormat", { defaultValue: "0 shots · {format}", format })
          : t("storyProjects.card.metaFormat", {
              defaultValue: "{shots, plural, one {# shot} other {# shots}} · {seconds} s · {format}",
              shots,
              seconds: filmSeconds(timeline),
              format,
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
    </article>
  );
}
