/**
 * The Projects list (D-65): search and sort the projects, open one, make one (optionally from a template), or
 * import a .chaosfilm bundle with the button or by dropping it on the list. Rename, export and delete sit in
 * each card's menu.
 */
import { useEffect, useRef, useState, type DragEvent } from "react";
import { useTranslation } from "react-i18next";
import { createStoryboardProject, deleteStoryboardProject, importStoryboardBundle, runStoryboardOp } from "../../api";
import { PlusIcon } from "../../components/icons/ActionIcons";
import { Panel } from "../../components/Panel";
import { ConfirmDialog } from "./ConfirmDialog";
import { ProjectCard } from "./ProjectCard";
import type { ModelChoice } from "./ProjectDefaultsPanel";
import { ProjectDialog, type ProjectDialogResult } from "./ProjectDialog";
import { newProjectDefaults, type VideoFitSource } from "./formatLogic";
import { downloadProjectBundle, formatBundleBytes } from "./projectFiles";
import { LIST_SORTS, filterAndSort, isBundleName, readListSort, writeListSort, type ListSort } from "./projectShellLogic";
import { templateById, templateShotArgs } from "./projectTemplates";
import { useStoryboardProjects, type ProjectCardData } from "./useStoryboardProjects";

export interface ProjectListTabProps {
  backendOnline: boolean;
  /** Opens a project; a new project opens on Overview. */
  onOpenProject: (id: string, created?: boolean) => void;
  /** Installed video models and the machine, for the fit hint in the New project dialog. */
  videoFit?: VideoFitSource;
  imageModels?: ModelChoice[];
  videoModels?: ModelChoice[];
}

export function ProjectListTab({ backendOnline, onOpenProject, videoFit, imageModels, videoModels }: ProjectListTabProps) {
  const { t, i18n } = useTranslation("studio");
  const { cards, loading, error, refresh } = useStoryboardProjects(backendOnline);
  const [dialog, setDialog] = useState<{ mode: "create" } | { mode: "rename"; card: ProjectCardData } | null>(null);
  const [dialogError, setDialogError] = useState<string | null>(null);
  const [toDelete, setToDelete] = useState<ProjectCardData | null>(null);
  const [working, setWorking] = useState(false);
  const [exportingId, setExportingId] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [problem, setProblem] = useState<{ headline: string; detail: string } | null>(null);
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<ListSort>(() => readListSort());
  const [dropOver, setDropOver] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const search = useRef<HTMLInputElement>(null);
  const now = Date.now();
  const shown = filterAndSort(cards, query, sort, i18n.language);

  // "/" focuses the search field, unless the user is typing somewhere.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (event.key !== "/" || event.metaKey || event.ctrlKey || event.altKey) return;
      if (target?.closest?.("input, textarea, select, [contenteditable='true'], .modal-overlay")) return;
      event.preventDefault();
      search.current?.focus();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  function openCreate() {
    setDialogError(null);
    setDialog({ mode: "create" });
  }

  async function createProject(result: ProjectDialogResult) {
    const defaults = { ...newProjectDefaults(result.aspect, result.resolution, result.fps), imageModel: result.imageModel, videoModel: result.videoModel };
    let payload = await createStoryboardProject(result.name, defaults);
    const shots = templateShotArgs(templateById(result.template), (shot) => ({
      title: t(`templates.shots.${shot.key}.title`, { defaultValue: shot.title }),
      stillPrompt: t(`templates.shots.${shot.key}.prompt`, { defaultValue: shot.stillPrompt }),
    }));
    for (const args of shots) {
      payload = await runStoryboardOp(payload.project.id, payload.project.revision, "add_shot", args);
    }
    return payload.project.id;
  }

  async function submitDialog(result: ProjectDialogResult) {
    if (!dialog) return;
    setWorking(true);
    setDialogError(null);
    try {
      if (dialog.mode === "create") {
        const id = await createProject(result);
        setDialog(null);
        onOpenProject(id, true);
      } else if (dialog.card.project) {
        await runStoryboardOp(dialog.card.id, dialog.card.project.revision, "edit_project", { name: result.name });
        setDialog(null);
        await refresh();
      }
    } catch (err) {
      setDialogError(err instanceof Error ? err.message : String(err));
    } finally {
      setWorking(false);
    }
  }

  async function confirmDelete() {
    if (!toDelete?.project) return;
    setWorking(true);
    try {
      await deleteStoryboardProject(toDelete.id, toDelete.project.revision);
      setToDelete(null);
      await refresh();
    } catch (err) {
      setToDelete(null);
      setProblem({
        headline: t("storyProjects.deleteFailed", { defaultValue: "The project could not be deleted." }),
        detail: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setWorking(false);
    }
  }

  async function exportBundle(card: ProjectCardData) {
    setExportingId(card.id);
    setProblem(null);
    try {
      const done = await downloadProjectBundle(card.id);
      setStatus(t("storyProjects.exported", { defaultValue: "Exported {name} · {size}", name: done.name, size: formatBundleBytes(done.bytes) }));
    } catch (err) {
      setProblem({
        headline: t("storyProjects.exportFailed", { defaultValue: "The bundle could not be exported." }),
        detail: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setExportingId(null);
    }
  }

  async function importFile(file: File) {
    setWorking(true);
    setProblem(null);
    setStatus(null);
    try {
      const payload = await importStoryboardBundle(file);
      await refresh();
      setStatus(t("storyProjects.imported", { defaultValue: "Imported {name}", name: payload.project.name }));
    } catch (err) {
      setProblem({
        headline: t("storyProjects.importFailed", { defaultValue: "This file is not a ChaosEngineAI project bundle." }),
        detail: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setWorking(false);
    }
  }

  function onDrop(event: DragEvent) {
    event.preventDefault();
    setDropOver(false);
    const file = Array.from(event.dataTransfer?.files ?? []).find((item) => isBundleName(item.name));
    if (file) void importFile(file);
  }

  const sortLabels: Record<ListSort, string> = {
    edited: t("projectList.sort.edited", { defaultValue: "Last edited" }),
    name: t("projectList.sort.name", { defaultValue: "Name" }),
    created: t("projectList.sort.created", { defaultValue: "Created" }),
    length: t("projectList.sort.length", { defaultValue: "Length" }),
  };

  const actions = (
    <>
      <button type="button" className="secondary-button" disabled={working} onClick={() => fileInput.current?.click()}>
        {t("storyProjects.import", { defaultValue: "Import bundle" })}
      </button>
      <button type="button" className="primary-button" onClick={openCreate}>
        <PlusIcon /> {t("storyProjects.new", { defaultValue: "New project" })}
      </button>
      <input
        ref={fileInput}
        type="file"
        accept=".chaosfilm,.zip"
        className="sr-only"
        tabIndex={-1}
        aria-label={t("storyProjects.import", { defaultValue: "Import bundle" })}
        onChange={(event) => {
          const file = event.target.files?.[0];
          event.target.value = "";
          if (file) void importFile(file);
        }}
      />
    </>
  );

  return (
    <div className="content-grid storyboard-page">
      <Panel
        title={t("projectList.title", { defaultValue: "Projects" })}
        subtitle={t("storyProjects.subtitle", {
          defaultValue: "{count, plural, one {# project} other {# projects}}",
          count: cards.length,
        })}
        actions={actions}
        className={`span-2 project-list${dropOver ? " project-list--over" : ""}`}
      >
        <div
          className="project-list__body"
          onDragOver={(event) => {
            if (!Array.from(event.dataTransfer?.types ?? []).includes("Files")) return;
            event.preventDefault();
            setDropOver(true);
          }}
          onDragLeave={(event) => event.currentTarget === event.target && setDropOver(false)}
          onDrop={onDrop}
        >
          {!backendOnline ? (
            <div className="empty-state">
              <p className="muted-text">{t("storyboard.offline", { defaultValue: "Waiting for the backend…" })}</p>
            </div>
          ) : null}
          {cards.length > 0 ? (
            <div className="project-list__tools">
              <input
                ref={search}
                type="search"
                className="text-input project-list__search"
                value={query}
                placeholder={t("projectList.search", { defaultValue: "Search projects" })}
                aria-label={t("projectList.search", { defaultValue: "Search projects" })}
                aria-keyshortcuts="/"
                onChange={(event) => setQuery(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Escape" && query) {
                    event.stopPropagation();
                    setQuery("");
                  }
                }}
              />
              <label className="project-list__sort">
                <span>{t("projectList.sortLabel", { defaultValue: "Sort" })}</span>
                <select
                  className="text-input"
                  value={sort}
                  onChange={(event) => {
                    const next = event.target.value as ListSort;
                    setSort(next);
                    writeListSort(next);
                  }}
                >
                  {LIST_SORTS.map((key) => (
                    <option key={key} value={key}>{sortLabels[key]}</option>
                  ))}
                </select>
              </label>
            </div>
          ) : null}
          {status ? <p className="muted-text storyboard-status" role="status">{status}</p> : null}
          {error ? <p className="error-text">{error}</p> : null}
          {problem ? (
            <div className="callout error" role="alert">
              <p>{problem.headline}</p>
              <details className="debug-details">
                <summary>{t("error.technicalDetails", { defaultValue: "Technical details" })}</summary>
                <pre>{problem.detail}</pre>
              </details>
            </div>
          ) : null}
          {backendOnline && loading && cards.length === 0 ? (
            <div className="empty-state">
              <p className="muted-text">{t("storyProjects.loading", { defaultValue: "Loading your projects…" })}</p>
            </div>
          ) : null}
          {backendOnline && !loading && cards.length === 0 && !error ? (
            <div className="empty-state">
              <p className="muted-text">
                {t("storyProjects.empty", {
                  defaultValue: "No projects yet. A project holds your shots, takes and sound in one folder you can move or back up.",
                })}
              </p>
              <div className="overview-tab__actions">
                <button type="button" className="primary-button" onClick={openCreate}>
                  {t("storyProjects.new", { defaultValue: "New project" })}
                </button>
                <button type="button" className="secondary-button" disabled={working} onClick={() => fileInput.current?.click()}>
                  {t("storyProjects.import", { defaultValue: "Import bundle" })}
                </button>
              </div>
              <p className="muted-text">{t("projectList.dropHint", { defaultValue: "Or drop a .chaosfilm file here." })}</p>
            </div>
          ) : null}
          {cards.length > 0 && shown.length === 0 ? (
            <p className="muted-text">{t("projectList.noMatch", { defaultValue: "No project matches {query}.", query })}</p>
          ) : null}
          {shown.length > 0 ? (
            <div className="project-grid" role="list">
              {shown.map((card) => (
                <div role="listitem" key={card.id}>
                  <ProjectCard
                    card={card}
                    now={now}
                    exporting={exportingId === card.id}
                    onOpen={() => onOpenProject(card.id)}
                    onExport={() => void exportBundle(card)}
                    onRename={() => { setDialogError(null); setDialog({ mode: "rename", card }); }}
                    onDelete={() => setToDelete(card)}
                  />
                </div>
              ))}
            </div>
          ) : null}
        </div>
      </Panel>
      {dialog ? (
        <ProjectDialog
          mode={dialog.mode}
          initialName={dialog.mode === "rename" ? dialog.card.name : ""}
          busy={working}
          error={dialogError}
          fitSource={videoFit}
          imageModels={imageModels}
          videoModels={videoModels}
          onSubmit={(result) => void submitDialog(result)}
          onCancel={() => setDialog(null)}
        />
      ) : null}
      {toDelete ? (
        <ConfirmDialog
          title={t("storyProjects.deleteTitle", { defaultValue: "Delete this project?" })}
          body={
            <p>
              {t("storyProjects.deleteBody", {
                defaultValue: "{name} and every take, still and sound in its folder will be deleted. This cannot be undone.",
                name: toDelete.name,
              })}
            </p>
          }
          confirmLabel={t("storyProjects.menu.delete", { defaultValue: "Delete" })}
          danger
          busy={working}
          onConfirm={() => void confirmDelete()}
          onCancel={() => setToDelete(null)}
        />
      ) : null}
    </div>
  );
}
