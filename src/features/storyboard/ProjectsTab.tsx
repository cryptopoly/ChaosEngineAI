/** Storyboard Projects: open, create, import or back up a project (UX spec D.4). */
import { useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  createStoryboardProject,
  deleteStoryboardProject,
  exportStoryboardBundle,
  fetchStoryboardFileBlobUrl,
  importStoryboardBundle,
  runStoryboardOp,
} from "../../api";
import { PlusIcon } from "../../components/icons/ActionIcons";
import { Panel } from "../../components/Panel";
import { ConfirmDialog } from "./ConfirmDialog";
import { ProjectCard } from "./ProjectCard";
import { ProjectDialog, type ProjectDialogResult } from "./ProjectDialog";
import { newProjectDefaults } from "./storyboardLogic";
import { useStoryboardProjects, type ProjectCardData } from "./useStoryboardProjects";

export interface ProjectsTabProps {
  backendOnline: boolean;
  onOpenProject: (id: string) => void;
}

function formatBytes(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

export function ProjectsTab({ backendOnline, onOpenProject }: ProjectsTabProps) {
  const { t } = useTranslation("studio");
  const { cards, loading, error, refresh } = useStoryboardProjects(backendOnline);
  const [dialog, setDialog] = useState<{ mode: "create" } | { mode: "rename"; card: ProjectCardData } | null>(null);
  const [dialogError, setDialogError] = useState<string | null>(null);
  const [toDelete, setToDelete] = useState<ProjectCardData | null>(null);
  const [working, setWorking] = useState(false);
  const [exportingId, setExportingId] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [problem, setProblem] = useState<{ headline: string; detail: string } | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const now = Date.now();

  async function submitDialog(result: ProjectDialogResult) {
    if (!dialog) return;
    setWorking(true);
    setDialogError(null);
    try {
      if (dialog.mode === "create") {
        const payload = await createStoryboardProject(result.name, newProjectDefaults(result.preset, result.fps));
        setDialog(null);
        onOpenProject(payload.project.id);
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
      const result = await exportStoryboardBundle(card.id);
      const url = await fetchStoryboardFileBlobUrl(card.id, result.file);
      const link = document.createElement("a");
      link.href = url;
      link.download = result.file.split("/").pop() ?? "project.chaosfilm";
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 10000);
      setStatus(
        t("storyProjects.exported", {
          defaultValue: "Exported {name} · {size}",
          name: link.download,
          size: formatBytes(result.bytes),
        }),
      );
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

  const actions = (
    <>
      <button type="button" className="primary-button" onClick={() => { setDialogError(null); setDialog({ mode: "create" }); }}>
        <PlusIcon /> {t("storyProjects.new", { defaultValue: "New project" })}
      </button>
      <button type="button" className="secondary-button" disabled={working} onClick={() => fileInput.current?.click()}>
        {t("storyProjects.import", { defaultValue: "Import bundle" })}
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
        title={t("storyProjects.title", { defaultValue: "Storyboard Projects" })}
        subtitle={t("storyProjects.subtitle", {
          defaultValue: "{count, plural, one {# project} other {# projects}}",
          count: cards.length,
        })}
        actions={actions}
        className="span-2"
      >
        {!backendOnline ? (
          <div className="empty-state">
            <p className="muted-text">{t("storyboard.offline", { defaultValue: "Waiting for the backend…" })}</p>
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
            <button type="button" className="primary-button" onClick={() => { setDialogError(null); setDialog({ mode: "create" }); }}>
              {t("storyProjects.new", { defaultValue: "New project" })}
            </button>
          </div>
        ) : null}
        {cards.length > 0 ? (
          <div className="project-grid" role="list">
            {cards.map((card) => (
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
      </Panel>
      {dialog ? (
        <ProjectDialog
          mode={dialog.mode}
          initialName={dialog.mode === "rename" ? dialog.card.name : ""}
          busy={working}
          error={dialogError}
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
