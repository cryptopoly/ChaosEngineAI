/**
 * The Export tab of a project (D-65): the export settings and render, the result, the export for an editor,
 * the films list, and the project bundle. The panel itself is the one the Mixer used to hold. Above it sits
 * "Ready to export?" (audit change 6): the guide's three groups, from `exportReadiness.ts`.
 */
import { isTauri } from "@tauri-apps/api/core";
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { exportStoryboardOtio, fetchStoryboardFileSize } from "../../api";
import type { StoryboardExportPreset, StoryboardGuide, StoryboardProject, StoryboardTimeline } from "../../types";
import { ExportPanel, type EditorExportState, type ExportResult } from "./mixer/ExportPanel";
import {
  aiLabelPatch,
  canRender,
  captionsExportPatch,
  hasClonedVoice,
  isRenderRunning,
  presetPatch,
  exportSize,
  exportedSeconds,
  filmCaptionCount,
  previewIsStale,
  readRenderedRevision,
} from "./mixer/mixerLogic";
import { upscaleSize } from "./formatLogic";
import {
  exportAllowed,
  exportAnyway,
  localLines,
  readinessGroups,
  stopReason,
  type ReadinessFix,
} from "./exportReadiness";
import { ReadinessCheck } from "./ReadinessCheck";
import { useExportGuide, useUpscaleAddon } from "./useExportChecks";
import { upscaleMethodOf, upscaleMethodPatch } from "./mixer/upscaleMethod";
import { copyProjectFilePath, downloadProjectBundle, formatBundleBytes, revealProjectFile } from "./projectFiles";
import type { ProjectTabId } from "./projectRoute";
import type { UseStoryboardResult } from "./useStoryboard";
import { useStoryboardRender, type UseStoryboardRenderResult } from "./useStoryboardRender";
import { RoughCutButton } from "./preview/RoughCutButton";
import { BrandCardsPanel } from "./BrandCardsPanel";
import { brandSeconds } from "./preview/brandCards";
import { VideoToolsPrompt } from "../videotools/VideoToolsPrompt";
import { useVideoTools } from "../videotools/useVideoTools";

export interface ExportTabProps {
  story: UseStoryboardResult;
  project: StoryboardProject;
  timeline: StoryboardTimeline;
  dataDirectory?: string;
  onTab: (tab: ProjectTabId) => void;
  /** The shell's render state, shared with the Mixer, so the last export and a running clock survive tab switches. */
  render?: UseStoryboardRenderResult;
  /** The payload's guide block, when the shell holds it; otherwise the tab reads it (`GET .../guide`) itself. */
  guide?: StoryboardGuide | null;
  /** Opens one shot's editor on the Storyboard tab; without it a shot fix opens the Storyboard. */
  onOpenShot?: (shotId: string) => void;
  /** Opens the app's Upscale tab (the add-on's install lives there); without it the note has no button. */
  onOpenUpscaleTab?: () => void;
}

export function ExportTab({ story, project, timeline, dataDirectory, onTab, render: shared, guide: givenGuide, onOpenShot, onOpenUpscaleTab }: ExportTabProps) {
  const { t } = useTranslation("studio");
  const own = useStoryboardRender(project.id, !shared, () => void story.reload());
  const render = shared ?? own;
  const videoTools = useVideoTools(true, () => render.clearError());
  const [exportBytes, setExportBytes] = useState<{ id: string; bytes: number | null } | null>(null);
  const [copied, setCopied] = useState(false);
  const [editorExport, setEditorExport] = useState<EditorExportState>({ busy: false, file: null, warnings: [], error: null });
  const [bundle, setBundle] = useState<{ busy: boolean; status: string | null; error: string | null }>({ busy: false, status: null, error: null });
  const [announce, setAnnounce] = useState("");

  const exportedId = render.exported?.renderId ?? null;
  const exportedFile = render.exported?.file ?? null;
  useEffect(() => {
    if (!exportedId || !exportedFile) return;
    let cancelled = false;
    fetchStoryboardFileSize(project.id, exportedFile)
      .then((bytes) => !cancelled && setExportBytes({ id: exportedId, bytes }))
      .catch(() => !cancelled && setExportBytes({ id: exportedId, bytes: null }));
    return () => {
      cancelled = true;
    };
  }, [project.id, exportedId, exportedFile]);

  async function reveal(file: string) {
    const problem = await revealProjectFile(project.id, file, dataDirectory);
    if (problem) setAnnounce(problem);
  }

  async function copyPath(file: string) {
    const fallback = await copyProjectFilePath(project.id, file, dataDirectory);
    if (fallback) {
      setAnnounce(fallback);
      return;
    }
    setCopied(true);
    setAnnounce(t("mixer.export.copied", { defaultValue: "Path copied" }));
    setTimeout(() => setCopied(false), 2000);
  }

  async function writeOtio() {
    setEditorExport({ busy: true, file: null, warnings: [], error: null });
    try {
      const done = await exportStoryboardOtio(project.id);
      setEditorExport({ busy: false, file: done.file, warnings: done.warnings, error: null });
    } catch (err) {
      setEditorExport({ busy: false, file: null, warnings: [], error: err instanceof Error ? err.message : String(err) });
    }
  }

  async function exportBundle() {
    setBundle({ busy: true, status: null, error: null });
    try {
      const done = await downloadProjectBundle(project.id);
      setBundle({
        busy: false,
        status: t("storyProjects.exported", { defaultValue: "Exported {name} · {size}", name: done.name, size: formatBundleBytes(done.bytes) }),
        error: null,
      });
    } catch (err) {
      setBundle({ busy: false, status: null, error: err instanceof Error ? err.message : String(err) });
    }
  }

  const rendering = isRenderRunning(render.job) || render.starting;
  const exportJob = render.job && !render.job.preview ? render.job : null;
  const exportResult: ExportResult | null = render.exported
    ? { job: render.exported, bytes: exportBytes?.id === render.exported.renderId ? exportBytes.bytes : null }
    : null;
  const rendered = readRenderedRevision(project.id);
  const film = exportSize(project.defaults, project.defaults.exportPreset);
  const scaled = upscaleSize(project.defaults.exportPreset, project.export.upscale, film);
  const guide = useExportGuide(project.id, project.revision, givenGuide);
  const readiness = guide?.readiness ?? null;
  const method = upscaleMethodOf(project);
  const addonInstalled = useUpscaleAddon(project.export.upscale !== "none");
  const groups = readinessGroups(
    readiness,
    localLines(project, {
      upscaledTo: scaled ? `${scaled.width} × ${scaled.height}` : null,
      method,
      previewStale: previewIsStale(rendered, project.revision),
      clonedVoice: hasClonedVoice(project),
      captions: filmCaptionCount(project, timeline),
    }),
  );
  const allowed = exportAllowed(readiness, canRender(timeline));
  function openFix(fix: ReadinessFix) {
    if (fix.kind === "shot" && fix.shotId && onOpenShot) onOpenShot(fix.shotId);
    else onTab(fix.tab);
  }
  const revealLabel = isTauri() ? t("mixer.export.reveal", { defaultValue: "Reveal file" }) : t("mixer.export.download", { defaultValue: "Download" });

  return (
    <div className="export-tab">
      <ReadinessCheck
        project={project}
        groups={groups}
        actions={<RoughCutButton project={project} timeline={timeline} quality={guide?.quality} />}
        onFix={openFix}
      />
      <VideoToolsPrompt
        status={videoTools.status}
        busy={videoTools.busy}
        percent={videoTools.percent}
        error={videoTools.error}
        forced={render.needsVideoTools}
        onInstall={() => void videoTools.install()}
        onChooseFile={(path) => void videoTools.chooseFile(path)}
        onDismissError={videoTools.clearError}
      />
      {render.error && !render.needsVideoTools ? (
        <div className="callout warning" role="alert">
          <p>{render.error}</p>
          <button type="button" className="secondary-button" onClick={render.clearError}>
            {t("storyboard.dismiss", { defaultValue: "Dismiss" })}
          </button>
        </div>
      ) : null}
      <BrandCardsPanel
        project={project}
        disabled={story.busy || rendering}
        onChange={(change) => story.op("edit_project", { export: change })}
        onOpenCast={() => onTab("cast")}
      />
      <ExportPanel
        project={project}
        filmSeconds={exportedSeconds(timeline) + (canRender(timeline) ? brandSeconds(project) : 0)}
        empty={!allowed}
        emptyReason={stopReason(readiness)}
        exportAnyway={exportAnyway(readiness)}
        upscaleMethod={method}
        onUpscaleMethod={(next) => void story.op("edit_project", upscaleMethodPatch(next))}
        upscaleAddonInstalled={addonInstalled}
        onOpenUpscaleTab={onOpenUpscaleTab}
        rendering={rendering}
        exportJob={exportJob}
        startedAt={render.startedAt}
        result={exportResult}
        disabled={story.busy}
        copied={copied}
        revealLabel={revealLabel}
        onPreset={(preset: StoryboardExportPreset) => void story.op("edit_project", presetPatch(preset))}
        onAiLabel={(change) => {
          const patch = aiLabelPatch(project, change);
          if (patch) void story.op("edit_project", patch);
        }}
        onUpscale={(upscale) => void story.op("edit_project", { export: { upscale } })}
        onLoudness={(lufs) => story.op("edit_project", { export: { loudnessLufs: lufs } })}
        onExport={() => void render.start(project.revision, false)}
        onCaptionOptions={(change) => {
          const patch = captionsExportPatch(change);
          if (patch) void story.op("edit_project", patch);
        }}
        editor={editorExport}
        onEditorExport={() => void writeOtio()}
        onCancel={() => void render.cancel()}
        onReveal={(file) => void reveal(file)}
        onCopyPath={(file) => void copyPath(file)}
      />
      <section className="export-tab__bundle">
        <h4>{t("exportTab.bundleTitle", { defaultValue: "Project bundle" })}</h4>
        <p className="muted-text">
          {t("exportTab.bundleBody", {
            defaultValue: "One .chaosfilm file with the shots, takes, cast pictures and sound, to back up the project or open it on another computer.",
          })}
        </p>
        <button type="button" className="secondary-button" disabled={bundle.busy} onClick={() => void exportBundle()}>
          {bundle.busy
            ? t("storyProjects.exporting", { defaultValue: "Exporting…" })
            : t("storyProjects.exportBundle", { defaultValue: "Export bundle" })}
        </button>
        {bundle.status ? <p className="muted-text" role="status">{bundle.status}</p> : null}
        {bundle.error ? <p className="error-text" role="alert">{bundle.error}</p> : null}
      </section>
      <p className="sr-only" role="status" aria-live="polite">{announce}</p>
    </div>
  );
}
