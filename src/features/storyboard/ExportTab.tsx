/**
 * The Export tab of a project (D-65): the export settings and render, the result, the export for an editor,
 * the films list, and the project bundle. The panel itself is the one the Mixer used to hold; the preflight
 * and the estimate on the button come in a later phase.
 */
import { isTauri } from "@tauri-apps/api/core";
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { exportStoryboardOtio, fetchStoryboardFileSize } from "../../api";
import type { StoryboardExportPreset, StoryboardProject, StoryboardTimeline } from "../../types";
import { ExportPanel, type EditorExportState, type ExportResult } from "./mixer/ExportPanel";
import {
  aiLabelPatch,
  canRender,
  captionsExportPatch,
  hasClonedVoice,
  isRenderRunning,
  leftOutShotCount,
  presetPatch,
  exportSize,
  previewIsStale,
  readRenderedRevision,
} from "./mixer/mixerLogic";
import { upscaleSize } from "./formatLogic";
import { preflight, type PreflightLine } from "./overviewLogic";
import { copyProjectFilePath, downloadProjectBundle, formatBundleBytes, revealProjectFile } from "./projectFiles";
import type { ProjectTabId } from "./projectRoute";
import type { UseStoryboardResult } from "./useStoryboard";
import { useStoryboardRender, type UseStoryboardRenderResult } from "./useStoryboardRender";
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
}

export function ExportTab({ story, project, timeline, dataDirectory, onTab, render: shared }: ExportTabProps) {
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
  const lines = preflight(project, {
    leftOut: leftOutShotCount(timeline),
    previewStale: previewIsStale(rendered, project.revision),
    clonedVoice: hasClonedVoice(project),
    format: `${film.width} × ${film.height} · ${project.defaults.fps} fps`,
    upscaledTo: scaled ? `${scaled.width} × ${scaled.height}` : null,
  });
  const lineText = (line: PreflightLine): string => {
    switch (line.id) {
      case "clips":
        return line.ok
          ? t("exportTab.line.clipsAll", { defaultValue: "{n, plural, one {The shot has a clip.} other {All # shots have a clip.}}", n: line.total })
          : line.n === 0
            ? t("exportTab.line.clipsNone", { defaultValue: "No shot has a clip yet, so there is nothing to export." })
            : t("exportTab.line.clipsSome", {
                defaultValue: "{n} of {total} shots have a clip; the others are left out.",
                n: line.n,
                total: line.total,
              });
      case "stale":
        return t("exportTab.stale", {
          defaultValue: "{n, plural, one {# shot is stale.} other {# shots are stale.}} The export uses the takes you have.",
          n: line.n,
        });
      case "captions":
        if (line.n === 0) return t("exportTab.line.noCaptions", { defaultValue: "No captions." });
        return line.detail === "srt+burn"
          ? t("exportTab.line.captionsBoth", { defaultValue: "{n, plural, one {# caption} other {# captions}}, burnt in and written as an SRT file.", n: line.n })
          : line.detail === "srt"
            ? t("exportTab.line.captionsSrt", { defaultValue: "{n, plural, one {# caption} other {# captions}}, written as an SRT file.", n: line.n })
            : line.detail === "burn"
              ? t("exportTab.line.captionsBurn", { defaultValue: "{n, plural, one {# caption} other {# captions}}, burnt into the picture.", n: line.n })
              : t("exportTab.line.captionsUnused", { defaultValue: "{n, plural, one {# caption} other {# captions}} not exported: tick SRT file or Burn in below.", n: line.n });
      case "preview":
        return t("exportTab.line.preview", { defaultValue: "The preview is out of date (that does not stop an export)." });
      case "format":
        return t("exportTab.line.format", { defaultValue: "The film is {format}.", format: line.detail });
      case "upscale":
        return t("exportTab.line.upscale", { defaultValue: "It is scaled up to {size} at the end: more pixels, not more detail.", size: line.detail });
      case "loudness":
        return line.detail
          ? t("exportTab.line.loudness", { defaultValue: "The sound is levelled to {lufs} LUFS.", lufs: line.detail })
          : t("exportTab.line.loudnessOff", { defaultValue: "The sound is left as mixed (no loudness target)." });
      case "aiTag":
        return line.n
          ? t("exportTab.line.aiTagOn", { defaultValue: "A machine-readable AI tag is written into the file." })
          : t("exportTab.line.aiTagOff", { defaultValue: "No AI tag is written." });
      default:
        return t("exportTab.line.cloned", { defaultValue: "A cloned voice is used, so the AI tag is always written." });
    }
  };
  const revealLabel = isTauri() ? t("mixer.export.reveal", { defaultValue: "Reveal file" }) : t("mixer.export.download", { defaultValue: "Download" });

  return (
    <div className="export-tab">
      <section className="export-preflight" aria-labelledby="export-preflight-title">
        <h4 id="export-preflight-title">{t("exportTab.before", { defaultValue: "Before you export" })}</h4>
        <ul>
          {lines.map((line) => (
            <li key={line.id} className={`export-preflight__line${line.ok ? " is-ok" : " is-warn"}`}>
              <span className="export-preflight__mark" aria-hidden="true">
                {line.ok ? (
                  <svg viewBox="0 0 24 24" width="11" height="11" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><polyline points="5 12 10 17 19 7" /></svg>
                ) : (
                  "!"
                )}
              </span>
              <span>{lineText(line)}</span>
              {line.tab && !line.ok ? (
                <button type="button" className="mixer__link" onClick={() => onTab(line.tab as ProjectTabId)}>
                  {line.tab === "storyboard"
                    ? t("exportTab.openStoryboard", { defaultValue: "Open in Storyboard" })
                    : t("exportTab.openMixer", { defaultValue: "Open the Mixer" })}
                </button>
              ) : null}
            </li>
          ))}
        </ul>
      </section>
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
      <ExportPanel
        project={project}
        filmSeconds={timeline.total}
        empty={!canRender(timeline)}
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
        onLoudness={(lufs) => void story.op("edit_project", { export: { loudnessLufs: lufs } })}
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
