/**
 * Reveal or download a file of a project, and copy its path: shared by the Mixer (captions) and the Export tab.
 * In the desktop app the file is shown in its folder; in a browser it is downloaded.
 */
import { isTauri } from "@tauri-apps/api/core";
import { exportStoryboardBundle, fetchStoryboardFileBlobUrl, storyboardFilePath } from "../../api";
import { fileNameOf, projectFilePath } from "./mixer/mixerLogic";

/** Shows the file in its folder (desktop) or downloads it. Returns an error message, or null. */
export async function revealProjectFile(projectId: string, file: string, dataDirectory?: string): Promise<string | null> {
  const absolute = projectFilePath(dataDirectory, projectId, file);
  if (isTauri()) {
    try {
      const [{ revealItemInDir }, { homeDir }] = await Promise.all([import("@tauri-apps/plugin-opener"), import("@tauri-apps/api/path")]);
      const resolved = absolute.startsWith("~/") ? `${(await homeDir()).replace(/[\\/]+$/, "")}/${absolute.slice(2)}` : absolute;
      await revealItemInDir(resolved);
      return null;
    } catch {
      // fall through to the download
    }
  }
  try {
    const url = await fetchStoryboardFileBlobUrl(projectId, file);
    const link = document.createElement("a");
    link.href = url;
    link.download = fileNameOf(file);
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
    return null;
  } catch (err) {
    return err instanceof Error ? err.message : String(err);
  }
}

/** Copies the file's path on this machine; when the clipboard refuses, returns the API path to show instead. */
export async function copyProjectFilePath(projectId: string, file: string, dataDirectory?: string): Promise<string | null> {
  try {
    await navigator.clipboard.writeText(projectFilePath(dataDirectory, projectId, file));
    return null;
  } catch {
    return storyboardFilePath(projectId, file);
  }
}

/** Downloads a project's .chaosfilm bundle. Returns the file name and size. */
export async function downloadProjectBundle(projectId: string): Promise<{ name: string; bytes: number }> {
  const result = await exportStoryboardBundle(projectId);
  const url = await fetchStoryboardFileBlobUrl(projectId, result.file);
  const link = document.createElement("a");
  link.href = url;
  link.download = result.file.split("/").pop() ?? "project.chaosfilm";
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 10000);
  return { name: link.download, bytes: result.bytes };
}

export function formatBundleBytes(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(bytes / 1024))} KB`;
}
