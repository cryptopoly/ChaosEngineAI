import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { FfmpegStatus } from "../../../api/ffmpeg";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import { StoryboardApiError, errorCodeOf } from "../../../api/storyboard";
import { VideoToolsPromptView } from "../VideoToolsPrompt";
import { VideoToolsRowView } from "../VideoToolsRow";
import {
  canRemove,
  formatMb,
  isMissingVideoTools,
  lastLine,
  promptPhase,
  sourceLabel,
} from "../videoToolsLogic";

const job = { phase: "idle", percent: 0, error: null, done: false, running: false, startedAt: 0 };

function status(extra: Partial<FfmpegStatus> = {}): FfmpegStatus {
  return {
    installed: false, path: null, source: null, version: null, licence: null, encoder: null,
    pin: null, canDownload: false, job, ...extra,
  };
}

const noop = () => undefined;

describe("promptPhase", () => {
  it("is hidden while unknown or installed", () => {
    expect(promptPhase(null, false, null)).toBe("hidden");
    expect(promptPhase(status({ installed: true }), false, null)).toBe("hidden");
  });
  it("asks when missing, or when a render just failed for want of ffmpeg", () => {
    expect(promptPhase(status(), false, null)).toBe("ask");
    expect(promptPhase(status({ installed: true }), false, null, true)).toBe("ask");
  });
  it("shows install progress and then the failure", () => {
    expect(promptPhase(status(), true, null)).toBe("installing");
    expect(promptPhase(status(), false, "no network")).toBe("failed");
  });
});

describe("helpers", () => {
  it("reads the install code from an API error", () => {
    const err = new StoryboardApiError("Video tools are needed", 409, "install-ffmpeg");
    expect(isMissingVideoTools(err)).toBe(true);
    expect(isMissingVideoTools(new StoryboardApiError("busy", 409))).toBe(false);
    expect(isMissingVideoTools(new Error("x"))).toBe(false);
    expect(errorCodeOf({ detail: { code: "install-ffmpeg", message: "m" } })).toBe("install-ffmpeg");
    expect(errorCodeOf({ detail: "text" })).toBeNull();
    expect(errorCodeOf(null)).toBeNull();
  });
  it("labels sources and decides what can be removed", () => {
    expect(sourceLabel("download").key).toBe("download");
    expect(sourceLabel("nope").key).toBe("unknown");
    expect(canRemove(status({ source: "download" }))).toBe(true);
    expect(canRemove(status({ source: "custom" }))).toBe(true);
    expect(canRemove(status({ source: "imageio" }))).toBe(false);
    expect(canRemove(null)).toBe(false);
  });
  it("formats sizes and trims a pip log", () => {
    expect(formatMb(168715328)).toBe("169");
    expect(lastLine("a\n\nERROR: no matching distribution\n")).toBe("ERROR: no matching distribution");
    expect(lastLine("x".repeat(300)).length).toBe(200);
  });
});

describe("VideoToolsPromptView", () => {
  const render = (phase: "hidden" | "ask" | "installing" | "failed", extra: { error?: string | null; percent?: number | null } = {}) =>
    renderToStaticMarkup(
      <VideoToolsPromptView phase={phase} error={extra.error ?? null} percent={extra.percent ?? null}
        onInstall={noop} onChooseFile={noop} onDismissError={noop} />,
    );

  it("renders nothing when hidden", () => {
    expect(render("hidden")).toBe("");
  });
  it("asks with one install button and the offline file field", () => {
    const html = render("ask");
    expect(html).toContain("Video tools are needed");
    expect(html.match(/Install video tools/g)).toHaveLength(1);
    expect(html).toContain("Offline? Use an ffmpeg file you already have");
    expect(html).not.toContain("<progress");
  });
  it("disables the button and shows progress while installing", () => {
    const html = render("installing", { percent: 40 });
    expect(html).toContain("Installing video tools...");
    expect(html).toContain("disabled");
    expect(html).toContain("<progress");
  });
  it("shows the failure message", () => {
    const html = render("failed", { error: "no network" });
    expect(html).toContain("Video tools did not install: no network");
    expect(html).toContain('role="alert"');
  });
});

describe("VideoToolsRowView", () => {
  const render = (s: FfmpegStatus | null, extra: { busy?: boolean; percent?: number | null; error?: string | null } = {}) =>
    renderToStaticMarkup(
      <VideoToolsRowView status={s} busy={extra.busy ?? false} percent={extra.percent ?? null} error={extra.error ?? null}
        onInstall={noop} onDownload={noop} onCancel={noop} onRemove={noop} />,
    );

  it("shows not installed with an install button", () => {
    const html = render(status());
    expect(html).toContain("Video tools (FFmpeg)");
    expect(html).toContain("Not installed");
    expect(html).toContain("Install video tools");
  });
  it("shows version, source, licence and encoder when installed", () => {
    const html = render(status({ installed: true, source: "imageio", version: "7.1", licence: "GPL", encoder: "libx264" }));
    expect(html).toContain("7.1 from imageio-ffmpeg package (GPL)");
    expect(html).toContain("H.264 encoder: libx264");
    expect(html).not.toContain("Install video tools");
    expect(html).not.toContain("Remove");
  });
  it("offers the pinned LGPL download only when there is a pin", () => {
    const pin = { url: "https://x/y", sha256: "a", size: 168715328, version: "9", licenceUrl: "https://l", sourceUrl: "https://s" };
    const html = render(status({ installed: true, source: "imageio", canDownload: true, pin }));
    expect(html).toContain("Download LGPL build (169 MB)");
    expect(html).toContain("https://l");
    expect(render(status({ installed: true, source: "download", canDownload: true, pin }))).not.toContain("Download LGPL build");
    expect(render(status({ installed: true, source: "imageio" }))).not.toContain("Download LGPL build");
  });
  it("offers Remove for a download and shows errors and progress", () => {
    expect(render(status({ installed: true, source: "download", version: "9.0.2" }))).toContain("Remove");
    const html = render(status(), { busy: true, percent: 12, error: "disk full" });
    expect(html).toContain("<progress");
    expect(html).toContain("Cancel");
    expect(html).toContain("disk full");
  });
  it("shows a checking state before the first answer", () => {
    expect(render(null)).toContain("Checking...");
  });
});
