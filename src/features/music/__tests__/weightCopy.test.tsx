import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { MusicWeightCopyInfo, MusicWeightCopyStatus } from "../../../types";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import { MusicWeightCopyView } from "../MusicWeightCopyPanel";
import { INITIAL_GENERATION, classifyMusicError, copyMinutes, formatCopyGb, generationReducer } from "../musicLogic";

const GB = 1024 ** 3;
const INFO: MusicWeightCopyInfo = {
  needsCopy: true,
  fileCount: 6,
  totalBytes: 14.3 * GB,
  source: "E:\\AI Models\\hf\\snapshots",
  target: "C:\\Users\\me\\.chaosengine\\music\\checkpoints",
  sourceDrive: "E:",
  targetDrive: "C:",
};

function status(over: Partial<MusicWeightCopyStatus> = {}): MusicWeightCopyStatus {
  return {
    phase: "running", running: true, bytesDone: 0, totalBytes: INFO.totalBytes, fileCount: 6, filesDone: 0,
    currentFile: "", error: null, etaSeconds: null, ...over,
  };
}

function html(props: Partial<Parameters<typeof MusicWeightCopyView>[0]> = {}): string {
  return renderToStaticMarkup(
    <MusicWeightCopyView info={INFO} status={null} started={false} refusal={null} onCopy={() => {}} onCancel={() => {}} {...props} />,
  );
}

describe("copy now panel states", () => {
  it("offers the button with size and minutes, and says why and the two ways to avoid it", () => {
    const markup = html();
    expect(markup).toContain("Copy now (14.3 GB, about 2 min)");
    expect(markup).toContain("6 files (14.3 GB)");
    expect(markup).toContain("Developer Mode");
    expect(markup).toContain("same drive");
    expect(markup).not.toContain("<progress");
  });

  it("shows a progress bar with bytes, file and a cancel button while copying", () => {
    const markup = html({
      started: true,
      status: status({ bytesDone: 7.15 * GB, currentFile: "model.safetensors", etaSeconds: 120 }),
    });
    expect(markup).toContain('value="50"');
    expect(markup).toContain("7.2 of 14.3 GB");
    expect(markup).toContain("model.safetensors");
    expect(markup).toContain("about 2 min left");
    expect(markup).toContain(">Cancel<");
    expect(markup).not.toContain("Copy now");
  });

  it("says Done, loading once finished", () => {
    const markup = html({ started: true, status: status({ phase: "done", running: false, bytesDone: INFO.totalBytes }) });
    expect(markup).toContain("Done, loading");
    expect(markup).not.toContain("<progress");
    expect(markup).not.toContain(">Cancel<");
  });

  it("goes back to a button after a cancel and keeps the finished files", () => {
    const markup = html({ started: true, status: status({ phase: "cancelled", running: false, bytesDone: 3 * GB }) });
    expect(markup).toContain("Copy again (14.3 GB, about 2 min)");
    expect(markup).toContain("Finished files are kept");
  });

  it("shows a failed copy's message and a refusal (not enough space)", () => {
    const failed = html({ started: true, status: status({ phase: "error", running: false, error: "The copy stopped: disk gone" }) });
    expect(failed).toContain("The copy stopped: disk gone");
    expect(failed).toContain("Copy again");
    expect(html({ refusal: "Not enough free space on C:" })).toContain("Not enough free space on C:");
  });

  it("does not show a stale finished job from an earlier copy before the button is pressed", () => {
    const markup = html({ started: false, status: status({ phase: "done", running: false }) });
    expect(markup).toContain("Copy now");
    expect(markup).not.toContain("Done, loading");
  });
});

describe("copy facts in a failed load", () => {
  it("keeps the copy facts on a 409 and drops them otherwise", () => {
    const withCopy = classifyMusicError("Setting up the music model needs 6 weight file(s)", 409, INFO);
    expect(withCopy.kind).toBe("notReady");
    expect(withCopy.copy?.fileCount).toBe(6);
    expect(classifyMusicError("Not downloaded yet", 409).copy).toBeUndefined();
    const started = generationReducer(INITIAL_GENERATION, { type: "start", at: 1, form: {} as never });
    expect(generationReducer(started, { type: "failed", error: withCopy }).error?.copy?.totalBytes).toBe(INFO.totalBytes);
  });

  it("rounds the first time estimate to whole minutes, at least one", () => {
    expect(copyMinutes(14.3 * GB)).toBe(2);
    expect(copyMinutes(1)).toBe(1);
    expect(copyMinutes(60 * GB)).toBe(10);
    expect(formatCopyGb(14.3 * GB)).toBe("14.3");
  });
});
