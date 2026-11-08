import { describe, expect, it } from "vitest";
import type { StoryboardProject, StoryboardRenderJob, StoryboardTimeline, VoiceLibrarySummary } from "../../../../types";
import {
  captionRows,
  captionsExportPatch,
  resumeProgress,
  EXPORT_PRESET_ORDER,
  RENDER_PHASES,
  accumulateNudge,
  addAudioArgs,
  aiLabelPatch,
  aiLabelState,
  canRender,
  exportSize,
  formatBytes,
  hasClonedVoice,
  isRenderRunning,
  leftOutShotCount,
  mixerKeyAction,
  nudgeArgs,
  parseLoudness,
  presetPatch,
  previewIsStale,
  projectFilePath,
  readRenderedRevision,
  readoutDue,
  resolveLibraryPath,
  writeRenderedRevision,
  renderPercent,
  renderPhase,
  renderUiState,
  seekedTime,
  voiceLibraryChoices,
  type MixerKeyInput,
} from "../mixerLogic";

type Parts = Pick<StoryboardProject, "audio" | "export">;

function parts(cloned = false, embedTag = true): Parts {
  return {
    audio: [{ source: { clonedVoice: cloned } }] as unknown as StoryboardProject["audio"],
    export: { aiLabel: { embedTag }, loudnessLufs: -14, captionsSrt: false, captionsBurn: false, stems: false, upscale: "none" },
  };
}

describe("export presets", () => {
  it("gives the pixels each preset produces for a 1280 x 720 project", () => {
    const defaults = { width: 1280, height: 720 };
    expect(exportSize(defaults, "16:9")).toEqual({ width: 1280, height: 720 });
    expect(exportSize(defaults, "9:16")).toEqual({ width: 720, height: 1280 });
    expect(exportSize(defaults, "1:1")).toEqual({ width: 720, height: 720 });
  });

  it("works from the long side whichever way the project was made", () => {
    const portrait = { width: 720, height: 1280 };
    expect(exportSize(portrait, "16:9")).toEqual({ width: 1280, height: 720 });
    expect(exportSize(portrait, "9:16")).toEqual({ width: 720, height: 1280 });
    expect(exportSize({ width: 1024, height: 1024 }, "16:9")).toEqual({ width: 1024, height: 576 });
  });

  it("makes a preview about 360 px on the short side, never larger than the film", () => {
    expect(exportSize({ width: 1280, height: 720 }, "16:9", true)).toEqual({ width: 640, height: 360 });
    expect(exportSize({ width: 1280, height: 720 }, "9:16", true)).toEqual({ width: 360, height: 640 });
    expect(exportSize({ width: 256, height: 144 }, "16:9", true)).toEqual({ width: 256, height: 144 });
  });

  it("always gives even sizes", () => {
    for (const preset of EXPORT_PRESET_ORDER) {
      const { width, height } = exportSize({ width: 1000, height: 563 }, preset);
      expect(width % 2).toBe(0);
      expect(height % 2).toBe(0);
    }
  });

  it("changes only the export preset, so no shot is regenerated", () => {
    expect(presetPatch("9:16")).toEqual({ defaults: { exportPreset: "9:16" } });
  });
});

describe("AI label rules (D-34)", () => {
  it("is on by default", () => {
    expect(aiLabelState(parts())).toEqual({ metadata: true, forced: false });
  });

  it("a cloned voice locks the tag on whatever the switch says", () => {
    const state = aiLabelState(parts(true, false));
    expect(state).toEqual({ metadata: true, forced: true });
    expect(hasClonedVoice(parts(true))).toBe(true);
    expect(hasClonedVoice(parts(false))).toBe(false);
  });

  it("turns the tag off for private use when there is no cloned voice", () => {
    expect(aiLabelPatch(parts(), { metadata: false })).toEqual({ export: { aiLabel: { embedTag: false } } });
    expect(aiLabelState(parts(false, false)).metadata).toBe(false);
  });

  it("refuses to change the tag of a cloned-voice project", () => {
    expect(aiLabelPatch(parts(true), { metadata: false })).toBeNull();
  });

  it("sends nothing when nothing changed", () => {
    expect(aiLabelPatch(parts(), {})).toBeNull();
  });
});

describe("loudness field", () => {
  it("accepts a number in range and rounds to one decimal", () => {
    expect(parseLoudness("-16")).toBe(-16);
    expect(parseLoudness("-14,26")).toBe(-14.3);
    expect(parseLoudness(" -9 ")).toBe(-9);
  });

  it("clamps to the model's range", () => {
    expect(parseLoudness("-80")).toBe(-40);
    expect(parseLoudness("3")).toBe(-5);
  });

  it("empty means off and words mean not usable", () => {
    expect(parseLoudness("")).toBeNull();
    expect(parseLoudness("   ")).toBeNull();
    expect(parseLoudness("loud")).toBeUndefined();
  });
});

function job(extra: Partial<StoryboardRenderJob> = {}): StoryboardRenderJob {
  return { renderId: "r_1", projectId: "p", preview: true, state: "running", progress: 0.25, message: "shots", ...extra };
}

describe("render state", () => {
  it("maps the assembler stages onto the four phases", () => {
    expect(RENDER_PHASES).toEqual(["normalise", "join", "mix", "encode"]);
    expect(renderPhase("shots")).toBe("normalise");
    expect(renderPhase("card")).toBe("normalise");
    expect(renderPhase("join")).toBe("join");
    expect(renderPhase("measure")).toBe("mix");
    expect(renderPhase("mix")).toBe("encode");
    expect(renderPhase("")).toBe("normalise");
    expect(renderPhase(undefined)).toBe("normalise");
    expect(renderPhase("something new")).toBe("normalise");
  });

  it("reads the screen state from the job", () => {
    expect(renderUiState(null)).toBe("idle");
    expect(renderUiState(job())).toBe("running");
    expect(renderUiState(job({ state: "done" }))).toBe("done");
    expect(renderUiState(job({ state: "failed", error: "x" }))).toBe("failed");
    expect(renderUiState(job({ state: "cancelled" }))).toBe("cancelled");
    expect(renderUiState(null, true)).toBe("running");
    expect(isRenderRunning(job())).toBe(true);
    expect(isRenderRunning(job({ state: "done" }))).toBe(false);
    expect(isRenderRunning(null)).toBe(false);
  });

  it("shows progress as a whole percent inside 0 to 100", () => {
    expect(renderPercent(job({ progress: 0.256 }))).toBe(26);
    expect(renderPercent(job({ progress: 2 }))).toBe(100);
    expect(renderPercent(job({ progress: -1 }))).toBe(0);
    expect(renderPercent(null)).toBe(0);
  });

  it("calls a preview stale only when it is known to be from another revision", () => {
    expect(previewIsStale(null, 5)).toBe(false);
    expect(previewIsStale(5, 5)).toBe(false);
    expect(previewIsStale(4, 5)).toBe(true);
  });

  it("counts the shots a render leaves out and needs one ready shot", () => {
    const base = { total: 8, audio: [], problems: [] };
    const span = (ready: boolean) => ({ id: "x", start: 0, end: 4, duration: 4, source_in: 0, source_out: 4, transition_type: "cut", transition_seconds: 0, measured: true, ready });
    const none: StoryboardTimeline = { ...base, shots: [span(false), span(false)] };
    const some: StoryboardTimeline = { ...base, shots: [span(true), span(false)] };
    expect(canRender(none)).toBe(false);
    expect(canRender(some)).toBe(true);
    expect(canRender(null)).toBe(false);
    expect(leftOutShotCount(some)).toBe(1);
    expect(leftOutShotCount(null)).toBe(0);
  });

  it("formats file sizes and best-effort paths", () => {
    expect(formatBytes(900)).toBe("1 KB");
    expect(formatBytes(8_303_793)).toBe("7.9 MB");
    expect(formatBytes(3 * 1024 ** 3)).toBe("3 GB");
    expect(projectFilePath("", "abc", "exports/f.mp4")).toBe("~/.chaosengine/storyboards/abc/exports/f.mp4");
    expect(projectFilePath("/data/", "abc", "exports/f.mp4")).toBe("/data/storyboards/abc/exports/f.mp4");
  });
});

function key(extra: Partial<MixerKeyInput>): MixerKeyInput {
  return { key: "", shift: false, alt: false, mod: false, target: "other", selectedClipId: null, focusedClipId: null, fps: 24, ...extra };
}

describe("keyboard map", () => {
  it("Space and K play or pause", () => {
    expect(mixerKeyAction(key({ key: " " }))).toEqual({ type: "toggle-play" });
    expect(mixerKeyAction(key({ key: "k" }))).toEqual({ type: "toggle-play" });
  });

  it("J and L move 5 s", () => {
    expect(mixerKeyAction(key({ key: "j" }))).toEqual({ type: "seek", deltaS: -5 });
    expect(mixerKeyAction(key({ key: "L" }))).toEqual({ type: "seek", deltaS: 5 });
  });

  it("arrows move the playhead 1 s, Shift 10 s, Alt one frame", () => {
    expect(mixerKeyAction(key({ key: "ArrowRight" }))).toEqual({ type: "seek", deltaS: 1 });
    expect(mixerKeyAction(key({ key: "ArrowLeft", shift: true }))).toEqual({ type: "seek", deltaS: -10 });
    const frame = mixerKeyAction(key({ key: "ArrowRight", alt: true })) as { deltaS: number };
    expect(frame.deltaS).toBeCloseTo(1 / 24);
  });

  it("arrows nudge the selected clip 0.1 s, Shift 1 s, when a clip has focus", () => {
    expect(mixerKeyAction(key({ key: "ArrowRight", target: "clip", selectedClipId: "a", focusedClipId: "a" }))).toEqual({ type: "nudge-clip", deltaS: 0.1 });
    expect(mixerKeyAction(key({ key: "ArrowLeft", target: "clip", selectedClipId: "a", focusedClipId: "a", shift: true }))).toEqual({ type: "nudge-clip", deltaS: -1 });
    // a clip that is not selected, or focus elsewhere, moves the playhead
    expect(mixerKeyAction(key({ key: "ArrowRight", target: "clip", selectedClipId: null }))).toEqual({ type: "seek", deltaS: 1 });
    expect(mixerKeyAction(key({ key: "ArrowRight", target: "other", selectedClipId: "a", focusedClipId: "a" }))).toEqual({ type: "seek", deltaS: 1 });
  });

  it("Delete and the arrows act on the focused clip only when it is the selected one", () => {
    // clip M is selected, Tab moved focus to clip V
    const onOther = { target: "clip" as const, selectedClipId: "m", focusedClipId: "v" };
    expect(mixerKeyAction(key({ key: "Delete", ...onOther }))).toBeNull();
    expect(mixerKeyAction(key({ key: "Backspace", ...onOther }))).toBeNull();
    expect(mixerKeyAction(key({ key: "ArrowRight", ...onOther }))).toEqual({ type: "seek", deltaS: 1 });
    // a picture block has focus (no audio clip id)
    expect(mixerKeyAction(key({ key: "Delete", target: "clip", selectedClipId: "m", focusedClipId: null }))).toBeNull();
    expect(mixerKeyAction(key({ key: "ArrowLeft", target: "clip", selectedClipId: "m", focusedClipId: null }))).toEqual({ type: "seek", deltaS: -1 });
  });

  it("Home and End jump to the ends, + and - zoom", () => {
    expect(mixerKeyAction(key({ key: "Home" }))).toEqual({ type: "seek-to", where: "start" });
    expect(mixerKeyAction(key({ key: "End" }))).toEqual({ type: "seek-to", where: "end" });
    expect(mixerKeyAction(key({ key: "+" }))).toEqual({ type: "zoom", direction: 1 });
    expect(mixerKeyAction(key({ key: "=" }))).toEqual({ type: "zoom", direction: 1 });
    expect(mixerKeyAction(key({ key: "-" }))).toEqual({ type: "zoom", direction: -1 });
  });

  it("Delete removes only from a focused, selected clip", () => {
    expect(mixerKeyAction(key({ key: "Delete", target: "clip", selectedClipId: "a", focusedClipId: "a" }))).toEqual({ type: "remove-clip" });
    expect(mixerKeyAction(key({ key: "Backspace", target: "clip", selectedClipId: "a", focusedClipId: "a" }))).toEqual({ type: "remove-clip" });
    expect(mixerKeyAction(key({ key: "Delete", target: "other", selectedClipId: "a", focusedClipId: "a" }))).toBeNull();
    expect(mixerKeyAction(key({ key: "Delete", target: "clip", selectedClipId: null }))).toBeNull();
  });

  it("[ and ] trim the selected clip at the playhead", () => {
    expect(mixerKeyAction(key({ key: "[", selectedClipId: "a", focusedClipId: "a" }))).toEqual({ type: "trim", side: "in" });
    expect(mixerKeyAction(key({ key: "]", selectedClipId: "a", focusedClipId: "a" }))).toEqual({ type: "trim", side: "out" });
    expect(mixerKeyAction(key({ key: "[" }))).toBeNull();
  });

  it("Cmd/Ctrl+Z undoes and with Shift redoes; Escape clears the selection", () => {
    expect(mixerKeyAction(key({ key: "z", mod: true }))).toEqual({ type: "undo" });
    expect(mixerKeyAction(key({ key: "Z", mod: true, shift: true }))).toEqual({ type: "redo" });
    expect(mixerKeyAction(key({ key: "Escape" }))).toEqual({ type: "deselect" });
  });

  it("leaves every key to a form control or button that has focus", () => {
    for (const k of [" ", "ArrowLeft", "Home", "Delete", "z", "-", "j"]) {
      expect(mixerKeyAction(key({ key: k, target: "control", selectedClipId: "a", focusedClipId: "a", mod: k === "z" }))).toBeNull();
    }
  });

  it("ignores other keys and modified shortcuts it does not own", () => {
    expect(mixerKeyAction(key({ key: "a" }))).toBeNull();
    expect(mixerKeyAction(key({ key: "c", mod: true }))).toBeNull();
  });

  it("keeps a seek inside the film", () => {
    expect(seekedTime(1, -5, 20)).toBe(0);
    expect(seekedTime(18, 5, 20)).toBe(20);
    expect(seekedTime(5, 0.1, 20)).toBe(5.1);
  });
});

describe("adding sound", () => {
  const spans = [{ id: "s1", start: 0, end: 4 }, { id: "s2", start: 4, end: 8 }];

  it("places a new clip at the playhead, anchored to the shot under it", () => {
    expect(addAudioArgs("music", { path: " /a/song.wav ", name: "song.wav" }, 5, spans)).toEqual({
      kind: "music", path: "/a/song.wav", label: "song", anchor: { shotId: "s2", offset: 1 },
    });
  });

  it("falls back to an absolute start with no shots, and names the clip from its path", () => {
    expect(addAudioArgs("sfx", { path: "/a/b/gull.mp3" }, 2.5, [])).toEqual({ kind: "sfx", path: "/a/b/gull.mp3", label: "gull", start: 2.5 });
    expect(addAudioArgs("sfx", { path: "C:\\clips\\boom.wav" }, 0, []).label).toBe("boom");
  });

  it("marks a cloned voice only on the voice lane", () => {
    expect(addAudioArgs("voice", { path: "/v.wav" }, 0, spans, true).clonedVoice).toBe(true);
    expect(addAudioArgs("voice", { path: "/v.wav" }, 0, spans, false).clonedVoice).toBeUndefined();
    expect(addAudioArgs("music", { path: "/v.wav" }, 0, spans, true).clonedVoice).toBeUndefined();
  });

  it("offers Voice Library speech clips that have audio, with their file in the library folder", () => {
    const item = (extra: Partial<VoiceLibrarySummary>): VoiceLibrarySummary => ({
      id: "1_a", kind: "speech", title: "At dusk", preview: "", createdAt: 0, updatedAt: 0, audio: ".wav", speakers: {}, speakerCount: 0, ...extra,
    });
    const choices = voiceLibraryChoices([
      item({}),
      item({ id: "2_b", title: "", audio: ".mp3" }),
      item({ id: "3_c", audio: null }),
      item({ id: "4_d", kind: "transcript" }),
    ]);
    expect(choices).toEqual([
      { id: "1_a", title: "At dusk", path: "~/.chaosengine/voice/library/1_a.wav" },
      { id: "2_b", title: "2_b", path: "~/.chaosengine/voice/library/2_b.mp3" },
    ]);
  });
});

describe("nudges", () => {
  const audio = (start: number) => ({
    id: "m", kind: "music" as const,
    source: { path: "p.wav", hash: "h", measuredSeconds: 10, clonedVoice: false, label: "" },
    anchor: null, start, trimIn: 0, trimOut: null, gainDb: 0, fadeIn: 0, fadeOut: 0, duck: false, loop: false, status: "ok" as const,
  });

  it("sums presses on the same clip and hands back the old one when the clip changes", () => {
    let state = accumulateNudge(null, "m", 0.1);
    for (let i = 0; i < 4; i += 1) state = accumulateNudge(state.pending, "m", 0.1);
    expect(state.flush).toBeNull();
    expect(state.pending.delta).toBeCloseTo(0.5);
    const next = accumulateNudge(state.pending, "v", 0.1);
    expect(next.flush).toEqual(state.pending);
    expect(next.pending).toEqual({ clipId: "v", delta: 0.1 });
  });

  it("works out each send from the project as it is, so quick nudges add up to +0.8", () => {
    let project = { audio: [audio(2)] };
    // flush 1 (+0.5) lands, then flush 2 (+0.3) is built from the newest project
    for (const delta of [0.5, 0.3]) {
      const args = nudgeArgs(project, { clipId: "m", delta });
      expect(args).not.toBeNull();
      project = { audio: [audio((args as { start: number }).start)] };
    }
    expect(project.audio[0].start).toBeCloseTo(2.8);
  });

  it("skips a clip that is gone", () => {
    expect(nudgeArgs({ audio: [] }, { clipId: "m", delta: 0.1 })).toBeNull();
  });
});

describe("preview freshness and playhead readout", () => {
  it("remembers the rendered revision per project and survives a missing store", () => {
    const data = new Map<string, string>();
    const store = { getItem: (k: string) => data.get(k) ?? null, setItem: (k: string, v: string) => void data.set(k, v) };
    expect(readRenderedRevision("p1", store)).toBeNull();
    writeRenderedRevision("p1", 7, store);
    expect(readRenderedRevision("p1", store)).toBe(7);
    expect(readRenderedRevision("p2", store)).toBeNull();
    expect(previewIsStale(readRenderedRevision("p1", store), 9)).toBe(true);
    const broken = { getItem: () => { throw new Error("blocked"); }, setItem: () => { throw new Error("blocked"); } };
    expect(readRenderedRevision("p1", broken)).toBeNull();
    expect(() => writeRenderedRevision("p1", 1, broken)).not.toThrow();
    expect(readRenderedRevision("p1", null)).toBeNull();
  });

  it("publishes the number at most ten times a second", () => {
    expect(readoutDue(0, 99)).toBe(false);
    expect(readoutDue(0, 100)).toBe(true);
    expect(readoutDue(1000, 1050)).toBe(false);
  });
});

describe("library path", () => {
  it("uses the server's real path, and the default folder only as a last resort", () => {
    expect(resolveLibraryPath("/Volumes/data/voice/library/1_a.wav", "~/.chaosengine/voice/library/1_a.wav")).toBe("/Volumes/data/voice/library/1_a.wav");
    expect(resolveLibraryPath(null, "~/x.wav")).toBe("~/x.wav");
    expect(resolveLibraryPath("  ", "~/x.wav")).toBe("~/x.wav");
  });
});

describe("captions, stems and resuming", () => {
  it("builds the export patch only for what changed", () => {
    expect(captionsExportPatch({})).toBeNull();
    expect(captionsExportPatch({ burn: true })).toEqual({ export: { captionsBurn: true } });
    expect(captionsExportPatch({ srt: false, stems: true })).toEqual({ export: { captionsSrt: false, stems: true } });
  });

  it("places a clip's captions on the timeline by the clip's start and trim, and flags trimmed-away ones", () => {
    const tl = { total: 10, shots: [], problems: [], audio: [{ id: "v1", start: 4, duration: 3, trim_in: 1, end: 7 }] } as unknown as StoryboardTimeline;
    const rows = captionRows(
      {
        captions: [
          { id: "late", clipId: "v1", start: 2, end: 3, text: "b" },
          { id: "gone", clipId: "v1", start: 0, end: 0.5, text: "x" },
          { id: "hand", clipId: null, start: 1, end: 2, text: "a" },
          { id: "cut", clipId: "v1", start: 0.5, end: 1.5, text: "c" },
        ],
      },
      tl,
    );
    expect(rows.map((r) => r.id)).toEqual(["hand", "cut", "late", "gone"]);
    expect(rows.find((r) => r.id === "late")).toMatchObject({ start: 5, end: 6, fromClip: true });
    expect(rows.find((r) => r.id === "cut")).toMatchObject({ start: 4, end: 4.5 });
    expect(rows.find((r) => r.id === "gone")).toMatchObject({ start: null, end: null });
  });

  it("reports how many shots an interrupted render keeps, and nothing for a running or finished one", () => {
    const job = { renderId: "r", projectId: "p", preview: false, state: "interrupted", progress: 0.3, message: "", resumable: true, shotsReady: 5, shotsTotal: 4 } as StoryboardRenderJob;
    expect(resumeProgress(job)).toEqual([4, 4]);
    expect(resumeProgress({ ...job, state: "running" })).toBeNull();
    expect(resumeProgress({ ...job, state: "done" })).toBeNull();
    expect(resumeProgress({ ...job, resumable: false })).toBeNull();
    expect(resumeProgress(null)).toBeNull();
  });
});
