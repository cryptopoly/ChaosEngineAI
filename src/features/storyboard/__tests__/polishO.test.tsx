import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { StoryboardEstimate, StoryboardPassEstimate } from "../../../api/storyboard";
import type { StoryboardProject, StoryboardShot, StoryboardSong } from "../../../types";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import { PlanBrief } from "../PlanBrief";
import { PlanShotList } from "../PlanShotList";
import { RunBar, draftPassTitle, type TFn } from "../RunBar";
import { SongPanel } from "../song/SongPanel";
import { songLength, songLonger } from "../song/songLogic";
import type { UseStoryboardResult } from "../useStoryboard";

const noop = () => undefined;

function shot(id: string, extra: Partial<StoryboardShot> = {}): StoryboardShot {
  return {
    id, order: 0, title: "", status: "idle", stillPrompt: "", motionPrompt: "", referenceImages: [],
    startFrom: "generate", endImage: null, seconds: 4, seed: null, locked: false, trimIn: 0, trimOut: null,
    speed: 1, takes: [], activeTake: null, transition: { type: "cut", seconds: 0 }, nativeSoundDb: 0, redo: false,
    ...extra,
  };
}

function song(seconds: number): StoryboardSong {
  return {
    path: "song.wav", hash: null, title: "Harbour", seconds, bpm: 120, beats: [],
    sections: [{ start: 0, end: seconds, label: "A", energy: 0.5 }] as StoryboardSong["sections"],
    lyrics: ["first line"], clipId: "a1", libraryId: null, rough: false, note: "", analysedAt: 0,
  };
}

function project(extra: Partial<StoryboardProject> = {}): StoryboardProject {
  return {
    schemaVersion: 3, revision: 4, id: "0123456789abcdef", name: "Tea shop", createdAt: 0,
    defaults: { width: 1280, height: 720, fps: 24, imageModel: null, videoModel: null, voice: null, exportPreset: "16:9", resolution: "720p" },
    references: { characters: [], places: [], styles: [] }, shots: [], audio: [],
    export: { aiLabel: { embedTag: true }, loudnessLufs: -16, captionsSrt: false, captionsBurn: false, stems: false, upscale: "none" },
    captions: [], batches: [], undoStack: [], redoStack: [], ...extra,
  } as StoryboardProject;
}

function story(p: StoryboardProject): UseStoryboardResult {
  return {
    projectId: p.id, project: p, timeline: null, loading: false, error: null, notFound: false, notice: null,
    dismissNotice: noop, busy: false, batch: null, reload: async () => undefined, op: vi.fn(async () => true),
    opFrom: vi.fn(async () => true), undo: async () => undefined, redo: async () => undefined,
    startBatch: async () => true, cancelBatch: async () => undefined, call: vi.fn(async () => true),
  } as unknown as UseStoryboardResult;
}

const brief = (p: StoryboardProject) =>
  renderToStaticMarkup(<PlanBrief story={story(p)} project={p} imageGallery={[]} canPropose proposing={false} elapsed={0} modelName={null} onPropose={noop} onOpenCast={noop} />);

describe("Plan shot rows: one roving Tab stop (S-O item 2)", () => {
  it("makes only the first row a Tab stop and names every row by number and title", () => {
    const p = project({ shots: [shot("s1", { title: "Steam" }), shot("s2"), shot("s3")] });
    const html = renderToStaticMarkup(<PlanShotList story={story(p)} project={p} canAsk={false} onModelMissing={noop} />);
    const rows = html.match(/<li[^>]*class="plan-shot[^"]*"[^>]*>/g) ?? [];
    expect(rows).toHaveLength(3);
    expect(rows[0]).toContain('tabindex="0"');
    expect(rows[0]).toContain('aria-label="Shot 1 · Steam"');
    expect(rows[1]).toContain('tabindex="-1"');
    expect(rows[2]).toContain('aria-label="Shot 3"');
    expect(rows[1]).toContain('data-plan-shot-id="s2"');
  });
});

describe("A song longer than the film (S-O item 3)", () => {
  it("says so beside Time the lyrics and offers the cut plan there", () => {
    const p = project({ shots: [shot("s1"), shot("s2")], song: song(64) });
    const html = renderToStaticMarkup(<SongPanel story={story(p)} project={p} />);
    expect(html).toContain("The song runs 1:04; the shots add up to 0:08. Lines after 0:08 do not show until the film is that long.");
    const longer = html.slice(html.indexOf("song-lyrics__longer"));
    expect(longer).toMatch(/<button[^>]*>Make a cut plan<\/button>/);
  });

  it("says nothing when the shots cover the song", () => {
    const p = project({ shots: Array.from({ length: 16 }, (_, i) => shot(`s${i}`)), song: song(64) });
    expect(renderToStaticMarkup(<SongPanel story={story(p)} project={p} />)).not.toContain("song-lyrics__longer");
    expect(songLonger(64, 64.5)).toBe(false);
    expect(songLonger(64, undefined)).toBe(false);
    expect(songLonger(66, 64)).toBe(true);
  });
});

describe("Length and an attached song (S-O item 4)", () => {
  it("offers the song's length when the Length chip says otherwise", () => {
    const html = brief(project({ song: song(64), plan: { brief: "", filmSeconds: 15 } } as Partial<StoryboardProject>));
    expect(html).toContain("The song runs 1 min 4 s.");
    expect(html).toMatch(/<button[^>]*>Use the song&#x27;s length<\/button>/);
  });

  it("stays quiet on Any, on the same length, and without a song; a song length shows as its own chip", () => {
    expect(brief(project({ song: song(64), plan: { brief: "", filmSeconds: null } } as Partial<StoryboardProject>))).not.toContain("Use the song");
    const same = brief(project({ song: song(64.2), plan: { brief: "", filmSeconds: 64 } } as Partial<StoryboardProject>));
    expect(same).not.toContain("Use the song");
    expect(same).toMatch(/<span class="plan-chip is-active">1 min 4 s<\/span>/);
    expect(brief(project({ plan: { brief: "", filmSeconds: 15 } } as Partial<StoryboardProject>))).not.toContain("Use the song");
    expect(songLength(900)).toBe(600);
    expect(songLength(3)).toBeNull();
  });
});

describe("Make drafts tooltip (S-O item 5)", () => {
  const draft: StoryboardPassEstimate = {
    shots: 3, stills: 3, clips: 3, seconds: 45, stillSeconds: 45, clipSeconds: 150, basis: "estimated",
    imageModel: "Iris-3B Turbo", videoModel: "Vid", width: 640, height: 360,
  };
  const t: TFn = (_key, options) =>
    String(options.defaultValue).replace(/\{(\w+)\}/g, (_m, name: string) => String(options[name] ?? ""));

  it("names the draft model, its size and the time per draft", () => {
    expect(draftPassTitle(t, draft)).toBe(
      "Quick stills with Iris-3B Turbo at 640 x 360. Choose the shots you like for final, then run the final pass. About 15 s per draft.",
    );
    expect(draftPassTitle(t, { ...draft, stillSeconds: null })).not.toContain("per draft");
    expect(draftPassTitle(t, { ...draft, imageModel: null })).toBeUndefined();
  });

  it("gives the run bar the same words", () => {
    const estimate = {
      what: "both", shots: 3, stills: 3, clips: 3, seconds: 200, peakGb: null, totalGb: null, diskMb: null, basis: "estimated",
      missing: [], imageModel: "Qwen", videoModel: "Vid",
      passes: { draft, final: { ...draft, imageModel: "Qwen", width: 1280, height: 720 } },
    } as StoryboardEstimate;
    const html = renderToStaticMarkup(
      <RunBar missing={3} stills={0} stale={0} unlocked={3} estimate={estimate} batch={null} running={false} busy={false}
        primary="drafts" onMakeMissing={noop} onMakeAll={noop} onCancel={noop} onMakeDrafts={noop} onFinalPass={noop} />,
    );
    expect(html).toContain("About 15 s per draft.");
  });
});
