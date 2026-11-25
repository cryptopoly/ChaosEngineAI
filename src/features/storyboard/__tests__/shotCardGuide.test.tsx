import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type {
  StoryboardBatchRecord,
  StoryboardFailure,
  StoryboardGuide,
  StoryboardLipSyncRow,
  StoryboardProject,
  StoryboardShot,
  StoryboardTake,
} from "../../../types";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import type { StoryboardEstimate } from "../../../api/storyboard";
import { BatchBar } from "../BatchBar";
import { lipSyncInUse, shotCardGuide, syncChip, type ShotCardGuide } from "../cardGuide";
import type { TFn } from "../qualityText";
import { RunBar, type RunBarProps } from "../RunBar";
import { ShotCard, type ShotCardProps } from "../ShotCard";
import { TakesStrip } from "../TakesStrip";

const noop = () => undefined;
const t: TFn = (_key, options) =>
  String(options.defaultValue).replace(/\{(\w+)\}/g, (_m, name: string) => String(options[name] ?? ""));

function take(id: string, extra: Partial<StoryboardTake> = {}): StoryboardTake {
  return { id, still: "media/s.png", clip: null, hashes: {}, settings: {}, measuredSeconds: 4, createdAt: 0, origin: "batch", basis: null, ...extra };
}
function shot(extra: Partial<StoryboardShot> = {}): StoryboardShot {
  return {
    id: "s_1", order: 0, title: "Arrival", status: "idle", stillPrompt: "", motionPrompt: "", referenceImages: [],
    startFrom: "generate", endImage: null, seconds: 4, seed: null, locked: false, trimIn: 0, trimOut: null,
    speed: 1, takes: [], activeTake: null, transition: { type: "cut", seconds: 0 }, nativeSoundDb: 0, redo: false,
    ...extra,
  };
}
function card(guide: ShotCardGuide | undefined, overrides: Partial<ShotCardProps> = {}) {
  const props: ShotCardProps = {
    projectId: "abc", shot: shot(), index: 1, count: 3, status: "idle", staleAfter: null, kept: false, failure: null,
    tabbable: true, dragOver: false, disabled: false, guide,
    onRename: noop, onPickTake: noop, onToggleLock: noop, onToggleRedo: noop, onTransition: noop, onMove: noop,
    onEdit: noop, onKeep: noop, onRerun: noop, onRetry: noop, onRemove: noop, onDragStart: noop, onDragEnd: noop,
    onDropOn: noop, onDragOverCard: noop,
    ...overrides,
  };
  return renderToStaticMarkup(<ShotCard {...props} />);
}

const row = (extra: Partial<StoryboardLipSyncRow> = {}): StoryboardLipSyncRow => ({
  shotId: "s_1", number: 2, songStart: 3, songEnd: 7, seconds: 4, state: "unassigned", code: "not-assigned", message: "backend",
  fix: "assign-vocals", minSeconds: 2, maxSeconds: 7.6, ...extra,
});
const failure = (code: string, action: string, extra: Partial<StoryboardFailure> = {}) =>
  ({ shotId: "s_1", kind: "clip", code, message: "backend sentence", action, error: "CUDA error: out of memory", batchId: "b1", ...extra }) as StoryboardFailure;

const draftTake = take("t1", { clip: "media/c.mp4", renderPass: "draft", settings: { stillWidth: 640, stillHeight: 384, clipWidth: 640, clipHeight: 384, clipFps: 24 } });

describe("quality tag on the shot card (change 2)", () => {
  it("tags a draft picture with its size and a plain explanation", () => {
    const html = card({ target: { width: 1280, height: 720 } }, { status: "done", shot: shot({ takes: [draftTake], activeTake: "t1" }) });
    expect(html).toContain(">Draft 640 x 384</span>");
    expect(html).toContain("The film is 1280 x 720");
    // The tag sits over the picture, so it also covers the clip player.
    expect(html.indexOf("quality-tag--corner")).toBeLessThan(html.indexOf("shot-card__play"));
  });

  it("prefers the guide's take label over the take's own settings", () => {
    const guide: ShotCardGuide = {
      takes: { t1: { pass: "draft", draft: true, still: { width: 704, height: 416 }, clip: null, stillStale: false, clipStale: false } },
    };
    expect(card(guide, { status: "done", shot: shot({ takes: [take("t1", { renderPass: "draft" })], activeTake: "t1" }) })).toContain(">Draft 704 x 416</span>");
  });

  it("gives a final take no tag, and a draft with no recorded size just Draft", () => {
    expect(card({}, { status: "done", shot: shot({ takes: [take("t1")], activeTake: "t1" }) })).not.toContain("quality-tag");
    expect(card(undefined, { status: "done", shot: shot({ takes: [take("t1", { renderPass: "draft" })], activeTake: "t1" }) })).toContain(">Draft</span>");
  });

  it("tags every draft take in the strip, with the size in its tooltip", () => {
    const html = renderToStaticMarkup(
      <TakesStrip projectId="abc" takes={[draftTake, take("t2")]} activeId="t2" onPick={noop} target={{ width: 1280, height: 720 }} />,
    );
    expect(html.match(/quality-tag--mini/g)).toHaveLength(1);
    expect(html).toContain("Take 1 · Draft 640 x 384");
    expect(html).toContain("made at 640 x 384");
  });
});

describe("failures on the shot card (change 5)", () => {
  it("shows the plain sentence and the one action, with the raw text behind Details", () => {
    const html = card({ failure: failure("out-of-memory", "smaller-size"), draftSize: { width: 640, height: 384 } }, {
      status: "failed", failure: "CUDA error: out of memory", onFailureAction: noop,
    });
    expect(html).toContain("There was not enough free memory to make this shot at this size.");
    expect(html).toContain("Make it at draft size (640 x 384)");
    expect(html).not.toContain("Try again or edit the prompts");
    expect(html.match(/shot-card__failed-actions"><button/g)).toHaveLength(1);
    expect(html).toMatch(/<details class="debug-details"><summary>Details<\/summary><pre>CUDA error: out of memory<\/pre>/);
    expect(html).not.toContain("Technical details");
  });

  it("offers Open Video Models for a missing video model, Install video tools, or Edit the prompts", () => {
    const open = card({ failure: failure("model-missing", "open-models", { kind: "clip" }) }, { status: "failed", onFailureAction: noop });
    expect(open).toContain("Open Video Models");
    const stillModel = card({ failure: failure("model-missing", "open-models", { kind: "still" }) }, { status: "failed", onFailureAction: noop });
    expect(stillModel).toContain("Open Image Models");
    expect(card({ failure: failure("video-tools-missing", "install-video-tools") }, { status: "failed", onFailureAction: noop })).toContain("Install video tools");
    expect(card({ failure: failure("no-prompt", "edit-shot") }, { status: "failed" })).toContain("Edit the prompts");
  });

  it("falls back to Retry when the screen has no handler for the action, and for an unknown failure", () => {
    const noHandler = card({ failure: failure("out-of-memory", "smaller-size") }, { status: "failed" });
    expect(noHandler).toContain(">Retry</button>");
    expect(noHandler).not.toContain("Make it at draft size");
    const unknown = card(undefined, { status: "failed", failure: "boom" });
    expect(unknown).toContain("The shot did not render.");
    expect(unknown).toContain("<pre>boom</pre>");
  });

  it("holds Retry on a locked shot but not Edit", () => {
    const retry = card({ failure: failure("cancelled", "retry") }, { status: "failed", shot: shot({ locked: true }) });
    expect(retry).toMatch(/<button type="button" class="secondary-button" disabled="">Retry/);
    const edit = card({ failure: failure("no-prompt", "edit-shot") }, { status: "failed", shot: shot({ locked: true }) });
    expect(edit).toMatch(/<button type="button" class="secondary-button">Edit the prompts/);
  });
});

describe("keeps (change 11)", () => {
  it("calls the pass button Choose for final and the stale button something that is not a keep", () => {
    const html = card({}, { status: "done", shot: shot({ takes: [take("t1")], activeTake: "t1", kept: true }), onToggleKept: noop });
    expect(html).toContain("Choose for final");
    expect(html).not.toContain("Keep for final");
    const stale = card({}, { status: "stale", shot: shot({ status: "stale", startFrom: "previous-shot" }), staleAfter: 0 });
    expect(stale).toContain("Hide this warning");
    expect(stale).not.toContain("Keep as is");
  });
});

describe("lip sync and character chips on the card (changes 3 and 4)", () => {
  it("says Sings for a synced shot and Will sing while its clip is waiting", () => {
    expect(syncChip(t, row({ state: "synced", code: null }))).toMatchObject({ text: "Sings", tone: "sings" });
    expect(syncChip(t, row({ state: "waiting", code: null }))).toMatchObject({ text: "Will sing", tone: "sings" });
    const html = card({ sync: row({ state: "synced", code: null }) });
    expect(html).toContain(">Sings</li>");
  });

  it("shows a quiet No lip sync chip with the reason for every skip reason", () => {
    const reasons: Record<string, string> = {
      "other-speed": "other speed", locked: "locked", "outside-song": "outside the song", "too-short": "too short",
      "too-long": "too long", "no-singing": "no singing here", "not-assigned": "no vocal yet",
    };
    for (const [code, word] of Object.entries(reasons)) {
      const chip = syncChip(t, row({ state: code === "not-assigned" ? "unassigned" : "skipped", code: code as StoryboardLipSyncRow["code"] }));
      expect(chip?.text).toBe(`No lip sync: ${word}`);
      expect(chip?.tone).toBe("quiet");
      expect(chip?.title).toContain("ordinary clip");
    }
    expect(syncChip(t, row({ state: "skipped", code: "too-short", minSeconds: 2 }))?.title).toContain("at least 2 s");
    expect(syncChip(t, row({ state: "skipped", code: "too-long", maxSeconds: 7.6 }))?.title).toContain("at most 7.6 s");
    expect(syncChip(t, null)).toBeNull();
    expect(card({ sync: row({ state: "skipped", code: "too-short" }) })).toContain("shot-card__chip--quiet");
  });

  it("shows No character when the guide says the shot has none, and nothing otherwise", () => {
    expect(card({ noCharacter: true })).toContain(">No character</li>");
    expect(card({ noCharacter: false })).not.toContain("No character");
    expect(card(undefined)).not.toContain("shot-card__tags");
  });
});

describe("shotCardGuide", () => {
  const project = {
    id: "p", references: { characters: [{ id: "c1", name: "Mara" }], places: [{ id: "pl1", name: "Roof" }] },
    shots: [shot({ id: "s_1" }), shot({ id: "s_2" })],
  } as unknown as StoryboardProject;
  const coverage = (source: "stem" | "song" | null, sings: number) => ({
    source, at: null, counts: { synced: sings, waiting: 0, skipped: 0, unassigned: 1, total: 2, sings },
    shots: [row({ shotId: "s_1", state: "synced", code: null }), row({ shotId: "s_2" })],
  });
  const guide = (extra: Partial<StoryboardGuide> = {}) => ({
    quality: { target: { width: 1280, height: 720, fps: 24 }, draftSize: { width: 640, height: 384 }, draftShots: [], staleShots: [], takes: { s_1: { t1: draftQuality() } } },
    coverage: null, failures: [failure("cancelled", "retry", { shotId: "s_2" })],
    readiness: {}, stages: {}, cast: { s_1: ["c1"], s_2: ["pl1"] }, ...extra,
  }) as unknown as StoryboardGuide;
  function draftQuality() {
    return { pass: "draft", draft: true, still: { width: 640, height: 384 }, clip: null, stillStale: false, clipStale: false };
  }

  it("cuts one shot's part out of the guide", () => {
    const one = shotCardGuide(project, guide(), project.shots[0]);
    expect(one.takes?.t1.draft).toBe(true);
    expect(one.target).toEqual({ width: 1280, height: 720, fps: 24 });
    expect(one.draftSize).toEqual({ width: 640, height: 384 });
    expect(one.failure).toBeNull();
    expect(shotCardGuide(project, guide(), project.shots[1]).failure?.code).toBe("cancelled");
  });

  it("calls a shot without a character only when the project has one and the effective cast has none", () => {
    expect(shotCardGuide(project, guide(), project.shots[0]).noCharacter).toBe(false);
    expect(shotCardGuide(project, guide(), project.shots[1]).noCharacter).toBe(true); // only a place
    const none = { ...project, references: { characters: [], places: [] } } as unknown as StoryboardProject;
    expect(shotCardGuide(none, guide({ cast: { s_1: [], s_2: [] } }), none.shots[0]).noCharacter).toBe(false);
    // An older payload without cast: unknown, so no chip.
    expect(shotCardGuide(project, guide({ cast: undefined as never }), project.shots[0]).noCharacter).toBe(false);
  });

  it("shows lip sync chips only when lip sync is in use", () => {
    expect(lipSyncInUse(guide({ coverage: null }))).toBe(false);
    expect(lipSyncInUse(guide({ coverage: coverage(null, 0) as never }))).toBe(false);
    expect(lipSyncInUse(guide({ coverage: coverage("stem", 0) as never }))).toBe(true);
    expect(lipSyncInUse(guide({ coverage: coverage(null, 1) as never }))).toBe(true);
    expect(shotCardGuide(project, guide({ coverage: coverage(null, 0) as never }), project.shots[0]).sync).toBeNull();
    expect(shotCardGuide(project, guide({ coverage: coverage("stem", 1) as never }), project.shots[0]).sync?.state).toBe("synced");
  });

  it("is empty without a guide", () => {
    expect(shotCardGuide(project, undefined, project.shots[0])).toEqual({});
  });
});

describe("run bar: sizes, one primary, More ways to make (changes 2 and 11)", () => {
  const passes = {
    draft: { shots: 3, stills: 3, clips: 3, seconds: 240, stillSeconds: 240, clipSeconds: 600, basis: "estimated" as const, imageModel: "Fast", videoModel: "Vid", width: 640, height: 384 },
    final: { shots: 3, stills: 3, clips: 3, seconds: 2280, stillSeconds: 900, clipSeconds: 1380, basis: "estimated" as const, imageModel: "Best", videoModel: "Vid", width: 1280, height: 720, assumedAllKept: false },
  };
  const estimate = (extra: Partial<StoryboardEstimate> = {}): StoryboardEstimate => ({
    what: "both", shots: 3, stills: 3, clips: 3, seconds: 2280, peakGb: 12, totalGb: 24, diskMb: 40, basis: "estimated", missing: [],
    imageModel: "Best", videoModel: "Vid", passes, ...extra,
  });
  const quality = { target: { width: 1280, height: 720, fps: 24 }, draftSize: { width: 640, height: 384 }, draftShots: [], staleShots: [], takes: {} };
  const bar = (extra: Partial<RunBarProps> = {}) =>
    renderToStaticMarkup(
      <RunBar missing={3} stills={0} stale={0} unlocked={3} estimate={estimate()} batch={null} running={false} busy={false}
        onMakeMissing={noop} onMakeAll={noop} onCancel={noop} onMakeDrafts={noop} onFinalPass={noop} quality={quality} {...extra} />,
    );

  it("states the draft-or-final choice in sizes and time", () => {
    const html = bar();
    expect(html).toContain("Drafts first: 640 x 384 · about 4 min · half size, to choose the shots you like. You then make those full size.");
    expect(html).toContain("Straight to final: 1280 x 720 · about 38 min · peak 12 GB of 24 GB");
  });

  it("has one primary button and puts the other ways under More ways to make", () => {
    const html = bar();
    expect(html.match(/class="primary-button"/g)).toHaveLength(1);
    const [visible, more] = html.split('<details class="run-bar__more">');
    expect(visible).toContain("Make 3 missing · about 38 min");
    expect(visible).not.toContain("Make all unlocked");
    expect(visible).not.toContain("Make drafts");
    expect(more).toContain("More ways to make");
    expect(more).toContain("Make all unlocked");
    expect(more).toContain("Make drafts");
  });

  it("makes Make drafts the one primary when told so", () => {
    const html = bar({ primary: "drafts" });
    expect(html.match(/class="primary-button"/g)).toHaveLength(1);
    const [visible, more] = html.split('<details class="run-bar__more">');
    expect(visible).toMatch(/class="primary-button"[^>]*>Make drafts · about 4 min</);
    expect(more).toContain("Make 3 missing");
    expect(more).not.toContain("primary-button");
  });

  it("makes the final pass primary when nothing is missing and shots are chosen, and says its size", () => {
    const html = bar({ missing: 0, kept: 2, finalShots: 2 });
    expect(html.match(/class="primary-button"/g)).toHaveLength(1);
    expect(html.split('<details class="run-bar__more">')[0]).toMatch(/class="primary-button"[^>]*>Final pass on 2 chosen shots/);
    expect(html).toContain("The final pass makes the chosen shots again at 1280 x 720, then their clips.");
    expect(bar({ missing: 0, kept: 0, finalShots: 0 })).not.toContain("primary-button");
  });

  it("keeps the memory acknowledgement reachable when Make drafts is the primary", () => {
    const html = bar({ primary: "drafts", estimate: estimate({ peakGb: 23.5, totalGb: 24 }) });
    expect(html.split('<details class="run-bar__more">')[1]).toContain("Run anyway.");
    expect(html.split('<details class="run-bar__more">')[1]).toMatch(/disabled="">Make 3 missing/);
  });
});

describe("batch bar failures (change 5)", () => {
  const batch = (results: StoryboardBatchRecord["results"]) =>
    ({ id: "b", what: "both", force: false, state: "running", createdAt: 0, finishedAt: null, results }) as StoryboardBatchRecord;

  it("says how many shots did not render while the run goes on", () => {
    const html = renderToStaticMarkup(
      <BatchBar
        batch={batch([
          { shotId: "a", kind: "still", state: "failed", takeId: null, error: "x" },
          { shotId: "a", kind: "clip", state: "failed", takeId: null, error: "x" },
          { shotId: "b", kind: "still", state: "running", takeId: null, error: null },
        ])}
        onCancel={noop}
      />,
    );
    expect(html).toContain("1 shot did not render. Its card says why and what to do.");
  });

  it("stays quiet when nothing failed", () => {
    const html = renderToStaticMarkup(<BatchBar batch={batch([{ shotId: "a", kind: "still", state: "running", takeId: null, error: null }])} onCancel={noop} />);
    expect(html).not.toContain("did not render");
  });
});
