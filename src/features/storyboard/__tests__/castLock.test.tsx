import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { StoryboardPlannedShot, StoryboardProject, StoryboardReference, StoryboardShot } from "../../../types";
import { CAST_OUT, IN_EVERY_SHOT, PLANNED_CHARACTERS } from "../../../types/storyboard";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import { castInShot, plannedShotFromProposal, startsInEveryShot, toggleCastOut } from "../castLock";
import { usedByCount } from "../pickerLogic";
import { ReferenceCard } from "../ReferenceCard";
import { ShotEditorDrawer } from "../ShotEditorDrawer";
import { draftFromShot, shotPatchFromDraft } from "../storyboardLogic";

const noop = () => undefined;

function ref(id: string, extra: Partial<StoryboardReference> = {}): StoryboardReference {
  return { id, name: id.toUpperCase(), description: "", images: [], ...extra };
}

function shot(id: string, extra: Partial<StoryboardShot> = {}): StoryboardShot {
  return {
    id, order: 0, title: "", status: "idle", stillPrompt: "", motionPrompt: "", referenceImages: [],
    startFrom: "generate", endImage: null, seconds: 4, seed: null, locked: false, trimIn: 0, trimOut: null,
    speed: 1, takes: [], activeTake: null, transition: { type: "cut", seconds: 0 }, nativeSoundDb: 0, redo: false,
    ...extra,
  };
}

function project(extra: Partial<StoryboardProject> = {}): StoryboardProject {
  return {
    schemaVersion: 3, revision: 1, id: "0123456789abcdef", name: "Song", createdAt: 0,
    defaults: { width: 1280, height: 720, fps: 24, imageModel: null, videoModel: null, voice: null, exportPreset: "16:9", resolution: "720p" },
    references: { characters: [], places: [], styles: [] }, shots: [], audio: [],
    export: {}, captions: [], batches: [], undoStack: [], redoStack: [], ...extra,
  } as unknown as StoryboardProject;
}

const song = { path: "media/song.flac", title: "Song", seconds: 180 } as StoryboardProject["song"];

describe("character lock: the switch default", () => {
  it("starts the first character of a song project in every shot", () => {
    expect(startsInEveryShot(project({ song }))).toBe(true);
  });

  it("leaves it off without a song, and for the second character", () => {
    expect(startsInEveryShot(project())).toBe(false);
    expect(startsInEveryShot(project({ song: null }))).toBe(false);
    const withOne = project({ song, references: { characters: [ref("c1")], places: [], styles: [] } });
    expect(startsInEveryShot(withOne)).toBe(false);
  });

  it("shows the switch with its explanation on a character card", () => {
    const html = renderToStaticMarkup(
      <ReferenceCard
        projectId="p" reference={ref("c1")} usedBy={3} disabled={false}
        onEdit={async () => true} onRemove={noop} onRemovePicture={noop} onOpenPicker={noop} onFiles={noop}
        inEveryShot onInEveryShot={noop}
      />,
    );
    expect(html).toContain('role="switch"');
    expect(html).toMatch(/role="switch"[^>]*checked=""/);
    expect(html).toContain("In every shot");
    expect(html).toContain("A shot can still leave them out");
  });

  it("shows no switch on a place card", () => {
    const html = renderToStaticMarkup(
      <ReferenceCard
        projectId="p" reference={ref("p1")} usedBy={0} disabled={false}
        onEdit={async () => true} onRemove={noop} onRemovePicture={noop} onOpenPicker={noop} onFiles={noop}
      />,
    );
    expect(html).not.toContain('role="switch"');
  });

  it("counts every shot that does not leave an in-every-shot character out", () => {
    const shots = [shot("a"), shot("b", { [CAST_OUT]: ["c1"] }), shot("c", { referenceImages: ["c1"] })];
    expect(usedByCount({ shots }, "c1", true)).toBe(2);
    expect(usedByCount({ shots }, "c1")).toBe(1);
  });
});

describe("character lock: leaving a character out of one shot", () => {
  const lead = ref("c1", { [IN_EVERY_SHOT]: true });
  const friend = ref("c2");

  it("puts in-every-shot characters in the shot unless it leaves them out", () => {
    expect(castInShot([lead, friend], { referenceImages: [], leftOut: [] }).map((c) => c.id)).toEqual(["c1"]);
    expect(castInShot([lead, friend], { referenceImages: ["c2"], leftOut: [] }).map((c) => c.id)).toEqual(["c1", "c2"]);
    expect(castInShot([lead, friend], { referenceImages: ["c1"], leftOut: ["c1"] }).map((c) => c.id)).toEqual([]);
  });

  it("toggles a character in and out of the leave-out list", () => {
    expect(toggleCastOut([], "c1")).toEqual(["c1"]);
    expect(toggleCastOut(["c1", "c2"], "c1")).toEqual(["c2"]);
  });

  it("saves only a changed leave-out list", () => {
    const s = shot("a", { [CAST_OUT]: ["c1"] });
    const draft = draftFromShot(s);
    expect(draft.leftOut).toEqual(["c1"]);
    expect(shotPatchFromDraft(s, draft)).toEqual({});
    expect(shotPatchFromDraft(s, { ...draft, leftOut: [] })).toEqual({ [CAST_OUT]: [] });
    expect(shotPatchFromDraft(shot("b"), { ...draftFromShot(shot("b")), leftOut: ["c1"] })).toEqual({ [CAST_OUT]: ["c1"] });
  });

  it("shows who is in the shot, the leave-out control and the lip-sync note in the shot editor", () => {
    const p = project({ references: { characters: [lead, friend], places: [], styles: [] } });
    const render = (s: StoryboardShot) =>
      renderToStaticMarkup(
        <ShotEditorDrawer
          project={p} shot={s} index={0} imageGallery={[]} videoGallery={[]}
          busy={false} batchRunning={false} op={async () => true} onMake={noop} onClose={noop}
        />,
      );
    const html = render(shot("a"));
    expect(html).toContain("In this shot: C1");
    expect(html).toContain("Leave out of this shot");
    expect(html).toContain("Lip sync is set for each shot on its own");
    const out = render(shot("a", { [CAST_OUT]: ["c1"] }));
    expect(out).toContain("No character in this shot.");
  });
});

describe("character lock: applying a plan proposal (PlanTab.use)", () => {
  const p = project({ references: { characters: [ref("c1"), ref("c2")], places: [ref("p1")], styles: [] } });

  it("carries each proposed shot's characters onto the shot as its references", () => {
    const proposed: StoryboardPlannedShot = { title: "Verse", stillPrompt: "A singer", motionPrompt: "sways", seconds: 4, [PLANNED_CHARACTERS]: ["c1", "c2"] };
    expect(plannedShotFromProposal(proposed, p)).toEqual({
      title: "Verse", stillPrompt: "A singer", motionPrompt: "sways", seconds: 4, referenceImages: ["c1", "c2"],
    });
  });

  it("drops ids the project does not have and repeats, and keeps a shot without characters plain", () => {
    const proposed: StoryboardPlannedShot = {
      title: "", stillPrompt: "", motionPrompt: "", seconds: 3, referenceImages: ["p1"], [PLANNED_CHARACTERS]: ["c1", "ghost", "c1"],
    };
    expect(plannedShotFromProposal(proposed, p).referenceImages).toEqual(["p1", "c1"]);
    expect(plannedShotFromProposal({ title: "", stillPrompt: "", motionPrompt: "", seconds: 3 }, p)).not.toHaveProperty("referenceImages");
  });

  it("forwards the characters a proposed shot leaves out, only ones the project has", () => {
    const proposed: StoryboardPlannedShot = {
      title: "", stillPrompt: "", motionPrompt: "", seconds: 3, [CAST_OUT]: ["c2", "ghost", "c2"],
    };
    expect(plannedShotFromProposal(proposed, p)[CAST_OUT]).toEqual(["c2"]);
    expect(plannedShotFromProposal({ title: "", stillPrompt: "", motionPrompt: "", seconds: 3 }, p)).not.toHaveProperty(CAST_OUT);
  });
});
