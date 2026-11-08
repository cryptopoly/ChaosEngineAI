import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { StoryboardRevisionConflict } from "../../../api/storyboard";
import type { StoryboardProject, StoryboardProjectPayload } from "../../../types";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import { SendToStoryboardMenu } from "../../../components/SendToStoryboardMenu";
import {
  NEW_SHOT,
  applyPlan,
  destinationTab,
  menuKeyAction,
  newShotId,
  pickRecentProjects,
  planSend,
  projectNameFor,
  resolveStep,
  sendIntoProject,
  sendMenuVisible,
  shotTitleFrom,
  targetsFor,
  type PlannedStep,
  type SendApi,
  type SendItem,
  type SendProject,
} from "../sendToStoryboardLogic";

const noShots = { shots: [] };
const twoShots = { shots: [{ id: "s_1" }, { id: "s_2" }] } as unknown as Pick<StoryboardProject, "shots">;

const image: SendItem = {
  kind: "image",
  path: "/out/lamp.png",
  prompt: "A lighthouse keeper lights the lamp at dusk, warm light on the stone",
};
const clip: SendItem = { kind: "video", path: "/out/storm.mp4", prompt: "Storm rolls in", seconds: 4.04 };
const voice: SendItem = { kind: "voice", path: "/lib/abc.wav", label: "Some nights the sea is quiet" };

describe("planSend", () => {
  it("makes a new shot for an image: prompt and a 40-character title, then the still as its take", () => {
    const steps = planSend(image, noShots, { type: "new-shot" });
    expect(steps).toEqual([
      { op: "add_shot", args: { title: "A lighthouse keeper lights the lamp at d", stillPrompt: image.prompt } },
      { op: "import_take", args: { shotId: NEW_SHOT, still: "/out/lamp.png" } },
    ]);
    expect((steps[0].args as { title: string }).title.length).toBeLessThanOrEqual(40);
  });

  it("adds the image as a new take of an existing shot, and nothing else", () => {
    expect(planSend(image, twoShots, { type: "new-take", shotId: "s_2" })).toEqual([
      { op: "import_take", args: { shotId: "s_2", still: "/out/lamp.png" } },
    ]);
  });

  it("makes a new shot for a clip with its length, then the clip as the take", () => {
    expect(planSend(clip, noShots, { type: "new-shot" })).toEqual([
      { op: "add_shot", args: { title: "Storm rolls in", stillPrompt: "Storm rolls in", seconds: 4.04 } },
      { op: "import_take", args: { shotId: NEW_SHOT, clip: "/out/storm.mp4" } },
    ]);
  });

  it("keeps a clip length inside what the server accepts, and leaves it out when unknown", () => {
    const long = planSend({ ...clip, seconds: 300 }, noShots, { type: "new-shot" })[0].args as { seconds?: number };
    const tiny = planSend({ ...clip, seconds: 0.1 }, noShots, { type: "new-shot" })[0].args as { seconds?: number };
    const unknown = planSend({ ...clip, seconds: null }, noShots, { type: "new-shot" })[0].args as { seconds?: number };
    expect(long.seconds).toBe(60);
    expect(tiny.seconds).toBe(0.5);
    expect(unknown.seconds).toBeUndefined();
  });

  it("adds a clip as a new take of an existing shot", () => {
    expect(planSend(clip, twoShots, { type: "new-take", shotId: "s_1" })).toEqual([
      { op: "import_take", args: { shotId: "s_1", clip: "/out/storm.mp4" } },
    ]);
  });

  it("anchors a voice clip to the first shot at offset 0", () => {
    expect(planSend(voice, twoShots, { type: "audio" })).toEqual([
      { op: "add_audio", args: { kind: "voice", path: "/lib/abc.wav", label: "Some nights the sea is quiet", anchor: { shotId: "s_1", offset: 0 } } },
    ]);
  });

  it("starts a voice clip at 0 s when the project has no shots", () => {
    const [step] = planSend(voice, noShots, { type: "audio" });
    expect(step.args).toMatchObject({ kind: "voice", start: 0 });
    expect(step.args).not.toHaveProperty("anchor");
  });

  it("sends music and effects to their own lanes", () => {
    expect(planSend({ kind: "music", path: "/m.wav" }, twoShots, { type: "audio" })[0].args).toMatchObject({ kind: "music" });
    expect(planSend({ kind: "sfx", path: "/s.wav" }, twoShots, { type: "audio" })[0].args).toMatchObject({ kind: "sfx" });
  });

  it("refuses to plan without a file", () => {
    expect(() => planSend({ kind: "image" }, noShots, { type: "new-shot" })).toThrow();
  });

  it("names a new project from the prompt, then the label, then the fallback", () => {
    expect(projectNameFor(image, "Untitled storyboard")).toBe("A lighthouse keeper lights the lamp at d");
    expect(projectNameFor(voice, "Untitled storyboard")).toBe("Some nights the sea is quiet");
    expect(projectNameFor({ kind: "music" }, "Untitled storyboard")).toBe("Untitled storyboard");
    expect(shotTitleFrom("  one \n two  ")).toBe("one two");
  });

  it("plans into a freshly made project the same way (no shots yet)", () => {
    const steps = planSend(image, { shots: [] }, { type: "new-shot" });
    expect(steps.map((step) => step.op)).toEqual(["add_shot", "import_take"]);
  });
});

function payload(revision: number, shotIds: string[]): StoryboardProjectPayload {
  return {
    project: { revision, shots: shotIds.map((id) => ({ id })) } as unknown as StoryboardProject,
    timeline: { total: 0, shots: [], audio: [], problems: [] },
  };
}

describe("applyPlan", () => {
  const steps: PlannedStep[] = planSend(image, noShots, { type: "new-shot" });

  it("runs the steps in order, each with the revision the last answer left, and names the new shot", async () => {
    const calls: { op: string; revision: number; shotId?: string }[] = [];
    const run = vi.fn(async (step: PlannedStep, revision: number) => {
      calls.push({ op: step.op, revision, shotId: (step.args as { shotId?: string }).shotId });
      return step.op === "add_shot" ? payload(revision + 1, ["s_new"]) : payload(revision + 1, ["s_new"]);
    });
    const done = await applyPlan({ revision: 4, shots: [] }, steps, run);
    expect(calls).toEqual([
      { op: "add_shot", revision: 4, shotId: undefined },
      { op: "import_take", revision: 5, shotId: "s_new" },
    ]);
    expect(done.revision).toBe(6);
  });

  it("retries a step once with the fresh revision after a conflict", async () => {
    let first = true;
    const seen: number[] = [];
    const run = async (step: PlannedStep, revision: number) => {
      seen.push(revision);
      if (first) {
        first = false;
        throw new StoryboardRevisionConflict("changed", payload(9, []).project);
      }
      return payload(revision + 1, ["s_new"]);
    };
    const done = await applyPlan({ revision: 4, shots: [] }, [steps[0]], run);
    expect(seen).toEqual([4, 9]);
    expect(done.revision).toBe(10);
  });

  it("gives up after a second conflict and raises it", async () => {
    const run = async () => {
      throw new StoryboardRevisionConflict("changed", payload(9, []).project);
    };
    await expect(applyPlan({ revision: 4, shots: [] }, [steps[0]], run)).rejects.toBeInstanceOf(StoryboardRevisionConflict);
  });

  it("does not retry other failures", async () => {
    const run = vi.fn(async () => {
      throw new Error("File not found");
    });
    await expect(applyPlan({ revision: 4, shots: [] }, [steps[0]], run)).rejects.toThrow("File not found");
    expect(run).toHaveBeenCalledTimes(1);
  });

  it("names the new shot by the id that appeared, not by its position", async () => {
    // the server puts the new shot first, so "the last shot" would be somebody else's
    const calls: (string | undefined)[] = [];
    const run = async (step: PlannedStep, revision: number) => {
      calls.push((step.args as { shotId?: string }).shotId);
      return payload(revision + 1, ["s_new", "s_old"]);
    };
    await applyPlan({ revision: 4, shots: [{ id: "s_old" }] as never }, steps, run);
    expect(calls).toEqual([undefined, "s_new"]);
  });

  it("finds the new id after a conflict retry too", async () => {
    let first = true;
    const calls: (string | undefined)[] = [];
    const run = async (step: PlannedStep, revision: number) => {
      calls.push((step.args as { shotId?: string }).shotId);
      if (first) {
        first = false;
        throw new StoryboardRevisionConflict("changed", payload(9, ["s_other"]).project);
      }
      return payload(revision + 1, ["s_new", "s_other"]);
    };
    await applyPlan({ revision: 4, shots: [] }, steps, run);
    expect(calls).toEqual([undefined, undefined, "s_new"]);
  });

  it("resolves a step with the created id, and refuses when there is none", () => {
    expect(resolveStep(steps[1], "s_2").args).toMatchObject({ shotId: "s_2" });
    expect(newShotId(["a", "b"], ["x", "a", "b"])).toBe("x");
    expect(newShotId(["a"], ["a"])).toBeNull();
    expect(() => resolveStep(steps[1], null)).toThrow();
  });
});

describe("projects and targets", () => {
  const rows = [
    { id: "a", createdAt: 10 },
    { id: "b", createdAt: 40 },
    { id: "c", createdAt: 30 },
    { id: "d", createdAt: 20 },
    { id: "bad", createdAt: 99, error: "damaged" },
  ];

  it("lists the three newest readable projects", () => {
    expect(pickRecentProjects(rows, null).map((row) => row.id)).toEqual(["b", "c", "d"]);
  });

  it("puts the open project first", () => {
    expect(pickRecentProjects(rows, "a").map((row) => row.id)).toEqual(["a", "b", "c"]);
  });

  const project: SendProject = { id: "p", name: "Lighthouse", shots: [{ id: "s_1", title: "Arrival", number: 1 }, { id: "s_2", title: "Lamp", number: 2 }] };

  it("offers a new shot and a new take per shot for pictures and clips", () => {
    expect(targetsFor(image, project)).toEqual([{ type: "new-shot" }, { type: "new-take", shotId: "s_1" }, { type: "new-take", shotId: "s_2" }]);
  });

  it("offers one target for sound", () => {
    expect(targetsFor(voice, project)).toEqual([{ type: "audio" }]);
  });

  it("opens the project's Storyboard tab for pictures and clips and its Mixer tab for sound", () => {
    expect(destinationTab("image")).toBe("storyboard");
    expect(destinationTab("video")).toBe("storyboard");
    expect(destinationTab("voice")).toBe("mixer");
    expect(destinationTab("music")).toBe("mixer");
  });
});

describe("menu key model", () => {
  it("moves with the arrows and wraps", () => {
    expect(menuKeyAction("ArrowDown", -1, 3)).toEqual({ type: "focus", index: 0 });
    expect(menuKeyAction("ArrowDown", 2, 3)).toEqual({ type: "focus", index: 0 });
    expect(menuKeyAction("ArrowUp", 0, 3)).toEqual({ type: "focus", index: 2 });
    expect(menuKeyAction("ArrowUp", 2, 3)).toEqual({ type: "focus", index: 1 });
  });

  it("jumps with Home and End and closes with Escape and Tab", () => {
    expect(menuKeyAction("Home", 2, 3)).toEqual({ type: "focus", index: 0 });
    expect(menuKeyAction("End", 0, 3)).toEqual({ type: "focus", index: 2 });
    expect(menuKeyAction("Escape", 1, 3)).toEqual({ type: "close" });
    expect(menuKeyAction("Tab", 1, 3)).toEqual({ type: "close" });
    expect(menuKeyAction("a", 1, 3)).toBeNull();
    expect(menuKeyAction("ArrowDown", 0, 0)).toBeNull();
  });
});

describe("SendToStoryboardMenu", () => {
  const projects: SendProject[] = [
    { id: "p1", name: "Lighthouse keeper", shots: [{ id: "s_1", title: "Arrival", number: 1 }, { id: "s_2", title: "Lamp", number: 2 }] },
    { id: "p2", name: "Coffee shop ad", shots: [] },
  ];
  const noop = () => undefined;

  it("sendMenuVisible needs the flag and a file", () => {
    expect(sendMenuVisible(false, image)).toBe(false);
    expect(sendMenuVisible(true, null)).toBe(false);
    expect(sendMenuVisible(true, { kind: "image" })).toBe(false);
    expect(sendMenuVisible(true, image)).toBe(true);
    expect(sendMenuVisible(true, voice)).toBe(true);
    expect(sendMenuVisible(true, { kind: "voice", resolvePath: async () => "/x.wav" })).toBe(true);
  });

  it("renders nothing when the Home Studio areas are switched off", () => {
    expect(renderToStaticMarkup(<SendToStoryboardMenu item={image} projects={projects} enabled={false} onSend={noop} />)).toBe("");
  });

  it("shows 'Use in storyboard' for pictures and clips and 'Send to storyboard' for sound", () => {
    const html = renderToStaticMarkup(<SendToStoryboardMenu item={image} projects={projects} enabled onSend={noop} />);
    expect(html).toContain("send-to-menu");
    expect(html).toContain("Use in storyboard");
    expect(html).toContain('aria-haspopup="menu"');
    expect(html).toContain('aria-expanded="false"');
    expect(html).not.toContain("send-to-menu__list");
    expect(renderToStaticMarkup(<SendToStoryboardMenu item={voice} projects={projects} enabled onSend={noop} />)).toContain("Send to storyboard");
  });

  it("reads 'Start a storyboard with this' when there is no project and opens no list", () => {
    const html = renderToStaticMarkup(<SendToStoryboardMenu item={image} projects={[]} loaded enabled onSend={noop} />);
    expect(html).toContain("Start a storyboard with this");
    expect(html).not.toContain("aria-haspopup");
  });

  it("lists the projects and 'New project…' as menu items when open", () => {
    const html = renderToStaticMarkup(<SendToStoryboardMenu item={image} projects={projects} enabled initialOpen onSend={noop} />);
    expect(html).toContain('role="menu"');
    expect(html).toContain('aria-expanded="true"');
    expect(html.match(/role="menuitem"/g)?.length).toBe(3);
    expect(html).toContain("Lighthouse keeper");
    expect(html).toContain("Coffee shop ad");
    expect(html).toContain("New project…");
  });

  it("lists New shot and one 'n · title (new take)' per shot once a project is chosen", () => {
    const html = renderToStaticMarkup(
      <SendToStoryboardMenu item={clip} projects={projects} enabled initialOpen initialProjectId="p1" onSend={noop} />,
    );
    expect(html).toContain("Back to projects");
    expect(html).toContain("New shot");
    expect(html).toContain("1 · Arrival (new take)");
    expect(html).toContain("2 · Lamp (new take)");
    expect(html).toContain('title="Lamp"');
  });

  it("shows a failure notice only after a failed send from this menu", () => {
    const html = renderToStaticMarkup(<SendToStoryboardMenu item={image} projects={projects} enabled notice="Could not add this" onSend={noop} />);
    expect(html).not.toContain("Could not add this");
  });

  it("disables the button while any send is running, and only the sending menu says so", () => {
    const html = renderToStaticMarkup(<SendToStoryboardMenu item={image} projects={projects} enabled busy onSend={noop} />);
    expect(html).toContain("disabled");
    expect(html).toContain("Use in storyboard");
  });
});

describe("sendIntoProject", () => {
  function api(overrides: Partial<SendApi> = {}): SendApi & { removed: [string, number][] } {
    const removed: [string, number][] = [];
    const made = { ...payload(1, []), project: { ...payload(1, []).project, id: "p_new" } as StoryboardProject };
    return {
      removed,
      create: vi.fn(async () => made),
      get: vi.fn(async () => ({ ...made, project: { ...made.project, revision: 2 } as StoryboardProject })),
      run: vi.fn(async (_id: string, step: PlannedStep, revision: number) => payload(revision + 1, ["s_new"])),
      remove: vi.fn(async (id: string, revision: number) => {
        removed.push([id, revision]);
      }),
      ...overrides,
    };
  }

  it("deletes the project it made when a step fails, using its newest revision", async () => {
    const fake = api({
      run: vi.fn(async (_id: string, step: PlannedStep, revision: number) => {
        if (step.op === "import_take") throw new Error("File not found");
        return payload(revision + 1, ["s_new"]);
      }),
    });
    await expect(sendIntoProject({ ...image }, null, { type: "new-shot" }, "Untitled", fake)).rejects.toThrow("File not found");
    expect(fake.removed).toEqual([["p_new", 2]]);
  });

  it("leaves an existing project alone when a step fails", async () => {
    const fake = api({ run: vi.fn(async () => { throw new Error("nope"); }) });
    await expect(sendIntoProject({ ...image }, "p_old", { type: "new-shot" }, "Untitled", fake)).rejects.toThrow("nope");
    expect(fake.remove).not.toHaveBeenCalled();
  });

  it("still reports the send's own failure when the clean-up fails", async () => {
    const fake = api({
      run: vi.fn(async () => { throw new Error("first failure"); }),
      remove: vi.fn(async () => { throw new Error("cannot delete"); }),
    });
    await expect(sendIntoProject({ ...image }, null, { type: "new-shot" }, "Untitled", fake)).rejects.toThrow("first failure");
  });

  it("returns the id of the project the item went into", async () => {
    const fake = api();
    await expect(sendIntoProject({ ...image }, null, { type: "new-shot" }, "Untitled", fake)).resolves.toBe("p_new");
    expect(fake.remove).not.toHaveBeenCalled();
  });
});
