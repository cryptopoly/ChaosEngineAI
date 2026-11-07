import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type {
  StoryboardAudioClip,
  StoryboardProject,
  StoryboardRenderJob,
  StoryboardShot,
  StoryboardTimeline,
} from "../../../../types";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import type { UseStoryboardResult } from "../../useStoryboard";
import { ClipInspector, ShotInspector } from "../ClipInspector";
import { AiLabelNotice, ExportPanel, RenderProgress, type ExportPanelProps } from "../ExportPanel";
import { MixerTab } from "../MixerTab";
import { Timeline } from "../Timeline";
import { Transport } from "../Transport";
import { buildLanes } from "../timelineGeometry";

const noop = () => undefined;
const asyncNoop = async () => true;

function clip(extra: Partial<StoryboardAudioClip> = {}): StoryboardAudioClip {
  return {
    id: "a1", kind: "music",
    source: { path: "media/audio/a1.wav", hash: "h", measuredSeconds: 10, clonedVoice: false, label: "lo-fi piano" },
    anchor: { shotId: "s1", offset: 0.5 }, start: null, trimIn: 0, trimOut: null, gainDb: -6, fadeIn: 0.5, fadeOut: 1.5,
    duck: true, loop: false, status: "ok", ...extra,
  };
}

function shot(id: string, title: string, extra: Partial<StoryboardShot> = {}): StoryboardShot {
  return {
    id, order: 0, title, status: "done", stillPrompt: "", motionPrompt: "", referenceImages: [], startFrom: "generate",
    endImage: null, seconds: 4, seed: null, locked: false, trimIn: 0, trimOut: null, speed: 1,
    takes: [{ id: "t", still: null, clip: `media/${id}/t/clip.mp4`, hashes: { clip: "hh" }, settings: {}, measuredSeconds: 4, createdAt: 0, origin: "upload", basis: null }],
    activeTake: "t", transition: { type: "cut", seconds: 0 }, nativeSoundDb: -3, redo: false, ...extra,
  };
}

function project(extra: Partial<StoryboardProject> = {}): StoryboardProject {
  return {
    schemaVersion: 1, revision: 4, id: "abc", name: "Lighthouse", createdAt: 0,
    defaults: { width: 1280, height: 720, fps: 24, imageModel: null, videoModel: null, voice: null, exportPreset: "16:9" },
    references: { characters: [], places: [] },
    shots: [shot("s1", "Arrival"), shot("s2", "")],
    audio: [clip()],
    export: { aiLabel: { embedTag: true, endCard: false }, loudnessLufs: -14, endCardText: "Made with AI", endCardSeconds: 2 },
    batches: [], undoStack: [], redoStack: [], ...extra,
  };
}

function timeline(): StoryboardTimeline {
  return {
    total: 8,
    shots: [
      { id: "s1", start: 0, end: 4, duration: 4, source_in: 0, source_out: 4, transition_type: "cut", transition_seconds: 0, measured: true, ready: true },
      { id: "s2", start: 3.5, end: 8, duration: 4.5, source_in: 0, source_out: 4.5, transition_type: "crossfade", transition_seconds: 0.5, measured: true, ready: true },
    ],
    audio: [{ id: "a1", start: 0.5, duration: 7.5, trim_in: 0, end: 8 }],
    problems: [],
  };
}

describe("Transport", () => {
  const base = { playing: false, timeS: 12.4, durationS: 32.4, loop: false, canPlay: true, canZoomIn: true, canZoomOut: true, onPlay: noop, onStop: noop, onLoop: noop, onZoom: noop };

  it("shows the time readout and play, stop, loop and zoom controls", () => {
    const html = renderToStaticMarkup(<Transport {...base} />);
    expect(html).toContain("00:12.4 / 00:32.4");
    for (const label of ["Play", "Stop", "Loop", "Zoom in", "Zoom out"]) expect(html).toContain(`aria-label="${label}"`);
  });

  it("offers Pause while playing and marks Loop pressed", () => {
    const html = renderToStaticMarkup(<Transport {...base} playing loop />);
    expect(html).toContain('aria-label="Pause"');
    expect(html).toContain('aria-pressed="true"');
  });

  it("disables play and stop until a preview has been rendered, and explains why", () => {
    const html = renderToStaticMarkup(<Transport {...base} canPlay={false} />);
    expect(html).toContain("Render a preview to play the film.");
    expect(html.match(/disabled=""/g)?.length).toBe(2);
  });

  it("disables zoom at the ends of the range", () => {
    const html = renderToStaticMarkup(<Transport {...base} canZoomIn={false} canZoomOut={false} />);
    expect(html.match(/disabled=""/g)?.length).toBe(2);
  });
});

describe("Timeline", () => {
  const p = project({ audio: [clip(), clip({ id: "a2", kind: "voice", anchor: null, start: 1, source: { path: "media/audio/a2.wav", hash: null, measuredSeconds: 2, clonedVoice: false, label: "Some nights" }, duck: false, gainDb: 0 }), clip({ id: "a3", kind: "sfx", status: "missing", source: { path: "media/audio/a3.wav", hash: null, measuredSeconds: 2, clonedVoice: false, label: "thunder" }, anchor: null, start: 5, duck: false })] });
  const tl = { ...timeline(), audio: [...timeline().audio, { id: "a2", start: 1, duration: 2, trim_in: 0, end: 3 }] };
  const lanes = buildLanes(p, tl);
  const labels = { picture: "Picture", native: "Shot sound", voice: "Voice", sfx: "Effects", music: "Music" } as const;
  const html = renderToStaticMarkup(
    <Timeline
      projectId="abc" durationS={8} pxPerSecond={40} playheadS={2} lanes={lanes} laneLabels={labels} selectedId="a1" selectedType="clip"
      onSelect={noop} onCommit={noop} onRelink={noop} onKeyDown={noop} onAdd={noop} onSeek={noop} onClearSelection={noop}
    />,
  );

  it("draws the ruler, five lanes as lists and an add button on each audio lane", () => {
    expect(html).toContain("time-ruler");
    expect((html.match(/role="list"/g) ?? []).length).toBe(5);
    for (const name of ["Picture", "Shot sound", "Voice", "Effects", "Music"]) expect(html).toContain(`aria-label="${name}"`);
    expect(html).toContain('aria-label="Add audio to Voice"');
    expect(html).toContain('aria-label="Add audio to Music"');
    expect(html).not.toContain('aria-label="Add audio to Picture"');
    expect(html).not.toContain('aria-label="Add audio to Shot sound"');
  });

  it("places blocks with custom properties from time and zoom, and the playhead from its time", () => {
    expect(html).toContain("--timeline-w:");
    expect(html).toContain("--playhead-x:80"); // 2 s at 40 px/s
    expect(html).toContain("--clip-x:20"); // the music clip: 0.5 s at 40 px/s
    expect(html).toContain("--clip-w:300"); // 7.5 s at 40 px/s
  });

  it("gives each clip the spec's screen reader label", () => {
    expect(html).toContain('aria-label="Music, lo-fi piano, from 0.5 to 8 seconds, -6 dB"');
    expect(html).toContain('aria-label="Voice, Some nights, from 1 to 3 seconds, 0 dB"');
  });

  it("marks the selected, ducked and missing clips and offers Relink on the missing one", () => {
    expect(html).toContain("timeline__clip--selected");
    expect(html).toContain("timeline__clip--ducked");
    expect(html).toContain("timeline__clip--missing");
    expect(html).toContain("Missing");
    expect(html).toContain("Relink");
  });

  it("shades the ducked music where voice plays and marks the crossfade overlap", () => {
    expect(html).toContain("timeline__duck");
    expect(html).toContain("timeline__xfade");
  });

  it("shows each shot's sound level and a Reading… state before peaks arrive", () => {
    expect(html).toContain("-3 dB");
    expect(html).toContain("timeline__level-fill");
    expect(html).toContain("Reading…");
  });

  it("has trim handles on editable audio clips only", () => {
    expect(html).toContain("timeline__clip-handle--start");
    expect(html).toContain("timeline__clip-handle--end");
    const picture = renderToStaticMarkup(
      <Timeline
        projectId="abc" durationS={8} pxPerSecond={40} playheadS={0} lanes={lanes.filter((l) => l.id === "picture")} laneLabels={labels}
        selectedId={null} selectedType={null} onSelect={noop} onCommit={noop} onRelink={noop} onKeyDown={noop} onAdd={noop} onSeek={noop} onClearSelection={noop}
      />,
    );
    expect(picture).not.toContain("timeline__clip-handle");
  });
});

describe("ClipInspector", () => {
  const shots = [shot("s1", "Arrival"), shot("s2", "")];
  const spans = timeline().shots;
  const props = { shots, shotSpans: spans, disabled: false, op: asyncNoop as never, onRemove: noop, onClose: noop };
  const span = { start: 0.5, duration: 7.5, end: 8, approximate: false };

  it("is a labelled region with gain, fades, trim, duck, loop, anchor and remove", () => {
    const html = renderToStaticMarkup(<ClipInspector clip={clip()} span={span} {...props} />);
    expect(html).toContain('role="region"');
    expect(html).toContain('aria-label="Selected clip"');
    for (const text of ["Gain", "Fade in", "Fade out", "Duck under voice", "Loop to fill", "Anchor", "Offset", "Remove from lane"]) {
      expect(html).toContain(text);
    }
    expect(html).toContain("-6 dB");
    expect(html).toContain("Shot 1: Arrival");
    expect(html).toContain("Shot 2: Shot 2");
  });

  it("offers the film start instead of an offset when the clip is not anchored", () => {
    const html = renderToStaticMarkup(<ClipInspector clip={clip({ anchor: null, start: 3 })} span={span} {...props} />);
    expect(html).toContain("Starts at");
    expect(html).not.toContain("Moves with shot");
  });

  it("cannot duck a voice clip and cannot trim the end of a looping one", () => {
    const voice = renderToStaticMarkup(<ClipInspector clip={clip({ kind: "voice", duck: false })} span={span} {...props} />);
    expect(voice).toMatch(/<input type="checkbox"[^>]*disabled=""[^>]*\/>\s*<svg/);
    const looped = renderToStaticMarkup(<ClipInspector clip={clip({ loop: true })} span={span} {...props} />);
    expect(looped.match(/disabled=""/g)?.length).toBeGreaterThanOrEqual(1);
  });

  it("locks every field while an edit is in flight", () => {
    const html = renderToStaticMarkup(<ClipInspector clip={clip()} span={span} {...props} disabled />);
    expect(html).not.toContain('<input type="range" min="-60" max="12" step="0.5" value="-6" aria-label="Gain"/>');
    expect(html.match(/disabled=""/g)?.length).toBeGreaterThan(6);
  });

  it("shows the shot's own sound level for a selected shot", () => {
    const html = renderToStaticMarkup(
      <ShotInspector shot={shots[1]} index={1} span={spans[1]} disabled={false} op={asyncNoop as never} onOpenBoard={noop} onClose={noop} />,
    );
    expect(html).toContain("Shot sound level for shot 2");
    expect(html).toContain("-3 dB");
    expect(html).toContain("Open on Board");
    expect(html).toContain("From 3.5 s, 4.5 s long");
  });
});

describe("AiLabelNotice and ExportPanel", () => {
  it("shows the tag switch on and the end card off by default", () => {
    const html = renderToStaticMarkup(<AiLabelNotice metadata endCard={false} forced={false} onChange={noop} />);
    expect(html).toContain("Add a machine-readable AI tag");
    expect(html).toContain("Add an AI-generated end card");
    expect(html).toContain('role="switch"');
    expect(html).toMatch(/role="switch" checked=""/);
    expect(html).not.toContain("cloned voice");
  });

  it("locks the tag on and says why when a cloned voice is in the project", () => {
    const html = renderToStaticMarkup(<AiLabelNotice metadata endCard={false} forced onChange={noop} />);
    expect(html).toMatch(/role="switch" disabled="" checked=""/);
    expect(html).toContain("This project uses a cloned voice, so the AI tag is always written.");
  });

  function panel(overrides: Partial<ExportPanelProps> = {}) {
    const props: ExportPanelProps = {
      project: project(), filmSeconds: 8, empty: false, rendering: false, exportJob: null, startedAt: 0, result: null, disabled: false,
      revealLabel: "Download", copied: false, onPreset: noop, onAiLabel: noop, onLoudness: noop, onExport: noop, onCancel: noop, onReveal: noop, onCopyPath: noop,
      ...overrides,
    };
    return renderToStaticMarkup(<ExportPanel {...props} />);
  }

  it("shows the three presets with the pixels of the chosen one", () => {
    const html = panel();
    for (const name of ["16:9", "9:16", "1:1"]) expect(html).toContain(`>${name}</button>`);
    expect(html).toContain("1280 × 720 px.");
    expect(html).toContain("Reframed at export; shots are not regenerated.");
    expect(html).toMatch(/aria-checked="true"[^>]*>16:9</);
    expect(panel({ project: project({ defaults: { ...project().defaults, exportPreset: "9:16" } }) })).toContain("720 × 1280 px.");
  });

  it("shows the loudness target, captions disabled with a note, and an unknown run estimate", () => {
    const html = panel();
    expect(html).toContain('value="-14"');
    expect(html).toContain("LUFS");
    expect(html).toMatch(/<fieldset class="export-panel__captions" disabled="">/);
    expect(html).toContain("Captions are coming");
    expect(html).toContain("Time: unknown until the first run");
    expect(html).toContain("Export MP4");
  });

  it("cannot export an empty film or while a render runs", () => {
    expect(panel({ empty: true })).toMatch(/<button type="button" class="primary-button" disabled="">Export MP4/);
    expect(panel({ rendering: true })).toMatch(/<button type="button" class="primary-button" disabled="">Export MP4/);
  });

  it("shows progress and Cancel while an export runs", () => {
    const running: StoryboardRenderJob = { renderId: "r", projectId: "abc", preview: false, state: "running", progress: 0.4, message: "join" };
    const html = panel({ exportJob: running, rendering: true });
    expect(html).toContain("Cancel export");
    expect(html).toContain("live-progress--storyboard");
    for (const phase of ["Normalise shots", "Join", "Mix sound", "Encode"]) expect(html).toContain(phase);
    expect(html).not.toContain("Export MP4");
  });

  it("shows the result card with size, length and the file actions", () => {
    const done: StoryboardRenderJob = { renderId: "r", projectId: "abc", preview: false, state: "done", progress: 1, message: "mix", file: "exports/Lighthouse-1.mp4" };
    const html = panel({ result: { job: done, bytes: 49_283_072 } });
    expect(html).toContain("Exported · 47 MB · 00:08");
    expect(html).toContain("Lighthouse-1.mp4");
    expect(html).toContain("Download");
    expect(html).toContain("Copy path");
    expect(panel({ result: { job: done, bytes: null }, copied: true })).toContain("Copied");
  });

  it("says a failed export wrote nothing and offers Retry and the technical detail", () => {
    const failed: StoryboardRenderJob = { renderId: "r", projectId: "abc", preview: false, state: "failed", progress: 0.2, message: "mix", error: "RenderError: boom" };
    const html = panel({ exportJob: failed });
    expect(html).toContain("Export failed. Nothing was written.");
    expect(html).toContain("RenderError: boom");
    expect(html).toContain("Retry");
  });

  it("puts the render progress on the storyboard accent for a preview too", () => {
    const job: StoryboardRenderJob = { renderId: "r", projectId: "abc", preview: true, state: "running", progress: 0.5, message: "shots" };
    const html = renderToStaticMarkup(<RenderProgress job={job} startedAt={0} filmSeconds={20} title="Rendering preview…" />);
    expect(html).toContain("Rendering preview…");
    expect(html).toContain("live-progress--storyboard");
  });
});

function story(overrides: Partial<UseStoryboardResult> = {}): UseStoryboardResult {
  return {
    projectId: "abc", project: project(), timeline: timeline(), loading: false, error: null, notFound: false, notice: null, dismissNotice: noop,
    busy: false, batch: null, open: noop, reload: asyncNoop as never, op: asyncNoop as never, undo: asyncNoop as never, redo: asyncNoop as never,
    startBatch: asyncNoop as never, cancelBatch: asyncNoop as never, ...overrides,
  };
}

describe("MixerTab states", () => {
  const render = (s: UseStoryboardResult, backendOnline = true) =>
    renderToStaticMarkup(<MixerTab backendOnline={backendOnline} story={s} onOpenProjects={noop} onOpenBoard={noop} />);

  it("asks for an open project like the Board does", () => {
    expect(render(story({ projectId: null, project: null, timeline: null }))).toContain("No project is open.");
    expect(render(story({ notFound: true, project: null, timeline: null }))).toContain("This project is not there any more.");
  });

  it("says it is waiting while offline and shows a load error with Reload", () => {
    expect(render(story({ project: null, timeline: null }), false)).toContain("Waiting for the backend…");
    const html = render(story({ project: null, timeline: null, error: "no route" }));
    expect(html).toContain("no route");
    expect(html).toContain("Reload");
  });

  it("says there is nothing to mix for a project with no shots", () => {
    const html = render(story({ project: project({ shots: [], audio: [] }), timeline: { total: 0, shots: [], audio: [], problems: [] } }));
    expect(html).toContain("Nothing to mix yet. Make shots on the Board first.");
    expect(html).toContain("Open Board");
    expect(html).not.toContain("timeline__inner");
  });

  it("shows the title, subtitle, preview size, lanes, transport, inspector hint and export", () => {
    const html = render(story());
    expect(html).toContain("Lighthouse · Mixer");
    expect(html).toContain("8 s · 16:9 · preview 640 × 360");
    expect(html).toContain("Render preview");
    expect(html).toContain("No preview yet.");
    expect(html).toContain("Select a clip on the timeline to edit its sound.");
    expect(html).toContain("Open on Board");
    expect(html).toContain("Transitions shorten the film by their length; the ruler shows the real total.");
    expect(html).toContain("Export MP4");
    expect(html).toContain("transport");
  });

  it("tells the user how many shots the preview and export leave out", () => {
    const tl = timeline();
    tl.shots[1].ready = false;
    expect(render(story({ timeline: tl }))).toContain("1 shot has no clip yet and is left out of the preview and the export.");
  });

  it("disables Undo and Redo with an empty history and shows the changed-elsewhere notice", () => {
    const html = render(story({ notice: "conflict" }));
    expect(html).toContain("Changed elsewhere; showing the latest");
    expect(html.match(/<button type="button" class="secondary-button" disabled="">/g)?.length).toBe(2);
  });
});
