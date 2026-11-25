import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { StoryboardGuideAction, StoryboardReadiness, StoryboardReadinessItem } from "../../../types";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import { readinessGroups } from "../exportReadiness";
import { ReadinessCheck } from "../ReadinessCheck";
import { ClipSoundSwitch } from "../mixer/ClipSoundSwitch";
import { ExportPanel, type ExportPanelProps } from "../mixer/ExportPanel";
import { Timeline } from "../mixer/Timeline";
import { buildLanes } from "../mixer/timelineGeometry";

const noop = () => undefined;
const shots = [{ id: "s1" }, { id: "s2" }, { id: "s3" }, { id: "s4" }] as never;

function item(code: StoryboardReadinessItem["code"], action: StoryboardGuideAction | null, message: string, shotIds: string[] = []): StoryboardReadinessItem {
  return { code, message, shotIds, action, data: {} };
}

function render(readiness: StoryboardReadiness | null, local = [] as Parameters<typeof readinessGroups>[1]) {
  return renderToStaticMarkup(
    <ReadinessCheck project={{ shots }} groups={readinessGroups(readiness, local)} onFix={noop} actions={<button type="button">Rough cut</button>} />,
  );
}

describe("ReadinessCheck", () => {
  it("shows the three group titles in order and says none when nothing stops or needs fixing", () => {
    const html = render({ ready: true, stops: [], worthFixing: [], forInfo: [item("format", null, "1280 x 720 · 24 fps")] });
    expect(html).toContain("Ready to export?");
    expect(html.indexOf("Stops the export")).toBeLessThan(html.indexOf("Worth fixing"));
    expect(html.indexOf("Worth fixing")).toBeLessThan(html.indexOf("For your information"));
    expect(html).toContain("Nothing stops the export.");
    expect(html).toContain("Nothing to fix.");
    expect(html).toContain("1280 x 720 · 24 fps");
    expect(html).toContain("Rough cut");
  });

  it("lists a stopping item in Stops the export with its fix button", () => {
    const html = render({ ready: false, stops: [item("no-clips", "make-clips", "No shot has a clip yet, so there is nothing to export.")], worthFixing: [], forInfo: [] });
    const stops = html.slice(html.indexOf('data-group="stops"'), html.indexOf('data-group="worthFixing"'));
    expect(stops).toContain("No shot has a clip yet, so there is nothing to export.");
    expect(stops).toContain("Open in Storyboard");
    expect(stops).not.toContain("Nothing stops the export.");
    expect(html).toContain("Nothing to fix.");
  });

  it("lists worth-fixing items with a button each, naming the shot a fix opens", () => {
    const html = render({
      ready: true,
      stops: [],
      worthFixing: [
        item("drafts", "choose-final", "2 shots are still a draft (640 x 384).", ["s1", "s2"]),
        item("lipsync-gap", "lengthen", "Shot 4 is too short. It is not lip-synced.", ["s4"]),
        item("song-length", "open-mixer", "The film is 0:28, the song 0:30."),
      ],
      forInfo: [],
    });
    const worth = html.slice(html.indexOf('data-group="worthFixing"'), html.indexOf('data-group="forInfo"'));
    expect(worth).toContain("2 shots are still a draft (640 x 384).");
    expect(worth).toContain("Open shot 4");
    expect(worth).toContain("Open the Mixer");
    expect((worth.match(/<button/g) ?? []).length).toBe(3);
    expect(html).toContain("Nothing stops the export.");
  });

  it("puts facts under For your information, with the app's own lines and no button for lines that point nowhere", () => {
    const html = render(
      { ready: true, stops: [], worthFixing: [], forInfo: [item("clip-sound-muted", "open-mixer", "Clip sound is muted under the song in 4 shots.", ["s1"]), item("lipsync-coverage", null, "9 of 12 shots are lip-synced.")] },
      [{ id: "loudness", lufs: -14 }, { id: "captions", count: 0, how: "none" }, { id: "aiTag", on: true, cloned: false }, { id: "preview" }],
    );
    const info = html.slice(html.indexOf('data-group="forInfo"'));
    expect(info).toContain("Clip sound is muted under the song in 4 shots.");
    expect(info).toContain("9 of 12 shots are lip-synced.");
    expect(info).toContain("levelled to -14 LUFS");
    expect(info).toContain("No captions.");
    expect(info).toContain("AI tag is written");
    expect(info).toContain("The preview is out of date");
    // the Mixer button on the muted-clip line and on the stale preview line only
    expect((info.match(/<button/g) ?? []).length).toBe(2);
  });
});

describe("ExportPanel upscale method and Export button", () => {
  const project = {
    id: "abc", name: "x", revision: 1,
    defaults: { width: 1280, height: 720, fps: 24, exportPreset: "16:9", resolution: "720p" },
    export: { aiLabel: { embedTag: true }, loudnessLufs: null, captionsSrt: false, captionsBurn: false, stems: false, upscale: "1440p" },
    shots: [], audio: [], captions: [], references: { characters: [], places: [] },
  } as never;

  function panel(extra: Partial<ExportPanelProps> = {}) {
    const props: ExportPanelProps = {
      project, filmSeconds: 8, empty: false, rendering: false, exportJob: null, startedAt: 0, result: null, disabled: false,
      revealLabel: "Download", copied: false, onPreset: noop, onUpscale: noop, onAiLabel: noop, onLoudness: noop, onExport: noop, onCaptionOptions: noop,
      editor: { busy: false, file: null, warnings: [], error: null }, onEditorExport: noop, onCancel: noop, onReveal: noop, onCopyPath: noop,
      upscaleMethod: "lanczos", onUpscaleMethod: noop, ...extra,
    };
    return renderToStaticMarkup(<ExportPanel {...props} />);
  }

  it("offers Sharper and Standard stretch next to the size choice, the chosen one checked", () => {
    const html = panel();
    expect(html).toContain("Sharper (AI, needs the Upscale add-on)");
    expect(html).toContain("Standard stretch");
    expect(html).toMatch(/aria-checked="true"[^>]*>Standard stretch</);
    expect(html).toMatch(/aria-checked="false"[^>]*>Sharper/);
    expect(panel({ upscaleMethod: "fast" })).toMatch(/aria-checked="true"[^>]*>Sharper/);
  });

  it("hides the choice when the film is not scaled up", () => {
    const plain = { ...(project as object), export: { ...((project as { export: object }).export), upscale: "none" } } as never;
    expect(panel({ project: plain })).not.toContain("Standard stretch");
  });

  it("notes a missing add-on with a button to the Upscale tab only for the sharper method", () => {
    const missing = panel({ upscaleMethod: "fast", upscaleAddonInstalled: false, onOpenUpscaleTab: noop });
    expect(missing).toContain("The Upscale add-on is not installed");
    expect(missing).toContain("Open the Upscale tab");
    expect(panel({ upscaleMethod: "fast", upscaleAddonInstalled: true, onOpenUpscaleTab: noop })).not.toContain("is not installed");
    expect(panel({ upscaleMethod: "lanczos", upscaleAddonInstalled: false, onOpenUpscaleTab: noop })).not.toContain("is not installed");
    expect(panel({ upscaleMethod: "fast", upscaleAddonInstalled: false })).not.toContain("Open the Upscale tab");
  });

  it("says Export anyway when things are worth fixing and shows the backend's reason when the export is stopped", () => {
    expect(panel({ exportAnyway: true })).toContain("Export anyway");
    expect(panel()).toContain("Export MP4");
    const stopped = panel({ empty: true, emptyReason: "The project has no shots yet." });
    expect(stopped).toContain("The project has no shots yet.");
    expect(stopped).toMatch(/<button[^>]*disabled=""[^>]*>Export MP4/);
  });
});

describe("the Shot sound lane header", () => {
  it("reads Muted under the song with the Let clip sound play switch off", () => {
    const html = renderToStaticMarkup(<ClipSoundSwitch state="muted" onToggle={noop} />);
    expect(html).toContain("Muted under the song");
    expect(html).toContain("Let clip sound play");
    expect(html).toContain('role="switch"');
    expect(html).not.toContain('checked=""');
  });

  it("shows the switch on when clips play, and in some shots when mixed", () => {
    expect(renderToStaticMarkup(<ClipSoundSwitch state="playing" onToggle={noop} />)).toMatch(/role="switch" checked=""/);
    const mixed = renderToStaticMarkup(<ClipSoundSwitch state="mixed" onToggle={noop} />);
    expect(mixed).toContain("Plays in some shots");
    expect(mixed).toMatch(/role="switch" checked=""/);
  });

  it("sits in the Shot sound lane only, when the timeline is given it", () => {
    const labels = { picture: "Picture", native: "Shot sound", voice: "Voice", sfx: "Effects", music: "Music" } as const;
    const project = {
      id: "abc", audio: [], captions: [],
      shots: [{ id: "s1", title: "A", status: "done", seconds: 4, nativeSoundDb: -60, takes: [], activeTake: null, transition: { type: "cut", seconds: 0 } }],
    } as never;
    const timeline = { total: 4, shots: [{ id: "s1", start: 0, end: 4, duration: 4, source_in: 0, source_out: 4, transition_type: "cut", transition_seconds: 0, measured: true, ready: false }], audio: [] } as never;
    const lanes = buildLanes(project, timeline, "Shot {n}");
    const props = {
      projectId: "abc", durationS: 4, pxPerSecond: 40, playheadS: 0, lanes, laneLabels: labels, selectedId: null, selectedType: null,
      onSelect: noop, onCommit: noop, onRelink: noop, onKeyDown: noop, onAdd: noop, onSeek: noop, onClearSelection: noop,
    } as const;
    const withSwitch = renderToStaticMarkup(<Timeline {...props} clipSound={{ state: "muted", onToggle: noop }} />);
    expect((withSwitch.match(/Let clip sound play/g) ?? []).length).toBe(1);
    expect(withSwitch).toContain("timeline__lane--clip-sound");
    expect(renderToStaticMarkup(<Timeline {...props} />)).not.toContain("Let clip sound play");
  });
});
