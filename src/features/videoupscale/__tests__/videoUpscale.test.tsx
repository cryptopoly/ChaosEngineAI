import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import i18n from "i18next";
import type { VideoUpscaleAddonStatus, VideoUpscaleJob, VideoUpscaleModel, VideoUpscaleProbe } from "../../../api/videoUpscale";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import { UpscaleInstallCard } from "../UpscaleInstallCard";
import { UpscaleJobCard, etaText, progressText, stateText } from "../UpscaleJobCard";
import { UpscaleRecentJobs, stateLabel } from "../UpscaleRecentJobs";
import {
  MAX_SIDE,
  NO_MEDIA,
  defaultModelId,
  defaultSizeChoice,
  deinterlaceDefault,
  durationText,
  etaParts,
  etaSeconds,
  even,
  fpsText,
  installSizeMb,
  installStep,
  jobButtons,
  missingModelIds,
  outputSize,
  percentDone,
  probeSize,
  qualityOption,
  showInstallCard,
  sizeOption,
  sizeRequest,
  type UpscaleMediaView,
} from "../upscaleLogic";
import { clearUpscaleRequest, peekUpscaleRequest, requestUpscale } from "../upscaleSource";
import { upscaleSourceOf } from "../../video/VideoGalleryTab";
import { tabViewFor } from "../../../app/tabViews";
import { VideoUpscaleView } from "../../../app/VideoTabs";

const noop = () => undefined;
const t = i18n.getFixedT("en", "studio");

function addon(extra: Partial<VideoUpscaleAddonStatus> = {}): VideoUpscaleAddonStatus {
  return {
    installed: false, packagesInstalled: false, folder: "C:/x/video-upscale", diskMb: 0,
    packages: ["spandrel==0.4.2", "einops==0.8.2"],
    models: [
      { id: "realesr-general-x4v3", name: "Real-ESRGAN general x4v3", label: "Fast", installed: false, downloadMb: 4.9, pinned: true, sha256Seen: null },
      { id: "realesrgan-x4plus", name: "Real-ESRGAN x4plus", label: "Fast, stronger clean-up", installed: false, downloadMb: 67.0, pinned: true, sha256Seen: null },
    ],
    job: { phase: "idle", error: null, running: false, models: [], done: false, startedAt: 0, finishedAt: 0 },
    ...extra,
  };
}

function model(extra: Partial<VideoUpscaleModel> = {}): VideoUpscaleModel {
  return {
    id: "realesr-general-x4v3", name: "Real-ESRGAN general x4v3", label: "Fast", author: "xinntao", scale: 4,
    default: true, licence: "BSD-3-Clause", licenceNote: "BSD-3-Clause (xinntao/Real-ESRGAN).", sizeBytes: 4885111,
    installed: true,
    fit: { status: "fits", needGb: 2, usableGb: 22, headroomPct: 90, reason: "2 GB", basis: "estimated" },
    ...extra,
  };
}

function job(extra: Partial<VideoUpscaleJob> = {}): VideoUpscaleJob {
  return {
    id: "abc123", state: "running", input: "C:/clips/tape.mp4", output: "C:/out/videos/upscaled/tape-x-1440x1152.mp4",
    modelId: "realesr-general-x4v3", width: 1440, height: 1152, framesDone: 120, framesTotal: 2400, fraction: 0.05,
    framesPerSecond: 20, segment: 1, segments: 4, resumedSegments: 0, error: null, notes: [], startedAt: 1,
    finishedAt: null, resumable: false, ...extra,
  };
}

describe("size choice against the limit", () => {
  it("rounds to even sides like the backend", () => {
    expect(even(721)).toBe(722);
    expect(even(1)).toBe(2);
    expect(outputSize("2x", { width: 720, height: 576 })).toEqual({ width: 1440, height: 1152 });
    expect(outputSize("1080p", { width: 640, height: 360 })).toEqual({ width: 1920, height: 1080 });
    expect(outputSize("2160p", { width: 720, height: 480 })).toEqual({ width: 3240, height: 2160 });
  });

  it("blocks a size past the limit with the backend's numbers", () => {
    const option = sizeOption("4x", { width: 1920, height: 1080 });
    expect(option.block).toBe("limit");
    expect(option.size).toEqual({ width: 7680, height: 4320 });
    expect(sizeOption("2x", { width: 1920, height: 1080 }).block).toBeNull();
    expect(sizeOption("4x", { width: 1024, height: 576 }).block).toBeNull(); // exactly 4096
    expect(MAX_SIDE).toBe(4096);
    const text = t("videoUpscale.size.limit", { width: 7680, height: 4320, max: MAX_SIDE });
    expect(text).toBe(
      "The result would be 7680x4320; the limit is 4096 pixels on a side. Choose a smaller scale or a smaller target size.",
    );
  });

  it("needs the source size for a target height, and a taller target", () => {
    expect(sizeOption("1080p", null).block).toBe("needsSize");
    expect(sizeOption("2x", null).block).toBeNull();
    expect(sizeOption("4x", null).block).toBeNull();
    expect(sizeOption("1080p", { width: 1920, height: 1080 }).block).toBe("notBigger");
    expect(sizeOption("2160p", { width: 3840, height: 1080 }).block).toBe("limit"); // a very wide source
  });

  it("asks for a scale or for both sides, never a blocked size", () => {
    expect(sizeRequest("2x", null)).toEqual({ scale: 2 });
    expect(sizeRequest("4x", { width: 640, height: 360 })).toEqual({ scale: 4 });
    expect(sizeRequest("1440p", { width: 640, height: 360 })).toEqual({ width: 2560, height: 1440 });
    expect(sizeRequest("1440p", null)).toBeNull();
    expect(sizeRequest("4x", { width: 1920, height: 1080 })).toBeNull();
    expect(defaultSizeChoice({ width: 2560, height: 1440 })).toBe("2160p"); // 2x is 5120 wide
  });
});

describe("fit verdict of the quality rows", () => {
  it("keeps fits, tight and unknown selectable; refuses won't fit and rows not downloaded", () => {
    expect(qualityOption(model()).selectable).toBe(true);
    expect(qualityOption(model({ fit: { ...model().fit!, status: "tight" } })).verdict).toBe("tight");
    expect(qualityOption(model({ fit: { ...model().fit!, status: "unknown" } })).selectable).toBe(true);
    expect(qualityOption(model({ fit: { ...model().fit!, status: "wont-fit" } })).selectable).toBe(false);
    expect(qualityOption(model({ installed: false })).selectable).toBe(false);
    expect(qualityOption(model({ fit: undefined })).verdict).toBe("none");
  });

  it("starts on the default row, or the first one that can run", () => {
    const fast = model();
    const strong = model({ id: "realesrgan-x4plus", label: "Fast, stronger clean-up", default: false });
    expect(defaultModelId([strong, fast], null)).toBe(fast.id);
    expect(defaultModelId([strong, fast], strong.id)).toBe(strong.id);
    expect(defaultModelId([strong, { ...fast, installed: false }], null)).toBe(strong.id);
    expect(defaultModelId([], null)).toBeNull();
  });
});

describe("job state to buttons", () => {
  it("offers Cancel while queued or running", () => {
    expect(jobButtons(job({ state: "queued" }))).toEqual({ cancel: true, resume: false, reveal: false, open: false, dismiss: false });
    expect(jobButtons(job())).toMatchObject({ cancel: true, resume: false });
  });
  it("offers Resume after a cancel, a restart or a failure", () => {
    for (const state of ["cancelled", "interrupted", "failed"] as const) {
      expect(jobButtons(job({ state, resumable: true }))).toEqual({ cancel: false, resume: true, reveal: false, open: false, dismiss: true });
    }
  });
  it("offers Open and Show in folder when done", () => {
    expect(jobButtons(job({ state: "done" }))).toEqual({ cancel: false, resume: false, reveal: true, open: true, dismiss: true });
  });
});

describe("progress and time left", () => {
  it("reads the time left from the measured rate", () => {
    expect(etaSeconds(job())).toBe(114);
    expect(etaSeconds(job({ framesPerSecond: null }))).toBeNull();
    expect(etaSeconds(job({ state: "queued" }))).toBeNull();
    expect(etaSeconds(job({ framesDone: 2400 }))).toBeNull();
  });

  it("words the time left", () => {
    expect(etaParts(42)).toEqual({ unit: "seconds", value: 42 });
    expect(etaParts(114)).toEqual({ unit: "minutes", value: 2 });
    expect(etaParts(3 * 3600 + 20 * 60)).toEqual({ unit: "hours", value: 3, minutes: 20 });
    expect(etaText(114, t)).toBe("about 2 min left");
    expect(etaText(null, t)).toBeNull();
  });

  it("shows frames done of total with the speed and the time left", () => {
    expect(percentDone(job())).toBe(5);
    expect(percentDone(job({ framesTotal: 0, fraction: 0.5 }))).toBe(50);
    expect(progressText(job(), t)).toBe("120 of 2,400 frames · 20 frames a second · about 2 min left");
    expect(progressText(job({ state: "cancelled", framesPerSecond: null }), t)).toBe("120 of 2,400 frames");
  });

  it("words each state", () => {
    expect(stateText(job({ state: "failed", error: "Out of memory" }), t)).toBe("Stopped with an error: Out of memory");
    expect(stateText(job({ state: "interrupted" }), t)).toContain("Resume");
  });
});

describe("install card", () => {
  it("shows until the add-on is in, and while it installs or after it failed", () => {
    expect(showInstallCard(null)).toBe(false);
    expect(showInstallCard(addon())).toBe(true);
    expect(showInstallCard(addon({ installed: true }))).toBe(false);
    expect(showInstallCard(addon({ installed: true, job: { ...addon().job, running: true, phase: "weights" } }))).toBe(true);
    expect(installStep("weights")).toBe(2);
    expect(installStep("done")).toBe(0);
    expect(installSizeMb(addon())).toBe(72);
    expect(missingModelIds(addon())).toEqual(["realesr-general-x4v3", "realesrgan-x4plus"]);
  });

  it("names the size, the licence and the Install button", () => {
    const html = renderToStaticMarkup(
      <UpscaleInstallCard status={addon()} licence="BSD-3-Clause" licenceNote="note" requestError={null} onInstall={noop} />,
    );
    expect(html).toContain("About 72 MB");
    expect(html).toContain("Model licence: BSD-3-Clause.");
    expect(html).toContain("Install (about 72 MB)");
  });

  it("shows the step while installing and a plain error when it failed", () => {
    const running = renderToStaticMarkup(
      <UpscaleInstallCard status={addon({ job: { ...addon().job, running: true, phase: "weights" } })} licence={null}
        licenceNote={null} requestError={null} onInstall={noop} />,
    );
    expect(running).toContain("Step 2 of 3: Downloading the upscaling models");
    expect(running).not.toContain("Install (about");
    const failed = renderToStaticMarkup(
      <UpscaleInstallCard status={addon({ job: { ...addon().job, phase: "error", error: "No network" } })} licence={null}
        licenceNote={null} requestError={null} onInstall={noop} />,
    );
    expect(failed).toContain("The install failed: No network");
    expect(failed).toContain("Try the install again");
  });
});

describe("job card", () => {
  const card = (value: VideoUpscaleJob, media: UpscaleMediaView = NO_MEDIA) => renderToStaticMarkup(
    <UpscaleJobCard job={value} modelLabel="Fast" media={media} busy={false} actionError={null}
      onCancel={noop} onResume={noop} onReveal={noop} onPlay={noop} onOpen={noop} onDismiss={noop} />,
  );
  const done = (extra: Partial<VideoUpscaleJob> = {}) => job({ state: "done", framesDone: 2400, fraction: 1, ...extra });

  it("shows progress and Cancel while running", () => {
    const html = card(job());
    expect(html).toContain("120 of 2,400 frames");
    expect(html).toContain("about 2 min left");
    expect(html).toContain(">Cancel<");
    expect(html).not.toContain(">Resume<");
  });

  it("offers Resume after a cancel", () => {
    const html = card(job({ state: "cancelled", resumable: true, framesPerSecond: null }));
    expect(html).toContain(">Resume<");
    expect(html).not.toContain(">Cancel<");
  });

  it("shows the output path, Play, Open and Show in folder when done", () => {
    const html = card(done());
    expect(html).toContain("Saved as C:/out/videos/upscaled/tape-x-1440x1152.mp4");
    expect(html).toContain("Play the result");
    expect(html).toContain("Open the result");
    expect(html).toContain("Show in folder");
    expect(html).toContain("After (1440 × 1152)");
  });

  it("shows the first frames side by side, and the film once played", () => {
    const html = card(done(), { ...NO_MEDIA, inputFrameUrl: "blob:in", outputFrameUrl: "blob:out", playUrl: "blob:film" });
    expect(html).toContain('<img src="blob:in"');
    expect(html).toContain('<img src="blob:out"');
    expect(html.indexOf("blob:in")).toBeLessThan(html.indexOf("blob:out"));
    expect(html).toContain('src="blob:film"');
    expect(card(done())).not.toContain("<video");
  });

  it("falls back to the placeholders, with Open and Show in folder, when the backend answers 404", () => {
    const html = card(done(), { ...NO_MEDIA, playState: "missing" });
    expect(html).not.toContain("<img");
    expect(html).toContain("No preview of this file in the app.");
    expect(html).toContain("Open the result to compare it at full size.");
    expect(html).toContain("it was restarted since");
    expect(html).toContain("Open the result</button>");
    expect(html).toContain("Show in folder</button>");
  });

  it("says the frames are loading, and shows a play error word for word", () => {
    expect(card(done(), { ...NO_MEDIA, loading: true })).toContain("Loading the first frame…");
    expect(card(done(), { ...NO_MEDIA, playState: "error", playError: "disk full" })).toContain("Could not load the film: disk full");
    expect(card(done(), { ...NO_MEDIA, playState: "loading" })).toContain("Loading the film…");
  });
});

describe("what the probe tells", () => {
  const probe = (extra: Partial<VideoUpscaleProbe> = {}): VideoUpscaleProbe => ({
    inputPath: "C:/clips/tape.mp4", width: 720, height: 576, fps: 25, durationSeconds: 3725, frames: 93125,
    hasAudio: true, interlaced: true, ...extra,
  });

  it("gives the size choices any file's size, with the limit and taller-than-source rules kept", () => {
    const size = probeSize(probe());
    expect(size).toEqual({ width: 720, height: 576 });
    expect(probeSize(null)).toBeNull();
    expect(sizeOption("1080p", size).block).toBeNull();
    expect(sizeRequest("2160p", size)).toEqual({ width: 2700, height: 2160 });
    expect(sizeOption("4x", probeSize(probe({ width: 1920, height: 1080 }))).block).toBe("limit");
    expect(sizeOption("1080p", probeSize(probe({ width: 1920, height: 1080 }))).block).toBe("notBigger");
  });

  it("ticks deinterlace only for a video that is interlaced", () => {
    expect(deinterlaceDefault(probe())).toBe(true);
    expect(deinterlaceDefault(probe({ interlaced: false }))).toBe(false);
    expect(deinterlaceDefault(probe({ interlaced: null }))).toBe(false);
    expect(deinterlaceDefault(null)).toBe(false);
  });

  it("words the length, the frame rate and the line", () => {
    expect(durationText(65)).toBe("1:05");
    expect(durationText(3725)).toBe("1:02:05");
    expect(durationText(0.4)).toBe("0:00");
    expect(fpsText(25)).toBe("25");
    expect(fpsText(29.97)).toBe("29.97");
    expect(fpsText(23.976023)).toBe("23.98");
    expect(t("videoUpscale.probe.line", { name: "tape.mp4", width: 720, height: 576, fps: "25", duration: "1:02:05", audio: "with sound" }))
      .toBe("tape.mp4: 720 × 576, 25 frames a second, 1:02:05, with sound");
  });
});

describe("recent jobs", () => {
  const list = (jobs: VideoUpscaleJob[], extra: { selectedId?: string | null; resumeDisabled?: boolean } = {}) => renderToStaticMarkup(
    <UpscaleRecentJobs jobs={jobs} selectedId={extra.selectedId ?? null} resumeDisabled={extra.resumeDisabled ?? false}
      onSelect={noop} onResume={noop} />,
  );

  it("shows nothing for an empty list", () => {
    expect(list([])).toBe("");
  });

  it("lists the jobs in the order given, with their state, and Resume only where the job can resume", () => {
    const html = list([
      job({ id: "a", state: "done", input: "C:/clips/new.mp4", framesDone: 2400 }),
      job({ id: "b", state: "interrupted", input: "C:/clips/old.mp4", resumable: true }),
      job({ id: "c", state: "failed", input: "C:/clips/bad.mp4", resumable: false }),
    ], { selectedId: "a" });
    expect(html.indexOf("new.mp4")).toBeLessThan(html.indexOf("old.mp4"));
    expect(html).toContain("Recent upscales");
    expect(html).toContain("Done");
    expect(html).toContain("Stopped");
    expect(html).toContain("Failed");
    expect(html.match(/>Resume</g)).toHaveLength(1);
    expect(html).toContain('aria-current="true"');
    expect(list([job({ state: "interrupted", resumable: true })], { resumeDisabled: true })).toContain("disabled");
  });

  it("names a running job's progress and every state", () => {
    expect(list([job()])).toContain("Upscaling");
    expect(stateLabel("queued", t)).toBe("Waiting");
    expect(stateLabel("cancelled", t)).toBe("Cancelled");
  });
});

describe("Video Gallery hand-off and tab wiring", () => {
  it("hands a clip's path, id and size to the Upscale tab", () => {
    const source = upscaleSourceOf({ artifactId: "v1", videoPath: "C:/v/clip.mp4", width: 768, height: 512 } as never);
    expect(source).toEqual({ path: "C:/v/clip.mp4", artifactId: "v1", width: 768, height: 512 });
    requestUpscale(source);
    expect(peekUpscaleRequest()).toEqual(source);
    clearUpscaleRequest();
    expect(peekUpscaleRequest()).toBeNull();
  });

  it("renders the Upscale view for its tab id", () => {
    expect(tabViewFor("video-upscale")).toBe(VideoUpscaleView);
  });
});
