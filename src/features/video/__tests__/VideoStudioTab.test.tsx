import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { VideoModelVariant, VideoRuntimeStatus } from "../../../types";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import { CUDA_PC, MAC, makeVideoFamily, makeVideoVariant } from "../../../components/models/__tests__/videoFixtures";
import { VideoStudioTab, type VideoStudioTabProps } from "../VideoStudioTab";
import { videoMenuItems } from "../videoMenu";
import { capacityLine, deviceLabel } from "../videoCapacity";

const noop = () => undefined;

function runtime(extra: Partial<VideoRuntimeStatus> = {}): VideoRuntimeStatus {
  return {
    activeEngine: "diffusers",
    realGenerationAvailable: true,
    message: "ready",
    device: "cuda",
    expectedDevice: "cuda",
    missingDependencies: [],
    deviceMemoryGb: 24,
    ...extra,
  };
}

function props(overrides: Partial<VideoStudioTabProps> = {}, variant?: Partial<VideoModelVariant>): VideoStudioTabProps {
  const selected = makeVideoVariant({ availableLocally: true, hasLocalData: true, ...variant });
  const family = makeVideoFamily({}, [selected]);
  return {
    videoCatalog: [family],
    hostSystem: CUDA_PC,
    selectedVideoModelId: selected.id,
    onSelectedVideoModelIdChange: noop,
    selectedVideoVariant: selected,
    selectedVideoFamily: family,
    selectedVideoLoaded: false,
    selectedVideoWillLoadOnGenerate: false,
    videoRuntimeLoadedDifferentModel: false,
    loadedVideoVariant: null,
    videoRuntimeStatus: runtime(),
    tauriBackend: null,
    busy: false,
    busyAction: null,
    videoBusy: false,
    videoBusyLabel: null,
    backendOnline: true,
    activeVideoDownloads: {},
    videoPrompt: "A lighthouse at dusk",
    onVideoPromptChange: noop,
    videoNegativePrompt: "blurry",
    videoStartImage: null,
    onVideoStartImageChange: noop,
    videoEndImage: null,
    onVideoEndImageChange: noop,
    imageGallery: [],
    onVideoNegativePromptChange: noop,
    videoUseRandomSeed: true,
    onVideoUseRandomSeedChange: noop,
    videoSeedInput: "",
    onVideoSeedInputChange: noop,
    videoWidth: 832,
    onVideoWidthChange: noop,
    videoHeight: 480,
    onVideoHeightChange: noop,
    videoNumFrames: 33,
    onVideoNumFramesChange: noop,
    videoFps: 16,
    onVideoFpsChange: noop,
    videoSteps: 30,
    onVideoStepsChange: noop,
    videoGuidance: 5,
    onVideoGuidanceChange: noop,
    videoUseNf4: false,
    onVideoUseNf4Change: noop,
    videoEnableLtxRefiner: false,
    onVideoEnableLtxRefinerChange: noop,
    videoCfgDecay: true,
    onVideoCfgDecayChange: noop,
    videoPreviewVae: false,
    onVideoPreviewVaeChange: noop,
    videoFp8LayerwiseCasting: false,
    onVideoFp8LayerwiseCastingChange: noop,
    videoCacheStrategy: "none",
    onVideoCacheStrategyChange: noop,
    videoCacheRelL1Thresh: null,
    onVideoCacheRelL1ThreshChange: noop,
    videoStgScale: 1,
    onVideoStgScaleChange: noop,
    videoFastPreview: false,
    onVideoFastPreviewChange: noop,
    onActiveTabChange: noop,
    onPreloadVideoModel: noop,
    onUnloadVideoModel: noop,
    onVideoDownload: noop,
    onGenerateVideo: noop,
    onOpenExternalUrl: noop,
    onRestartServer: noop,
    onInstallVideoOutputDeps: vi.fn(async () => ({ ok: true, output: "", capabilities: {} })) as VideoStudioTabProps["onInstallVideoOutputDeps"],
    onInstallVideoGpuRuntime: vi.fn(async () => ({ ok: true, output: "", capabilities: {} })) as VideoStudioTabProps["onInstallVideoGpuRuntime"],
    longLiveStatus: null,
    installingLongLive: false,
    onRefreshLongLiveStatus: noop,
    onInstallLongLive: vi.fn(async () => ({ ok: true, output: "", capabilities: {} })) as VideoStudioTabProps["onInstallLongLive"],
    longLiveJob: null,
    mlxVideoStatus: null,
    installingMlxVideo: false,
    onRefreshMlxVideoStatus: noop,
    onInstallMlxVideo: vi.fn(async () => ({ ok: true, output: "", capabilities: {} })) as VideoStudioTabProps["onInstallMlxVideo"],
    gpuBundleJob: null,
    ...overrides,
  };
}

const render = (overrides: Partial<VideoStudioTabProps> = {}, variant?: Partial<VideoModelVariant>) =>
  renderToStaticMarkup(<VideoStudioTab {...props(overrides, variant)} />);

/** The markup of the disabled attribute on the Generate video button. */
function generateButton(html: string): string {
  const match = html.match(/<button[^>]*>Generate video<\/button>/);
  expect(match, "Generate video button").not.toBeNull();
  return match![0];
}

describe("Video Studio busy and loading feedback", () => {
  it("shows the busy label as a status next to the estimate while a model loads or unloads", () => {
    const html = render({ videoBusy: true, videoBusyLabel: "Loading Wan 2.1…" });
    expect(html).toMatch(/<p class="busy-indicator" role="status"><span class="busy-dot"><\/span>Loading Wan 2.1…<\/p>/);
    expect(render({ videoBusy: true, videoBusyLabel: "Unloading…" })).toContain("Unloading…");
    expect(render()).not.toContain("busy-indicator");
  });

  it("says the model will be loaded on the next generate", () => {
    expect(render({ selectedVideoWillLoadOnGenerate: true })).toContain("The selected model will be loaded into memory on the next generate.");
    expect(render()).not.toContain("will be loaded into memory");
  });

  it("keeps the device and capacity facts as the memory item's tooltip", () => {
    const html = render();
    expect(html).toMatch(/run-estimate__item--memory[^>]*title="GPU · 24 GB total · model ≈ [^"]+, this run peak ≈ [^"]+"/);
    const apple = render({ hostSystem: MAC, videoRuntimeStatus: runtime({ device: "mps", expectedDevice: "mps", deviceMemoryGb: 64 }) });
    expect(apple).toContain("Apple Silicon · 64 GB total");
    // The guess comes from the webview's own platform, so pin it to a Mac for this assertion.
    vi.stubGlobal("navigator", { platform: "MacIntel", userAgent: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)" });
    try {
      const guess = render({ hostSystem: MAC, videoRuntimeStatus: runtime({ device: null, expectedDevice: null, deviceMemoryGb: null }) });
      expect(guess).toContain("Apple Silicon (detected)");
      expect(guess).toContain("default — restart backend for real detection");
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("announces only the estimate figures, not the Generate button", () => {
    const html = render({ videoBusy: true, videoBusyLabel: "Generating 3/30" });
    expect(html).toMatch(/<div class="run-estimate__items" aria-live="polite">/);
    expect(html).not.toMatch(/<div class="run-estimate run-estimate--\w+" aria-live/);
  });
});

describe("Video Studio more actions", () => {
  const t = (_key: string, options: Record<string, unknown>) =>
    String(options.defaultValue).replace(/\{(\w+)\}/g, (_m, name) => String(options[name]));
  const variant = makeVideoVariant({ availableLocally: true });
  const other = makeVideoVariant({ id: "other/video", name: "Other video" });
  const actions = { openLink: vi.fn(), preload: vi.fn(), unload: vi.fn() };
  const base = { selected: variant, loadedVariant: null, isDownloaded: true, selectedLoaded: false, videoBusy: false, realGenerationAvailable: true };

  it("is in the header, closed", () => {
    expect(render()).toContain('aria-label="More actions"');
  });

  it("offers the model card and Load into memory for an installed model that is not loaded", () => {
    const items = videoMenuItems(t, base, actions);
    expect(items.map((item) => item.id)).toEqual(["model-card", "load"]);
    expect(items[1].label).toBe("Load into memory");
    expect(items[1].disabled).toBe(false);
    items[1].onSelect();
    expect(actions.preload).toHaveBeenCalledWith(variant);
  });

  it("offers Unload for the loaded model and Unload {name} for another loaded one", () => {
    expect(videoMenuItems(t, { ...base, selectedLoaded: true }, actions).map((item) => item.id)).toEqual(["model-card", "unload"]);
    const named = videoMenuItems(t, { ...base, loadedVariant: other }, actions);
    expect(named.map((item) => item.id)).toEqual(["model-card", "load", "unload-named"]);
    expect(named[2].label).toBe("Unload Other video");
  });

  it("disables load and unload while busy, and Load while the engine is not ready", () => {
    expect(videoMenuItems(t, { ...base, videoBusy: true, selectedLoaded: true }, actions).find((item) => item.id === "unload")?.disabled).toBe(true);
    expect(videoMenuItems(t, { ...base, realGenerationAvailable: false }, actions).find((item) => item.id === "load")?.disabled).toBe(true);
  });

  it("has no Load for a model that is not downloaded, and no items without a model", () => {
    expect(videoMenuItems(t, { ...base, isDownloaded: false }, actions).map((item) => item.id)).toEqual(["model-card"]);
    expect(videoMenuItems(t, { ...base, selected: null }, actions)).toEqual([]);
  });
});

describe("Video Studio danger gate", () => {
  // 4K x 161 frames on a 24 GB card is past the safe envelope for any model
  const heavy = { videoWidth: 3840, videoHeight: 2160, videoNumFrames: 161 };

  it("keeps Generate off at danger level until the box is ticked, and says why", () => {
    const html = render(heavy);
    expect(html).toContain("Run anyway");
    expect(generateButton(html)).toContain("disabled");
    expect(html).toContain("This configuration is likely to crash the backend.");
  });

  it("leaves Generate on for a safe run and shows no tick box", () => {
    const html = render();
    expect(html).not.toContain("Run anyway");
    expect(generateButton(html)).not.toContain("disabled");
  });
});

describe("Video Studio fixed-schedule models", () => {
  it("hides steps and guidance for a distilled model that runs its own schedule", () => {
    const html = render({}, { fixedSampler: true, runtime: "diffusers LTX2Pipeline" });
    expect(html).not.toContain("Guidance");
    expect(html).not.toContain("CFG decay");
  });

  it("keeps them for a model whose steps and guidance apply", () => {
    const html = render({}, { fixedSampler: false, runtime: "diffusers LTX2Pipeline" });
    expect(html).toContain("Guidance");
  });
});

describe("Video Studio start and end image pickers", () => {
  it("shows no picture picker for a model without image input", () => {
    const html = render({}, { acceptsImageInput: false, acceptsEndImage: false });
    expect(html).not.toContain("Start image");
    expect(html).not.toContain("End image (optional)");
  });

  it("shows only the start picker for a model that follows the first frame (Wan 2.2 TI2V 5B)", () => {
    const html = render({}, { acceptsImageInput: true, acceptsEndImage: false });
    expect(html).toContain("Start image");
    expect(html).not.toContain("End image (optional)");
  });

  it("shows both pickers for a model that takes a first and a last frame (LTX-2)", () => {
    const html = render({}, { acceptsImageInput: true, acceptsEndImage: true });
    expect(html).toContain("Start image");
    expect(html).toContain("End image (optional)");
  });
});

describe("Video Studio controls moved into Options and Expert", () => {
  it("still has every control, each under a disclosure", () => {
    const html = render({ videoUseRandomSeed: false, videoSeedInput: "7" }, { acceptsImageInput: true });
    for (const label of [
      "Options",
      "Expert",
      "Negative prompt",
      "Width",
      "Height",
      "Frames",
      "FPS",
      "Steps",
      "Guidance",
      "Random seed",
      "Diffusion cache",
      "CFG decay",
      "Preview VAE",
      "FP8 layerwise (CUDA Ada+)",
      "4-bit (NF4)",
    ]) {
      expect(html, label).toContain(label);
    }
    expect(html).toContain('id="disclosure-options"');
    expect(html).toContain('id="disclosure-expert"');
  });
});

describe("capacity line", () => {
  const t = (_key: string, options: Record<string, unknown>) =>
    String(options.defaultValue).replace(/\{(\w+)\}/g, (_m, name) => String(options[name]));
  const facts = { device: "cuda", activeEngine: "diffusers", effectiveDevice: "cuda" as const, deviceMemoryGb: 24, memoryReported: true, modelFootprintGb: 8.4, estimatedPeakGb: 12.2 };

  it("names the device the backend reported, and tags a guess as detected", () => {
    expect(deviceLabel(t, facts)).toBe("GPU");
    expect(deviceLabel(t, { ...facts, device: "mps", effectiveDevice: "mps" })).toBe("Apple Silicon");
    expect(deviceLabel(t, { ...facts, device: "mlx", activeEngine: "mlx-video" })).toBe("Apple Silicon (MLX)");
    expect(deviceLabel(t, { ...facts, device: null })).toBe("GPU (detected)");
    expect(deviceLabel(t, { ...facts, device: null, effectiveDevice: "cpu" })).toBe("CPU (detected)");
    expect(deviceLabel(t, { ...facts, device: null, effectiveDevice: "mps" })).toBe("Apple Silicon (detected)");
  });

  it("reads device, memory, model and run peak, and marks a default memory figure", () => {
    expect(capacityLine(t, facts)).toBe("GPU · 24 GB total · model ≈ 8.4 GB, this run peak ≈ 12 GB");
    expect(capacityLine(t, { ...facts, modelFootprintGb: 0 })).toBe("GPU · 24 GB total · this run peak ≈ 12 GB");
    expect(capacityLine(t, { ...facts, memoryReported: false })).toContain("~24 GB (default — restart backend for real detection) total");
  });
});
