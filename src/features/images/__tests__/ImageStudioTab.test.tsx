import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { ImageModelVariant, ImageRuntimeStatus } from "../../../types";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import { makeImageFamily, makeImageVariant } from "../../../components/models/__tests__/fixtures";
import { imageMenuItems } from "../imageMenu";
import { ImageStudioTab, type ImageStudioTabProps } from "../ImageStudioTab";

const noop = () => undefined;
const CUDA_PC = { platform: "Windows", arch: "x86_64", gpuVramTotalGb: 24 };

function runtime(extra: Partial<ImageRuntimeStatus> = {}): ImageRuntimeStatus {
  return {
    activeEngine: "diffusers",
    realGenerationAvailable: true,
    message: "ready",
    device: "cuda",
    expectedDevice: "cuda",
    missingDependencies: [],
    deviceMemoryGb: 24,
    ...extra,
  } as ImageRuntimeStatus;
}

function props(overrides: Partial<ImageStudioTabProps> = {}, variant?: Partial<ImageModelVariant>): ImageStudioTabProps {
  const selected = makeImageVariant({ availableLocally: true, ...variant });
  const family = makeImageFamily({}, [selected]);
  return {
    imageCatalog: [family],
    selectedImageModelId: selected.id,
    onSelectedImageModelIdChange: noop,
    selectedImageVariant: selected,
    selectedImageFamily: family,
    selectedImageLoaded: false,
    selectedImageWillLoadOnGenerate: false,
    imageRuntimeLoadedDifferentModel: false,
    loadedImageVariant: null,
    imageRuntimeStatus: runtime(),
    tauriBackend: null,
    busy: false,
    busyAction: null,
    imageBusy: false,
    imageBusyLabel: null,
    backendOnline: true,
    activeImageDownloads: {},
    imagePrompt: "A lighthouse",
    onImagePromptChange: noop,
    imageNegativePrompt: "",
    imageReferences: [],
    onImageReferencesChange: noop,
    imageReferenceStrength: 0.6,
    onImageReferenceStrengthChange: noop,
    onImageNegativePromptChange: noop,
    imageQualityPreset: "balanced",
    imageRatioId: "square",
    imageWidth: 1024,
    onImageWidthChange: noop,
    imageHeight: 1024,
    onImageHeightChange: noop,
    imageSteps: 24,
    onImageStepsChange: noop,
    imageGuidance: 5,
    onImageGuidanceChange: noop,
    imageBatchSize: 1,
    onImageBatchSizeChange: noop,
    imageUseRandomSeed: true,
    onImageUseRandomSeedChange: noop,
    imageSeedInput: "",
    onImageSeedInputChange: noop,
    imageOutputs: [],
    recentImageOutputs: [],
    onActiveTabChange: noop,
    onOpenImageStudio: noop,
    onOpenImageGallery: noop,
    onSubmitImageGeneration: noop,
    onApplyImageRatioPreset: noop,
    onApplyImageQuality: noop,
    imageDraftMode: false,
    onImageDraftModeChange: noop,
    imageSampler: "default" as ImageStudioTabProps["imageSampler"],
    onImageSamplerChange: noop,
    imageCacheStrategy: "none",
    onImageCacheStrategyChange: noop,
    imageCacheRelL1Thresh: null,
    onImageCacheRelL1ThreshChange: noop,
    imageCfgDecay: true,
    onImageCfgDecayChange: noop,
    imagePreviewVae: false,
    onImagePreviewVaeChange: noop,
    imageFp8LayerwiseCasting: false,
    onImageFp8LayerwiseCastingChange: noop,
    isAppleSiliconHost: false,
    hostSystem: CUDA_PC,
    onPreloadImageModel: noop,
    onUnloadImageModel: noop,
    onInstallImageRuntime: vi.fn(async () => ({ ok: true, output: "", capabilities: {} })) as ImageStudioTabProps["onInstallImageRuntime"],
    gpuBundleJob: null,
    onImageDownload: noop,
    onCancelImageDownload: noop,
    onDeleteImageDownload: noop,
    onOpenExternalUrl: noop,
    onRestartServer: noop,
    onUseSameImageSettings: noop,
    onVaryImageSeed: noop,
    onRevealPath: noop,
    onDeleteImageArtifact: noop,
    ...overrides,
  };
}

const render = (overrides: Partial<ImageStudioTabProps> = {}, variant?: Partial<ImageModelVariant>) =>
  renderToStaticMarkup(<ImageStudioTab {...props(overrides, variant)} />);

function generateButton(html: string): string {
  const match = html.match(/<button[^>]*>Generate image<\/button>/);
  expect(match, "Generate button").not.toBeNull();
  return match![0];
}

describe("Image Studio loading feedback", () => {
  it("shows the busy label as a status right by the run estimate", () => {
    const html = render({ imageBusy: true, imageBusyLabel: "Loading FLUX…" });
    expect(html).toMatch(/<p class="busy-indicator" role="status"><span class="busy-dot"><\/span>Loading FLUX…<\/p>/);
    expect(html.indexOf("Loading FLUX…")).toBeLessThan(html.indexOf('id="disclosure-options"'));
  });

  it("badges a model that loads on the first generate", () => {
    expect(render({ selectedImageWillLoadOnGenerate: true })).toContain("Loads On First Generate");
  });
});

describe("Image Studio danger gate", () => {
  it("keeps Generate off at danger level until the box is ticked", () => {
    const html = render({ imageWidth: 8192, imageHeight: 8192, hostSystem: CUDA_PC, imageRuntimeStatus: runtime({ deviceMemoryGb: 8 }) });
    expect(html).toContain("Run anyway");
    expect(generateButton(html)).toContain("disabled");
    expect(html).toContain("Model ≈");
  });

  it("leaves Generate on for a safe run", () => {
    const html = render();
    expect(html).not.toContain("Run anyway");
    expect(generateButton(html)).not.toContain("disabled");
  });
});

describe("Image Studio controls moved into Options and Expert", () => {
  it("still has the controls, each under a disclosure", () => {
    const html = render({ imageUseRandomSeed: false, imageSeedInput: "7" });
    for (const label of ["Options", "Expert", "Negative prompt", "Width", "Height", "Steps", "Guidance"]) {
      expect(html, label).toContain(label);
    }
    expect(html).toContain('id="disclosure-options"');
    expect(html).toContain('id="disclosure-expert"');
  });
});

describe("Image Studio more actions", () => {
  const t = (_key: string, options: Record<string, unknown>) =>
    String(options.defaultValue).replace(/\{(\w+)\}/g, (_m, name) => String(options[name]));
  const variant = makeImageVariant({ availableLocally: true });
  const other = makeImageVariant({ id: "other/model", name: "Other model" });
  const actions = { openLink: vi.fn(), preload: vi.fn(), unload: vi.fn() };
  const base = { selected: variant, loadedVariant: null, selectedLoaded: false, realGenerationAvailable: true, menuDisabled: false };

  it("offers the model card and Preload for an installed model that is not loaded", () => {
    const items = imageMenuItems(t, base, actions);
    expect(items.map((item) => item.id)).toEqual(["model-card", "load"]);
    expect(items[1].disabled).toBe(false);
  });

  it("offers Unload for the loaded model and Unload {name} for another loaded one", () => {
    expect(imageMenuItems(t, { ...base, selectedLoaded: true }, actions).map((item) => item.id)).toEqual(["model-card", "unload"]);
    const named = imageMenuItems(t, { ...base, loadedVariant: other }, actions);
    expect(named.map((item) => item.id)).toEqual(["model-card", "load", "unload-named"]);
    expect(named[2].label).toBe("Unload Other model");
  });

  it("disables load and unload while busy, and hides them without a working runtime", () => {
    const busy = imageMenuItems(t, { ...base, menuDisabled: true }, actions);
    expect(busy.find((item) => item.id === "load")?.disabled).toBe(true);
    expect(imageMenuItems(t, { ...base, realGenerationAvailable: false }, actions).map((item) => item.id)).toEqual(["model-card"]);
    expect(imageMenuItems(t, { ...base, selected: null }, actions)).toEqual([]);
  });
});
