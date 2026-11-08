import { renderToStaticMarkup } from "react-dom/server";
import { createElement } from "react";
import { describe, expect, it, vi } from "vitest";
import type { VideoModelVariant } from "../../../types";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import { FormatFields } from "../FormatFields";
import { ProjectDialog } from "../ProjectDialog";
import {
  FPS_CHOICES,
  RESOLUTIONS,
  fitContext,
  formatPatch,
  fpsNeedsConversion,
  modelSizeFor,
  newProjectDefaults,
  resolutionFit,
  sizeFor,
  upscaleSize,
  type VideoFitContext,
} from "../formatLogic";

function model(extra: Partial<VideoModelVariant> = {}): VideoModelVariant {
  return {
    id: "m", familyId: "f", name: "Test model", repo: "org/test", runtime: "diffusers", sizeGb: 20, runtimeFootprintGb: 22,
    availableLocally: true, ...extra,
  } as unknown as VideoModelVariant;
}

function ctx(m: VideoModelVariant | null, memoryGb: number, device = "cuda"): VideoFitContext {
  return { model: m, device, deviceMemoryGb: memoryGb, residentGb: null };
}

describe("size table", () => {
  it("matches the backend per aspect ratio", () => {
    expect(RESOLUTIONS.map((r) => sizeFor("16:9", r))).toEqual([
      { width: 854, height: 480 }, { width: 1280, height: 720 }, { width: 1920, height: 1080 },
      { width: 2560, height: 1440 }, { width: 3840, height: 2160 },
    ]);
    expect(sizeFor("9:16", "1080p")).toEqual({ width: 1080, height: 1920 });
    expect(sizeFor("1:1", "480p")).toEqual({ width: 480, height: 480 });
  });

  it("offers 24, 25, 30, 48, 50 and 60 fps", () => {
    expect([...FPS_CHOICES]).toEqual([24, 25, 30, 48, 50, 60]);
  });

  it("builds the new-project and edit arguments from the format", () => {
    expect(newProjectDefaults("9:16", "1080p", 25)).toEqual({ width: 1080, height: 1920, resolution: "1080p", fps: 25, exportPreset: "9:16" });
    expect(formatPatch({ exportPreset: "9:16", resolution: "720p" }, { resolution: "480p" })).toEqual({
      defaults: { width: 480, height: 854, exportPreset: "9:16", resolution: "480p" },
    });
  });

  it("upscales only to a larger size", () => {
    expect(upscaleSize("16:9", "none", sizeFor("16:9", "720p"))).toBeNull();
    expect(upscaleSize("16:9", "2160p", sizeFor("16:9", "720p"))).toEqual({ width: 3840, height: 2160 });
    expect(upscaleSize("9:16", "1440p", sizeFor("9:16", "2160p"))).toBeNull();
  });
});

describe("fit hint", () => {
  it("caps a size to the model's own limit and says it", () => {
    const wan = model({ maxSide: 1280, maxPixels: 1280 * 720 });
    expect(modelSizeFor(wan, sizeFor("16:9", "720p"))).toBeNull();
    const capped = modelSizeFor(wan, sizeFor("16:9", "2160p"));
    expect(capped && capped.width <= 1280 && capped.width * capped.height <= 1280 * 720).toBe(true);
    expect(capped && capped.width % 16 === 0 && capped.height % 16 === 0).toBe(true);
    expect(resolutionFit(ctx(wan, 24), "16:9", "2160p", 24).modelSize).toEqual(capped);
  });

  it("grades by the device memory: bigger sizes need more", () => {
    const m = model({ maxSide: 4096, maxPixels: 4096 * 4096 });
    const small = resolutionFit(ctx(m, 24), "16:9", "480p", 24);
    const big = resolutionFit(ctx(m, 24), "16:9", "2160p", 24);
    expect(small.peakGb).not.toBeNull();
    expect(big.peakGb as number).toBeGreaterThan(small.peakGb as number);
    expect(big.level).toBe("wontfit");
    expect(resolutionFit(ctx(m, 192), "16:9", "480p", 24).level).toBe("safe");
  });

  it("says nothing without a model", () => {
    expect(resolutionFit(ctx(null, 24), "16:9", "720p", 24).level).toBe("unknown");
  });

  it("uses the project's model, or the first installed one for Automatic", () => {
    const a = model({ id: "a" });
    const b = model({ id: "b" });
    const source = { variants: [a, b], device: "cuda", deviceMemoryGb: 24, residentGb: null };
    expect(fitContext(source, "b").model?.id).toBe("b");
    expect(fitContext(source, null).model?.id).toBe("a");
    expect(fitContext(source, "gone").model).toBeNull();
    expect(fitContext(undefined, null).model).toBeNull();
  });

  it("flags a frame rate the model does not make natively, only when the catalog states one", () => {
    expect(fpsNeedsConversion(model({ nativeFps: 16 }), 24)).toBe(true);
    expect(fpsNeedsConversion(model({ nativeFps: 24 }), 24)).toBe(false);
    expect(fpsNeedsConversion(model(), 24)).toBe(false);
  });
});

describe("format fields", () => {
  const html = (props: Partial<Parameters<typeof FormatFields>[0]> = {}) =>
    renderToStaticMarkup(
      createElement(FormatFields, {
        aspect: "16:9", resolution: "720p", fps: 24, fit: ctx(model({ maxSide: 1280, maxPixels: 1280 * 720, nativeFps: 16 }), 24),
        onAspect: () => undefined, onResolution: () => undefined, onFps: () => undefined, ...props,
      }),
    );

  it("names the groups Aspect ratio, Resolution and Frame rate, with no Shape", () => {
    const out = html();
    expect(out).toContain("Aspect ratio");
    expect(out).toContain("Resolution");
    expect(out).toContain("Frame rate");
    expect(out).not.toContain("Shape");
  });

  it("offers every resolution and frame rate, none disabled by the machine", () => {
    const out = html({ fit: ctx(model({ sizeGb: 200, runtimeFootprintGb: 220 }), 8) });
    for (const r of ["480p", "720p", "1080p", "1440p", "2160p (4K)"]) expect(out).toContain(r);
    for (const f of FPS_CHOICES) expect(out).toContain(`${f} fps`);
    expect(out).not.toContain("disabled");
    expect(out).toContain("Will not fit");
  });

  it("notes a capped size and a frame-rate conversion", () => {
    const out = html({ resolution: "2160p", fps: 30 });
    expect(out).toContain("scaled up to 3840 × 2160");
    expect(out).toContain("makes 16 fps clips");
    expect(out).toContain("not interpolated");
  });
});

describe("new project dialog", () => {
  it("defaults to 16:9, 720p and 24 fps and says Aspect ratio", () => {
    const out = renderToStaticMarkup(
      createElement(ProjectDialog, { mode: "create", onSubmit: () => undefined, onCancel: () => undefined }),
    );
    expect(out).toContain("Aspect ratio");
    expect(out).not.toContain("Shape");
    expect(out).toMatch(/aria-checked="true"[^>]*><span class="preset-pill-label">720p/);
    expect(out).toMatch(/aria-checked="true"[^>]*><span class="preset-pill-label">24 fps/);
  });
});
