import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { VideoUpscaleQualityModel, VideoUpscaleQualityStatus } from "../../../api/videoUpscale";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import { UpscaleQualityTier } from "../UpscaleQualityTier";
import {
  qualityInstallPercent,
  qualityNeedGb,
  qualityRow,
  qualitySizeBlock,
  showQualityInstall,
} from "../upscaleLogic";

const noop = () => undefined;

function row(extra: Partial<VideoUpscaleQualityModel> = {}, fit: Partial<VideoUpscaleQualityModel["qualityFit"]> = {}): VideoUpscaleQualityModel {
  return {
    id: "flashvsr-plus-turbo", name: "FlashVSR+ Turbo", label: "Quality", author: "OpenImagingLab", scale: 4, default: false,
    licence: { id: "apache-2.0", name: "Apache 2.0", commercial: "yes", gated: false, summary: "Free to use." },
    licenceNote: "Apache-2.0", sizeBytes: 7016472997, installed: false, tier: "quality", minCardGb: 12,
    sizeTiers: [{ width: 1280, height: 1024, gpuGb: 10 }, { width: 1920, height: 1088, gpuGb: 12 }, { width: 2560, height: 1440, gpuGb: 20 }],
    maxOutputWidth: 2560, maxOutputHeight: 1440,
    qualityFit: { offered: true, code: null, reason: "", cardGb: 24, maxOutput: [2560, 1440], ...fit },
    ...extra,
  };
}

function status(extra: Partial<VideoUpscaleQualityStatus> = {}, job: Partial<VideoUpscaleQualityStatus["job"]> = {}): VideoUpscaleQualityStatus {
  return {
    available: true, unavailableReason: null, installed: false, folder: "C:/x", diskMb: 0, downloadGb: 10.4, weightsGb: 7,
    minFreeGb: 20,
    job: { phase: "idle", phaseIndex: 0, phaseTotal: 6, error: null, running: false, done: false, bytesDone: 0, bytesTotal: 0,
      log: [], startedAt: 0, finishedAt: 0, ...job },
    ...extra,
  };
}

describe("quality tier rules", () => {
  it("hides the tier where it cannot run", () => {
    expect(qualityRow([row()], status({ available: false }))).toBeNull();
    expect(qualityRow([], status())).toBeNull();
    expect(qualityRow(undefined, null)).toBeNull();
    expect(qualityRow([row()], status())?.id).toBe("flashvsr-plus-turbo");
  });

  it("blocks 4K, and sizes above what this card allows", () => {
    expect(qualitySizeBlock({ width: 3840, height: 2160 }, row())).toBe("qualityTooBig");
    expect(qualitySizeBlock({ width: 2560, height: 1440 }, row())).toBeNull();
    expect(qualitySizeBlock({ width: 1440, height: 2560 }, row())).toBeNull();
    const small = row({}, { cardGb: 16, maxOutput: [1920, 1088] });
    expect(qualitySizeBlock({ width: 2560, height: 1440 }, small)).toBe("qualityNeedsMore");
    expect(qualitySizeBlock({ width: 1920, height: 1080 }, small)).toBeNull();
    expect(qualitySizeBlock(null, small)).toBeNull();
  });

  it("names the memory a size needs", () => {
    expect(qualityNeedGb({ width: 1280, height: 960 }, row())).toBe(10);
    expect(qualityNeedGb({ width: 1920, height: 1080 }, row())).toBe(12);
    expect(qualityNeedGb({ width: 2560, height: 1440 }, row())).toBe(20);
    expect(qualityNeedGb({ width: 3840, height: 2160 }, row())).toBeNull();
  });

  it("counts the install by phase and by the weights' bytes", () => {
    expect(qualityInstallPercent({ phase: "idle", phaseIndex: 0, phaseTotal: 6, bytesDone: 0, bytesTotal: 0 })).toBe(0);
    expect(qualityInstallPercent({ phase: "packages", phaseIndex: 3, phaseTotal: 6, bytesDone: 0, bytesTotal: 0 })).toBe(33);
    expect(qualityInstallPercent({ phase: "weights", phaseIndex: 5, phaseTotal: 6, bytesDone: 50, bytesTotal: 100 })).toBe(75);
    expect(qualityInstallPercent({ phase: "done", phaseIndex: 0, phaseTotal: 6, bytesDone: 0, bytesTotal: 0 })).toBe(100);
  });

  it("shows the install card until it is in, and while it runs or failed", () => {
    expect(showQualityInstall(status())).toBe(true);
    expect(showQualityInstall(status({ installed: true }))).toBe(false);
    expect(showQualityInstall(status({ installed: true }, { phase: "error", error: "x" }))).toBe(true);
    expect(showQualityInstall(status({ available: false }))).toBe(false);
  });
});

describe("quality tier card", () => {
  const render = (props: Partial<Parameters<typeof UpscaleQualityTier>[0]> = {}) =>
    renderToStaticMarkup(
      <UpscaleQualityTier row={row()} status={status()} tier="fast" onTier={noop} requestError={null} onInstall={noop}
        onCancelInstall={noop} {...props} />,
    );

  it("offers Fast as the default and Quality as the other choice, without the install card until chosen", () => {
    const html = render();
    expect(html).toContain("Fast (default)");
    expect(html).toContain("FlashVSR+ Turbo");
    expect(html).toContain('aria-checked="true"');
    expect(html).not.toContain("Install the quality upscaler");
    expect(html).not.toContain("invents detail");
  });

  it("warns that it invents detail and points family tapes at the fast tier", () => {
    const html = render({ tier: "quality" });
    expect(html).toContain("This upscaler invents detail");
    expect(html).toContain("family tapes");
    expect(html).toContain("timecode");
    expect(html).toContain("RTX 4090");
    expect(html).toContain("On this 24 GB card it makes up to 2560 × 1440. 4K is not offered.");
    expect(html).toContain("Install the quality upscaler (about 10 GB)");
    expect(html).toContain("Model licence: Apache 2.0.");
  });

  it("shows the step and a Cancel button while installing", () => {
    const html = render({ tier: "quality", status: status({}, { phase: "weights", phaseIndex: 5, running: true, bytesDone: 1, bytesTotal: 2 }) });
    expect(html).toContain("Step 5 of 6: Downloading and checking the model (7 GB)…");
    expect(html).toContain("Cancel the install");
    expect(html).not.toContain("Install the quality upscaler (about");
  });

  it("says why it is not offered on a small card and disables the choice", () => {
    const html = render({ row: row({}, { offered: false, code: "smallCard", cardGb: 8, maxOutput: null }) });
    expect(html).toContain("needs a graphics card with at least 12 GB, and this one has 8 GB");
    expect(html).toMatch(/aria-checked="false"[^>]*disabled=""/);
  });

  it("shows no install card once installed", () => {
    const html = render({ tier: "quality", status: status({ installed: true }) });
    expect(html).not.toContain("Install the quality upscaler");
    expect(html).toContain("invents detail");
  });

  it("shows a failed install's reason and a retry", () => {
    const html = render({ tier: "quality", status: status({}, { phase: "error", error: "Only 3.0 GB free" }) });
    expect(html).toContain("The install failed: Only 3.0 GB free");
    expect(html).toContain("Try the install again");
  });
});
