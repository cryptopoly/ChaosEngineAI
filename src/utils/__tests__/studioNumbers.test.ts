// Review B (v0.10.0) follow-ups: download sizes (D-108, FU-471), Studio estimates and defaults (FU-472, FU-475),
// the default image row (FU-478) and Make without a model (FU-474).
import { describe, expect, it } from "vitest";
import { cautionCoveredByFit, type Fit } from "../../components/models/fit";
import { imageSizeSortKey } from "../../components/models/adapters/image";
import { videoSizeSortKey } from "../../components/models/adapters/video";
import { stepsForPickedVariant } from "../../features/image/studioPresets";
import { modelsMissingReason } from "../../features/storyboard/storyboardLogic";
import { defaultNumFramesFor } from "../../hooks/videoStateDefaults";
import type { ImageModelFamily, ImageModelVariant } from "../../types/image";
import type { VideoModelVariant } from "../../types/video";
import { compareDiscoverVariants } from "../discoverSort";
import { imagePrimarySizeLabel, imageSecondarySizeLabel, videoPrimarySizeLabel, videoSecondarySizeLabel } from "../format";
import { defaultImageVariantFor, imageVariantEngineMissing } from "../images";

const image = (extra: Partial<ImageModelVariant>): ImageModelVariant => ({
  id: "row", familyId: "f", name: "Row", provider: "P", repo: "o/r", link: "", runtime: "diffusers", styleTags: [],
  taskSupport: ["txt2img"], sizeGb: 0, recommendedResolution: "1024x1024", note: "", availableLocally: false,
  estimatedGenerationSeconds: null, ...extra,
} as ImageModelVariant);

const video = (extra: Partial<VideoModelVariant>): VideoModelVariant => ({
  id: "v", familyId: "f", name: "V", provider: "P", repo: "o/v", link: "", runtime: "diffusers", styleTags: [],
  taskSupport: ["txt2video"], sizeGb: 0, recommendedResolution: "768x512", defaultDurationSeconds: 4, note: "",
  availableLocally: false, estimatedGenerationSeconds: null, ...extra,
} as VideoModelVariant);

describe("download size before a download (D-108)", () => {
  it("shows the catalogue size, not the sum of every weights file in the Hub repo", () => {
    // Review B, H1: LTX-Video read 236.4 GB for a 2.0 GB download; Qwen-Image-2.1 Turbo sd.cpp 60.4 for 3.9 on disk.
    const ltx = video({ sizeGb: 2.0, coreWeightsGb: 236.4, repoSizeGb: 240 });
    expect(videoPrimarySizeLabel(ltx)).toBe("2.0 GB download");
    expect(videoSecondarySizeLabel(ltx)).toBeNull();
    expect(videoSizeSortKey(ltx)).toBe(2.0);
    const qwen = image({ sizeGb: 4.2, coreWeightsGb: 60.4, repoSizeGb: 61 });
    expect(imagePrimarySizeLabel(qwen)).toBe("4.2 GB download");
    expect(imageSecondarySizeLabel(qwen)).toBeNull();
    expect(imageSizeSortKey(qwen)).toBe(4.2);
  });

  it("keeps what is on disk first once downloaded", () => {
    expect(videoPrimarySizeLabel(video({ sizeGb: 71.1, coreWeightsGb: 162.4, onDiskGb: 66.2 }))).toBe("66.2 GB on disk");
    expect(imagePrimarySizeLabel(image({ sizeGb: 20.3, onDiskGb: 19.2 }))).toBe("19.2 GB on disk");
  });

  it("falls back to the Hub figures only for a row with no catalogue size", () => {
    expect(imagePrimarySizeLabel(image({ sizeGb: 0, coreWeightsGb: 12 }))).toBe("12.0 GB weights");
    expect(compareDiscoverVariants("size", { name: "a", sizeGb: 2, coreWeightsGb: 236 }, { name: "b", sizeGb: 10 })).toBeGreaterThan(0);
  });
});

describe("one verdict per row (FU-472)", () => {
  const tight: Fit = { status: "tight", needGb: 20, usableGb: 22.5, headroomPct: 11, reason: "", basis: "measured" };
  it("drops the caution at or under the recommended size and clip", () => {
    expect(cautionCoveredByFit(tight, { width: 1024, height: 1024 }, { recommendedResolution: "1024x1024" })).toBe(true);
    expect(cautionCoveredByFit(tight, { width: 768, height: 512, frames: 97 }, { recommendedResolution: "768x512", defaultFrames: 97 })).toBe(true);
  });
  it("keeps it for a larger run, a longer clip, or a row that will not fit", () => {
    expect(cautionCoveredByFit(tight, { width: 2048, height: 2048 }, { recommendedResolution: "1024x1024" })).toBe(false);
    expect(cautionCoveredByFit(tight, { width: 768, height: 512, frames: 121 }, { recommendedResolution: "768x512", defaultFrames: 97 })).toBe(false);
    expect(cautionCoveredByFit({ ...tight, status: "wont-fit" }, { width: 1024, height: 1024 }, { recommendedResolution: "1024x1024" })).toBe(false);
    expect(cautionCoveredByFit(null, { width: 1024, height: 1024 }, { recommendedResolution: "1024x1024" })).toBe(false);
  });
});

describe("a distilled row starts at its own steps (FU-472)", () => {
  it("applies FLUX 3 Klein's 8 steps and guidance 1, not the Balanced preset's 24", () => {
    expect(stepsForPickedVariant({ defaultSteps: 8, cfgOverride: 1.0 }, "balanced", false)).toEqual({ steps: 8, guidance: 1 });
  });
  it("goes back to the preset after a distilled row, and otherwise leaves the user's steps alone", () => {
    expect(stepsForPickedVariant({}, "balanced", true)).toEqual({ steps: 24, guidance: 6 });
    expect(stepsForPickedVariant({}, "balanced", false)).toBeNull();
  });
});

describe("Video Studio opens on the clip its chip names (FU-475)", () => {
  it("makes 4.0 s at 24 fps 97 frames, 5.0 s 121, and keeps 33 without a duration", () => {
    expect(defaultNumFramesFor(4)).toBe(97);
    expect(defaultNumFramesFor(5)).toBe(121);
    expect(defaultNumFramesFor(2)).toBe(49);
    expect(defaultNumFramesFor(undefined)).toBe(33);
  });
});

describe("Image Studio never opens on a row whose engine is missing (FU-478)", () => {
  const nunchaku = image({ id: "nunchaku", nunchakuRepo: "mit-han-lab/svdq-int4-flux.1-schnell", availableLocally: true });
  const gguf = image({ id: "gguf", availableLocally: true });
  const families: ImageModelFamily[] = [
    { id: "f", name: "F", provider: "P", headline: "", summary: "", updatedLabel: "", badges: [], defaultVariantId: "nunchaku", variants: [nunchaku, gguf] },
  ] as ImageModelFamily[];
  it("skips the Nunchaku row while the package is not installed", () => {
    expect(imageVariantEngineMissing(nunchaku, false)).toBe(true);
    expect(defaultImageVariantFor(families, false)?.id).toBe("gguf");
  });
  it("keeps it when the package is there or the backend has not said yet", () => {
    expect(defaultImageVariantFor(families, true)?.id).toBe("nunchaku");
    expect(imageVariantEngineMissing(nunchaku, null)).toBe(false);
  });
});

describe("Make without a model (FU-474)", () => {
  it("joins the estimate's sentences, or says nothing when every model is installed", () => {
    expect(modelsMissingReason({ missing: ["No image model is installed.", "No video model is installed."] }))
      .toBe("No image model is installed. No video model is installed.");
    expect(modelsMissingReason({ missing: [] })).toBeNull();
    expect(modelsMissingReason(null)).toBeNull();
  });
});
