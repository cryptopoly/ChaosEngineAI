import { describe, expect, it } from "vitest";
import {
  IMAGE_ASPECTS,
  VIDEO_ASPECTS,
  fitsLimit,
  nearestAspect,
  nearestMegapixels,
  nearestVideoResolution,
  sizeForPixels,
  sizeForShortEdge,
} from "../sizePresets";

const aspect = (id: string) => [...IMAGE_ASPECTS, ...VIDEO_ASPECTS].find((a) => a.id === id)!;

describe("sizePresets", () => {
  it("gives multiples of 64 near the asked pixel count", () => {
    const one = sizeForPixels(aspect("1:1"), 1_000_000);
    expect(one).toEqual({ width: 1024, height: 1024 });
    const wide = sizeForPixels(aspect("16:9"), 2_000_000);
    expect(wide.width % 64).toBe(0);
    expect(wide.height % 64).toBe(0);
    expect(Math.abs(wide.width * wide.height - 2_000_000)).toBeLessThan(250_000);
    expect(wide.width).toBeGreaterThan(wide.height);
  });

  it("names video resolutions by the short edge, both orientations", () => {
    expect(sizeForShortEdge(aspect("16:9"), 1080)).toEqual({ width: 1920, height: 1088 });
    expect(sizeForShortEdge(aspect("9:16"), 1080)).toEqual({ width: 1088, height: 1920 });
    expect(sizeForShortEdge(aspect("1:1"), 720)).toEqual({ width: 704, height: 704 });
  });

  it("greys out sizes past the model limits", () => {
    const limit = { maxSide: 2048, maxPixels: 4_194_304 };
    expect(fitsLimit({ width: 2048, height: 2048 }, limit)).toBe(true);
    expect(fitsLimit({ width: 2560, height: 1408 }, limit)).toBe(false);
    expect(fitsLimit({ width: 2048, height: 2112 }, limit)).toBe(false);
    expect(fitsLimit({ width: 512, height: 512 }, { minSide: 768 })).toBe(false);
  });

  it("finds the matching button for the current size", () => {
    expect(nearestAspect(1344, 768, IMAGE_ASPECTS)?.id).toBe("16:9");
    expect(nearestAspect(1000, 1200, IMAGE_ASPECTS)).toBeNull();
    expect(nearestMegapixels(1024, 1024)?.id).toBe("1");
    expect(nearestMegapixels(2048, 2048)?.id).toBe("4");
    expect(nearestVideoResolution(1280, 704)?.id).toBe("hd");
    expect(nearestVideoResolution(768, 512)?.id).toBe("sd");
    expect(nearestVideoResolution(640, 384)).toBeNull();
  });
});
