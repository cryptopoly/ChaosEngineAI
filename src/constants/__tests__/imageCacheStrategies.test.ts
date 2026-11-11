import { describe, expect, it } from "vitest";
import { imageCacheStrategiesForRepo, isCacheUselessImageRepo } from "../image";

const ids = (repo: string) => imageCacheStrategiesForRepo(repo).map((s) => s.id);

describe("imageCacheStrategiesForRepo", () => {
  it("offers only Off on FLUX.2 Klein and other step-distilled rows", () => {
    for (const repo of [
      "black-forest-labs/FLUX.2-klein-4B",
      "black-forest-labs/FLUX.1-schnell",
      "Tongyi-MAI/Z-Image-Turbo",
    ]) {
      expect(isCacheUselessImageRepo(repo)).toBe(true);
      expect(ids(repo)).toEqual(["none"]);
    }
  });

  it("keeps the caches on FLUX.1-dev and on UNet rows as before", () => {
    expect(ids("black-forest-labs/FLUX.1-dev")).toContain("teacache");
    expect(ids("black-forest-labs/FLUX.1-dev")).toContain("fbcache");
    expect(ids("stabilityai/stable-diffusion-xl-base-1.0")).toEqual(["none"]);
    expect(ids("Qwen/Qwen-Image")).not.toContain("teacache");
  });
});
