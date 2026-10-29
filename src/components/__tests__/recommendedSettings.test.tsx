import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

import type { RecommendedSettings } from "../../types/recommendations";
import { RecommendedSettingsPanel } from "../RecommendedSettingsPanel";
import {
  applyRecommendationRows,
  clampStudioSide,
  recommendationRows,
  type RecommendationAppliers,
} from "../recommendedSettings";

function makeRec(overrides: Partial<RecommendedSettings> = {}): RecommendedSettings {
  return {
    hardwareBucket: "apple-64",
    hardwareLabel: "Apple Silicon, 64 GB",
    basis: "estimated",
    basisByKey: { steps: "measured", width: "measured", height: "measured", cacheStrategy: "estimated", cacheThreshold: "estimated" },
    measuredOn: "Apple M4 Max, 64 GB",
    settings: { steps: 8, width: 1024, height: 1024, cacheStrategy: "fbcache", cacheThreshold: 0.12 },
    reasons: { steps: "Distilled to 8 steps.", width: "Native size.", cacheStrategy: "Skips repeated work." },
    ...overrides,
  };
}

function makeAppliers(): Required<Pick<RecommendationAppliers, "steps" | "width" | "height" | "cacheStrategy" | "cacheThreshold">> {
  return {
    steps: vi.fn(),
    width: vi.fn(),
    height: vi.fn(),
    cacheStrategy: vi.fn(),
    cacheThreshold: vi.fn(),
  };
}

describe("recommendationRows", () => {
  it("groups width/height and cache strategy/threshold into one row each", () => {
    const rows = recommendationRows(makeRec(), makeAppliers());
    expect(rows.map((row) => row.id)).toEqual(["steps", "size", "cache"]);
    expect(rows.find((row) => row.id === "cache")?.keys).toEqual(["cacheStrategy", "cacheThreshold"]);
  });

  it("drops a setting the Studio cannot apply (no setter means the control is hidden)", () => {
    const appliers: RecommendationAppliers = { ...makeAppliers(), cacheStrategy: undefined };
    const rows = recommendationRows(makeRec(), appliers);
    expect(rows.map((row) => row.id)).toEqual(["steps", "size"]);
  });

  it("labels a row measured only when every key of it is measured", () => {
    const rows = recommendationRows(
      makeRec({ basisByKey: { steps: "measured", width: "measured", height: "estimated" } }),
      makeAppliers(),
    );
    expect(rows.find((row) => row.id === "steps")?.basis).toBe("measured");
    expect(rows.find((row) => row.id === "size")?.basis).toBe("estimated");
  });

  it("falls back to the overall basis when there is no per-key basis", () => {
    const rows = recommendationRows(makeRec({ basis: "estimated", basisByKey: undefined }), makeAppliers());
    expect(rows.every((row) => row.basis === "estimated")).toBe(true);
  });

  it("marks rows whose values the Studio already uses", () => {
    const rows = recommendationRows(makeRec(), makeAppliers(), {
      steps: 8,
      width: 1024,
      height: 768,
      cacheStrategy: "fbcache",
      cacheThreshold: 0.12,
    });
    expect(Object.fromEntries(rows.map((row) => [row.id, row.inUse]))).toEqual({
      steps: true,
      size: false,
      cache: true,
    });
  });

  it("returns nothing for a missing recommendation", () => {
    expect(recommendationRows(null, makeAppliers())).toEqual([]);
    expect(recommendationRows(undefined, makeAppliers())).toEqual([]);
  });
});

describe("applyRecommendationRows", () => {
  it("applies every value of the given rows through the Studio setters", () => {
    const appliers = makeAppliers();
    const rows = recommendationRows(makeRec(), appliers);
    const applied = applyRecommendationRows(rows, appliers);
    expect(applied).toEqual(["steps", "width", "height", "cacheStrategy", "cacheThreshold"]);
    expect(appliers.steps).toHaveBeenCalledWith(8);
    expect(appliers.width).toHaveBeenCalledWith(1024);
    expect(appliers.height).toHaveBeenCalledWith(1024);
    expect(appliers.cacheStrategy).toHaveBeenCalledWith("fbcache");
    expect(appliers.cacheThreshold).toHaveBeenCalledWith(0.12);
  });

  it("applies one row only, for the per-row Use action", () => {
    const appliers = makeAppliers();
    const rows = recommendationRows(makeRec(), appliers);
    applyRecommendationRows(rows.filter((row) => row.id === "steps"), appliers);
    expect(appliers.steps).toHaveBeenCalledTimes(1);
    expect(appliers.width).not.toHaveBeenCalled();
    expect(appliers.cacheStrategy).not.toHaveBeenCalled();
  });

  it("applies a quality preset before explicit steps so the steps win", () => {
    const calls: string[] = [];
    const appliers: RecommendationAppliers = {
      qualityPreset: () => calls.push("preset"),
      steps: () => calls.push("steps"),
    };
    const rows = recommendationRows(
      makeRec({ settings: { steps: 8, qualityPreset: "fast" }, basisByKey: undefined }),
      appliers,
    );
    applyRecommendationRows([...rows].reverse(), appliers);
    expect(calls).toEqual(["preset", "steps"]);
  });
});

describe("clampStudioSide", () => {
  it("caps at the model's maxSide and defaults to 4096", () => {
    expect(clampStudioSide(5000, { maxSide: 2048 }, 1024)).toBe(2048);
    expect(clampStudioSide(5000, {}, 1024)).toBe(4096);
    expect(clampStudioSide(3000, { maxSide: null }, 1024)).toBe(3000);
  });

  it("raises to the model's minSide (never below 256) and replaces invalid input", () => {
    expect(clampStudioSide(512, { minSide: 768 }, 1024)).toBe(768);
    expect(clampStudioSide(100, {}, 1024)).toBe(256);
    expect(clampStudioSide(Number.NaN, { maxSide: 2048 }, 832)).toBe(832);
  });
});

describe("RecommendedSettingsPanel", () => {
  it("renders the hardware label, each row with its reason and a measured / estimated chip", () => {
    const html = renderToStaticMarkup(
      <RecommendedSettingsPanel recommendation={makeRec()} appliers={makeAppliers()} current={{ steps: 8 }} />,
    );
    expect(html).toContain("Recommended for this machine");
    expect(html).toContain("Apple Silicon, 64 GB");
    expect(html).toContain("Distilled to 8 steps.");
    expect(html).toContain("1024 x 1024");
    expect(html).toContain("Measured");
    expect(html).toContain("Estimated");
    expect(html).toContain("In use");
    expect(html).toContain("Apply recommended");
  });

  it("renders nothing when no recommended setting can be applied", () => {
    expect(
      renderToStaticMarkup(<RecommendedSettingsPanel recommendation={makeRec()} appliers={{}} current={{}} />),
    ).toBe("");
    expect(
      renderToStaticMarkup(<RecommendedSettingsPanel recommendation={null} appliers={makeAppliers()} current={{}} />),
    ).toBe("");
  });
});
