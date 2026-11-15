import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { StoryboardEstimate } from "../../../api/storyboard";
import { RunBar, type RunBarProps } from "../RunBar";

const noop = () => undefined;
const estimate = (extra: Partial<StoryboardEstimate> = {}): StoryboardEstimate => ({
  what: "both", shots: 3, stills: 2, clips: 3, seconds: 190, peakGb: 12, totalGb: 48, diskMb: 40, basis: "estimated",
  missing: [], imageModel: "Img", videoModel: "Vid", ...extra,
});
const bar = (extra: Partial<RunBarProps> = {}) =>
  renderToStaticMarkup(
    <RunBar missing={3} stills={1} stale={0} unlocked={3} estimate={estimate()} batch={null} running={false} busy={false}
      onMakeMissing={noop} onMakeAll={noop} onCancel={noop} {...extra} />,
  );

describe("Storyboard run bar", () => {
  it("puts the count and the time on Make, and says a still-only shot needs its clip", () => {
    const html = bar();
    expect(html).toContain("Make 3 missing · about 3 min 10 s");
    expect(html).toContain("3 shots need work: 1 has a still and needs its clip");
    expect(html).toContain("Estimated");
  });

  it("names a missing model and keeps the count without a time", () => {
    const html = bar({ estimate: estimate({ seconds: null, peakGb: null, missing: ["No video model is installed."] }) });
    expect(html).toContain("No video model is installed.");
    expect(html).toMatch(/class="primary-button">Make 3 missing</);
  });

  it("turns red and holds Make until the risk is accepted when it will not fit", () => {
    const html = bar({ estimate: estimate({ peakGb: 47, totalGb: 48 }) });
    expect(html).toContain("run-bar--danger");
    expect(html).toContain("Run anyway. I accept that the backend may crash.");
    expect(html).toMatch(/class="primary-button" disabled="">Make 3 missing/);
  });

  it("says when nothing is missing", () => {
    const html = bar({ missing: 0, stills: 0, estimate: null });
    expect(html).toContain("Every unlocked shot has a take.");
    expect(html).toMatch(/disabled="">Make missing/);
  });
});

describe("Storyboard run bar: draft and final passes", () => {
  const passes = {
    draft: { shots: 3, stills: 3, clips: 3, seconds: 45, stillSeconds: 45, clipSeconds: 150, basis: "estimated" as const,
      imageModel: "Iris-3B Turbo", videoModel: "Vid", width: 1280, height: 720 },
    final: { shots: 1, stills: 1, clips: 1, seconds: 500, stillSeconds: 450, clipSeconds: 50, basis: "estimated" as const,
      imageModel: "Qwen-Image 2.1", videoModel: "Vid", width: 1280, height: 720, assumedAllKept: false },
  };

  it("shows both passes with their times and models", () => {
    const html = bar({ estimate: estimate({ passes }), kept: 1, finalShots: 1, onMakeDrafts: noop, onFinalPass: noop });
    expect(html).toContain("Make drafts · about 45 s");
    expect(html).toContain("Final pass on 1 kept shot · about 8 min 20 s");
    expect(html).toContain("Quick stills with Iris-3B Turbo at 1280 x 720");
    expect(html).toContain("again with Qwen-Image 2.1");
    expect(html).not.toContain("Mark shots with Keep");
  });

  it("holds the final pass until a shot is kept, and says how", () => {
    const html = bar({ estimate: estimate({ passes: { ...passes, final: { ...passes.final, assumedAllKept: true } } }),
      kept: 0, finalShots: 0, onMakeDrafts: noop, onFinalPass: noop });
    expect(html).toMatch(/disabled="" title="[^"]*">Final pass on 0 kept shots</);
    expect(html).toContain("Mark shots with Keep to choose them for the final pass.");
  });

  it("has no passes row when the screen does not offer them", () => {
    expect(bar()).not.toContain("Make drafts");
  });
});
