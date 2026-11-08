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
