import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { PerformancePreview } from "../PerformancePreview";
import { emptyWorkspace } from "../../defaults";

const base = { ...emptyWorkspace.preview, baselineCacheGb: 4, optimizedCacheGb: 1.8, estimatedTokS: 30 };
const render = (bits: number, qualityPercent: number) =>
  renderToStaticMarkup(
    <PerformancePreview preview={{ ...base, bits, qualityPercent }} availableMemoryGb={32} totalMemoryGb={64} />,
  );

describe("PerformancePreview quality", () => {
  it("marks the quality of a compressed cache as an estimate", () => {
    const html = render(3, 97.6);
    expect(html).toContain("97.6%");
    expect(html).toContain('data-testid="perf-quality-estimate"');
  });

  it("shows no estimate mark for the full-precision cache, which is the reference", () => {
    expect(render(0, 100)).not.toContain('data-testid="perf-quality-estimate"');
  });
});
