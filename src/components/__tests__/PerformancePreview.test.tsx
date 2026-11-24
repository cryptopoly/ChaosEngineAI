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

describe("PerformancePreview memory bar", () => {
  const withMemory = (overrides: Partial<typeof base> & Record<string, unknown> = {}, props: Record<string, unknown> = {}) =>
    renderToStaticMarkup(
      <PerformancePreview
        preview={{
          ...base,
          bits: 0,
          qualityPercent: 100,
          diskSizeGb: 15,
          memory: { kind: "metal", limitGb: 51.8, totalGb: 64, freeGb: null },
          ...overrides,
        }}
        availableMemoryGb={38.6}
        totalMemoryGb={64}
        {...props}
      />,
    );

  it("splits the bar into what is in use now, the model and the cache, and says what it needs", () => {
    const html = withMemory({ optimizedCacheGb: 25.3 }, { fitCacheGb: 25.3, actualDiskSizeGb: 15 });
    expect(html).toContain("In use now 25.4 GB");
    expect(html).toContain("Model 15.0 GB");
    expect(html).toContain("Cache 25.3 GB");
    expect(html).toContain("Needs 40.3 GB. 38.6 GB free now. GPU limit 51.8 GB.");
  });

  it("warns when the launch is more than the free memory right now, without saying it cannot start", () => {
    const html = withMemory({ optimizedCacheGb: 25.3 }, { fitCacheGb: 25.3, actualDiskSizeGb: 15 });
    expect(html).toContain("More than the free memory right now");
    expect(html).not.toContain("will not start as set");
  });

  it("says so plainly when the launch is more than the ceiling allows", () => {
    const html = withMemory({ optimizedCacheGb: 40 }, { fitCacheGb: 40, actualDiskSizeGb: 15 });
    expect(html).toContain("will not start as set");
  });

  it("measures an NVIDIA host against the card, not system RAM", () => {
    const html = withMemory(
      { optimizedCacheGb: 3, memory: { kind: "vram", limitGb: 24, totalGb: 24, freeGb: 20 } },
      { actualDiskSizeGb: 15 },
    );
    expect(html).toContain("In use now 4.0 GB");
    expect(html).toContain("Card memory 24.0 GB");
  });
});
