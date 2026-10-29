import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { BestFitRibbon, FitBadge } from "../FitBadge";
import { fitSortRank, readBestFitVariantId, readFit, readVariantFit, type Fit } from "../fit";

function makeFit(overrides: Partial<Fit> = {}): Fit {
  return {
    status: "fits",
    needGb: 18.4,
    usableGb: 51.8,
    headroomPct: 64,
    reason: "Weights 15.9 GB + 2.5 GB cache at 8K context",
    basis: "estimated",
    ...overrides,
  };
}

describe("readFit", () => {
  it("reads a well-formed fit", () => {
    expect(readFit(makeFit())).toEqual(makeFit());
  });

  it("returns null for missing or malformed values", () => {
    expect(readFit(undefined)).toBeNull();
    expect(readFit(null)).toBeNull();
    expect(readFit("fits")).toBeNull();
    expect(readFit({ status: "maybe" })).toBeNull();
    expect(readFit({ reason: "no status" })).toBeNull();
  });

  it("keeps an unknown verdict with null numbers", () => {
    expect(readFit({ status: "unknown", needGb: null, usableGb: 51.8, headroomPct: null, reason: "Mac-only engine", basis: "estimated" })).toEqual({
      status: "unknown",
      needGb: null,
      usableGb: 51.8,
      headroomPct: null,
      reason: "Mac-only engine",
      basis: "estimated",
    });
  });

  it("treats anything but measured as estimated and fills missing fields", () => {
    expect(readFit({ status: "tight", basis: "guess" })).toEqual({
      status: "tight",
      needGb: null,
      usableGb: 0,
      headroomPct: null,
      reason: "",
      basis: "estimated",
    });
  });

  it("reads the fit of a variant and the family's best fit id", () => {
    expect(readVariantFit({ id: "a", fit: makeFit({ status: "tight" }) })?.status).toBe("tight");
    expect(readVariantFit({ id: "a" })).toBeNull();
    expect(readBestFitVariantId({ bestFitVariantId: "variant-1" })).toBe("variant-1");
    expect(readBestFitVariantId({ bestFitVariantId: null })).toBeNull();
    expect(readBestFitVariantId({ bestFitVariantId: "" })).toBeNull();
    expect(readBestFitVariantId({})).toBeNull();
  });

  it("ranks fits before tight before unknown before won't fit", () => {
    const ranks = (["fits", "tight", "unknown", "wont-fit"] as const).map((status) => fitSortRank(makeFit({ status })));
    expect(ranks).toEqual([0, 1, 2, 3]);
    expect(fitSortRank(null)).toBe(4);
  });
});

describe("FitBadge", () => {
  it("renders nothing without a verdict", () => {
    expect(renderToStaticMarkup(<FitBadge fit={null} />)).toBe("");
    expect(renderToStaticMarkup(<FitBadge fit={undefined} />)).toBe("");
  });

  it("shows Fits with the reason as tooltip and marks an estimate visibly", () => {
    const markup = renderToStaticMarkup(<FitBadge fit={makeFit()} />);
    expect(markup).toContain("fit-badge--fits");
    expect(markup).toContain(">Fits<");
    expect(markup).toContain('class="fit-badge__basis">est.<');
    expect(markup).toContain("Weights 15.9 GB + 2.5 GB cache at 8K context");
    expect(markup).toContain("Needs about 18 GB of 52 GB usable on this machine.");
    expect(markup).toContain("Estimated from the model size");
    expect(markup).toContain('data-fit-basis="estimated"');
  });

  it("does not mark a measured verdict as an estimate", () => {
    const markup = renderToStaticMarkup(<FitBadge fit={makeFit({ status: "tight", basis: "measured" })} />);
    expect(markup).toContain("fit-badge--tight");
    expect(markup).toContain(">Tight<");
    expect(markup).not.toContain("fit-badge__basis");
    expect(markup).toContain("Measured on this machine.");
  });

  it("shows Won't fit from the status, not from the sign of the headroom", () => {
    // 95 % of usable memory: still positive headroom, but the backend says it will not fit.
    const markup = renderToStaticMarkup(<FitBadge fit={makeFit({ status: "wont-fit", needGb: 49.2, headroomPct: 5 })} />);
    expect(markup).toContain("fit-badge--wont-fit");
    expect(markup).toContain("Won&#x27;t fit");
  });

  it("shows unknown as a neutral chip with the reason, without numbers or an estimate mark", () => {
    const markup = renderToStaticMarkup(
      <FitBadge fit={makeFit({ status: "unknown", needGb: null, headroomPct: null, reason: "Runs only on Apple Silicon" })} />,
    );
    expect(markup).toContain("fit-badge--unknown");
    expect(markup).toContain("Not for this machine");
    expect(markup).toContain("Runs only on Apple Silicon");
    expect(markup).not.toContain("Needs about");
    expect(markup).not.toContain("fit-badge__basis");
  });

  it("renders the Best fit ribbon with an explanation", () => {
    const markup = renderToStaticMarkup(<BestFitRibbon />);
    expect(markup).toContain("best-fit-ribbon");
    expect(markup).toContain("Best fit");
    expect(markup).toContain("newest release that fits comfortably");
  });
});
