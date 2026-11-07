import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { RunEstimateStrip } from "../RunEstimateStrip";
import { ProjectChip, ProjectFilter } from "../ProjectFilter";
import { ALL_PROJECTS, filterByProject, projectOptions } from "../../utils/galleryProject";

const items = [
  { id: "a", projectId: "p1", projectName: "Harbour" },
  { id: "b", projectId: "p1", projectName: "Harbour" },
  { id: "c", projectId: "p2", projectName: "Alpine" },
  { id: "d" },
  { id: "e", projectId: null, projectName: null },
];

describe("gallery project helpers", () => {
  it("lists projects with outputs, most first, ignoring untagged items", () => {
    expect(projectOptions(items)).toEqual([
      { id: "p1", name: "Harbour", count: 2 },
      { id: "p2", name: "Alpine", count: 1 },
    ]);
    expect(projectOptions([{}])).toEqual([]);
  });

  it("filters by project id and passes everything for 'all'", () => {
    expect(filterByProject(items, ALL_PROJECTS)).toHaveLength(5);
    expect(filterByProject(items, "p1").map((i) => i.id)).toEqual(["a", "b"]);
    expect(filterByProject(items, "nope")).toEqual([]);
  });
});

describe("ProjectFilter and ProjectChip", () => {
  it("renders the select with an All option and one per project; nothing without projects", () => {
    const html = renderToStaticMarkup(<ProjectFilter options={projectOptions(items)} value="all" onChange={() => {}} />);
    expect(html).toContain("All projects");
    expect(html).toContain("Harbour");
    expect(html).toContain("Alpine");
    expect(renderToStaticMarkup(<ProjectFilter options={[]} value="all" onChange={() => {}} />)).toBe("");
  });

  it("shows the From project chip only when named", () => {
    expect(renderToStaticMarkup(<ProjectChip name="Harbour" />)).toContain("From project Harbour");
    expect(renderToStaticMarkup(<ProjectChip name={null} />)).toBe("");
  });
});

describe("RunEstimateStrip busy note", () => {
  const base = { seconds: 10, peakGb: 10, totalGb: 64, diskMb: null, basis: "estimated" as const, risk: "safe" as const, action: null };
  const advice = { level: "risky" as const, message: "Estimated peak 60.0 GB of 64.0 GB RAM.", jobs: [], combinedPeakGb: 60, totalGb: 64, availableGb: 4, footprintGb: 10 };

  it("shows the advice message under the strip", () => {
    expect(renderToStaticMarkup(<RunEstimateStrip {...base} busy={advice} />)).toContain("Estimated peak 60.0 GB of 64.0 GB RAM.");
  });

  it("shows nothing for ok or missing advice", () => {
    expect(renderToStaticMarkup(<RunEstimateStrip {...base} busy={{ ...advice, level: "ok", message: "" }} />)).not.toContain("run-estimate__busy");
    expect(renderToStaticMarkup(<RunEstimateStrip {...base} />)).not.toContain("run-estimate__busy");
  });
});
