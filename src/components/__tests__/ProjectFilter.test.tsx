import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { RunEstimateStrip } from "../RunEstimateStrip";
import { ProjectChip, ProjectFilter, ProjectHeader } from "../ProjectFilter";
import { ALL_PROJECTS, NO_PROJECT, filterByProject, projectOptions, searchProjects, untaggedCount } from "../../utils/galleryProject";

const items = [
  { id: "a", projectId: "p1", projectName: "Harbour", createdAt: "2026-11-01T00:00:00Z" },
  { id: "b", projectId: "p1", projectName: "Harbour", createdAt: "2026-11-02T00:00:00Z" },
  { id: "c", projectId: "p2", projectName: "Alpine", createdAt: "2026-11-05T00:00:00Z" },
  { id: "d" },
  { id: "e", projectId: null, projectName: null },
];

describe("gallery project helpers", () => {
  it("lists projects with outputs, newest output first, ignoring untagged items", () => {
    expect(projectOptions(items)).toEqual([
      { id: "p2", name: "Alpine", count: 1, latest: Date.parse("2026-11-05T00:00:00Z") },
      { id: "p1", name: "Harbour", count: 2, latest: Date.parse("2026-11-02T00:00:00Z") },
    ]);
    expect(projectOptions([{}])).toEqual([]);
  });

  it("reads unix seconds (music, voice) as well as ISO text, and falls back to name order", () => {
    const rows = [
      { projectId: "a", projectName: "Zed", createdAt: 1_800_000_000 },
      { projectId: "b", projectName: "Amy", createdAt: 1_700_000_000 },
      { projectId: "c", projectName: "Bob", createdAt: 1_700_000_000 },
    ];
    expect(projectOptions(rows).map((o) => o.id)).toEqual(["a", "b", "c"]);
    expect(projectOptions([{ projectId: "x", projectName: "No date" }])[0].latest).toBe(0);
  });

  it("counts untagged items and filters them with NO_PROJECT", () => {
    expect(untaggedCount(items)).toBe(2);
    expect(filterByProject(items, NO_PROJECT).map((i) => i.id)).toEqual(["d", "e"]);
  });

  it("searches project names case-insensitively", () => {
    const options = projectOptions(items);
    expect(searchProjects(options, " harb ").map((o) => o.id)).toEqual(["p1"]);
    expect(searchProjects(options, "")).toHaveLength(2);
    expect(searchProjects(options, "zzz")).toEqual([]);
  });

  it("filters by project id and passes everything for 'all'", () => {
    expect(filterByProject(items, ALL_PROJECTS)).toHaveLength(5);
    expect(filterByProject(items, "p1").map((i) => i.id)).toEqual(["a", "b"]);
    expect(filterByProject(items, "nope")).toEqual([]);
  });
});

describe("ProjectFilter and ProjectChip", () => {
  it("shows the current choice closed; nothing without projects", () => {
    const closed = renderToStaticMarkup(<ProjectFilter options={projectOptions(items)} value="p1" onChange={() => {}} />);
    expect(closed).toContain("Harbour");
    expect(closed).not.toContain("project-filter__panel");
    expect(renderToStaticMarkup(<ProjectFilter options={[]} value="all" onChange={() => {}} />)).toBe("");
  });

  it("lists All projects, No project and each project with its count when open", () => {
    const html = renderToStaticMarkup(
      <ProjectFilter options={projectOptions(items)} value="all" onChange={() => {}} untagged={2} total={5} initialOpen />,
    );
    expect(html).toContain("All projects");
    expect(html).toContain("No project");
    expect(html.indexOf("Alpine")).toBeLessThan(html.indexOf("Harbour"));
    expect(html).toContain(">2</span>");
    expect(html).not.toContain('type="search"');
  });

  it("leaves out No project when nothing is untagged and adds a search box for many projects", () => {
    expect(renderToStaticMarkup(<ProjectFilter options={projectOptions(items)} value="all" onChange={() => {}} initialOpen />)).not.toContain("No project");
    const many = Array.from({ length: 9 }, (_, i) => ({ id: `p${i}`, name: `Project ${i}`, count: 1, latest: i }));
    expect(renderToStaticMarkup(<ProjectFilter options={many} value="all" onChange={() => {}} initialOpen />)).toContain('type="search"');
  });

  it("header shows the name, the counts and the Storyboard link only when given a handler", () => {
    const withLink = renderToStaticMarkup(<ProjectHeader name="Harbour" counts={{ stills: 3, clips: 1 }} onOpen={() => {}} />);
    expect(withLink).toContain("Harbour");
    expect(withLink).toContain("3 stills");
    expect(withLink).toContain("1 clip");
    expect(withLink).toContain("Open in Storyboard");
    const without = renderToStaticMarkup(<ProjectHeader name="Harbour" counts={{ stills: 0, tracks: 2 }} />);
    expect(without).toContain("2 tracks");
    expect(without).not.toContain("still");
    expect(without).not.toContain("Open in Storyboard");
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
