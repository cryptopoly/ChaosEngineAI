import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { RunEstimateStrip } from "../RunEstimateStrip";
import { ProjectChip, ProjectFilter, ProjectHeader, ProjectList, projectRows } from "../ProjectFilter";
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

  it("names a project as on its newest output", () => {
    const rows = [
      { projectId: "a", projectName: "Old name", createdAt: "2026-11-01T00:00:00Z" },
      { projectId: "a", projectName: "New name", createdAt: "2026-11-03T00:00:00Z" },
      { projectId: "a", projectName: "Middle", createdAt: "2026-11-02T00:00:00Z" },
    ];
    expect(projectOptions(rows)[0].name).toBe("New name");
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

  it("lists All projects, No project and each project with its count in the open list", () => {
    const rows = projectRows(projectOptions(items), "", { all: "All projects", none: "No project" }, 5, 2);
    expect(rows.map((r) => [r.id, r.count])).toEqual([["all", 5], ["__none__", 2], ["p2", 1], ["p1", 2]]);
    const html = renderToStaticMarkup(<ProjectList id="l" rows={rows} value="p1" searchable={false} query="" onQuery={() => {}} onPick={() => {}} />);
    expect(html).toContain('role="listbox"');
    expect(html).toContain('<button type="button" role="option" aria-selected="true"');
    expect(html.indexOf("Alpine")).toBeLessThan(html.indexOf("Harbour"));
    expect(html).not.toContain('type="search"');
  });

  it("leaves out No project when nothing is untagged, hides All and None while searching, and adds a search box on request", () => {
    const labels = { all: "All projects", none: "No project" };
    expect(projectRows(projectOptions(items), "", labels, 5, 0).map((r) => r.id)).toEqual(["all", "p2", "p1"]);
    expect(projectRows(projectOptions(items), "harb", labels, 5, 2).map((r) => r.id)).toEqual(["p1"]);
    const html = renderToStaticMarkup(<ProjectList id="l" rows={[]} value="all" searchable query="zz" onQuery={() => {}} onPick={() => {}} />);
    expect(html).toContain('type="search"');
    expect(html).toContain("No project matches.");
  });

  it("can drop the visible label and keeps an accessible name", () => {
    const html = renderToStaticMarkup(<ProjectFilter options={projectOptions(items)} value="p1" onChange={() => {}} showLabel={false} />);
    expect(html).not.toContain("project-filter__label");
    expect(html).toContain('aria-label="Project: Harbour"');
  });

  it("header shows the name, the counts and the Storyboard link only when given a handler", () => {
    const withLink = renderToStaticMarkup(<ProjectHeader name="Harbour" counts={{ stills: 3, clips: 1 }} onOpen={() => {}} />);
    expect(withLink).toContain("Harbour");
    expect(withLink).toContain("3 stills");
    expect(withLink).toContain("1 clip");
    expect(withLink).toContain("Open in Storyboard");
    const without = renderToStaticMarkup(<ProjectHeader name="Harbour" counts={{ stills: 0, tracks: 2, transcripts: 1 }} />);
    expect(without).toContain("2 tracks");
    expect(without).toContain("1 transcript");
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
