import type { ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { StoryboardProject } from "../../../../types";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import { ProjectDialog } from "../../ProjectDialog";
import { ProjectShellHeader } from "../../ProjectShellHeader";

const base = {
  id: "p1", name: "Night Drive", revision: 1, undoStack: [], redoStack: [],
  defaults: { exportPreset: "16:9", resolution: "720p", fps: 24 },
  references: { characters: [], places: [] },
  shots: [],
} as unknown as StoryboardProject;

function header(project: StoryboardProject, tab: "overview" | "freestyle" = "overview", rail?: ReactNode) {
  return renderToStaticMarkup(
    <ProjectShellHeader name="Night Drive" project={project} tab={tab} busy={false} conflict={false}
      onDismissConflict={() => undefined} onBack={() => undefined} onRename={() => undefined}
      onUndo={() => undefined} onRedo={() => undefined} onTab={() => undefined} rail={rail} />,
  );
}

describe("project header with the stage rail and More (audit changes 1 and 12)", () => {
  it("puts the rail between the top row and the tabs", () => {
    const html = header(base, "overview", <nav className="stage-rail">rail</nav>);
    expect(html.indexOf("stage-rail")).toBeGreaterThan(html.indexOf("project-shell__top"));
    expect(html.indexOf("stage-rail")).toBeLessThan(html.indexOf('role="tablist"'));
  });

  it("keeps Board behind More until it has cards, then shows it as a tab", () => {
    const empty = header(base);
    expect(empty).not.toContain('id="project-tab-freestyle"');
    expect(empty).toContain(">More<");
    const used = header({ ...base, board: { pos: { s1: { x: 10, y: 20 } } } } as StoryboardProject);
    expect(used).toContain('id="project-tab-freestyle"');
    expect(used).not.toContain(">More<");
  });

  it("shows Board as the selected tab while it is open, from a remembered route", () => {
    const html = header(base, "freestyle");
    expect(html).toMatch(/id="project-tab-freestyle" aria-selected="true"/);
    expect(html).not.toContain(">More<");
  });
});

describe("new project dialog (audit change 9)", () => {
  it("offers Music video after Blank", () => {
    const html = renderToStaticMarkup(<ProjectDialog mode="create" onSubmit={() => undefined} onCancel={() => undefined} />);
    const blank = html.indexOf(">Blank<");
    const music = html.indexOf(">Music video<");
    expect(blank).toBeGreaterThan(0);
    expect(music).toBeGreaterThan(blank);
    expect(html).toContain("Starts with the song");
  });
});
