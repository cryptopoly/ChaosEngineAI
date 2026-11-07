import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { Sidebar } from "../Sidebar";
import { arrangeRows, buildItems, resolveGroupTarget, visibleSidebarTabs } from "../sidebarModel";
import { HOME_STUDIO_ENABLED, tabs } from "../../constants";
import type { TabId } from "../../types";

function render(props: { activeTab?: TabId; keepAllOpen?: boolean; homeStudioEnabled?: boolean; mode?: "collapsible" | "tabs" } = {}) {
  return renderToStaticMarkup(
    <Sidebar
      activeTab={props.activeTab ?? "image-studio"}
      onTabChange={() => {}}
      platform="Darwin"
      backendOnline
      engineLabel="MLX"
      mode={props.mode ?? "collapsible"}
      collapsedGroups={new Set()}
      onToggleGroupCollapsed={() => {}}
      lastChildByGroup={{}}
      onRememberLastChild={() => {}}
      keepAllOpen={props.keepAllOpen}
      homeStudioEnabled={props.homeStudioEnabled}
    />,
  );
}

function openGroupIds(html: string): string[] {
  return [...html.matchAll(/class="nav-group open"/g)].map(() => "open");
}

describe("home studio visibility", () => {
  it("ships with the flag off", () => {
    expect(HOME_STUDIO_ENABLED).toBe(false);
  });

  it("hides Music and Storyboard when the flag is false", () => {
    const html = render({ homeStudioEnabled: false });
    expect(html).not.toContain("Music");
    expect(html).not.toContain("Storyboard");
    expect(visibleSidebarTabs(tabs, "Darwin", false).some((t) => t.group === "music" || t.group === "storyboard")).toBe(false);
  });

  it("shows both groups when the flag is true", () => {
    const html = render({ homeStudioEnabled: true });
    expect(html).toContain("Music");
    expect(html).toContain("Storyboard");
  });

  it("opens Music on Models by default and on Studio once it is ready", () => {
    const children = tabs.filter((t) => t.group === "music");
    expect(resolveGroupTarget("music", children, {})).toBe("music-models");
    expect(resolveGroupTarget("music", children, {}, new Set(["music"]))).toBe("music-studio");
    expect(resolveGroupTarget("music", children, { music: "music-library" }, new Set(["music"]))).toBe("music-library");
    const story = tabs.filter((t) => t.group === "storyboard");
    expect(resolveGroupTarget("storyboard", story, {})).toBe("story-projects");
    expect(resolveGroupTarget("storyboard", story, {}, new Set(["storyboard"]))).toBe("story-board");
  });
});

describe("sidebar sections and accordion", () => {
  it("puts Chat on top, then Make and Run section labels around the groups", () => {
    const rows = arrangeRows(buildItems(visibleSidebarTabs(tabs, "Darwin", true)));
    const names = rows.map((row) => (row.kind === "section" ? `#${row.section}` : row.kind === "group" ? row.id : row.tab.id));
    expect(names).toEqual([
      "dashboard", "chat",
      "#make", "images", "video", "voice", "music", "storyboard",
      "#run", "benchmarks", "tools",
      "server", "logs", "settings",
    ]);
  });

  it("renders thin non-clickable labels", () => {
    const html = render();
    expect(html).toContain('<div class="nav-section-label">Make</div>');
    expect(html).toContain('<div class="nav-section-label">Run</div>');
    expect(html).not.toContain("<button type=\"button\" class=\"nav-section-label");
  });

  it("opens only the active group by default", () => {
    const html = render({ activeTab: "video-studio" });
    expect(openGroupIds(html)).toHaveLength(1);
    expect(html).toContain('aria-expanded="true"');
    expect(html.match(/aria-expanded="false"/g)?.length).toBeGreaterThan(1);
  });

  it("keeps every group open when the setting is on", () => {
    const html = render({ activeTab: "video-studio", keepAllOpen: true });
    expect(openGroupIds(html).length).toBeGreaterThan(3);
    expect(html).not.toContain('aria-expanded="false"');
  });

  it("keeps group headers keyboard reachable with aria attributes", () => {
    const html = render({ activeTab: "chat" });
    expect(html).toMatch(/<button type="button" class="nav-group-header" aria-expanded="(true|false)" aria-controls="nav-group-/);
  });
});
