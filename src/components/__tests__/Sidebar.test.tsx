import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { Sidebar } from "../Sidebar";
import { arrangeRows, buildItems, COMPACT_SIDEBAR_QUERY, resolveGroupTarget, trapFocusIndex, visibleSidebarTabs } from "../sidebarModel";
import { HOME_STUDIO_ENABLED, tabs } from "../../constants";
import type { ActiveGeneration, TabId } from "../../types";

function render(props: { loadedModelName?: string | null; activeGenerations?: ActiveGeneration[]; activeTab?: TabId; keepAllOpen?: boolean; homeStudioEnabled?: boolean; mode?: "collapsible" | "tabs" } = {}) {
  return renderToStaticMarkup(
    <Sidebar
      activeTab={props.activeTab ?? "image-studio"}
      onTabChange={() => {}}
      platform="Darwin"
      backendOnline
      engineLabel="MLX"
      loadedModelName={props.loadedModelName}
      activeGenerations={props.activeGenerations}
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
  it("ships with the flag on", () => {
    expect(HOME_STUDIO_ENABLED).toBe(true);
  });

  it("hides Music and Projects when the flag is false", () => {
    const html = render({ homeStudioEnabled: false });
    expect(html).not.toContain("Music");
    expect(html).not.toContain("Storyboard films");
    expect(visibleSidebarTabs(tabs, "Darwin", false).some((t) => t.group === "music" || t.id === "storyboard")).toBe(false);
  });

  it("shows Music and the one Projects entry when the flag is true", () => {
    const html = render({ homeStudioEnabled: true });
    expect(html).toContain("Music and Audio");
    expect(html).toContain("Projects");
    expect(html).toContain("Storyboard films: shots, sound, export");
  });

  it("opens Music on Models by default and on Studio once it is ready", () => {
    const children = tabs.filter((t) => t.group === "music");
    expect(resolveGroupTarget("music", children, {})).toBe("music-models");
    expect(resolveGroupTarget("music", children, {}, new Set(["music"]))).toBe("music-studio");
    expect(resolveGroupTarget("music", children, { music: "music-library" }, new Set(["music"]))).toBe("music-library");
    const voice = tabs.filter((t) => t.group === "voice");
    expect(resolveGroupTarget("voice", voice, {})).toBe("voice-models");
    expect(resolveGroupTarget("voice", voice, {}, new Set(["voice"]))).toBe("voice-studio");
  });
});

describe("sidebar sections and accordion", () => {
  it("puts Chat on top, then Make and Run section labels around the groups", () => {
    const rows = arrangeRows(buildItems(visibleSidebarTabs(tabs, "Darwin", true)));
    const names = rows.map((row) => (row.kind === "section" ? `#${row.section}` : row.kind === "group" ? row.id : row.tab.id));
    expect(names).toEqual([
      "dashboard", "chat",
      // Projects is a standalone tab listed inside Make, after the groups (D-65).
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
    expect(html).not.toMatch(/class="nav-group-header" aria-expanded="false"/);
  });

  it("keeps group headers keyboard reachable with aria attributes", () => {
    const html = render({ activeTab: "chat" });
    expect(html).toMatch(/<button type="button" class="nav-group-header" aria-expanded="(true|false)" aria-controls="nav-group-/);
  });
});

describe("footer status while an engine is busy", () => {
  const still: ActiveGeneration = { kind: "image", runLabel: "Qwen-Image 2.1", phase: "diffusing", message: "", step: 27, totalSteps: 40, elapsedSeconds: 90 };

  it("says No model loaded when nothing runs", () => {
    expect(render()).toContain("No model loaded");
  });

  it("names the busy engine and its step instead of No model loaded", () => {
    const html = render({ activeGenerations: [still] });
    expect(html).toContain("Image engine busy: Qwen-Image 2.1 · step 27 of 40");
    expect(html).not.toContain("No model loaded");
  });

  it("keeps the loaded chat model name beside the busy line", () => {
    const html = render({ loadedModelName: "Gemma 4", activeGenerations: [still] });
    expect(html).toContain("Gemma 4");
    expect(html).toContain("Image engine busy");
  });
});

describe("compact layout (below 1180 px)", () => {
  it("renders a closed menu button that controls the nav panel", () => {
    const html = render();
    expect(html).toMatch(/<button[^>]*class="sidebar-menu-button"[^>]*aria-expanded="false"[^>]*aria-controls="sidebar-panel"[^>]*aria-label="Menu"/);
    expect(html).toContain('id="sidebar-panel"');
    // The nav and the footer live in the panel; no backdrop while closed.
    expect(html.indexOf('id="sidebar-panel"')).toBeLessThan(html.indexOf('class="nav-list"'));
    expect(html).not.toContain("sidebar-backdrop");
  });

  it("switches below 1180 px only", () => {
    expect(COMPACT_SIDEBAR_QUERY).toBe("(max-width: 1179.98px)");
  });

  it("keeps Tab inside the drawer", () => {
    // 4 items: the menu button then three drawer controls.
    expect(trapFocusIndex(3, 4, false)).toBe(0);
    expect(trapFocusIndex(0, 4, true)).toBe(3);
    expect(trapFocusIndex(1, 4, false)).toBeNull();
    expect(trapFocusIndex(2, 4, true)).toBeNull();
    expect(trapFocusIndex(-1, 4, false)).toBe(0);
    expect(trapFocusIndex(-1, 4, true)).toBe(3);
    expect(trapFocusIndex(0, 0, false)).toBeNull();
  });
});
