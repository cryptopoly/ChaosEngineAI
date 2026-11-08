import { describe, expect, it } from "vitest";
import { sidebarGroups } from "../sidebarGroups";
import { tabs } from "../tabs";
import { LEGACY_TAB_REDIRECTS } from "../tabRedirects";

describe("sidebarGroups", () => {
  it("each group's defaultChild is a real tab in that group", () => {
    for (const group of sidebarGroups) {
      const child = tabs.find((t) => t.id === group.defaultChild);
      expect(child, `group "${group.id}" defaultChild "${group.defaultChild}" not found`).toBeDefined();
      expect(child?.group, `defaultChild "${group.defaultChild}" must belong to group "${group.id}"`).toBe(group.id);
    }
  });

  it("every tab.group references a known group", () => {
    const groupIds = new Set(sidebarGroups.map((g) => g.id));
    for (const tab of tabs) {
      if (!tab.group) continue;
      expect(groupIds.has(tab.group), `tab "${tab.id}" references unknown group "${tab.group}"`).toBe(true);
    }
  });

  it("every group has at least one tab", () => {
    for (const group of sidebarGroups) {
      const children = tabs.filter((t) => t.group === group.id);
      expect(children.length, `group "${group.id}" has no tabs`).toBeGreaterThan(0);
    }
  });

  it("grouped tabs have a shortLabel", () => {
    for (const tab of tabs) {
      if (!tab.group) continue;
      expect(tab.shortLabel, `tab "${tab.id}" in group "${tab.group}" missing shortLabel`).toBeTruthy();
    }
  });

  it("has no Discover / My Models tabs left: one Models page per area (FU-129)", () => {
    const ids = tabs.map((tab) => tab.id as string);
    for (const legacy of Object.keys(LEGACY_TAB_REDIRECTS)) {
      expect(ids, `legacy tab "${legacy}" is still registered`).not.toContain(legacy);
    }
    for (const [group, page] of [["chat", "chat-models"], ["images", "image-models"], ["video", "video-models"]] as const) {
      const tab = tabs.find((candidate) => candidate.id === page);
      expect(tab?.group, `${page} must live in group ${group}`).toBe(group);
      expect(tab?.shortLabel).toBe("Models");
    }
  });

  it("registers Music with its tabs and ready child, and Projects as one entry in Make (D-65)", () => {
    const music = sidebarGroups.find((g) => g.id === "music");
    expect(music?.label).toBe("Music and Audio");
    expect(music?.defaultChild).toBe("music-models");
    expect(music?.readyChild).toBe("music-studio");
    expect(tabs.find((t) => t.id === music?.readyChild)?.group).toBe("music");
    expect(tabs.filter((t) => t.group === "music").map((t) => t.id)).toEqual(["music-models", "music-studio", "music-library"]);
    expect(sidebarGroups.map((g) => g.id as string)).not.toContain("storyboard");
    const projects = tabs.find((t) => t.id === "storyboard");
    expect(projects).toMatchObject({ label: "Projects", section: "make" });
    expect(projects?.group).toBeUndefined();
  });

  it("lists Voice like the other areas: Models first, opening on Models until the pack is installed", () => {
    const voice = sidebarGroups.find((g) => g.id === "voice");
    expect(voice?.defaultChild).toBe("voice-models");
    expect(voice?.readyChild).toBe("voice-studio");
    expect(tabs.filter((t) => t.group === "voice").map((t) => t.id)).toEqual(["voice-models", "voice-studio", "voice-gallery"]);
  });

  it("opens Images and Video on their Models page and Chat on the chat", () => {
    const defaults = Object.fromEntries(sidebarGroups.map((group) => [group.id, group.defaultChild]));
    expect(defaults.images).toBe("image-models");
    expect(defaults.video).toBe("video-models");
    expect(defaults.chat).toBe("chat");
    for (const legacy of Object.keys(LEGACY_TAB_REDIRECTS)) {
      expect(Object.values(defaults)).not.toContain(legacy);
    }
  });
});
