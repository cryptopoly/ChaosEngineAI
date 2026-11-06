import { describe, expect, it } from "vitest";
import { tabViewFor } from "../tabViews";
import { MusicStudioView } from "../MusicTabs";
import { StoryBoardView } from "../StoryboardTabs";
import { DashboardView } from "../SystemTabs";
import { HOME_STUDIO_TABS } from "../../constants/features";

describe("tabViewFor and the home studio flag", () => {
  it("sends Music and Storyboard ids to the dashboard while the flag is off", () => {
    for (const tab of HOME_STUDIO_TABS) {
      expect(tabViewFor(tab, false)).toBe(DashboardView);
    }
    expect(tabViewFor("music-studio")).toBe(DashboardView);
  });

  it("renders the placeholder views when the flag is on", () => {
    expect(tabViewFor("music-studio", true)).toBe(MusicStudioView);
    expect(tabViewFor("story-board", true)).toBe(StoryBoardView);
    for (const tab of HOME_STUDIO_TABS) {
      expect(tabViewFor(tab, true)).not.toBe(DashboardView);
    }
  });
});
