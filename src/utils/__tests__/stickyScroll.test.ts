import { describe, expect, it } from "vitest";
import { newStickyScroll, stickToBottom, updateStickyScroll } from "../stickyScroll";

const box = (scrollHeight: number, scrollTop: number, clientHeight = 400) => ({ scrollHeight, scrollTop, clientHeight });

describe("stickyScroll", () => {
  it("keeps following when one chunk grows the content past the threshold", () => {
    const state = newStickyScroll();
    updateStickyScroll(box(1000, 600), state);
    // content grows 500px, scrollTop unchanged: no scroll event moves it up
    expect(updateStickyScroll(box(1500, 600), state)).toBe(true);
  });

  it("stops following when the user scrolls up, resumes at the bottom", () => {
    const state = newStickyScroll();
    updateStickyScroll(box(1000, 600), state);
    expect(updateStickyScroll(box(1000, 300), state)).toBe(false);
    expect(updateStickyScroll(box(1200, 300), state)).toBe(false);
    expect(updateStickyScroll(box(1200, 790), state)).toBe(true);
  });

  it("stickToBottom turns the follow back on", () => {
    const state = newStickyScroll();
    state.stuck = false;
    const el = box(2000, 0);
    stickToBottom(el, state);
    expect(el.scrollTop).toBe(2000);
    expect(state.stuck).toBe(true);
  });
});
