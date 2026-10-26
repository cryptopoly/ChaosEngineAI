/**
 * Follow-the-bottom state for streaming output.
 *
 * Measuring "am I near the bottom" after the content has grown loses the
 * follow whenever one chunk adds more than the threshold (a tall reasoning
 * block, a code fence). Content growth never moves ``scrollTop``, so the
 * follow only ends when the user scrolls up, and resumes when they come
 * back to the bottom or press Latest.
 */
export interface StickyScroll {
  top: number;
  stuck: boolean;
}

const NEAR_BOTTOM_PX = 32;

export function newStickyScroll(): StickyScroll {
  return { top: 0, stuck: true };
}

/** Update from a scroll event; returns whether the follow is still on. */
export function updateStickyScroll(
  element: { scrollHeight: number; scrollTop: number; clientHeight: number },
  state: StickyScroll,
): boolean {
  const distance = element.scrollHeight - element.scrollTop - element.clientHeight;
  if (distance < NEAR_BOTTOM_PX) {
    state.stuck = true;
  } else if (element.scrollTop < state.top) {
    state.stuck = false;
  }
  state.top = element.scrollTop;
  return state.stuck;
}

/** Scroll to the end and turn the follow on. */
export function stickToBottom(
  element: { scrollHeight: number; scrollTop: number },
  state: StickyScroll,
): void {
  element.scrollTop = element.scrollHeight;
  state.stuck = true;
  state.top = element.scrollTop;
}
