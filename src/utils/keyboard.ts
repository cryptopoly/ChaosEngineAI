import type { KeyboardEvent } from "react";

export function handleActionKeyDown(
  event: KeyboardEvent<HTMLElement>,
  action: () => void,
) {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    action();
  }
}

/** The tab to move to for an arrow / Home / End key in a horizontal tab list (arrows wrap), or null for other keys. */
export function tabKeyTarget(current: number, count: number, key: string): number | null {
  if (count <= 0) return null;
  switch (key) {
    case "ArrowRight":
    case "ArrowDown":
      return (current + 1) % count;
    case "ArrowLeft":
    case "ArrowUp":
      return (current - 1 + count) % count;
    case "Home":
      return 0;
    case "End":
      return count - 1;
    default:
      return null;
  }
}
