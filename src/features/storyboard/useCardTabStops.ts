/**
 * Roving tab stops for the shot list (FU-188 item 2). Only the current card is in the Tab order with its inner
 * controls; every other card is one arrow-key step away and its controls are taken out of the Tab order
 * (`tabindex="-1"`, restored when the card becomes current). Pointer use is unchanged: a click still reaches
 * every control, and focusing one makes its card current. 30 shots are about 10 Tab stops, not about 250.
 */
import { useLayoutEffect, type RefObject } from "react";

const FOCUSABLE = "a[href], button, input, select, textarea, summary, video[controls], audio[controls], [tabindex]";
const SAVED = "data-roving-tabindex";
const NONE = "none";

/** Takes `root`'s focusable descendants out of the Tab order (`active` false) or puts them back (`active` true). */
export function setInnerTabStops(root: HTMLElement, active: boolean): void {
  for (const el of Array.from(root.querySelectorAll<HTMLElement>(FOCUSABLE))) {
    // A dialog opened from the card (Compare takes) keeps its own focus order, and the takes strip runs its own
    // roving stop (it takes `tabStops` from the card), so React and this hook never fight over one attribute.
    if (el.closest("[role='dialog'], [role='radiogroup']")) continue;
    const saved = el.getAttribute(SAVED);
    if (active) {
      if (saved == null) continue;
      if (saved === NONE) el.removeAttribute("tabindex");
      else el.setAttribute("tabindex", saved);
      el.removeAttribute(SAVED);
      continue;
    }
    const current = el.getAttribute("tabindex");
    // Already taken out and nobody changed it since: nothing to do. When React set a new value meanwhile,
    // keep that value for the restore.
    if (saved != null && current === "-1") continue;
    el.setAttribute(SAVED, current ?? NONE);
    el.setAttribute("tabindex", "-1");
  }
}

/** Runs after every render of the card, so controls that appear later (a stale badge's buttons) follow too. */
export function useCardTabStops(ref: RefObject<HTMLElement | null>, active: boolean): void {
  useLayoutEffect(() => {
    if (ref.current) setInnerTabStops(ref.current, active);
  });
}
