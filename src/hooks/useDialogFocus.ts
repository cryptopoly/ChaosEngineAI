/**
 * Focus management for a modal dialog: focus moves in when it opens, Tab and Shift+Tab wrap inside it,
 * Escape calls `onEscape` (when given), and focus goes back to what had it before when it closes.
 */
import { useEffect, useRef, type RefObject } from "react";

export const DIALOG_FOCUSABLE =
  'button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

/** Where Tab should land to stay inside a dialog with `count` stops. Null: let the browser move focus. */
export function trapTabTarget(count: number, index: number, shift: boolean): number | null {
  if (count <= 0) return null;
  if (index < 0) return shift ? count - 1 : 0;
  if (shift && index === 0) return count - 1;
  if (!shift && index === count - 1) return 0;
  return null;
}

export interface DialogFocusOptions {
  /** Escape closes the dialog. Leave unset for a dialog that must be answered. */
  onEscape?: () => void;
  /** "first" (default): the first control, or the one marked data-autofocus. "container": the dialog itself (give it tabIndex -1). */
  initialFocus?: "first" | "container";
}

export function useDialogFocus(ref: RefObject<HTMLElement | null>, options: DialogFocusOptions = {}) {
  const escape = useRef(options.onEscape);
  escape.current = options.onEscape;
  const initial = options.initialFocus ?? "first";

  useEffect(() => {
    const dialog = ref.current;
    const back = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    if (dialog && !dialog.contains(document.activeElement)) {
      if (initial === "container") dialog.focus();
      else (dialog.querySelector<HTMLElement>("[data-autofocus]") ?? dialog.querySelector<HTMLElement>(DIALOG_FOCUSABLE))?.focus();
    }
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape" && escape.current) {
        event.preventDefault();
        escape.current();
        return;
      }
      if (event.key !== "Tab" || !dialog) return;
      const items = Array.from(dialog.querySelectorAll<HTMLElement>(DIALOG_FOCUSABLE));
      const target = trapTabTarget(items.length, items.indexOf(document.activeElement as HTMLElement), event.shiftKey);
      if (target !== null) {
        event.preventDefault();
        items[target].focus();
      } else if (items.length === 0) {
        event.preventDefault();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
      if (back && back.isConnected) back.focus();
    };
  }, [ref, initial]);
}
