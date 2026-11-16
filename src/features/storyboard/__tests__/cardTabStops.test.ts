import { describe, expect, it } from "vitest";
import { setInnerTabStops } from "../useCardTabStops";

/** A minimal element: attributes plus `closest` for the dialog and radiogroup checks. */
class FakeEl {
  attrs = new Map<string, string>();
  constructor(public role: string | null = null, tabindex?: string) {
    if (tabindex != null) this.attrs.set("tabindex", tabindex);
  }
  getAttribute(name: string) {
    return this.attrs.get(name) ?? null;
  }
  setAttribute(name: string, value: string) {
    this.attrs.set(name, value);
  }
  removeAttribute(name: string) {
    this.attrs.delete(name);
  }
  closest(selector: string) {
    return this.role && selector.includes(`'${this.role}'`) ? this : null;
  }
}

function root(els: FakeEl[]) {
  return { querySelectorAll: () => els } as unknown as HTMLElement;
}

describe("setInnerTabStops", () => {
  it("takes inner controls out of the Tab order and puts them back as they were", () => {
    const button = new FakeEl();
    const custom = new FakeEl(null, "0");
    const card = root([button, custom]);
    setInnerTabStops(card, false);
    expect(button.getAttribute("tabindex")).toBe("-1");
    expect(custom.getAttribute("tabindex")).toBe("-1");
    // Running again while inactive changes nothing.
    setInnerTabStops(card, false);
    setInnerTabStops(card, true);
    expect(button.getAttribute("tabindex")).toBeNull();
    expect(custom.getAttribute("tabindex")).toBe("0");
    expect(button.getAttribute("data-roving-tabindex")).toBeNull();
  });

  it("keeps a value React set while the card was inactive", () => {
    const el = new FakeEl();
    const card = root([el]);
    setInnerTabStops(card, false);
    el.setAttribute("tabindex", "0");
    setInnerTabStops(card, false);
    expect(el.getAttribute("tabindex")).toBe("-1");
    setInnerTabStops(card, true);
    expect(el.getAttribute("tabindex")).toBe("0");
  });

  it("leaves dialogs and the takes strip alone", () => {
    const radio = new FakeEl("radiogroup", "0");
    const dialog = new FakeEl("dialog");
    setInnerTabStops(root([radio, dialog]), false);
    expect(radio.getAttribute("tabindex")).toBe("0");
    expect(dialog.getAttribute("tabindex")).toBeNull();
  });
});
