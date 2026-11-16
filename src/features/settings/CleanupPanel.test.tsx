import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import type { CleanupDeleteResult, CleanupItems } from "../../api/cleanup";
import { CleanupView } from "./CleanupPanel";
import type { CleanupViewProps } from "./CleanupPanel";
import { emptySelection, setGroup, toggleId } from "./cleanupLogic";

const items: CleanupItems = {
  projects: [{ id: "0123456789abcdef", name: "Lighthouse", shots: 1, bytes: 2_500_000, lastEdited: 1_790_000_000 }],
  threads: [{ id: "session-aa", title: "Plans", messages: 3, bytes: 4000, lastUsed: 1_790_000_000 }],
  outputs: [{ id: "image:2026-11-01/a.json", kind: "image", name: "a lighthouse", bytes: 500, createdAt: "2026-11-01T10:00:00Z" }],
};

const noop = () => undefined;
function view(overrides: Partial<CleanupViewProps> = {}) {
  const props: CleanupViewProps = {
    items, loading: false, error: null, selection: emptySelection(), busy: false, result: null, deleteError: null, locale: "en",
    onToggle: noop, onToggleGroup: noop, onRefresh: noop, onAskDelete: noop,
    ...overrides,
  };
  return renderToStaticMarkup(<CleanupView {...props} />);
}

describe("CleanupView", () => {
  it("lists the three groups with size and date and ticks nothing", () => {
    const html = view();
    for (const text of ["Storyboard projects (1)", "Chat threads (1)", "Gallery pictures and clips (1)", "Lighthouse", "1 shot", "3 messages", "a lighthouse", "Picture"]) {
      expect(html).toContain(text);
    }
    expect(html).toContain("2.5 MB");
    expect(html).toContain("Nothing ticked");
    expect(html).not.toContain('checked=""');
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*>Delete the ticked items/);
  });

  it("groups are fieldsets with a legend and each row has a labelled checkbox", () => {
    const html = view();
    expect(html.match(/<fieldset/g)?.length).toBe(3);
    expect(html.match(/<legend>/g)?.length).toBe(3);
    expect(html).toContain("Select all: Chat threads");
    // every checkbox sits inside a label, so its name is the row text
    expect(html.match(/<label/g)?.length).toBe(html.match(/type="checkbox"/g)?.length);
  });

  it("shows ticks, the count and size, and enables Delete", () => {
    const selection = setGroup(toggleId(emptySelection(), "outputs", "image:2026-11-01/a.json"), items, "projects", true);
    const html = view({ selection });
    expect(html).toContain("2 items ticked, 2.5 MB");
    // two rows plus the two "select all" boxes whose one-item groups are fully ticked
    expect(html.match(/checked=""/g)?.length).toBe(4);
    expect(html).not.toMatch(/<button[^>]*disabled=""[^>]*>Delete the ticked items/);
  });

  it("disables the controls while deleting", () => {
    const html = view({ busy: true, selection: setGroup(emptySelection(), items, "threads", true) });
    expect(html).toContain("Deleting...");
    expect(html.match(/<fieldset[^>]*disabled=""/g)?.length).toBe(3);
  });

  it("says what was deleted and names each refused item with its reason", () => {
    const result: CleanupDeleteResult = {
      deleted: 1, refused: 1, freedBytes: 4000,
      results: [
        { kind: "thread", id: "session-aa", name: "Plans", ok: true, bytes: 4000, reason: null },
        { kind: "project", id: "0123456789abcdef", name: "Lighthouse", ok: false, bytes: 0, reason: "A batch is running for this project. Cancel it or wait, then delete." },
      ],
    };
    const html = view({ result });
    expect(html).toContain("Deleted 1 item and freed 4 kB");
    expect(html).toContain("1 item was not deleted:");
    expect(html).toContain("Lighthouse</strong>: A batch is running");
    expect(html).toContain('role="status"');
  });

  it("shows an empty group message and a load error as an alert", () => {
    const html = view({ items: { projects: [], threads: [], outputs: [] }, error: "offline" });
    expect(html.match(/Nothing here\./g)?.length).toBe(3);
    expect(html).toContain('role="alert"');
    expect(html).toContain("Could not load the list: offline");
  });
});
