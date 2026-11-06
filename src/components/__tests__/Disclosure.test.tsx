import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Disclosure } from "../Disclosure";
import { StudioShell } from "../StudioShell";
import {
  DisclosureMemoryContext,
  disclosureStorageKey,
  readDisclosures,
  writeDisclosures,
} from "../studioDisclosureMemory";

function fakeStorage(initial: Record<string, string> = {}) {
  const data = new Map(Object.entries(initial));
  return {
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => void data.set(key, value),
    data,
  };
}

afterEach(() => vi.unstubAllGlobals());

describe("Disclosure", () => {
  it("is a native details with a title and a muted value summary", () => {
    const html = renderToStaticMarkup(
      <Disclosure id="options" title="Options" summary="BPM 120 · Key A minor">
        <p>body</p>
      </Disclosure>,
    );
    expect(html).toContain("<details");
    expect(html).toContain("<summary");
    expect(html).toContain('class="disclosure__title">Options<');
    expect(html).toContain('class="disclosure__values">BPM 120 · Key A minor<');
    expect(html).not.toMatch(/<details[^>]* open/);
  });

  it("starts open with defaultOpen and omits the values span without a summary", () => {
    const html = renderToStaticMarkup(
      <Disclosure id="expert" title="Expert" defaultOpen>
        <p>body</p>
      </Disclosure>,
    );
    expect(html).toMatch(/<details[^>]* open/);
    expect(html).toContain("is-open");
    expect(html).not.toContain("disclosure__values");
  });

  it("follows the open prop when controlled, whatever defaultOpen says", () => {
    const closed = renderToStaticMarkup(
      <Disclosure id="a" title="A" defaultOpen open={false}>
        x
      </Disclosure>,
    );
    const opened = renderToStaticMarkup(
      <Disclosure id="a" title="A" open>
        x
      </Disclosure>,
    );
    expect(closed).not.toMatch(/<details[^>]* open/);
    expect(opened).toMatch(/<details[^>]* open/);
  });

  it("opens from what the shell remembered, over defaultOpen", () => {
    const memory = { isOpen: (id: string) => (id === "options" ? true : undefined), setOpen: () => {} };
    const html = renderToStaticMarkup(
      <DisclosureMemoryContext.Provider value={memory}>
        <Disclosure id="options" title="Options">
          x
        </Disclosure>
        <Disclosure id="expert" title="Expert" defaultOpen>
          x
        </Disclosure>
      </DisclosureMemoryContext.Provider>,
    );
    const [first, second] = html.split("</details>");
    expect(first).toMatch(/<details[^>]* open/);
    expect(second).toMatch(/<details[^>]* open/);
  });
});

describe("disclosure memory", () => {
  it("stores open disclosures under chaos.studio.<id>.disclosures", () => {
    const storage = fakeStorage();
    vi.stubGlobal("window", { localStorage: storage });
    writeDisclosures("music", { options: true, expert: false });
    expect(disclosureStorageKey("music")).toBe("chaos.studio.music.disclosures");
    expect(storage.data.get("chaos.studio.music.disclosures")).toBe('{"options":true,"expert":false}');
    expect(readDisclosures("music")).toEqual({ options: true, expert: false });
    expect(readDisclosures("image")).toEqual({});
  });

  it("ignores junk and survives storage that throws or is missing", () => {
    vi.stubGlobal("window", { localStorage: fakeStorage({ "chaos.studio.a.disclosures": "not json" }) });
    expect(readDisclosures("a")).toEqual({});
    vi.stubGlobal("window", { localStorage: fakeStorage({ "chaos.studio.a.disclosures": '{"x":1,"y":true}' }) });
    expect(readDisclosures("a")).toEqual({ y: true });
    vi.stubGlobal("window", {
      localStorage: {
        getItem: () => {
          throw new Error("blocked");
        },
        setItem: () => {
          throw new Error("blocked");
        },
      },
    });
    expect(readDisclosures("a")).toEqual({});
    expect(() => writeDisclosures("a", { x: true })).not.toThrow();
    vi.unstubAllGlobals();
    expect(readDisclosures("a")).toEqual({});
    expect(() => writeDisclosures("a", { x: true })).not.toThrow();
  });
});

describe("StudioShell", () => {
  it("lays out the slots in order and skips empty ones", () => {
    const html = renderToStaticMarkup(
      <StudioShell
        studioId="music"
        header={<h2>Head</h2>}
        inputs={<p>Inputs</p>}
        presets={<p>Presets</p>}
        estimate={<p>Estimate</p>}
        options={<p>Options</p>}
        expert={<p>Expert</p>}
        results={<p>Results</p>}
      />,
    );
    const order = ["studio-shell__header", "studio-shell__inputs", "studio-shell__presets", "studio-shell__estimate", "studio-shell__options", "studio-shell__results"];
    const positions = order.map((name) => html.indexOf(name));
    expect(positions.every((p) => p >= 0)).toBe(true);
    expect([...positions].sort((a, b) => a - b)).toEqual(positions);

    const bare = renderToStaticMarkup(<StudioShell studioId="x" header="h" inputs="i" />);
    expect(bare).not.toContain("studio-shell__presets");
    expect(bare).not.toContain("studio-shell__options");
    expect(bare).not.toContain("studio-shell__results");
  });

  it("opens a child disclosure the user left open last time", () => {
    vi.stubGlobal("window", { localStorage: fakeStorage({ "chaos.studio.music.disclosures": '{"options":true}' }) });
    const html = renderToStaticMarkup(
      <StudioShell
        studioId="music"
        header="h"
        inputs="i"
        options={
          <Disclosure id="options" title="Options">
            x
          </Disclosure>
        }
      />,
    );
    expect(html).toMatch(/<details[^>]* open/);
  });
});
