import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import { SpeakResultRow, type SpeakOutput } from "../SpeakResultRow";

const output = (over: Partial<SpeakOutput> = {}): SpeakOutput => ({
  url: "blob:x",
  blob: new Blob(["x"], { type: "audio/wav" }),
  durationS: 12,
  text: "Welcome to the studio, everyone. Today we talk about voices and how they work.",
  voice: "af_heart",
  voiceName: "Heart",
  voiceKind: "builtin",
  speed: 1,
  cloned: false,
  libraryId: "lib-1",
  ...over,
});

describe("SpeakResultRow", () => {
  it("heads the clip with its first words and voice, and puts Show file beside Saved to Library", () => {
    const html = renderToStaticMarkup(<SpeakResultRow output={output()} onError={() => {}} onOpenLibrary={() => {}} />);
    expect(html).toContain("&quot;Welcome to the studio, everyone. Today…&quot; · Heart");
    expect(html).toContain("0:12");
    expect(html.indexOf("Saved to Library")).toBeGreaterThan(-1);
    expect(html.indexOf("Show file")).toBeGreaterThan(html.indexOf("Saved to Library"));
  });

  it("offers Save to Library, not Show file, while the clip is not in the Library", () => {
    const html = renderToStaticMarkup(<SpeakResultRow output={output({ libraryId: null })} onError={() => {}} />);
    expect(html).toContain("Save to Library");
    expect(html).not.toContain("Show file");
  });
});
