import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import { SongVocals } from "../SongVocals";
import type { UseStoryboardResult } from "../../useStoryboard";

describe("SongVocals (FU-371)", () => {
  it("offers the sung cut, an optional stem and the attach button", () => {
    const story = { call: vi.fn(), busy: false } as unknown as UseStoryboardResult;
    const html = renderToStaticMarkup(<SongVocals story={story} disabled={false} onCutForSinging={() => undefined} />);
    expect(html).toContain("Lip-sync the song");
    expect(html).toContain("Cut for singing (3 to 5 s shots)");
    expect(html).toContain("Vocal stem (optional, WAV or FLAC)");
    expect(html).toContain("Give each shot its vocal");
    expect(html).toContain("0.4 s before the shot");
  });
});
