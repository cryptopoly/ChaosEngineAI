import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import type { VoiceLibrarySummary } from "../../types";
import type { UseSendToStoryboard } from "../storyboard/useSendToStoryboard";
import { LibraryCard } from "./VoiceLibraryTab";

const storyboard: UseSendToStoryboard = {
  enabled: false,
  projects: [],
  loaded: false,
  busy: false,
  notice: null,
  dismissNotice: () => {},
  loadProjects: async () => {},
  send: async () => false,
};

function item(over: Partial<VoiceLibrarySummary>): VoiceLibrarySummary {
  return {
    id: "1_aaaaaaaa",
    kind: "speech",
    title: "Good evening",
    preview: "Good evening.",
    createdAt: 1_790_000_000,
    updatedAt: 1_790_000_000,
    audio: ".wav",
    speakers: {},
    speakerCount: 0,
    durationS: 42,
    ...over,
  };
}

function render(entry: VoiceLibrarySummary, speakAgain = true) {
  return renderToStaticMarkup(
    <LibraryCard
      item={entry}
      selected={false}
      onSelect={() => {}}
      onOpen={() => {}}
      onDelete={() => {}}
      onSpeakAgain={speakAgain ? () => {} : undefined}
      storyboard={storyboard}
    />,
  );
}

describe("Voice Library speech card", () => {
  it("names the voice and its kind instead of the raw id, with Speak again", () => {
    const html = render(item({ voice: "qwen3:Ryan", voiceName: "Ryan", voiceKind: "preset" }));
    expect(html).toContain("Ryan · ");
    expect(html).not.toContain("qwen3:Ryan");
    expect(html).toContain(">Natural<");
    expect(html).toContain("Speak again");
    expect(html).not.toContain("AI-made");
  });

  it("marks cloned speech as AI-made", () => {
    const html = render(item({ voice: "clone:1_bbbbbbbb", voiceName: "Ada", voiceKind: "clone", clonedVoice: true }));
    expect(html).toContain("Ada · ");
    expect(html).toContain(">Cloned<");
    expect(html).toContain("AI-made, cloned voice");
  });

  it("still renders older items that only have the voice id, or no voice at all", () => {
    const old = render(item({ voice: "af_heart" }));
    expect(old).toContain("Heart · ");
    expect(old).toContain(">Quick<");
    expect(old).not.toContain("af_heart");
    const oldClone = render(item({ voice: "clone:1_bbbbbbbb", clonedVoice: true }));
    expect(oldClone).toContain(">Cloned<");
    expect(oldClone).toContain("AI-made, cloned voice");
    expect(oldClone).not.toContain("clone:1_bbbbbbbb");
    const bare = render(item({}));
    expect(bare).toContain("Good evening");
    expect(bare).not.toContain("badge subtle\">Quick");
  });

  it("offers no Speak again on transcripts", () => {
    const html = render(item({ kind: "transcript", audio: null }));
    expect(html).not.toContain("Speak again");
    expect(html).toContain("Open");
  });
});
