import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { AudioPlayer } from "../AudioPlayer";
import { claimPlayback, playbackOwnerId, releasePlayback } from "../audioPlayback";
import { AUDIO_EXTENSIONS, InputAudioPicker, addInputAudio } from "../InputAudioPicker";
import { MediaLibraryCard } from "../MediaLibraryCard";
import { WaveformCanvas, barsFor } from "../WaveformCanvas";
import { LiveProgress } from "../LiveProgress";

describe("one player at a time", () => {
  it("stops the previous owner when another player starts", () => {
    const stopA = vi.fn();
    const stopB = vi.fn();
    claimPlayback("a", stopA);
    expect(playbackOwnerId()).toBe("a");
    claimPlayback("b", stopB);
    expect(stopA).toHaveBeenCalledTimes(1);
    expect(stopB).not.toHaveBeenCalled();
    expect(playbackOwnerId()).toBe("b");
    claimPlayback("b", stopB);
    expect(stopB).not.toHaveBeenCalled();
    releasePlayback("a");
    expect(playbackOwnerId()).toBe("b");
    releasePlayback("b");
    expect(playbackOwnerId()).toBeNull();
  });
});

describe("AudioPlayer", () => {
  it("names the play button, exposes loop state and a spoken time", () => {
    const html = renderToStaticMarkup(<AudioPlayer src="blob:x" durationS={30} label="lo-fi piano" />);
    expect(html).toContain('aria-label="Play lo-fi piano, 30 seconds"');
    expect(html).toContain('aria-pressed="false"');
    expect(html).toContain('aria-label="Loop"');
    expect(html).toContain('aria-label="0:00 of 0:30"');
    expect(html).toContain("0:00 / 0:30");
    expect(html).toContain("audio-player__wave");
    expect(html).toContain('preload="none"');
  });

  it("starts with loop on when asked", () => {
    const html = renderToStaticMarkup(<AudioPlayer src="blob:x" durationS={5} label="rain" loop />);
    expect(html).toContain('aria-pressed="true"');
  });
});

describe("WaveformCanvas", () => {
  it("is a decorative canvas", () => {
    const html = renderToStaticMarkup(<WaveformCanvas peaks={[]} />);
    expect(html).toContain("<canvas");
    expect(html).toContain('aria-hidden="true"');
    expect(html).toContain('class="waveform"');
  });

  it("reduces peaks to the bar count by bucket maximum and goes flat without peaks", () => {
    expect(barsFor([0.1, 0.9, 0.2, 0.4], 2)).toEqual([0.9, 0.4]);
    expect(barsFor([0.5], 3)).toEqual([0.5, 0.5, 0.5]);
    expect(barsFor([], 3)).toEqual([0, 0, 0]);
    expect(barsFor([2, -3], 1)).toEqual([1]);
    expect(barsFor([0.5], 0)).toEqual([]);
  });
});

describe("MediaLibraryCard", () => {
  it("renders select box, title, badge and meta, optional preview, tags, player, error and actions", () => {
    const html = renderToStaticMarkup(
      <MediaLibraryCard
        title="lo-fi piano"
        badge="Track"
        meta="2 Nov · 0:30"
        preview="warm and mellow"
        selected
        onSelect={() => {}}
        selectLabel="Select lo-fi piano"
        tags={["lo-fi", "piano"]}
        player={<audio controls />}
        error="boom"
        actions={<button type="button">Play</button>}
      />,
    );
    expect(html).toContain("media-library-card media-library-card--selected");
    expect(html).toContain('aria-label="Select lo-fi piano"');
    expect(html).toContain("<strong>lo-fi piano</strong>");
    expect(html).toContain('<span class="badge subtle">Track</span> 2 Nov · 0:30');
    expect(html).toContain("warm and mellow");
    expect(html).toContain("media-library-card__tags");
    expect(html).toContain("<audio controls");
    expect(html).toContain('role="alert"');
    expect(html).toContain('<div class="button-row"><button type="button">Play</button></div>');
  });

  it("omits what is not given, as the Voice card did", () => {
    const html = renderToStaticMarkup(
      <MediaLibraryCard title="t" meta="m" selected={false} onSelect={() => {}} selectLabel="s" actions={null} />,
    );
    expect(html).not.toContain("media-library-card--selected");
    expect(html).not.toContain("__preview");
    expect(html).not.toContain("__tags");
    expect(html).not.toContain("badge");
    expect(html).not.toContain("__error");
  });
});

describe("InputAudioPicker", () => {
  it("adds a path once and keeps the newest when over the limit", () => {
    const a = { path: "/a.wav", name: "a.wav" };
    const b = { path: "/b.flac", name: "b.flac" };
    expect(addInputAudio([a], a, 2)).toEqual([a]);
    expect(addInputAudio([a], b, 2)).toEqual([a, b]);
    expect(addInputAudio([a], b, 1)).toEqual([b]);
  });

  it("offers wav, mp3, flac and m4a and the .input-image layout classes", () => {
    expect(AUDIO_EXTENSIONS).toEqual(["wav", "mp3", "flac", "m4a"]);
    const html = renderToStaticMarkup(
      <InputAudioPicker
        label="Reference audio"
        audio={[{ path: "/music/a.wav", name: "a.wav" }]}
        max={1}
        onChange={() => {}}
        library={[{ id: "1", title: "rain", path: "/lib/rain.wav" }]}
      />,
    );
    expect(html).toContain("input-audio-picker");
    expect(html).toContain("input-image-list");
    expect(html).toContain("Choose file");
    expect(html).toContain("From library");
    expect(html).toContain("a.wav");
    // The list is full (max 1), so both pickers are disabled.
    expect(html.match(/disabled=""/g)?.length).toBeGreaterThanOrEqual(2);
  });

  it("disables From library when the library is empty", () => {
    const html = renderToStaticMarkup(
      <InputAudioPicker label="Reference audio" audio={[]} max={1} onChange={() => {}} library={[]} />,
    );
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*>From library/);
  });
});

describe("LiveProgress accents", () => {
  const phases = [{ id: "compose", label: "Compose", estimatedSeconds: 10 }];
  for (const accent of ["audio", "storyboard"] as const) {
    it(`accepts the ${accent} accent and reuses the waveform animation`, () => {
      const html = renderToStaticMarkup(<LiveProgress title="Making" phases={phases} startedAt={Date.now()} accent={accent} />);
      expect(html).toContain(`live-progress--${accent}`);
      expect(html).toContain("live-progress__waveform");
    });
  }

  it("keeps the default accent for existing callers", () => {
    const html = renderToStaticMarkup(<LiveProgress title="Making" phases={phases} startedAt={Date.now()} />);
    expect(html).toContain("live-progress--benchmark");
  });
});
