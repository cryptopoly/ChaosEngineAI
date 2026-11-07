import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { LiveProgress, progressValueNow } from "../LiveProgress";
import { RunEstimateStrip } from "../RunEstimateStrip";
import { Sidebar } from "../Sidebar";
import { SubtabBar } from "../SubtabBar";
import { ConfirmDialog } from "../../features/storyboard/ConfirmDialog";
import { TermsDialog } from "../../features/terms/TermsOfUse";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

const phases = [
  { id: "load", label: "Loading the model", estimatedSeconds: 5 },
  { id: "compose", label: "Composing", estimatedSeconds: 20 },
];

describe("progressValueNow", () => {
  it("rounds and clamps, and is absent while the length is unknown", () => {
    expect(progressValueNow(41.6, false)).toBe(42);
    expect(progressValueNow(140, false)).toBe(100);
    expect(progressValueNow(-3, false)).toBe(0);
    expect(progressValueNow(100, true)).toBeUndefined();
  });
});

describe("LiveProgress semantics", () => {
  const html = renderToStaticMarkup(<LiveProgress title="Making a track" phases={phases} startedAt={Date.now()} accent="audio" />);

  it("exposes the bar as a named progressbar", () => {
    expect(html).toContain('role="progressbar"');
    expect(html).toContain('aria-label="Making a track"');
    expect(html).toContain('aria-valuemin="0"');
    expect(html).toContain('aria-valuemax="100"');
  });

  it("announces the phase, not the running clock, and hides the decoration", () => {
    expect(html).toContain('<p class="sr-only" role="status">Making a track: Loading the model</p>');
    expect(html).toContain('class="live-progress__elapsed" aria-hidden="true"');
    expect(html).toContain('class="live-progress__waveform" aria-hidden="true"');
    expect(html).toContain('aria-current="step"');
  });
});

describe("RunEstimateStrip live region", () => {
  const props = { seconds: 40, peakGb: 9, totalGb: 64, diskMb: 12, basis: null, risk: "safe" as const, action: <button type="button">Go</button> };

  it("keeps the notes region mounted when there is nothing to say", () => {
    expect(renderToStaticMarkup(<RunEstimateStrip {...props} />)).toContain('<div class="run-estimate__notes" aria-live="polite"></div>');
  });

  it("puts the busy line inside it", () => {
    const html = renderToStaticMarkup(<RunEstimateStrip {...props} busy={{ level: "slower", message: "Another job is using the GPU" } as never} />);
    expect(html).toMatch(/run-estimate__notes" aria-live="polite"><p class="run-estimate__queue run-estimate__busy">Another job/);
  });

  it("says the memory verdict in words, not only colour", () => {
    expect(renderToStaticMarkup(<RunEstimateStrip {...props} peakGb={62} />)).toContain("danger: may not fit");
    expect(renderToStaticMarkup(<RunEstimateStrip {...props} peakGb={50} />)).toContain("caution: close to the limit");
    expect(renderToStaticMarkup(<RunEstimateStrip {...props} />)).not.toContain("sr-only");
  });
});

describe("navigation state", () => {
  it("marks the current page in the sidebar", () => {
    const html = renderToStaticMarkup(
      <Sidebar
        activeTab="image-studio"
        onTabChange={() => {}}
        platform="Darwin"
        backendOnline
        engineLabel="MLX"
        mode="collapsible"
        collapsedGroups={new Set()}
        onToggleGroupCollapsed={() => {}}
        lastChildByGroup={{}}
        onRememberLastChild={() => {}}
        keepAllOpen
      />,
    );
    expect(html.match(/aria-current="page"/g)?.length).toBe(1);
    expect(html).toContain('<img src="/logo.svg" alt=""');
  });

  it("gives only the selected sub tab a tab stop", () => {
    const html = renderToStaticMarkup(<SubtabBar activeTab="image-studio" onTabChange={() => {}} platform="Darwin" onRememberLastChild={() => {}} />);
    const tabs = html.match(/<button[^>]*role="tab"[^>]*>/g) ?? [];
    expect(tabs.length).toBeGreaterThan(1);
    expect(tabs.filter((tab) => tab.includes('tabindex="0"')).length).toBe(1);
    expect(tabs.filter((tab) => tab.includes('tabindex="-1"')).length).toBe(tabs.length - 1);
  });
});

describe("dialogs", () => {
  it("starts the first-run terms on the dialog and describes it by the points", () => {
    const html = renderToStaticMarkup(<TermsDialog mode="accept" onAgree={() => {}} onQuit={() => {}} />);
    expect(html).toContain('role="dialog"');
    expect(html).toContain('aria-modal="true"');
    expect(html).toContain('aria-describedby="terms-points"');
    expect(html).toContain('id="terms-points"');
    expect(html).toContain('tabindex="-1"');
  });

  it("starts a destructive confirm on Cancel and a plain one on the confirm button", () => {
    const danger = renderToStaticMarkup(<ConfirmDialog title="Delete" body="Sure?" confirmLabel="Delete" danger onConfirm={() => {}} onCancel={() => {}} />);
    expect(danger).toMatch(/<button[^>]*data-autofocus=""[^>]*>Cancel</);
    const plain = renderToStaticMarkup(<ConfirmDialog title="Export" body="Go?" confirmLabel="Export" onConfirm={() => {}} onCancel={() => {}} />);
    expect(plain).toMatch(/<button[^>]*data-autofocus=""[^>]*>Export</);
  });
});
