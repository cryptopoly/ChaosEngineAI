import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ChatSession, KeepGoingRun } from "../../../types";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import { initI18n } from "../../../i18n";
import { ActiveChip, ChatKeepGoing, KeepGoingEyebrow, KeepGoingPanel, stopReason } from "../ChatKeepGoing";
import { ChatThread, type ChatThreadProps } from "../ChatThread";
import {
  KEEP_GOING_POLL_MS,
  announcementKey,
  clampCap,
  clampHours,
  followKeepGoing,
  formatTimeLeft,
  isActive,
  needsRefresh,
  nudgeBody,
  refreshKey,
} from "../keepGoing";

function run(over: Partial<KeepGoingRun> = {}): KeepGoingRun {
  return {
    runId: "kg-1",
    sessionId: "s1",
    state: "running",
    phase: "turn",
    nudges: 3,
    cap: 20,
    goal: "Make a 2 minute ad",
    projectId: "p1",
    startedAt: 1,
    updatedAt: 2,
    noProgress: 0,
    stopCode: null,
    stopReason: null,
    stopDetail: null,
    stoppedAt: null,
    resumable: false,
    grant: { mode: "auto", until: 100, secondsLeft: 5 * 3600 + 40 * 60 },
    gpuMode: "auto",
    turnRunning: true,
    pendingPlans: 0,
    messageCount: 6,
    ...over,
  };
}

const t = ((key: string, opts: Record<string, unknown> = {}) => {
  let text = String(opts.defaultValue ?? key);
  for (const [name, value] of Object.entries(opts)) text = text.replace(`{${name}}`, String(value));
  return text;
}) as unknown as Parameters<typeof stopReason>[0];

describe("keep going helpers", () => {
  it("clamps the cap and the grant hours", () => {
    expect(clampCap(0)).toBe(1);
    expect(clampCap(500)).toBe(100);
    expect(clampCap(Number.NaN)).toBe(20);
    expect(clampHours(48)).toBe(12);
    expect(clampHours(0)).toBe(1);
  });

  it("formats the grant's time left", () => {
    expect(formatTimeLeft(5 * 3600 + 40 * 60 + 12)).toEqual({ hours: 5, minutes: 40 });
    expect(formatTimeLeft(null)).toBeNull();
  });

  it("knows active runs", () => {
    expect(isActive(run())).toBe(true);
    expect(isActive(run({ state: "armed" }))).toBe(true);
    expect(isActive(run({ state: "stopped" }))).toBe(false);
    expect(isActive(null)).toBe(false);
  });

  it("reloads the thread only when the backend has messages this window lacks and it is not streaming", () => {
    expect(needsRefresh(run({ messageCount: 7 }), 6, false)).toBe(true);
    expect(needsRefresh(run({ messageCount: 7 }), 6, true)).toBe(false);
    expect(needsRefresh(run({ messageCount: 6 }), 6, false)).toBe(false);
    expect(refreshKey(run())).not.toBe(refreshKey(run({ turnRunning: false })));
  });

  it("strips the nudge's first line for the thread", () => {
    expect(nudgeBody("Keep going (automatic message 2 of 20 from the app, not from the user).\nNext step: X")).toBe("Next step: X");
    expect(nudgeBody("hello")).toBe("hello");
  });

  it("announces state changes once", () => {
    expect(announcementKey(run())).not.toBe(announcementKey(run({ nudges: 4 })));
    expect(announcementKey(null)).toBe("");
  });

  it("shows the app's words for a stop code, with details", async () => {
    const i18n = await initI18n({ forceLocale: "en" });
    const tr = i18n.getFixedT("en", "chat") as unknown as typeof t;
    expect(stopReason(tr, run({ state: "stopped", stopCode: "cap_reached", cap: 20 }))).toBe("reached the limit of 20 nudges.");
    expect(stopReason(t, run({ state: "stopped", stopCode: "cap_reached", stopReason: "Reached the limit of 20 nudges." }))).toContain("20 nudges");
    expect(stopReason(tr, run({ state: "stopped", stopCode: "repeated_error", stopDetail: "no video model" }))).toContain("no video model");
    expect(stopReason(t, run({ state: "stopped", stopCode: "new_code", stopReason: "English reason" }))).toBe("English reason");
    // FU-198: a stall says why nothing changed.
    expect(stopReason(tr, run({ state: "stopped", stopCode: "calls_unreadable", stopDetail: "storyboard_audio: x" }))).toBe(
      "the model's tool calls could not be read 3 turns in a row, so they never ran: storyboard_audio: x",
    );
    expect(stopReason(tr, run({ state: "stopped", stopCode: "call_refused", stopDetail: "storyboard_audio: y" }))).toContain(
      "the same call was refused",
    );
  });
});

describe("followKeepGoing", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("polls while active and stops once the run stops", async () => {
    const answers = [run(), run({ nudges: 4 }), run({ state: "stopped", stopCode: "complete" })];
    const fetchRun = vi.fn(async () => answers.shift() ?? null);
    const seen: Array<KeepGoingRun | null> = [];
    const stop = followKeepGoing("s1", (r) => seen.push(r), fetchRun);
    await vi.advanceTimersByTimeAsync(0);
    await vi.advanceTimersByTimeAsync(KEEP_GOING_POLL_MS);
    await vi.advanceTimersByTimeAsync(KEEP_GOING_POLL_MS);
    await vi.advanceTimersByTimeAsync(KEEP_GOING_POLL_MS * 3);
    expect(fetchRun).toHaveBeenCalledTimes(3);
    expect(seen.map((r) => r?.state)).toEqual(["running", "running", "stopped"]);
    stop();
  });

  it("retries while the backend restarts and stops on cleanup", async () => {
    const fetchRun = vi.fn().mockRejectedValueOnce(new Error("offline")).mockResolvedValue(run());
    const onRun = vi.fn();
    const stop = followKeepGoing("s1", onRun, fetchRun);
    await vi.advanceTimersByTimeAsync(0);
    expect(onRun).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(5000);
    expect(onRun).toHaveBeenCalledTimes(1);
    stop();
    await vi.advanceTimersByTimeAsync(KEEP_GOING_POLL_MS * 5);
    expect(fetchRun).toHaveBeenCalledTimes(2);
  });
});

describe("Keep going in the composer", () => {
  it("shows nothing outside Studio chats and the toggle inside", () => {
    const off = renderToStaticMarkup(<ChatKeepGoing sessionId="s1" studioTools={false} messageCount={0} ownStreaming={false} onRefresh={() => undefined} />);
    expect(off).toBe("");
    const on = renderToStaticMarkup(<ChatKeepGoing sessionId="s1" studioTools messageCount={0} ownStreaming={false} onRefresh={() => undefined} />);
    expect(on).toContain("Keep going");
    expect(on).toContain('aria-haspopup="dialog"');
    expect(on).toContain('aria-live="polite"');
    expect(on).not.toContain("<span>&#");  // monochrome SVG icon, never an emoji
    expect(on).toContain("<svg");
  });

  it("asks for the grant explicitly: nothing preselected, Turn on disabled until a choice", () => {
    const html = renderToStaticMarkup(<KeepGoingPanel mode="start" run={null} busy={false} onCancel={() => undefined} onSubmit={async () => undefined} />);
    expect(html).toContain('role="dialog"');
    expect(html).toContain("I approve each plan card");
    expect(html).toContain("Approve renders automatically for this run");
    expect(html).toContain("at most 12");
    expect(html).not.toContain("checked");
    expect(html).toMatch(/<button[^>]*disabled[^>]*>Turn on<\/button>/);
    expect(html).toContain('value="20"');
    expect(html).toContain('value="6"');
  });

  it("offers the GPU swap setting (Auto by default) when turning on", () => {
    const html = renderToStaticMarkup(<KeepGoingPanel mode="start" run={null} busy={false} onCancel={() => undefined} onSubmit={async () => undefined} />);
    expect(html).toContain("Free GPU memory while rendering");
    expect(html).toMatch(/<option value="auto" selected="">Auto<\/option>/);
    expect(html).toContain('value="never"');
    const chip = renderToStaticMarkup(<ActiveChip run={run({ phase: "gpu_swap", swapPhase: "rendering" })} busy={false} onStop={() => undefined} />);
    expect(chip).toContain("chat model unloaded while a render runs");
  });

  it("the resume panel keeps the run's cap", () => {
    const html = renderToStaticMarkup(<KeepGoingPanel mode="resume" run={run({ cap: 40 })} busy={false} onCancel={() => undefined} onSubmit={async () => undefined} />);
    expect(html).toContain("Resume Keep going");
    expect(html).toContain('value="40"');
  });

  it("the active chip shows the counter, the phase, the grant and Stop", () => {
    const html = renderToStaticMarkup(<ActiveChip run={run({ phase: "awaiting_approval" })} busy={false} onStop={() => undefined} />);
    expect(html).toContain("nudge 3 of 20");
    expect(html).toContain("waiting for your approval");
    expect(html).toContain("renders auto-approved, 5 h 40 min left");
    expect(html).toContain("Stop");
    const manual = renderToStaticMarkup(<ActiveChip run={run({ grant: { mode: "manual", until: null, secondsLeft: null }, state: "armed", phase: "armed" })} busy={false} onStop={() => undefined} />);
    expect(manual).toContain("you approve each plan");
    expect(manual).toContain("send your brief");
    expect(manual).not.toContain("nudge 3");
  });
});

describe("Keep going messages in the thread", () => {
  it("are labelled as the app's and look different from the user's", () => {
    const session = {
      id: "s1",
      title: "Ad",
      updatedAt: "now",
      model: "Qwen",
      messages: [
        { role: "user", text: "Make an ad" },
        { role: "user", text: "Keep going (automatic message 2 of 20 from the app, not from the user).\nNext step: render.", keepGoing: { nudge: 2, cap: 20 } },
      ],
    } as unknown as ChatSession;
    const noop = () => undefined;
    const props = {
      activeChat: session, chatBusySessionId: null, chatScrollRef: { current: null }, serverLoading: null,
      engineLabel: "MLX", launchSettings: { contextTokens: 8192 }, busy: false, loadedModelRef: "q",
      onChatFileDrop: noop, onCopyMessage: noop, onRetryMessage: noop, onDeleteMessage: noop, onForkAtMessage: noop,
      warmModels: [], onAddVariant: noop, onDelveMessage: noop, onDetailsToggle: noop, onCancelGeneration: noop, onLoadModel: noop,
    } as unknown as ChatThreadProps;
    const html = renderToStaticMarkup(<ChatThread {...props} />);
    expect(html).toContain("message-bubble--keep-going");
    expect(html).toContain("Keep going: app message 2 of 20");
    expect(html).toContain("Next step: render.");
    expect(html).not.toContain("automatic message 2 of 20");
    expect(html.match(/message-bubble--keep-going/g)).toHaveLength(1);
    expect(renderToStaticMarkup(<KeepGoingEyebrow />)).toContain("Keep going: app message");
  });
});
