import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { StudioPlan, ToolCallInfo } from "../../../types";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

const decideStudioPlan = vi.fn().mockResolvedValue(undefined);
vi.mock("../../../api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../../api")>()),
  decideStudioPlan: (...args: unknown[]) => decideStudioPlan(...args),
}));

vi.mock("../../voice/ComposerMicButton", () => ({ ComposerMicButton: () => null }));

import { generateChatStream } from "../../../api";
import { ChatComposer, type ChatComposerProps } from "../ChatComposer";
import { StudioJobCard } from "../StudioJobCard";
import { StudioPlanCard } from "../StudioPlanCard";
import {
  isFinished,
  jobPercent,
  libraryTab,
  mediaKind,
  planDecision,
  readStudioAutoApprove,
  readStudioToolsEnabled,
  studioJobId,
  submitPlanDecision,
  upsertPlan,
  upsertToolCall,
  writeStudioAutoApprove,
  writeStudioToolsEnabled,
  AUTO_APPROVE_HOURS,
} from "../studioChat";

function plan(over: Partial<StudioPlan> = {}): StudioPlan {
  return {
    planId: "plan-1",
    sessionId: "s1",
    items: [
      { callId: "c1", tool: "generate_image", kind: "image", prompt: "Rose poster", model: "FLUX.1-schnell", size: "1024x1536", estimatedSeconds: 40, peakGb: 12 },
      { callId: "c2", tool: "generate_image", kind: "image", prompt: "Tulip poster", model: "FLUX.1-schnell", size: "1024x1536", estimatedSeconds: 40, peakGb: 12 },
    ],
    memory: { chatModel: "Qwen3.8-27B", freeGb: 30, level: "ok", advice: null },
    status: "pending",
    ...over,
  };
}

function toolCall(over: Partial<ToolCallInfo> = {}): ToolCallInfo {
  return { id: "c1", name: "generate_image", arguments: { prompt: "Rose" }, result: "{}", elapsed: 0.1, renderAs: "studio", data: { jobId: "job-1" }, ...over };
}

function memoryStorage() {
  const store = new Map<string, string>();
  return {
    getItem: (k: string) => store.get(k) ?? null,
    setItem: (k: string, v: string) => void store.set(k, v),
    removeItem: (k: string) => void store.delete(k),
    key: (i: number) => [...store.keys()][i] ?? null,
    get length() {
      return store.size;
    },
  };
}

beforeEach(() => {
  vi.stubGlobal("window", { localStorage: memoryStorage(), sessionStorage: memoryStorage(), dispatchEvent: () => true });
  decideStudioPlan.mockClear();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

function composer(over: Partial<ChatComposerProps> = {}) {
  const noop = () => undefined;
  const props = {
    draftMessage: "", pendingImages: [], loadedModelRef: "m", loadedModelCapabilities: null, thinkingMode: "off",
    reasoningEffort: "medium", enableTools: false, chatBusySessionId: null, activeChat: undefined, warmModels: [],
    oneTurnOverride: null, onOneTurnOverrideChange: noop, launchSettings: { temperature: 0.7, cacheStrategy: "native", cacheBits: 0 },
    temperatureOverride: null, samplerOverrides: {}, kvStrategyOverride: null, onKvStrategyOverrideChange: noop,
    availableCacheStrategies: [], showSlashMenu: false, slashMatches: [], slashIndex: 0, setSlashIndex: noop,
    onDraftMessageChange: noop, onPendingImagesChange: noop, onSendMessage: noop, onCancelGeneration: noop, onClearDraft: noop,
    onChatFileDrop: noop, onToggleTools: noop, onSetError: noop, onTemperatureOverrideChange: noop, onSamplerOverridesChange: noop,
    runSlashCommand: noop, handleEffortOff: noop, handleEffortChange: noop, onToggleStudioTools: noop,
    ...over,
  } as unknown as ChatComposerProps;
  return renderToStaticMarkup(<ChatComposer {...props} />);
}

describe("Studio switch in the composer", () => {
  it("shows off by default and on when set", () => {
    const off = composer();
    expect(off).toContain('role="switch"');
    expect(off).toContain('aria-checked="false"');
    expect(off).toContain(">Studio<");
    const on = composer({ studioTools: true });
    expect(on).toContain('aria-checked="true"');
    expect(on).toContain("Studio ON");
  });

  it("is hidden for a model without tool support", () => {
    const html = composer({ loadedModelCapabilities: { supportsTools: false, supportsVision: false, supportsReasoning: false } as never });
    expect(html).not.toContain('role="switch"');
  });
});

describe("Studio switch storage", () => {
  it("is off by default and remembered when turned on", () => {
    expect(readStudioToolsEnabled()).toBe(false);
    writeStudioToolsEnabled(true);
    expect(readStudioToolsEnabled()).toBe(true);
    writeStudioToolsEnabled(false);
    expect(readStudioToolsEnabled()).toBe(false);
  });

  it("does not break when storage is blocked", () => {
    vi.stubGlobal("window", {});
    expect(readStudioToolsEnabled()).toBe(false);
    expect(() => writeStudioToolsEnabled(true)).not.toThrow();
  });
});

describe("plan decisions", () => {
  it("approve all, approve this, deny and edit build the right bodies", () => {
    expect(planDecision(plan(), { type: "approveAll" }, false)).toEqual({ decisions: { c1: "approve", c2: "approve" }, edit: false, alwaysAllow: false });
    expect(planDecision(plan(), { type: "approveOne", callId: "c2" }, false).decisions).toEqual({ c1: "deny", c2: "approve" });
    expect(planDecision(plan(), { type: "deny" }, true)).toEqual({ decisions: { c1: "deny", c2: "deny" }, edit: false, alwaysAllow: false });
    expect(planDecision(plan(), { type: "edit" }, false)).toEqual({ decisions: { c1: "deny", c2: "deny" }, edit: true, alwaysAllow: false });
  });

  it("approve sends the click and remembers Always allow for this chat only", async () => {
    await submitPlanDecision(plan(), { type: "approveAll" }, true, "s1");
    expect(decideStudioPlan).toHaveBeenCalledWith("plan-1", { decisions: { c1: "approve", c2: "approve" }, edit: false, alwaysAllow: true });
    expect(readStudioAutoApprove("s1")).toBe(true);
    expect(readStudioAutoApprove("s2")).toBe(false);
  });

  it("Always allow expires after the set hours and when the Studio switch goes off", () => {
    const start = 1_000_000;
    writeStudioAutoApprove("s1", true, start);
    expect(readStudioAutoApprove("s1", start + 60_000)).toBe(true);
    expect(readStudioAutoApprove("s1", start + (AUTO_APPROVE_HOURS * 3600_000) + 1)).toBe(false);
    expect(readStudioAutoApprove("s1", start + 60_000)).toBe(false);  // the expired entry was removed
    writeStudioAutoApprove("s1", true);
    writeStudioAutoApprove("s2", true);
    writeStudioToolsEnabled(false);
    expect(readStudioAutoApprove("s1")).toBe(false);
    expect(readStudioAutoApprove("s2")).toBe(false);
    writeStudioAutoApprove("s3", true);
    writeStudioAutoApprove("s3", false);
    expect(readStudioAutoApprove("s3")).toBe(false);
  });

  it("deny never turns Always allow on, and a failed send is reported", async () => {
    await submitPlanDecision(plan(), { type: "deny" }, true, "s1");
    expect(readStudioAutoApprove("s1")).toBe(false);
    decideStudioPlan.mockRejectedValueOnce(new Error("This plan is no longer waiting"));
    await expect(submitPlanDecision(plan(), { type: "approveAll" }, false, "s1")).rejects.toThrow("no longer waiting");
  });

  it("upserts plans and tool calls by id", () => {
    const pending = upsertPlan(undefined, plan());
    const resolved = upsertPlan(pending, plan({ status: "decided", decisions: { c1: "approve", c2: "deny" } }));
    expect(resolved).toHaveLength(1);
    expect(resolved[0].status).toBe("decided");
    expect(upsertToolCall([toolCall()], toolCall({ result: "new" }))).toEqual([toolCall({ result: "new" })]);
    expect(upsertToolCall([toolCall()], toolCall({ id: "c2" }))).toHaveLength(2);
  });
});

describe("plan card states", () => {
  it("pending: lists what will be made with estimates and all four actions", () => {
    const html = renderToStaticMarkup(<StudioPlanCard plan={plan()} sessionId="s1" />);
    expect(html).toContain("Plan to approve");
    expect(html).toContain("Rose poster");
    expect(html).toContain("1024x1536");
    expect(html).toContain("about 40 s");
    expect(html).toContain("about 12 GB");
    expect(html).toContain("Qwen3.8-27B stays loaded: about 30 GB free.");
    for (const label of ["Approve all", "Approve this", "Deny", "Edit by chat", "Always allow renders in this chat"]) {
      expect(html).toContain(label);
    }
    expect(html).not.toContain("checked");
  });

  it("each Approve this has its own accessible name", () => {
    const html = renderToStaticMarkup(<StudioPlanCard plan={plan()} sessionId="s1" />);
    expect(html).toContain('aria-label="Approve only this Image: Rose poster"');
    expect(html).toContain('aria-label="Approve only this Image: Tulip poster"');
  });

  it("shows input images, takes, frames, steps, shot counts and the full prompt toggle", () => {
    const html = renderToStaticMarkup(<StudioPlanCard plan={plan({
      items: [
        { callId: "c1", tool: "generate_video", kind: "video", prompt: "Pan", promptFull: "Pan across the shop", inputs: ["id.jpg"], frames: 121, steps: 30, estimatedSeconds: 200, estimateAtRecommended: true },
        { callId: "c2", tool: "make_music", kind: "music", prompt: "Jingle", variations: 4 },
        { callId: "c3", tool: "storyboard_generate", kind: "storyboard", prompt: "p1", shots: 6, force: true },
      ],
    })} sessionId="s1" />);
    for (const text of ["input images: id.jpg", "121 frames", "30 steps", "4 takes", "6 shots", "includes locked shots", "at the recommended size", ">more<", ">Storyboard<"]) {
      expect(html).toContain(text);
    }
  });

  it("one item: Approve, no per-row button", () => {
    const html = renderToStaticMarkup(<StudioPlanCard plan={plan({ items: plan().items.slice(0, 1) })} sessionId="s1" />);
    expect(html).toContain(">Approve<");
    expect(html).not.toContain("Approve this");
  });

  it("decided: shows the outcome per row and no buttons", () => {
    const html = renderToStaticMarkup(<StudioPlanCard plan={plan({ status: "decided", decisions: { c1: "approve", c2: "deny" } })} sessionId="s1" />);
    expect(html).toContain("Approved 1 of 2.");
    expect(html).toContain("Declined");
    expect(html).not.toContain("Approve all");
  });

  it("edit and unanswered say nothing was made", () => {
    expect(renderToStaticMarkup(<StudioPlanCard plan={plan({ status: "edit" })} sessionId="s1" />)).toContain("You asked to change the plan");
    expect(renderToStaticMarkup(<StudioPlanCard plan={plan({ status: "unanswered" })} sessionId="s1" />)).toContain("Not answered");
  });

  it("a risky memory note is an alert", () => {
    const html = renderToStaticMarkup(
      <StudioPlanCard plan={plan({ memory: { chatModel: "Qwen", freeGb: 4, level: "risky", advice: "Estimated peak 70 GB." } })} sessionId="s1" />,
    );
    expect(html).toContain('role="alert"');
    expect(html).toContain("Estimated peak 70 GB.");
  });
});

describe("job cards", () => {
  it("only job-starting tools get a job card", () => {
    expect(studioJobId(toolCall())).toBe("job-1");
    expect(studioJobId(toolCall({ name: "wait_for_job" }))).toBeNull();
    expect(studioJobId(toolCall({ data: { declined: true } }))).toBeNull();
    expect(studioJobId(toolCall({ data: { error: "x" } }))).toBeNull();
  });

  it("declined calls render a declined card", () => {
    const html = renderToStaticMarkup(<StudioJobCard toolCall={toolCall({ data: { declined: true } })} />);
    expect(html).toContain("Declined. Nothing was made.");
  });

  it("a fresh job renders queued with a progress bar", () => {
    const html = renderToStaticMarkup(<StudioJobCard toolCall={toolCall()} />);
    expect(html).toContain('data-job-id="job-1"');
    expect(html).toContain("Queued");
    expect(html).toContain('role="progressbar"');
  });

  it("composer shows the auto-approve chip with Turn off only while it is on", () => {
    expect(composer({ studioTools: true, activeChat: { id: "s9" } as never })).not.toContain("Renders auto-approved");
    writeStudioAutoApprove("s9", true);
    const html = composer({ studioTools: true, activeChat: { id: "s9" } as never });
    expect(html).toContain("Renders auto-approved");
    expect(html).toContain("Turn off");
  });

  it("media kind, library tab, percent, finished", () => {
    expect(mediaKind({ kind: "image", output: { path: "/a.png" } })).toBe("image");
    expect(mediaKind({ kind: "voice", output: { path: "/a.wav" } })).toBe("audio");
    expect(mediaKind({ kind: "music", output: { tracks: [] } })).toBe("audio");
    expect(mediaKind({ kind: "video", output: undefined })).toBe("none");
    expect(libraryTab({ kind: "image" })).toBe("image-gallery");
    expect(libraryTab({ kind: "music" })).toBe("music-library");
    expect(libraryTab({ kind: "video" }, "storyboard_render")).toBe("storyboard");
    expect(jobPercent({ progress: { step: 3, totalSteps: 4 } })).toBe(75);
    expect(jobPercent({})).toBeNull();
    expect(isFinished({ state: "running" })).toBe(false);
    expect(isFinished({ state: "cancelled" })).toBe(true);
  });
});

describe("stream events", () => {
  it("passes plan, resolved plan and live tool results to the callbacks", async () => {
    vi.unstubAllGlobals();
    const events = [
      { studioPlan: plan() },
      { studioPlanResolved: plan({ status: "decided", decisions: { c1: "approve", c2: "deny" } }) },
      { toolCallResult: { id: "c1", name: "generate_image", arguments: { prompt: "Rose" }, result: "{}", elapsed: 0.2, renderAs: "studio", data: { jobId: "job-1" } } },
    ];
    const encoder = new TextEncoder();
    vi.stubGlobal("fetch", vi.fn().mockImplementation((url: string) => {
      if (url.includes("/api/auth/session")) return Promise.resolve(new Response(JSON.stringify({ apiToken: null }), { status: 200 }));
      return Promise.resolve(new Response(new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(encoder.encode(": keepalive\n\n"));
          for (const e of events) controller.enqueue(encoder.encode(`data: ${JSON.stringify(e)}\n`));
          controller.close();
        },
      }), { status: 200 }));
    }));
    const onStudioPlan = vi.fn();
    const onStudioPlanResolved = vi.fn();
    const onToolCallResult = vi.fn();
    await generateChatStream({ prompt: "go", studioTools: true } as never, {
      onToken: vi.fn(), onDone: vi.fn(), onError: vi.fn(), onStudioPlan, onStudioPlanResolved, onToolCallResult,
    });
    expect(onStudioPlan.mock.calls[0][0].planId).toBe("plan-1");
    expect(onStudioPlanResolved.mock.calls[0][0].decisions).toEqual({ c1: "approve", c2: "deny" });
    expect(onToolCallResult.mock.calls[0][0]).toMatchObject({ id: "c1", renderAs: "studio", data: { jobId: "job-1" } });
  });
});
