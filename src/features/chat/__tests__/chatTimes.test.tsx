import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { ToolCallInfo } from "../../../types";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import { JobProgressLine } from "../../../components/JobProgressLine";
import { ToolCallCard } from "../../../components/ToolCallCard";
import { applyToolProgress, mergeToolCalls, upsertToolCall } from "../studioChat";

function call(over: Partial<ToolCallInfo> = {}): ToolCallInfo {
  return { id: "c1", name: "wait_for_job", arguments: { job_id: "job-11" }, result: "{}", elapsed: 2, ...over };
}

describe("one card for repeated waits on a job", () => {
  it("collapses 24 polls of job-11 into one card and keeps the first start time", () => {
    const polls = Array.from({ length: 24 }, (_, i) =>
      call({ id: `c${i}`, startedAt: 1_000 + i * 60_000, elapsed: 60, data: { tool: "wait_for_job", state: i === 23 ? "done" : "running" } }),
    );
    const merged = mergeToolCalls(polls);
    expect(merged).toHaveLength(1);
    expect(merged[0].id).toBe("c0");
    expect(merged[0].startedAt).toBe(1_000);
    expect(merged[0].polls).toBe(24);
    expect(merged[0].elapsed).toBe(24 * 60);
    expect(merged[0].data?.state).toBe("done");
  });

  it("keeps other tools and other jobs apart and in order", () => {
    const merged = mergeToolCalls([
      call({ id: "a", arguments: { job_id: "job-1" } }),
      call({ id: "b", name: "generate_image", arguments: { prompt: "x" } }),
      call({ id: "c", arguments: { job_id: "job-2" } }),
      call({ id: "d", arguments: { job_id: "job-1" } }),
    ]);
    expect(merged.map((m) => m.id)).toEqual(["a", "b", "c"]);
    expect(merged[0].polls).toBe(2);
    expect(merged[2].polls).toBe(1);
  });

  it("passes an old session through, including a call with null arguments", () => {
    const old = [call({ id: "x", arguments: null as never }), call({ id: "y", name: "calculator", arguments: {} })];
    expect(mergeToolCalls(old).map((m) => m.id)).toEqual(["x", "y"]);
    expect(mergeToolCalls(undefined)).toEqual([]);
  });
});

describe("running tool cards", () => {
  it("the result keeps the start time of the running card", () => {
    const running = call({ id: "r", running: true, startedAt: 5_000, elapsed: 0, result: "" });
    const [done] = upsertToolCall([running], call({ id: "r", startedAt: null, elapsed: 9 }));
    expect(done.startedAt).toBe(5_000);
    expect(done.running).toBeFalsy();
  });

  it("progress lands only on a running card", () => {
    const running = call({ id: "r", running: true, startedAt: 5_000 });
    const [updated] = applyToolProgress([running], { id: "r", data: { state: "running", progress: { step: 2 } } });
    expect(updated.data?.progress).toEqual({ step: 2 });
    expect(updated.renderAs).toBe("studio");
    const finished = call({ id: "f" });
    expect(applyToolProgress([finished], { id: "f", data: { state: "x" } })[0]).toBe(finished);
    expect(applyToolProgress([running], { id: "other", data: {} })).toEqual([running]);
  });
});

describe("ToolCallCard timing", () => {
  const started = new Date(2026, 10, 9, 7, 31).getTime();

  it("shows when the call started and how long it took", () => {
    const html = renderToStaticMarkup(<ToolCallCard toolCall={call({ startedAt: started, elapsed: 252 })} />);
    expect(html).toContain("started ");
    expect(html).toContain("4 min 12 sec");
    expect(html).toContain('data-testid="tool-call-timing"');
  });

  it("shows the number of merged checks", () => {
    const html = renderToStaticMarkup(<ToolCallCard toolCall={{ ...call({ startedAt: started, elapsed: 90 }), polls: 3 }} />);
    expect(html).toContain("3 checks");
  });

  it("an old call without a start time still shows its elapsed seconds", () => {
    const html = renderToStaticMarkup(<ToolCallCard toolCall={call({ elapsed: 1.5 })} />);
    expect(html).toContain("1.5s");
    expect(html).not.toContain("started ");
  });

  it("a running wait shows its live progress line", () => {
    const html = renderToStaticMarkup(
      <ToolCallCard
        toolCall={call({
          running: true,
          startedAt: Date.now() - 5_000,
          elapsed: 0,
          data: { progress: { phase: "clip 2 of 4", elapsedSeconds: 400, etaSeconds: 740, etaBasis: "clips" } },
        })}
      />,
    );
    expect(html).toContain("clip 2 of 4");
    expect(html).toContain("about ");
    expect(html).toContain("(from the clips made so far)");
  });
});

describe("JobProgressLine", () => {
  it("shows elapsed only when there is no basis for an estimate", () => {
    const html = renderToStaticMarkup(<JobProgressLine progress={{ phase: "loading", elapsedSeconds: 75 }} running />);
    expect(html).toContain("loading");
    expect(html).toContain("1 min 15 sec elapsed");
    expect(html).not.toContain("left");
  });

  it("shows the step and an estimated time left with its basis", () => {
    const html = renderToStaticMarkup(
      <JobProgressLine progress={{ phase: "diffusing", step: 4, totalSteps: 8, elapsedSeconds: 30, etaSeconds: 20, etaBasis: "steps" }} running />,
    );
    expect(html).toContain("step 4 of 8");
    expect(html).toContain("about 20 sec left");
    const catalog = renderToStaticMarkup(<JobProgressLine progress={{ elapsedSeconds: 10, etaSeconds: 90, etaBasis: "catalog" }} running />);
    expect(catalog).toContain("(estimated for this model)");
  });

  it("a film render's percent is not shown as steps, and a finished job shows no estimate", () => {
    const render = renderToStaticMarkup(<JobProgressLine progress={{ phase: "joining", step: 40, totalSteps: 100, elapsedSeconds: 5 }} running />);
    expect(render).not.toContain("step 40");
    const finished = renderToStaticMarkup(<JobProgressLine progress={{ elapsedSeconds: 10, etaSeconds: 90, etaBasis: "clips" }} running={false} />);
    expect(finished).not.toContain("left");
  });

  it("shows nothing for a job with no progress at all", () => {
    expect(renderToStaticMarkup(<JobProgressLine progress={null} running />)).toBe("");
  });
});
