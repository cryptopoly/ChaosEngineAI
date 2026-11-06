import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { RunEstimateStrip, memoryRisk, type RunEstimateStripProps } from "../RunEstimateStrip";

function strip(overrides: Partial<RunEstimateStripProps> = {}) {
  return renderToStaticMarkup(
    <RunEstimateStrip
      seconds={40}
      peakGb={9}
      totalGb={64}
      diskMb={12}
      basis="estimated"
      risk="safe"
      action={<button type="button">Make track</button>}
      {...overrides}
    />,
  );
}

describe("RunEstimateStrip", () => {
  it("reads time, memory of total and disk in that order, announced politely", () => {
    const html = strip();
    expect(html).toContain('aria-live="polite"');
    const time = html.indexOf("About 40 s");
    const memory = html.indexOf("Peak 9 GB of 64 GB");
    const disk = html.indexOf("Writes 12 MB");
    expect(time).toBeGreaterThan(-1);
    expect(memory).toBeGreaterThan(time);
    expect(disk).toBeGreaterThan(memory);
    expect(html).toContain("Make track");
  });

  it("formats minutes and gigabytes of disk", () => {
    const html = strip({ seconds: 140, diskMb: 1500 });
    expect(html).toContain("About 2 min 20 s");
    expect(html).toContain("Writes 1.5 GB");
  });

  it("never leaves an unknown value blank", () => {
    const html = strip({ seconds: null, peakGb: null, totalGb: null, diskMb: null, basis: null });
    expect(html).toContain("Time: unknown until the first run");
    expect(html).toContain("Memory: not estimated");
    expect(html).toContain("Disk: not estimated");
    expect(html).not.toContain("run-estimate__basis");
  });

  it("shows the basis chip with its tooltip", () => {
    expect(strip({ basis: "estimated" })).toContain("Estimated");
    const measured = strip({ basis: "measured" });
    expect(measured).toContain(">Measured<");
    expect(measured).toContain("Timed on this machine for this model and preset.");
  });

  it("adds a queue line when another job is running", () => {
    const html = strip({ queue: { what: "Image Studio", secondsLeft: 60 } });
    expect(html).toContain("Image Studio is running, about 1 min left. Yours starts after it.");
    expect(strip({ queue: { what: "", secondsLeft: null, position: 3 } })).toContain("position 3");
    expect(strip()).not.toContain("run-estimate__queue");
  });

  it("renders the acknowledgement checkbox inside the strip only at danger", () => {
    const ack = { checked: false, onChange: () => {} };
    const danger = strip({ risk: "danger", ack, peakGb: 62 });
    expect(danger).toContain("run-estimate--danger");
    expect(danger).toContain('type="checkbox"');
    expect(danger).toContain("Run anyway. I accept that the backend may crash.");
    expect(strip({ risk: "safe", ack })).not.toContain('type="checkbox"');
    expect(strip({ risk: "caution", ack })).toContain("run-estimate--caution");
  });

  it("colours memory by share of total: under 70 % safe, 70 to 92 % caution, above danger", () => {
    expect(memoryRisk(40, 64, "safe")).toBe("safe");
    expect(memoryRisk(45, 64, "safe")).toBe("caution");
    expect(memoryRisk(60, 64, "safe")).toBe("danger");
    expect(memoryRisk(null, 64, "caution")).toBe("caution");
    expect(strip({ peakGb: 50 })).toContain("run-estimate__item--memory run-estimate__item--caution");
  });
});
