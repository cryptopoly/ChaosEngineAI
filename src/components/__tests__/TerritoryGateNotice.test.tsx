import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { TerritoryModel } from "../../api/licenceGate";
import { TerritoryGateNotice, territoryLicence } from "../TerritoryGateNotice";

function model(overrides: Partial<TerritoryModel> = {}): TerritoryModel {
  return {
    key: "minimax-h3",
    name: "MiniMax H3",
    kinds: ["video"],
    excludedTerritories: ["European Union", "United Kingdom", "Republic of Korea", "United States"],
    licenceName: "MiniMax H3 Community License",
    licenceUrl: "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/x/LICENSE",
    summaryLines: ["Only outside the excluded territories."],
    confirmSentence: "I confirm that I am not there.",
    note: "Not tested by the maintainers.",
    confirmed: false,
    confirmedAt: null,
    ...overrides,
  };
}

const api = { get: vi.fn(async () => ({ models: [] })), confirm: vi.fn(), withdraw: vi.fn() };

function render(models: TerritoryModel[], kind: "image" | "video" = "video") {
  return renderToStaticMarkup(
    <TerritoryGateNotice kind={kind} onChanged={vi.fn()} onOpenExternalUrl={vi.fn()} initial={{ models }} api={api} />,
  );
}

describe("TerritoryGateNotice", () => {
  it("renders nothing when no territory-limited model is released for the kind", () => {
    expect(render([])).toBe("");
    expect(render([model()], "image")).toBe("");
  });

  it("is a collapsed notice with Review until the statement is confirmed", () => {
    const html = render([model()]);
    expect(html).toMatch(/^<details class="territory-gate-notice">/);
    expect(html).toContain("Models with territory limits");
    expect(html).toContain("MiniMax H3");
    expect(html).toContain("Hidden until you confirm");
    expect(html).toContain(">Review<");
    expect(html).not.toContain(">Withdraw<");
  });

  it("offers Withdraw once confirmed", () => {
    const html = render([model({ confirmed: true, confirmedAt: "2026-11-14T10:00:00+0000" })]);
    expect(html).toContain(">Withdraw<");
    expect(html).not.toContain(">Review<");
  });

  it("maps the model to a territory-limited licence for the screen and chip", () => {
    const licence = territoryLicence(model());
    expect(licence.territoryRestricted).toBe(true);
    expect(licence.gated).toBe(false);
    expect(licence.url).toContain("MiniMax-H3");
  });
});
