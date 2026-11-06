import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { LicenceChip, type ModelLicence } from "../LicenceChip";
import { LicenceScreen } from "../LicenceScreen";
import { MakesChips } from "../MakesChips";

function makeLicence(overrides: Partial<ModelLicence> = {}): ModelLicence {
  return {
    id: "stability-community",
    name: "Stability Community License",
    commercial: "limited",
    gated: true,
    summary: "Free under US$1M a year.",
    summaryLines: ["Free for personal use", "Show a credit where you publish"],
    ...overrides,
  };
}

const noop = () => {};

describe("LicenceChip", () => {
  it("states commercial use in words and puts the summary in the tooltip", () => {
    const yes = renderToStaticMarkup(<LicenceChip licence={makeLicence({ name: "MIT", commercial: "yes", gated: false })} />);
    expect(yes).toContain("MIT · commercial");
    expect(yes).toContain("licence-chip--yes");
    expect(yes).toContain('title="Free under US$1M a year."');
    expect(renderToStaticMarkup(<LicenceChip licence={makeLicence({ commercial: "no", gated: false })} />)).toContain("no commercial use");
    expect(renderToStaticMarkup(<LicenceChip licence={makeLicence({ gated: false })} />)).toContain("limited commercial");
  });

  it("prefixes Gated and becomes a button only when it can open the screen", () => {
    const html = renderToStaticMarkup(<LicenceChip licence={makeLicence()} onOpen={noop} />);
    expect(html).toContain("licence-chip--gated");
    expect(html).toContain(">Gated<");
    expect(html).toMatch(/^<button type="button"/);
    expect(renderToStaticMarkup(<LicenceChip licence={makeLicence({ gated: false })} />)).toMatch(/^<span/);
  });
});

describe("LicenceScreen", () => {
  const props = { entry: { name: "Stable Audio Open Small" }, onAccept: noop, onDismiss: noop, onOpenHub: noop, onOpenFull: noop };

  it("is a labelled modal dialog with the bullet lines taken from the licence prop", () => {
    const html = renderToStaticMarkup(<LicenceScreen {...props} licence={makeLicence()} accepted={false} />);
    expect(html).toContain('role="dialog"');
    expect(html).toContain('aria-modal="true"');
    expect(html).toMatch(/aria-labelledby="[^"]+"/);
    expect(html).toMatch(/aria-describedby="[^"]+"/);
    expect(html).toContain("Stable Audio Open Small");
    expect(html).toContain("Stability Community License");
    expect(html).toContain("What this licence asks of you");
    expect(html).toContain("<li>Free for personal use</li>");
    expect(html).toContain("<li>Show a credit where you publish</li>");
  });

  it("offers the Hub steps for a gated, unaccepted model", () => {
    const html = renderToStaticMarkup(<LicenceScreen {...props} licence={makeLicence()} accepted={false} />);
    for (const text of ["Read the full licence", "Open on Hugging Face", "Not now", "I have accepted", "This model is gated"]) {
      expect(html).toContain(text);
    }
  });

  it("shows only OK for a notice, and after the accept", () => {
    const notice = renderToStaticMarkup(
      <LicenceScreen {...props} licence={makeLicence({ gated: false, summaryLines: [], note: "Trained on data the authors describe as legally compliant." })} accepted={false} />,
    );
    expect(notice).toContain("Trained on data the authors describe as legally compliant.");
    expect(notice).toContain(">OK<");
    expect(notice).not.toContain("I have accepted");
    expect(notice).not.toContain("What this licence asks of you");
    const done = renderToStaticMarkup(<LicenceScreen {...props} licence={makeLicence()} accepted />);
    expect(done).toContain(">OK<");
    expect(done).not.toContain("I have accepted");
  });
});

describe("MakesChips", () => {
  it("renders one chip per kind with its label", () => {
    const html = renderToStaticMarkup(<MakesChips makes={["songs", "instrumental", "sfx"]} />);
    expect(html.match(/makes-chip"/g)).toHaveLength(3);
    expect(html).toContain("Songs with vocals");
    expect(html).toContain("Sound effects");
    expect(renderToStaticMarkup(<MakesChips makes={[]} />)).toBe("");
  });
});
