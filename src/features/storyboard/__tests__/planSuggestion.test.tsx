import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { PlanSuggestion } from "../PlanSuggestion";
import { briefKind, briefSeconds, shotSuggestion } from "../planLogic";

describe("shot count suggestion rule (FU-188 item 5)", () => {
  it("reads a length from the brief in digits or words", () => {
    expect(briefSeconds("A 30 s ad for a tea shop")).toBe(30);
    expect(briefSeconds("a 30-second spot")).toBe(30);
    expect(briefSeconds("A one-minute advert for a small tea shop")).toBe(60);
    expect(briefSeconds("ninety seconds of slow mornings")).toBe(90);
    expect(briefSeconds("2 min trailer")).toBe(120);
    expect(briefSeconds("about 45s")).toBe(45);
    expect(briefSeconds("Four shots of the harbour")).toBeNull();
    expect(briefSeconds("a calm film")).toBeNull();
  });

  it("knows an ad and a trailer from a few words", () => {
    expect(briefKind("An advert for tea")).toBe("ad");
    expect(briefKind("a commercial")).toBe("ad");
    expect(briefKind("a game trailer")).toBe("trailer");
    expect(briefKind("a quiet film about the sea")).toBe("film");
    expect(briefKind("a shadow over the road")).toBe("film");
  });

  it("gives 6 to 8 shots for 30 s, the brief's length first, then the Length choice, then the usual length", () => {
    expect(shotSuggestion("A 30 s ad", 60)).toEqual({ seconds: 30, source: "brief", kind: "ad", low: 6, high: 8 });
    expect(shotSuggestion("A tea shop film", 60)).toEqual({ seconds: 60, source: "target", kind: "film", low: 12, high: 15 });
    expect(shotSuggestion("A trailer for a game", null)).toEqual({ seconds: 60, source: "typical", kind: "trailer", low: 12, high: 15 });
    expect(shotSuggestion("", 15)).toEqual({ seconds: 15, source: "target", kind: "film", low: 3, high: 4 });
    expect(shotSuggestion("A 4 s sting", null)).toMatchObject({ low: 1, high: 1 });
    expect(shotSuggestion("  ", null)).toBeNull();
  });
});

describe("PlanSuggestion", () => {
  const lengthText = (s: number) => (s >= 60 && s % 60 === 0 ? `${s / 60} min` : `${s} s`);
  const render = (brief: string, target: number | null) =>
    renderToStaticMarkup(<PlanSuggestion brief={brief} target={target} lengthText={lengthText} onUseLength={() => undefined} />);

  it("labels the line as a suggestion and offers the brief's length", () => {
    const html = render("A 30 s ad for a tea shop", null);
    expect(html).toContain("<strong>Suggestion:</strong> An ad of 30 s is about 6 to 8 shots.");
    expect(html).toContain(">Use 30 s</button>");
  });

  it("uses the Length choice without offering it again", () => {
    const html = render("A quiet film", 60);
    expect(html).toContain("A film of 1 min is about 12 to 15 shots.");
    expect(html).not.toContain("Use ");
  });

  it("says the usual length when the brief states none, and stays away with no brief", () => {
    expect(render("A trailer for a game", null)).toContain("A trailer usually runs about 1 min, which is about 12 to 15 shots.");
    expect(render("A 4 s sting", null)).toContain("is about 1 shot.");
    expect(render("", null)).toBe("");
  });
});
