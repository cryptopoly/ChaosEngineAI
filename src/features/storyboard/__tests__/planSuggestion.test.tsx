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
    expect(html).toContain("Propose writes at most 8 shots; add the rest on the Storyboard tab.");
    expect(render("A 30 s ad", null)).not.toContain("Propose writes at most");
  });

  it("says the usual length when the brief states none, and stays away with no brief", () => {
    expect(render("A trailer for a game", null)).toContain("A trailer usually runs about 1 min, which is about 12 to 15 shots.");
    expect(render("A 4 s sting", null)).toContain("is about 1 shot.");
    expect(render("", null)).toBe("");
  });
});

describe("length and kind words in the shipping languages (S-N)", () => {
  it("reads a stated length in German, French, Spanish, Portuguese, Russian, Japanese, Chinese and Korean", () => {
    expect(briefSeconds("Ein Werbespot von 30 Sekunden für ein Café")).toBe(30);
    expect(briefSeconds("eine Minute über den Hafen")).toBe(60);
    expect(briefSeconds("une publicité de 45 secondes")).toBe(45);
    expect(briefSeconds("un anuncio de un minuto")).toBe(60);
    expect(briefSeconds("um comercial de 20 segundos")).toBe(20);
    expect(briefSeconds("рекламный ролик на 30 секунд")).toBe(30);
    expect(briefSeconds("30-секундный ролик")).toBe(30);
    expect(briefSeconds("одну минуту")).toBe(60);
    expect(briefSeconds("30秒の広告")).toBe(30);
    expect(briefSeconds("1分間の予告編")).toBe(60);
    expect(briefSeconds("一个1分钟的广告")).toBe(60);
    expect(briefSeconds("30초 광고")).toBe(30);
    expect(briefSeconds("2,5 min")).toBe(150);
    // Han numerals (S-Q).
    expect(briefSeconds("三十秒的广告")).toBe(30);
    expect(briefSeconds("一分钟的广告")).toBe(60);
    expect(briefSeconds("十五秒のCM")).toBe(15);
    expect(briefSeconds("九十秒の予告編")).toBe(90);
    expect(briefSeconds("两分鐘")).toBe(120);
    expect(briefSeconds("一个故事")).toBeNull();
  });

  it("still ignores numbers that are not a length", () => {
    expect(briefSeconds("Vier Einstellungen vom Hafen")).toBeNull();
    expect(briefSeconds("Paris 2024")).toBeNull();
    expect(briefSeconds("a film about bus 15")).toBeNull();
  });

  it("knows an ad or a trailer in the shipping languages", () => {
    expect(briefKind("Werbung für ein Café")).toBe("ad");
    expect(briefKind("une publicité pour le thé")).toBe("ad");
    expect(briefKind("un anuncio de té")).toBe("ad");
    expect(briefKind("рекламный ролик")).toBe("ad");
    expect(briefKind("お茶の広告")).toBe("ad");
    expect(briefKind("茶馆广告")).toBe("ad");
    expect(briefKind("차 광고")).toBe("ad");
    expect(briefKind("une bande-annonce")).toBe("trailer");
    expect(briefKind("трейлер игры")).toBe("trailer");
    expect(briefKind("ゲームの予告編")).toBe("trailer");
    expect(briefKind("A film about a pub")).toBe("film");
    expect(shotSuggestion("30秒の広告", null)).toEqual({ seconds: 30, source: "brief", kind: "ad", low: 6, high: 8 });
  });
});
