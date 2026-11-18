/**
 * Review L-1: the first-launch banner asked "Switch the app to Deutsch?"
 * in German, on an app already running in German.
 *
 * Decision: the banner states what happened, in the app's language (the
 * detected one), and offers "Keep {endonym}" and "Use English". "Use
 * English" is always in English so it can be read by someone who cannot
 * read the detected language. This file pins the wording contract; the
 * behaviour on a German system is in e2e spec 34.
 */
import { describe, expect, it } from "vitest";
import { changeLocale, i18n, type SupportedLocale } from "../../i18n";
import enCommon from "../../locales/en/common.json";
import enMeta from "../../locales/en/meta.json";

const SHIPPING: SupportedLocale[] = ["zh-CN", "zh-TW", "ja", "de", "ru", "ko", "fr", "es", "pt-BR"];
const endonyms = new Map(
  (enMeta as { supportedLocales: Array<{ code: string; endonym: string }> }).supportedLocales.map((entry) => [entry.code, entry.endonym]),
);

describe("first-launch language banner wording (review L-1)", () => {
  it("no longer has the switch question", () => {
    expect(Object.keys(enCommon.localeBanner).sort()).toEqual(["ariaLabel", "changeHint", "inUse", "keepCurrent", "useEnglish"]);
  });

  it.each(SHIPPING)("%s: states the language in use, in that language, and does not ask to switch to it", async (locale) => {
    await changeLocale(locale);
    const endonym = endonyms.get(locale) as string;
    const banner = i18n.getResourceBundle(locale, "common").localeBanner as Record<string, string>;
    const inUse = i18n.t("localeBanner.inUse", { endonym });
    const keep = i18n.t("localeBanner.keepCurrent", { endonym });

    // Translated, with the language's own name filled in.
    expect(banner.inUse).not.toBe(enCommon.localeBanner.inUse);
    expect(banner.changeHint).not.toBe(enCommon.localeBanner.changeHint);
    expect(banner.keepCurrent).not.toBe(enCommon.localeBanner.keepCurrent);
    expect(inUse).toContain(endonym);
    expect(keep).toContain(endonym);
    expect(`${inUse}${keep}`).not.toMatch(/\{|\}/);
    // A statement, not a question (the old text ended "…{endonym}?").
    expect(inUse).not.toMatch(/[?？]/);
    // The way back is readable without knowing the detected language.
    expect(i18n.t("localeBanner.useEnglish", { lng: "en" })).toBe("Use English");
    expect(banner.useEnglish).toBe("Use English");
    await changeLocale("en");
  });
});
