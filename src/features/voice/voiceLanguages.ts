/** The dictation languages offered (English fallback names) and a language's name in the UI language. */
import i18next from "i18next";

export const VOICE_LANGUAGES: Array<[string, string]> = [
  ["", "Detect automatically"],
  ["en", "English"],
  ["de", "German"],
  ["fr", "French"],
  ["es", "Spanish"],
  ["it", "Italian"],
  ["pt", "Portuguese"],
  ["nl", "Dutch"],
  ["pl", "Polish"],
  ["ru", "Russian"],
  ["uk", "Ukrainian"],
  ["ja", "Japanese"],
  ["zh", "Chinese"],
  ["ko", "Korean"],
  ["hi", "Hindi"],
  ["ar", "Arabic"],
  ["tr", "Turkish"],
];

/** A language's name in the UI language (the list above is the English fallback). */
export function voiceLanguageName(code: string, fallback: string, uiLanguage: string): string {
  if (!code) return i18next.t("studio:voice.languages.auto", { defaultValue: fallback });
  try {
    const name = new Intl.DisplayNames([uiLanguage], { type: "language" }).of(code);
    return name ? name.charAt(0).toLocaleUpperCase(uiLanguage) + name.slice(1) : fallback;
  } catch {
    return fallback;
  }
}
