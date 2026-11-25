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

type T = (key: string, options?: Record<string, unknown>) => string;

const tStudio: T = (key, options) => i18next.t(`studio:${key}`, options) as string;

/**
 * Speech to text models that never take a language setting: they find the language themselves (the Language note that
 * sat under the old dictation models list, now said per model). Every other model uses the language when one is
 * chosen and finds it when Detect is chosen.
 */
const FINDS_LANGUAGE_ITSELF = new Set(["parakeet-tdt-v3", "granite-speech-5.0-2b", "nemotron-3.5-transcribe-30b"]);

/** True for a model that ignores the Language setting, by catalog key. */
export function sttFindsLanguageItself(key: string | undefined): boolean {
  return key !== undefined && FINDS_LANGUAGE_ITSELF.has(key);
}

export type SttLanguageMode = "auto" | "choice";

/** What the model will do for this language setting: find the language itself, or use the one chosen. */
export function sttLanguageMode(key: string | undefined, language: string): SttLanguageMode {
  return sttFindsLanguageItself(key) || !language ? "auto" : "choice";
}

/** The words beside a model in a list: "finds the language itself" or "uses your choice". */
export function sttLanguageModeText(mode: SttLanguageMode, t: T = tStudio): string {
  return mode === "auto"
    ? t("voice.transcribe.languageFindsItself", { defaultValue: "finds the language itself" })
    : t("voice.transcribe.languageUsesChoice", { defaultValue: "uses your choice" });
}
