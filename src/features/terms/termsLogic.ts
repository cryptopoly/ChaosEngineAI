/** First-run terms of use. Bump TERMS_VERSION when the text changes in a way people must accept again. */
export const TERMS_VERSION = 1;

/** The points, in order; the text lives in the `common` namespace under `terms.points.<id>`. */
export const TERMS_POINT_IDS = ["responsibility", "disclosure", "impersonation", "voices", "licences"] as const;

export interface TermsSettings {
  termsAcceptedVersion?: number;
  termsAcceptedAt?: string;
}

/** True when the splash must show: settings are loaded and the user has not accepted this version. */
export function termsNeedAcceptance(settings: TermsSettings | null | undefined): boolean {
  if (!settings) return false;
  return (settings.termsAcceptedVersion ?? 0) < TERMS_VERSION;
}

/** The accepted date as a short local date, or "" when there is none or it does not parse. */
export function formatAcceptedDate(iso: string | null | undefined, locale?: string): string {
  if (!iso) return "";
  const when = new Date(iso);
  if (Number.isNaN(when.getTime())) return "";
  return when.toLocaleDateString(locale, { year: "numeric", month: "short", day: "numeric" });
}
