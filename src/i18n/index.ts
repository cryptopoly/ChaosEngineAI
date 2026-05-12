/*
 * FU-042 — frontend i18n bootstrap.
 *
 * Loads namespaced JSON catalogs under ``src/locales/<lang>/*.json`` into
 * ``i18next``, plumbs ICU MessageFormat for plurals / select, and exposes
 * a negotiation helper that maps OS / browser BCP-47 tags onto the set
 * we actually ship.
 *
 * Top-level wiring lives in ``src/main.tsx`` — it calls
 * ``initI18n({ persistedLocale })`` *before* React mounts so the very
 * first render of ``<App />`` already has the right strings.  The
 * Tauri-side persisted ``settings.locale`` flows in via the existing
 * settings hydration call ``initialSettingsFromRustShell`` (see
 * ``src-tauri/src/settings.rs``); when absent the chain falls back to
 * the OS locale → ``navigator.language`` → ``en`` baseline.
 *
 * Per FU-042 plan §6 (translation update workflow): missing keys in a
 * non-en locale silently fall back to ``en`` — ``returnEmptyString:
 * false`` + ``fallbackLng: 'en'``.  This lets feature work ship en-only
 * without breaking other locales while follow-up PRs catch up.
 */

import i18n, { type PostProcessorModule } from "i18next";
import IntlMessageFormat from "intl-messageformat";
import { initReactI18next } from "react-i18next";

/**
 * ICU MessageFormat post-processor.
 *
 * The published `i18next-icu` 2.4.x plugin expects i18next v22's
 * positional `parse(res, options, lng, ns, key, info)` signature, but
 * i18next v24 calls `parse(res, dataObject)`. The mismatch leaves
 * `{var}` and `{count, plural, …}` strings unsubstituted at render
 * time. Until upstream catches up we wire `intl-messageformat`
 * ourselves via a post-processor — runs after i18next resolves the
 * key (or returns the defaultValue) and re-runs ICU substitution
 * with the call-site variables.
 *
 * The cache keys on `${lng}::${value}` so repeated calls reuse the
 * compiled IntlMessageFormat instance — formatter construction is
 * the dominant cost.
 */
const icuCache = new Map<string, IntlMessageFormat>();

const icuPostProcessor: PostProcessorModule = {
  type: "postProcessor",
  name: "icu",
  process(value: string, _key: string | string[], options: Record<string, unknown>, translator: { language?: string }): string {
    if (typeof value !== "string") return value as unknown as string;
    if (!value.includes("{")) return value;
    const lng = (options.lng as string | undefined) ?? translator.language ?? "en";
    const cacheKey = `${lng}::${value}`;
    let formatter = icuCache.get(cacheKey);
    if (!formatter) {
      try {
        formatter = new IntlMessageFormat(value, lng, undefined, { ignoreTag: true });
        icuCache.set(cacheKey, formatter);
      } catch {
        return value; // malformed pattern — fall through with literal string
      }
    }
    try {
      const out = formatter.format(options as Record<string, unknown>);
      return typeof out === "string" ? out : value;
    } catch {
      return value;
    }
  },
};

// Eagerly bundle all namespaces for English (source of truth) so that
// the moment ``initI18n`` resolves we can render *something* without a
// flash-of-empty-strings.  Other locales lazy-load via dynamic import
// once selected — keeps the initial bundle tight.
//
// Vite supports ``import.meta.glob('./...', { eager: true })`` to inline
// JSON at build time, but to keep the type story simple (no glob types
// in tsconfig) we list them.
import enCommon       from "../locales/en/common.json";
import enChat         from "../locales/en/chat.json";
import enStudio       from "../locales/en/studio.json";
import enLibrary      from "../locales/en/library.json";
import enSetup        from "../locales/en/setup.json";
import enDiagnostics  from "../locales/en/diagnostics.json";
import enRuntime      from "../locales/en/runtime.json";
import enErrors       from "../locales/en/errors.json";
import enMeta         from "../locales/en/meta.json";
import enDashboard    from "../locales/en/dashboard.json";

export const SUPPORTED_LOCALES = [
  "en",
  "zh-CN",
  "zh-TW",
  "ja",
  "de",
  "ru",
  "ko",
  "fr",
  "es",
  "pt-BR",
] as const;

export type SupportedLocale = (typeof SUPPORTED_LOCALES)[number];

export const NAMESPACES = [
  "common",
  "chat",
  "studio",
  "library",
  "setup",
  "diagnostics",
  "runtime",
  "errors",
  "meta",
  "dashboard",
] as const;

export type Namespace = (typeof NAMESPACES)[number];

const DEFAULT_NAMESPACE: Namespace = "common";

/**
 * Map a raw BCP-47 tag (``en-US``, ``zh``, ``pt``, ``zh-Hant-TW``…) onto
 * one of the locales we actually ship.  Normalisation rules per the
 * FU-042 plan:
 *
 *   - ``zh-Hant-*``, ``zh-HK``, ``zh-TW``          → ``zh-TW``
 *   - ``zh-Hans-*``, ``zh-CN``, bare ``zh``        → ``zh-CN``
 *   - ``pt`` (no region) / ``pt-BR``               → ``pt-BR``
 *   - ``pt-PT`` (Iberian)                          → ``pt-BR`` (closest until we ship pt-PT)
 *   - ``en-*``                                     → ``en``
 *   - anything else with a registered base match   → that base
 *   - otherwise                                    → ``en`` baseline
 */
export function normaliseLocale(raw: string | null | undefined): SupportedLocale {
  if (!raw) return "en";
  const lower = raw.toLowerCase();
  // Traditional Chinese family
  if (lower === "zh-tw" || lower === "zh-hk" || lower === "zh-mo" || lower.startsWith("zh-hant")) {
    return "zh-TW";
  }
  // Simplified Chinese family (bare ``zh`` defaults to Simplified — mainland
  // is the larger cohort and matches our anchor-tier shipping order).
  if (lower === "zh" || lower === "zh-cn" || lower === "zh-sg" || lower.startsWith("zh-hans")) {
    return "zh-CN";
  }
  // Brazilian / Iberian Portuguese both → pt-BR until pt-PT ships
  if (lower === "pt" || lower === "pt-br" || lower === "pt-pt") {
    return "pt-BR";
  }
  // English family
  if (lower === "en" || lower.startsWith("en-")) return "en";
  // Try a region-stripped match against the registered list
  const base = lower.split("-", 1)[0];
  const direct = SUPPORTED_LOCALES.find((tag) => tag.toLowerCase() === lower);
  if (direct) return direct;
  const byBase = SUPPORTED_LOCALES.find((tag) => tag.toLowerCase() === base);
  if (byBase) return byBase;
  return "en";
}

/**
 * Resolve the locale to use, honouring (in priority order):
 *
 *   1. An explicit override (e.g. the user-persisted ``settings.locale``
 *      coming back from Rust on startup).
 *   2. ``navigator.languages`` — first hit that we ship.
 *   3. ``navigator.language``.
 *   4. ``en`` baseline.
 *
 * The Tauri OS locale (``tauri-plugin-os::locale()``) is preferred over
 * ``navigator.*`` when running inside the desktop shell; callers pass it
 * in via ``override``.  In browser dev mode (vite serve), there's no
 * Tauri call so we fall back to the navigator chain — that's fine for
 * debugging.
 */
export function detectLocale(override?: string | null): SupportedLocale {
  if (override) {
    const explicit = normaliseLocale(override);
    return explicit;
  }
  if (typeof navigator !== "undefined") {
    const langs = navigator.languages ?? [navigator.language];
    for (const candidate of langs) {
      const matched = normaliseLocale(candidate);
      if (matched !== "en" || candidate.toLowerCase().startsWith("en")) {
        return matched;
      }
    }
  }
  return "en";
}

/** Map of locale → {namespace → bundle}.  Mutated as locales load. */
const loadedBundles = new Map<SupportedLocale, Record<Namespace, unknown>>();

const enBundles: Record<Namespace, unknown> = {
  common: enCommon,
  chat: enChat,
  studio: enStudio,
  library: enLibrary,
  setup: enSetup,
  diagnostics: enDiagnostics,
  runtime: enRuntime,
  errors: enErrors,
  meta: enMeta,
  dashboard: enDashboard,
};
loadedBundles.set("en", enBundles);

/**
 * Eagerly resolve every locale's JSON catalogs at build time via
 * ``import.meta.glob`` so Vite emits them as static assets the bundle
 * actually contains.
 *
 * The previous implementation used ``import(\`../locales/${locale}/${ns}.json\`)``
 * with a ``/* @vite-ignore *\/`` comment, which told Vite *not* to
 * analyse the path — so the production build never emitted the
 * per-locale chunks.  At runtime the dynamic import then 404'd inside
 * the packaged app and ``i18n.addResourceBundle`` was never called,
 * which fell back to en for every key.  Switching to
 * ``import.meta.glob`` keeps the catalogs lazy-loaded (each file is
 * its own chunk, loaded only when the locale is picked) but lets Vite
 * see + emit them.
 */
const LOCALE_GLOB = import.meta.glob("../locales/*/*.json") as Record<
  string,
  () => Promise<{ default: unknown }>
>;

async function loadLocale(locale: SupportedLocale): Promise<void> {
  if (loadedBundles.has(locale)) return;
  const bundles: Partial<Record<Namespace, unknown>> = {};
  await Promise.all(
    NAMESPACES.map(async (ns) => {
      const key = `../locales/${locale}/${ns}.json`;
      const loader = LOCALE_GLOB[key];
      if (!loader) return; // Locale file not present yet — fallback to en at runtime.
      try {
        const mod = await loader();
        bundles[ns] = mod.default ?? mod;
      } catch {
        // Parse / network error — fallback to en at runtime.
      }
    }),
  );
  loadedBundles.set(locale, bundles as Record<Namespace, unknown>);
  NAMESPACES.forEach((ns) => {
    const bundle = bundles[ns];
    if (bundle !== undefined) i18n.addResourceBundle(locale, ns, bundle, true, true);
  });
}

export interface InitI18nOptions {
  /** Persisted user preference from ``settings.json``, or ``"system"`` to auto-detect. */
  persistedLocale?: string | null;
  /** OS locale reported by ``tauri-plugin-os::locale()``; ignored in browser dev. */
  osLocale?: string | null;
  /** Force a specific locale (used by pseudo-locale dev mode + tests). */
  forceLocale?: SupportedLocale | null;
  /** Enable i18next debug logging in development. */
  debug?: boolean;
}

export async function initI18n(opts: InitI18nOptions = {}): Promise<typeof i18n> {
  let initial: SupportedLocale;
  if (opts.forceLocale) {
    initial = opts.forceLocale;
  } else if (opts.persistedLocale && opts.persistedLocale !== "system") {
    initial = normaliseLocale(opts.persistedLocale);
  } else {
    initial = detectLocale(opts.osLocale);
  }

  await i18n
    .use(icuPostProcessor)
    .use(initReactI18next)
    .init({
      lng: initial,
      fallbackLng: "en",
      defaultNS: DEFAULT_NAMESPACE,
      ns: [...NAMESPACES],
      // Custom ICU post-processor (icuPostProcessor above) handles
      // `{var}` + `{count, plural, …}` substitution via intl-messageformat.
      // Applied to every t() call by default; opt out per-call with
      // `{ postProcess: false }`.
      postProcess: ["icu"],
      interpolation: {
        // React already escapes its render output; double-escaping breaks CJK.
        escapeValue: false,
      },
      returnEmptyString: false,
      // i18next types ResourceLanguage as Record<string, ResourceKey>
      // where ResourceKey is recursive — the imported JSON modules
      // are typed as `unknown` by Vite's default JSON loader, so we
      // cast through `Record<string, object>` to satisfy the i18next
      // typings without losing the namespace shape.
      resources: {
        en: NAMESPACES.reduce<Record<string, object>>((acc, ns) => {
          acc[ns] = enBundles[ns] as object;
          return acc;
        }, {}),
      },
      debug: opts.debug ?? false,
    });

  if (initial !== "en") await loadLocale(initial);
  syncDocumentLang(initial);
  return i18n;
}

export async function changeLocale(locale: SupportedLocale): Promise<void> {
  await loadLocale(locale);
  await i18n.changeLanguage(locale);
  syncDocumentLang(locale);
}

function syncDocumentLang(locale: SupportedLocale): void {
  if (typeof document === "undefined") return;
  document.documentElement.setAttribute("lang", locale);
  // RTL hook — none of the top-10 are RTL, but the attribute is cheap
  // and prepares us for FU-046 Arabic/Hebrew.
  const rtl = (enMeta as { supportedLocales?: Array<{ code: string; rtl: boolean }> }).supportedLocales?.find(
    (entry) => entry.code === locale,
  )?.rtl;
  document.documentElement.setAttribute("dir", rtl ? "rtl" : "ltr");
}

export { i18n };
