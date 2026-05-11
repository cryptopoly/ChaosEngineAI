// FU-042 — Tauri shell i18n bootstrap.
//
// `rust-i18n` is a compile-time macro that bundles every YAML/JSON file
// under the directory passed to `i18n!()` into the binary as static
// data, then exposes `t!("dot.path.key")` for lookup with the current
// locale (default `en`).  Wiring is intentionally tiny — the Tauri
// shell only needs localized strings for the native menu / tray /
// updater dialog text; everything else lives in the React layer.
//
// Usage from `lib.rs` (or wherever the menu is built):
//
//     use crate::i18n;
//     i18n::set_locale("zh-CN");
//     let label = i18n::t("menu.file");
//
// `set_locale` is idempotent and threadsafe — rust-i18n stores the
// active tag in an atomic.  Calls from the React layer over IPC drive
// it; see the future `set_app_locale` Tauri command.
//
// For dynamic strings that need ICU plural/select (e.g. "{n, plural,
// one {# minute remaining} other {# minutes remaining}}") use the
// `fluent-bundle` path instead — rust-i18n's interpolation is
// `%{name}`-style and doesn't handle Slavic plural categories.

rust_i18n::i18n!("locales", fallback = "en");

/// Set the active locale.  Accepts any string `rust-i18n` recognises;
/// unknown tags silently fall back to the compiled-in fallback.
pub fn set_locale(locale: &str) {
    rust_i18n::set_locale(locale);
}

/// Shorthand for `rust_i18n::t!` from non-macro contexts.
#[allow(dead_code)]
pub fn t(key: &str) -> String {
    rust_i18n::t!(key).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_to_english() {
        set_locale("en");
        assert_eq!(t("menu.file"), "File");
    }

    #[test]
    fn unknown_locale_falls_back_to_en() {
        set_locale("xx-ZZ");
        assert_eq!(t("menu.file"), "File");
    }
}
