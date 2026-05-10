//! Desktop settings + port selection.
//!
//! Reads the user-saved settings JSON at `~/.chaosengine/settings.json`
//! (or `%APPDATA%\.chaosengine\settings.json` on Windows) and resolves
//! the backend port + bind host from the values stored there. The
//! `select_backend_port` helper falls back to an OS-assigned port when
//! the preferred one is busy so users get a deterministic startup
//! even on machines where another service has claimed 8876.
//!
//! Extracted from `lib.rs` as part of the v0.8.0 refactor.

use std::env;
use std::fs;
use std::net::TcpListener;
use std::path::PathBuf;

use serde::Deserialize;

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SavedDesktopSettings {
    pub preferred_server_port: Option<u16>,
    pub allow_remote_connections: Option<bool>,
    // Redirects HuggingFace cache to a user-chosen drive. We read it here
    // so we can set HF_HOME on the backend child BEFORE huggingface_hub
    // is first imported — setting it post-import is a no-op.
    pub hf_cache_path: Option<String>,
    // `auto_start_server` was previously read here to gate Python sidecar
    // bootstrap. The sidecar now always starts (it's required for /api/*),
    // and that toggle only controls the inference engine inside the backend.
}

pub fn settings_path() -> Option<PathBuf> {
    let base = if cfg!(windows) {
        env::var_os("APPDATA").map(PathBuf::from)
    } else {
        env::var_os("HOME").map(PathBuf::from)
    };
    base.map(|dir| dir.join(".chaosengine").join("settings.json"))
}

pub fn saved_backend_port() -> Option<u16> {
    let path = settings_path()?;
    let payload = fs::read_to_string(path).ok()?;
    let settings: SavedDesktopSettings = serde_json::from_str(&payload).ok()?;
    settings
        .preferred_server_port
        .filter(|port| (1024..=65535).contains(port))
}

pub fn saved_allow_remote_connections() -> Option<bool> {
    let path = settings_path()?;
    let payload = fs::read_to_string(path).ok()?;
    let settings: SavedDesktopSettings = serde_json::from_str(&payload).ok()?;
    settings.allow_remote_connections
}

// Read the user-configured HuggingFace cache path from settings.json.
// Returns None when the setting is missing / empty (falls through to HF's
// platform default). Expands `~` to the user profile so the value is
// directly usable as HF_HOME by Rust/Python consumers that don't call
// expanduser themselves (e.g. huggingface_hub internals).
pub fn saved_hf_cache_path() -> Option<String> {
    let path = settings_path()?;
    let payload = fs::read_to_string(path).ok()?;
    let settings: SavedDesktopSettings = serde_json::from_str(&payload).ok()?;
    let raw = settings.hf_cache_path?.trim().to_string();
    if raw.is_empty() {
        return None;
    }
    // Home-directory lookup uses platform env vars rather than pulling in
    // the `dirs` crate just for this one call — USERPROFILE on Windows,
    // HOME on Unix is enough for the `~` expansion we need here.
    let home_dir = || -> Option<PathBuf> {
        #[cfg(windows)]
        {
            std::env::var_os("USERPROFILE").map(PathBuf::from)
        }
        #[cfg(not(windows))]
        {
            std::env::var_os("HOME").map(PathBuf::from)
        }
    };
    if let Some(rest) = raw.strip_prefix("~/").or_else(|| raw.strip_prefix("~\\")) {
        if let Some(home) = home_dir() {
            return Some(home.join(rest).to_string_lossy().into_owned());
        }
    } else if raw == "~" {
        if let Some(home) = home_dir() {
            return Some(home.to_string_lossy().into_owned());
        }
    }
    Some(raw)
}

pub fn selected_bind_host(allow_remote_connections: bool) -> &'static str {
    if allow_remote_connections {
        "0.0.0.0"
    } else {
        "127.0.0.1"
    }
}

/// Try to bind the preferred port; fall back to an OS-assigned port if busy.
/// Returns `(port, warning)` — `warning` is set when the preferred port was
/// unavailable so the caller can surface it to the user.
pub fn select_backend_port(preferred: u16, allow_remote_connections: bool) -> (u16, Option<String>) {
    let bind_host = selected_bind_host(allow_remote_connections);
    if TcpListener::bind((bind_host, preferred)).is_ok() {
        return (preferred, None);
    }
    match TcpListener::bind((bind_host, 0)) {
        Ok(listener) => {
            if let Ok(addr) = listener.local_addr() {
                let alt = addr.port();
                (alt, Some(format!(
                    "Port {preferred} is in use. Using port {alt} instead."
                )))
            } else {
                (preferred, Some(format!(
                    "Port {preferred} is in use and no alternative could be determined."
                )))
            }
        }
        Err(_) => (preferred, Some(format!(
            "Port {preferred} is in use and no alternative port could be allocated."
        ))),
    }
}
