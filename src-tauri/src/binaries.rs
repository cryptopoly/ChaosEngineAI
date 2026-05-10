//! Resolves paths to bundled binaries (llama-server, sd, etc.).
//!
//! Each `resolve_*` honours an env-var override first, falls back to
//! `~/.chaosengine/bin/<name>` for the standard / turbo / sd-cli
//! managed installs, then finally walks `PATH`. The two utility
//! helpers (`resolve_candidate`, `find_in_path`) are shared with
//! `lib.rs` for cases that don't need the per-binary fallback chain.
//!
//! Extracted from `src-tauri/src/lib.rs` as part of the v0.8.0
//! Phase 3 refactor.

use std::env;
use std::path::{Path, PathBuf};


pub fn resolve_llama_server(_workspace_root: &Path) -> Option<PathBuf> {
    if let Some(value) = env::var_os("CHAOSENGINE_LLAMA_SERVER") {
        if let Some(path) = resolve_candidate(value) {
            return Some(path);
        }
    }

    find_in_path(&["llama-server"])
}

pub fn resolve_llama_server_turbo(_workspace_root: &Path) -> Option<PathBuf> {
    if let Some(value) = env::var_os("CHAOSENGINE_LLAMA_SERVER_TURBO") {
        if let Some(path) = resolve_candidate(value) {
            return Some(path);
        }
    }

    // Check ~/.chaosengine/bin/ first (ChaosEngineAI-managed installs),
    // then fall back to PATH.
    if let Ok(home) = env::var("HOME") {
        let managed = PathBuf::from(home)
            .join(".chaosengine")
            .join("bin")
            .join("llama-server-turbo");
        if managed.exists() {
            return Some(managed);
        }
    }

    find_in_path(&["llama-server-turbo"])
}

pub fn resolve_llama_cli(_workspace_root: &Path) -> Option<PathBuf> {
    if let Some(value) = env::var_os("CHAOSENGINE_LLAMA_CLI") {
        if let Some(path) = resolve_candidate(value) {
            return Some(path);
        }
    }

    find_in_path(&["llama-cli"])
}

pub fn resolve_sd_cpp(_workspace_root: &Path) -> Option<PathBuf> {
    if let Some(value) = env::var_os("CHAOSENGINE_SDCPP_BIN") {
        if let Some(path) = resolve_candidate(value) {
            return Some(path);
        }
    }

    if let Ok(home) = env::var("HOME") {
        let managed = PathBuf::from(home).join(".chaosengine").join("bin").join("sd");
        if managed.exists() {
            return Some(managed);
        }
    }

    find_in_path(&["sd"])
}

pub fn resolve_candidate(value: impl Into<PathBuf>) -> Option<PathBuf> {
    let candidate = value.into();
    if candidate.exists() {
        return Some(candidate);
    }
    if candidate.components().count() == 1 {
        return find_in_path(&[candidate.to_string_lossy().as_ref()]);
    }
    None
}

pub fn find_in_path(names: &[&str]) -> Option<PathBuf> {
    let path_var = env::var_os("PATH")?;
    for directory in env::split_paths(&path_var) {
        for name in names {
            let candidate = directory.join(name);
            if candidate.exists() {
                return Some(candidate);
            }
            #[cfg(windows)]
            {
                let exe_candidate = directory.join(format!("{name}.exe"));
                if exe_candidate.exists() {
                    return Some(exe_candidate);
                }
            }
        }
    }
    None
}
