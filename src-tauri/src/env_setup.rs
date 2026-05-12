//! Environment-variable + path-list helpers for spawning the
//! Python sidecar.
//!
//! Three pure utilities lifted out of `lib.rs`:
//!
//! * `apply_library_path` — set a `*_LIBRARY_PATH` env var on a
//!   `Command` builder, prepending the supplied entries to whatever
//!   the parent process inherits.
//! * `join_paths` — `OsString` join with the platform path separator.
//! * `prepend_env_paths` — read an env var, prepend the supplied
//!   entries, return the merged value.
//!
//! Extracted from `src-tauri/src/lib.rs` as part of the v0.8.0
//! Phase 3-2 refactor.

use std::env;
use std::ffi::OsString;
use std::path::PathBuf;
use std::process::Command;


pub fn apply_library_path(command: &mut Command, variable: &str, entries: &[PathBuf]) {
    if let Some(value) = prepend_env_paths(variable, entries) {
        command.env(variable, value);
    }
}

pub fn join_paths(entries: &[PathBuf]) -> Option<OsString> {
    if entries.is_empty() {
        return None;
    }
    env::join_paths(entries).ok()
}

pub fn prepend_env_paths(variable: &str, entries: &[PathBuf]) -> Option<OsString> {
    if entries.is_empty() {
        return env::var_os(variable);
    }
    let mut combined = entries.to_vec();
    if let Some(existing) = env::var_os(variable) {
        combined.extend(env::split_paths(&existing));
    }
    env::join_paths(combined).ok()
}
