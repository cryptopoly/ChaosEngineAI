//! Managed-backend lease persistence + stale-cleanup.
//!
//! When Tauri spawns the Python sidecar, we write `(pid, port)` to
//! `managed-backend.json` in the app data dir. On the next launch
//! `cleanup_stale_managed_backend` reads this lease, probes the
//! recorded port, and — if a previous backend is still running —
//! asks it to shut down so we don't end up with two competing
//! sidecars on consecutive launches (e.g. user closed window without
//! Quit, app re-opened from Dock).
//!
//! Two-phase safety check before issuing the shutdown:
//! 1. `probe_chaosengine_backend` confirms `/api/health` returns
//!    `{"status": "ok"}` — guards against killing an unrelated
//!    process that reused the port.
//! 2. The workspace-root check is intentionally permissive (treats
//!    `None` as "ours") because dev builds frequently report no
//!    workspace and we don't want to leave sidecars wedged.
//!
//! Extracted from `lib.rs` as part of the v0.8.0 refactor.

use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager};

use crate::probe::{
    fetch_backend_api_token, probe_chaosengine_backend, request_backend_shutdown,
};

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ManagedBackendLease {
    pub pid: u32,
    pub port: u16,
}

pub fn managed_backend_lease_path(app: &AppHandle) -> Option<PathBuf> {
    app.path().app_data_dir().ok().map(|path| path.join("managed-backend.json"))
}

pub fn write_managed_backend_lease(app: &AppHandle, lease: &ManagedBackendLease) {
    let Some(path) = managed_backend_lease_path(app) else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(payload) = serde_json::to_vec(lease) {
        let _ = fs::write(path, payload);
    }
}

pub fn read_managed_backend_lease(app: &AppHandle) -> Option<ManagedBackendLease> {
    let path = managed_backend_lease_path(app)?;
    let payload = fs::read(path).ok()?;
    serde_json::from_slice(&payload).ok()
}

pub fn clear_managed_backend_lease(app: &AppHandle) {
    if let Some(path) = managed_backend_lease_path(app) {
        let _ = fs::remove_file(path);
    }
}

pub fn cleanup_stale_managed_backend(app: &AppHandle) {
    let Some(lease) = read_managed_backend_lease(app) else {
        return;
    };

    // Only shut down the process on the leased port if it is actually a
    // ChaosEngineAI backend (probe_chaosengine_backend verifies /api/health
    // returns {"status": "ok"}).  This prevents killing unrelated services
    // that happen to reuse the same port number.
    if let Some(probe) = probe_chaosengine_backend(lease.port) {
        // Extra safety: if we know the workspace root, only shut down if it
        // matches — another ChaosEngineAI instance on a different workspace
        // should be left alone.
        let dominated = probe.workspace_root.is_none()
            || app
                .path()
                .app_data_dir()
                .ok()
                .and_then(|dir| dir.parent().map(|p| p.to_path_buf()))
                .is_none();
        if dominated {
            let api_token = fetch_backend_api_token(lease.port);
            let _ = request_backend_shutdown(lease.port, api_token.as_deref());
        }
    }

    clear_managed_backend_lease(app);
}
