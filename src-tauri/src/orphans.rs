//! Orphan-process cleanup at app launch.
//!
//! Sweeps any leftover ChaosEngineAI subprocess (Python sidecar +
//! llama.cpp children) whose parent died without tearing down its
//! subprocess tree — a 28 GB llama-server orphan otherwise stays
//! wedged until the user manually kills it.
//!
//! Three platform variants:
//! - **Unix** (Linux + macOS): walk `ps -axo pid,ppid,command` and
//!   kill anything with `ppid == 1` whose command line matches one of
//!   our markers (`backend_service.app`, `llama-server*`, `llama-cli`).
//! - **Windows**: `wmic process` queries filter by command-line
//!   substring or image name, then `tasklist` checks parent liveness
//!   before `taskkill /F /T` removes the orphan tree.
//! - **Other** (BSDs, illumos, etc.): no-op.
//!
//! Extracted from `lib.rs` as part of the v0.8.0 refactor.

#[cfg(windows)]
use std::os::windows::process::CommandExt;
use std::process::Command;
#[cfg(unix)]
use std::{thread, time::Duration};

#[cfg(unix)]
pub fn terminate_process_group(pid: i32) {
    unsafe {
        libc::killpg(pid, libc::SIGTERM);
    }
    thread::sleep(Duration::from_millis(300));
    unsafe {
        libc::killpg(pid, libc::SIGKILL);
    }
}

// Substrings / image names that identify a process as a ChaosEngineAI
// subprocess. Ordered turbo-first because substring matching on Unix
// uses `.contains()` — if `llama-server` matched first, it would
// swallow `llama-server-turbo` which has different kill semantics
// down the road (e.g. we might want to preserve turbo logs).
#[cfg(unix)]
const ORPHAN_COMMAND_MARKERS: &[&str] = &[
    "backend_service.app",
    "llama-server-turbo",
    "llama-server",
    "llama-cli",
];

#[cfg(unix)]
pub fn cleanup_orphaned_backends() {
    // Sweep processes re-parented to init (ppid==1) whose command line
    // matches a ChaosEngineAI marker. Covers both the Python sidecar
    // AND its llama.cpp children — when the sidecar crashes before
    // tearing down its subprocess tree, the llama-server processes
    // (which can be 3-30 GB each for large models) otherwise stay
    // wedged until the user manually task-kills them.
    let output = match Command::new("ps")
        .args(["-axo", "pid=,ppid=,command="])
        .output()
    {
        Ok(output) => output,
        Err(_) => return,
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut parts = trimmed.split_whitespace();
        let Some(pid_raw) = parts.next() else {
            continue;
        };
        let Some(ppid_raw) = parts.next() else {
            continue;
        };
        let command = parts.collect::<Vec<_>>().join(" ");
        let Ok(pid) = pid_raw.parse::<i32>() else {
            continue;
        };
        let Ok(ppid) = ppid_raw.parse::<i32>() else {
            continue;
        };
        if ppid != 1 {
            continue;
        }
        if !ORPHAN_COMMAND_MARKERS.iter().any(|marker| command.contains(marker)) {
            continue;
        }
        terminate_process_group(pid);
    }
}

#[cfg(windows)]
pub fn cleanup_orphaned_backends() {
    // Sweep orphaned ChaosEngineAI subprocesses whose parent is gone.
    // Unlike Unix, Windows keeps the orphan's original PPID around, so
    // we check parent liveness via tasklist rather than relying on a
    // re-parent-to-init signal.
    //
    // Two separate WMIC queries because the filters don't compose
    // cleanly in a single `where` clause (commandline LIKE and name=
    // each pull different WMI fields) and the cost of two invocations
    // on startup is tolerable.
    //
    // 1. Python sidecar orphans — matched by commandline containing
    //    `backend_service.app`.
    sweep_orphans_by_wmic_filter("commandline like '%backend_service.app%'");
    // 2. llama.cpp binary orphans — matched by image name. Covers both
    //    the standard `llama-server.exe` and the TurboQuant fork at
    //    `llama-server-turbo.exe`, plus `llama-cli.exe` in case a
    //    future feature uses it. These are the big memory hogs when
    //    they leak (the user reported two 28 GB processes surviving
    //    app close).
    sweep_orphans_by_wmic_filter(
        "name='llama-server.exe' or name='llama-server-turbo.exe' or name='llama-cli.exe'",
    );
}

#[cfg(windows)]
fn sweep_orphans_by_wmic_filter(filter: &str) {
    let output = match Command::new("wmic")
        .args([
            "process",
            "where",
            filter,
            "get",
            "processid,parentprocessid",
            "/format:csv",
        ])
        .creation_flags(0x08000000) // CREATE_NO_WINDOW
        .output()
    {
        Ok(output) => output,
        Err(_) => return,
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        let parts: Vec<&str> = line.split(',').collect();
        // CSV format: Node,ParentProcessId,ProcessId
        if parts.len() < 3 {
            continue;
        }
        let Ok(ppid) = parts[1].trim().parse::<u32>() else {
            continue;
        };
        let Ok(pid) = parts[2].trim().parse::<u32>() else {
            continue;
        };
        // Check if parent is still running. If tasklist itself fails
        // (Windows Defender hook, permissions, etc.) we conservatively
        // assume the parent IS alive so we don't kill a legitimate
        // child of a running backend.
        let parent_alive = Command::new("tasklist")
            .args(["/FI", &format!("PID eq {ppid}"), "/NH"])
            .creation_flags(0x08000000)
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).contains(&ppid.to_string()))
            .unwrap_or(true);
        if !parent_alive {
            let _ = Command::new("taskkill")
                .args(["/F", "/T", "/PID", &pid.to_string()])
                .creation_flags(0x08000000)
                .output();
        }
    }
}

#[cfg(not(any(unix, windows)))]
pub fn cleanup_orphaned_backends() {}
