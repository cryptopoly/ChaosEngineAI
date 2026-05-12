//! Windows-only Job Object management for orphan prevention.
//!
//! A Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` is a
//! kernel-level mechanism that guarantees every process in the job
//! dies when the job handle closes. We create one job at first spawn,
//! hold the handle for the entire Tauri process lifetime (so it closes
//! only when Tauri exits), and assign every spawned backend child to
//! it. Child processes inherit job membership automatically on
//! Windows 8+, so llama-server grandchildren land in the same job
//! without us tracking them individually.
//!
//! Why this matters: our graceful shutdown taskkill can be skipped if
//! Tauri itself is SIGKILL'd (Task Manager's End Task, OOM killer,
//! power event, Rust panic before state.shutdown fires). Job Objects
//! protect against every one of those paths — the kernel itself does
//! the kill, not user-mode code.
//!
//! Extracted from `lib.rs` as part of the v0.8.0 refactor.

#![cfg(windows)]

use std::mem;
use std::os::windows::io::AsRawHandle;
use std::ptr;
use std::sync::OnceLock;

use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
    SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
};

// HANDLE is a raw pointer so not naturally Send/Sync. We store it as
// usize — the kernel object it references is global to the process,
// so concurrent access from multiple threads is safe (the Windows API
// is thread-safe for these calls). OnceLock gives us lazy init with
// a single atomic store.
static JOB_HANDLE: OnceLock<usize> = OnceLock::new();

/// Create (or reuse) the singleton kill-on-close Job Object and
/// return its handle. Returns None only if Job Object creation
/// itself fails, which effectively never happens on modern Windows
/// outside of heavily locked-down environments (e.g. Windows
/// Sandbox with Jobs disabled).
fn ensure_job() -> Option<HANDLE> {
    let handle = JOB_HANDLE.get_or_init(|| unsafe {
        let job = CreateJobObjectW(ptr::null(), ptr::null());
        if job.is_null() {
            return 0;
        }

        // Flip the KILL_ON_JOB_CLOSE bit. Everything else in the
        // limit block stays zeroed so we don't impose memory / CPU
        // caps on the backend (it manages its own resource use).
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = mem::zeroed();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;

        let ok = SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            &info as *const _ as *const _,
            mem::size_of_val(&info) as u32,
        );
        if ok == 0 {
            let _ = CloseHandle(job);
            return 0;
        }
        job as usize
    });
    if *handle == 0 {
        None
    } else {
        Some(*handle as HANDLE)
    }
}

/// Assign a spawned child process to our kill-on-close Job. Returns
/// true on success. Non-fatal on failure — the reactive
/// `cleanup_orphaned_backends` sweep still catches anything that
/// leaks through.
///
/// Timing note: Rust's `Command::spawn()` creates the process
/// running (no CREATE_SUSPENDED). There's a theoretical race where
/// the child could spawn its own children before we assign it.
/// In practice this is safe because the child is a Python
/// interpreter that takes ~100ms+ to finish importing the FastAPI
/// stack before it even calls subprocess.Popen for llama-server.
pub fn assign_to_kill_on_close_job(child: &std::process::Child) -> bool {
    let Some(job) = ensure_job() else {
        return false;
    };
    let process_handle = child.as_raw_handle() as HANDLE;
    unsafe { AssignProcessToJobObject(job, process_handle) != 0 }
}
