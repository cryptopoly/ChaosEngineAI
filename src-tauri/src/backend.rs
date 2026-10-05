//! `BackendManager` impl — sidecar lifecycle.
//!
//! Owns the bootstrap → spawn → wait_for_port → probe sequence for
//! the bundled Python backend. Drives the runtime extraction
//! (`runtime::resolve_embedded_runtime`), settings (`settings::*`),
//! lease persistence (`lease::*`), HTTP probe (`probe::*`), and
//! orphaned-process cleanup (`orphans::cleanup_orphaned_backends`).
//!
//! Extracted from `src-tauri/src/lib.rs` as part of the v0.8.0
//! Phase 3-4 refactor. The struct itself stays in `lib.rs`.

use std::env;
use std::fs;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
#[cfg(windows)]
use std::os::windows::process::CommandExt;
use std::path::PathBuf;
use std::process::{Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use tauri::AppHandle;

use crate::binaries::{
    resolve_llama_cli, resolve_llama_server, resolve_llama_server_turbo, resolve_sd_cpp,
};
use crate::env_setup::join_paths;
use crate::lease::{
    cleanup_stale_managed_backend, clear_managed_backend_lease, write_managed_backend_lease,
    ManagedBackendLease,
};
use crate::orphans::cleanup_orphaned_backends;
use crate::probe::{
    fetch_backend_api_token, port_responding, probe_chaosengine_backend,
    request_backend_shutdown, wait_for_port,
};
use crate::runtime::{
    apply_embedded_runtime_env, chaosengine_extras_site_packages_for_python,
    cleanup_legacy_extraction_root, ensure_extras_site_packages_for_python,
    resolve_embedded_runtime, resolve_python_executable, resolve_workspace_root,
    source_workspace_root,
};
use crate::settings::{
    saved_allow_remote_connections, saved_backend_port, saved_hf_cache_path,
    select_backend_port, selected_bind_host,
};
use crate::{BackendManager, BackendRuntimeInfo, BACKEND_POLL_INTERVAL, BACKEND_START_TIMEOUT,
    DEFAULT_BACKEND_PORT, open_log_file, read_log_tail};

/// What the window shows when the sidecar is gone before it served requests.
fn exited_message(status: ExitStatus, log_tail: &str) -> String {
    if log_tail.is_empty() {
        format!("The backend sidecar exited with status {status}.")
    } else {
        format!("The backend sidecar exited with status {status}. {log_tail}")
    }
}

impl BackendManager {
    pub(crate) fn bootstrap(&self, app: &AppHandle) {
        let log_path;
        let port;
        let embedded_runtime = resolve_embedded_runtime(app);
        cleanup_orphaned_backends();
        cleanup_stale_managed_backend(app);
        // Remove the pre-0.6.2 unsuffixed extraction dir if it's still
        // on disk from a previous install. Idempotent — once it's
        // gone this is a no-op forever after.
        cleanup_legacy_extraction_root();

        {
            let mut inner = self.inner.lock().expect("backend lock poisoned");
            if inner.child.is_some() {
                return;
            }

            inner.info.managed_by_tauri = true;
            let allow_remote_connections = saved_allow_remote_connections().unwrap_or(false);
            let bind_host = selected_bind_host(allow_remote_connections);
            let preferred_port = saved_backend_port().unwrap_or(DEFAULT_BACKEND_PORT);
            if let Some(existing) = probe_chaosengine_backend(preferred_port) {
                inner.info.port = preferred_port;
                inner.info.api_base = format!("http://127.0.0.1:{}", inner.info.port);
                inner.info.api_token = fetch_backend_api_token(preferred_port);
                inner.info.process_running = true;
                inner.info.started = true;
                inner.info.startup_error = None;
                inner.info.workspace_root = existing.workspace_root;
                inner.info.python_executable = existing.python_executable;
                inner.info.log_path = None;
                inner.info.launcher_mode = "attached".to_string();
                return;
            }
            let (selected_port, port_warning) = select_backend_port(preferred_port, allow_remote_connections);
            inner.info.port = selected_port;
            inner.info.api_base = format!("http://127.0.0.1:{}", inner.info.port);
            inner.startup_note_soft = port_warning.is_some();
            inner.info.startup_error = port_warning;
            port = inner.info.port;

            let workspace_root = if let Some(runtime) = embedded_runtime.as_ref() {
                runtime.backend_root.clone()
            } else {
                match resolve_workspace_root(app) {
                    Some(path) => path,
                    None => {
                        inner.startup_note_soft = false;
                        inner.info.startup_error =
                            Some("Could not locate the ChaosEngineAI backend workspace.".to_string());
                        return;
                    }
                }
            };

            let python_executable = if let Some(runtime) = embedded_runtime.as_ref() {
                runtime.python_binary.clone()
            } else {
                match resolve_python_executable(&workspace_root) {
                    Some(path) => path,
                    None => {
                        inner.info.workspace_root = Some(workspace_root.display().to_string());
                        inner.startup_note_soft = false;
                        inner.info.startup_error =
                            Some("Could not find a Python runtime for the backend sidecar.".to_string());
                        return;
                    }
                }
            };

            let log_candidate =
                env::temp_dir().join(format!("chaosengine-backend-{}.log", inner.info.port));
            if let Some(parent) = log_candidate.parent() {
                let _ = fs::create_dir_all(parent);
            }

            inner.info.workspace_root = Some(workspace_root.display().to_string());
            inner.info.python_executable = Some(python_executable.display().to_string());
            inner.info.log_path = Some(log_candidate.display().to_string());
            inner.info.launcher_mode = if embedded_runtime.is_some() {
                "embedded".to_string()
            } else if workspace_root == source_workspace_root() {
                "source".to_string()
            } else {
                "bundled".to_string()
            };

            let mut command = Command::new(&python_executable);
            command
                .arg("-m")
                .arg("backend_service.app")
                .current_dir(&workspace_root)
                .env("CHAOSENGINE_HOST", bind_host)
                .env("CHAOSENGINE_PORT", inner.info.port.to_string())
                .env("CHAOSENGINE_MLX_PYTHON", python_executable.as_os_str());

            // Make the persistent extras site-packages path visible to the
            // backend whether we're in embedded-runtime or dev-source mode.
            // The install-gpu-bundle endpoint always writes to this path so
            // users can switch between dev builds / packaged builds without
            // redownloading 2 GB of CUDA torch every time. The path is
            // namespaced by Python ``major.minor`` so cp311 wheels can't
            // shadow a cp312 runtime (or vice versa).
            let python_version_hint = embedded_runtime
                .as_ref()
                .and_then(|runtime| runtime.python_version.as_deref());
            if let Some(extras) =
                ensure_extras_site_packages_for_python(&python_executable, python_version_hint)
            {
                command.env("CHAOSENGINE_EXTRAS_SITE_PACKAGES", extras.as_os_str());
            }

            // FU-038 (2026-05-10): silence the macOS MallocStackLogging
            // banner spam that floods the backend log file. The macOS
            // hardened runtime (which we ship under
            // ``bundle.macOS.hardenedRuntime: true``) sometimes inherits
            // a ``MallocStackLogging`` style flag from the Tauri parent
            // process, and every Python subprocess prints
            // ``Python(PID) MallocStackLogging: can't turn off malloc stack
            // logging because it was not enabled.`` at startup. Three
            // lines per spawn, hundreds per minute when polling system
            // metrics — drowns out the actual INFO / ERROR lines the
            // Diagnostics tab is meant to surface. ``env_remove`` drops
            // the variable from the child's environment entirely (setting
            // it to "0" still counts as "set" to the malloc allocator,
            // which is what triggers the warning in the first place).
            // Pure stderr noise; no behaviour change.
            command.env_remove("MallocStackLogging");
            command.env_remove("MallocStackLoggingNoCompact");
            command.env_remove("MallocScribble");

            // Inject HF_HOME when the user has configured a non-default
            // HuggingFace cache location (typically because the system
            // drive is full). This MUST be set before the backend process
            // starts — huggingface_hub reads HF_HOME at module import, so
            // setting it later via os.environ has no effect.
            if let Some(hf_home) = saved_hf_cache_path() {
                command.env("HF_HOME", &hf_home);
            }

            if let Some(runtime) = embedded_runtime.as_ref() {
                apply_embedded_runtime_env(&mut command, runtime);
                if let Some(llama_server) = runtime.llama_server.as_ref() {
                    command.env("CHAOSENGINE_LLAMA_SERVER", llama_server.as_os_str());
                }
                if let Some(llama_server_turbo) = runtime.llama_server_turbo.as_ref() {
                    command.env("CHAOSENGINE_LLAMA_SERVER_TURBO", llama_server_turbo.as_os_str());
                }
                if let Some(llama_cli) = runtime.llama_cli.as_ref() {
                    command.env("CHAOSENGINE_LLAMA_CLI", llama_cli.as_os_str());
                }
                if let Some(sd_cpp) = runtime.sd_cpp.as_ref() {
                    command.env("CHAOSENGINE_SDCPP_BIN", sd_cpp.as_os_str());
                }
            } else {
                // Source-workspace mode: the backend runs against the
                // developer's .venv so Python auto-loads .venv/site-packages
                // at startup. We still want extras (the persistent
                // ``~/.chaosengine/extras/site-packages`` dir populated by
                // /api/setup/install-gpu-bundle) to WIN over anything in
                // .venv — otherwise a stale CPU torch hanging around in
                // the dev venv would shadow the freshly-installed CUDA
                // torch in extras, which is exactly the failure the user
                // hit on Windows (video gen silently ran on CPU despite
                // a successful CUDA install).
                //
                // apply_embedded_runtime_env already does this for the
                // embedded path; this is the matching source-workspace
                // branch. No-op if extras doesn't exist yet.
                if let Some(extras) =
                    chaosengine_extras_site_packages_for_python(&python_executable, python_version_hint)
                        .filter(|p| p.is_dir())
                {
                    if let Some(python_path) = join_paths(&[extras]) {
                        command.env("PYTHONPATH", python_path);
                    }
                }
                if let Some(llama_server) = resolve_llama_server(&workspace_root) {
                    command.env("CHAOSENGINE_LLAMA_SERVER", llama_server.as_os_str());
                }
                if let Some(llama_server_turbo) = resolve_llama_server_turbo(&workspace_root) {
                    command.env("CHAOSENGINE_LLAMA_SERVER_TURBO", llama_server_turbo.as_os_str());
                }
                if let Some(llama_cli) = resolve_llama_cli(&workspace_root) {
                    command.env("CHAOSENGINE_LLAMA_CLI", llama_cli.as_os_str());
                }
                if let Some(sd_cpp) = resolve_sd_cpp(&workspace_root) {
                    command.env("CHAOSENGINE_SDCPP_BIN", sd_cpp.as_os_str());
                }
            }

            if let Some(stdout) = open_log_file(&log_candidate) {
                command.stdout(Stdio::from(stdout));
            } else {
                command.stdout(Stdio::null());
            }
            if let Some(stderr) = open_log_file(&log_candidate) {
                command.stderr(Stdio::from(stderr));
            } else {
                command.stderr(Stdio::null());
            }

            // Put the backend in its own process group on Unix so we can
            // kill the whole tree (Python + MLX worker subprocess) on shutdown.
            //
            // On Linux we ALSO set PR_SET_PDEATHSIG so the kernel delivers
            // SIGKILL to the backend if the Tauri parent dies for any
            // reason — including SIGKILL from the OOM killer, a crash, or
            // a force-close from a system activity monitor — before the
            // in-Python watchdog even runs. This closes the race where
            // the parent dies between the watchdog's 500ms polls.
            //
            // macOS has no PR_SET_PDEATHSIG equivalent, so it relies on
            // the Python watchdog (backend_service.app::_watch_parent_and_exit)
            // which detects parent death via getppid() polling and
            // killpg's the whole session. Gap is ~500ms worst case.
            #[cfg(unix)]
            unsafe {
                command.pre_exec(|| {
                    libc::setsid();
                    #[cfg(target_os = "linux")]
                    libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL);
                    Ok(())
                });
            }

            // On Windows, prevent the Python backend from opening a visible
            // console window.  CREATE_NO_WINDOW = 0x08000000.
            #[cfg(windows)]
            {
                command.creation_flags(0x08000000);
            }

            match command.spawn() {
                Ok(child) => {
                    // Windows: assign the spawned Python to our
                    // kill-on-close Job Object so its entire subprocess
                    // tree (llama-server, llama-server-turbo, any future
                    // native children) dies automatically when Tauri
                    // exits — even on a hard kill where our graceful
                    // shutdown code never runs. See
                    // windows_job::assign_to_kill_on_close_job for the
                    // mechanism. The call is best-effort: if Job Object
                    // creation fails we still have the reactive
                    // cleanup_orphaned_backends sweep on next launch.
                    #[cfg(windows)]
                    {
                        let _ = crate::windows_job::assign_to_kill_on_close_job(&child);
                    }

                    let lease = ManagedBackendLease {
                        pid: child.id(),
                        port: inner.info.port,
                    };
                    write_managed_backend_lease(app, &lease);
                    inner.info.process_running = true;
                    inner.child = Some(child);
                }
                Err(error) => {
                    clear_managed_backend_lease(app);
                    inner.info.process_running = false;
                    inner.startup_note_soft = false;
                    inner.info.startup_error = Some(format!("Failed to start the backend sidecar: {error}"));
                    return;
                }
            }

            log_path = log_candidate;
        }

        let started = wait_for_port(port, BACKEND_START_TIMEOUT, BACKEND_POLL_INTERVAL, || {
            self.sidecar_running()
        });

        let mut inner = self.inner.lock().expect("backend lock poisoned");
        inner.info.started = started;
        if started {
            inner.info.api_token = fetch_backend_api_token(port);
            return;
        }

        let detail = read_log_tail(&log_path);
        let exit_status = inner
            .child
            .as_mut()
            .and_then(|child| child.try_wait().ok().flatten());
        if let Some(status) = exit_status {
            // The sidecar died before it ever answered: say so now rather than
            // letting the window spin until the deadline.
            inner.info.process_running = false;
            inner.startup_note_soft = false;
            inner.info.startup_error = Some(exited_message(status, &detail));
            return;
        }
        // Alive but silent: a slow cold start, not a failure. runtime_info()
        // drops this note once the port answers.
        inner.startup_note_soft = true;
        inner.info.startup_error = Some(if detail.is_empty() {
            "The backend sidecar has not answered yet and may still be starting.".to_string()
        } else {
            format!("The backend sidecar has not answered yet and may still be starting. {detail}")
        });
    }

    fn sidecar_running(&self) -> bool {
        let mut inner = self.inner.lock().expect("backend lock poisoned");
        match inner.child.as_mut() {
            Some(child) => matches!(child.try_wait(), Ok(None)),
            None => false,
        }
    }

    pub(crate) fn runtime_info(&self) -> BackendRuntimeInfo {
        let mut inner = self.inner.lock().expect("backend lock poisoned");
        let log_path = inner.info.log_path.clone().map(PathBuf::from);

        if let Some(child) = inner.child.as_mut() {
            match child.try_wait() {
                Ok(Some(status)) => {
                    inner.info.process_running = false;
                    inner.info.started = false;
                    if inner.info.startup_error.is_none() || inner.startup_note_soft {
                        let tail = log_path
                            .as_ref()
                            .map(|path| read_log_tail(path))
                            .unwrap_or_default();
                        inner.startup_note_soft = false;
                        inner.info.startup_error = Some(exited_message(status, &tail));
                    }
                }
                Ok(None) => {
                    inner.info.process_running = true;
                    inner.info.started = port_responding(inner.info.port);
                    if inner.info.started && inner.startup_note_soft {
                        inner.startup_note_soft = false;
                        inner.info.startup_error = None;
                    }
                    if inner.info.started && inner.info.api_token.is_none() {
                        inner.info.api_token = fetch_backend_api_token(inner.info.port);
                    }
                }
                Err(error) => {
                    inner.info.process_running = false;
                    inner.info.started = false;
                    inner.info.startup_error =
                        Some(format!("Could not inspect the backend sidecar process: {error}"));
                }
            }
        } else {
            if inner.info.managed_by_tauri && inner.info.launcher_mode == "attached" {
                let responding = port_responding(inner.info.port);
                inner.info.process_running = responding;
                inner.info.started = responding;
                if responding && inner.info.api_token.is_none() {
                    inner.info.api_token = fetch_backend_api_token(inner.info.port);
                }
                if !responding {
                    inner.info.startup_error = Some("The attached backend is no longer reachable.".to_string());
                }
            } else {
                inner.info.process_running = false;
            }
        }

        inner.info.clone()
    }

    pub(crate) fn shutdown(&self) {
        let mut inner = self.inner.lock().expect("backend lock poisoned");
        let attached_backend = if inner.child.is_none() && inner.info.managed_by_tauri && inner.info.started {
            Some((inner.info.port, inner.info.api_token.clone()))
        } else {
            None
        };
        inner.info.process_running = false;
        inner.info.started = false;
        inner.info.startup_error = None;
        inner.startup_note_soft = false;
        // The Python sidecar generates a fresh API token on each startup.
        // Wipe our cached copy now so the next runtime_info call re-fetches
        // instead of handing the frontend a token that no longer unlocks
        // the new backend.
        inner.info.api_token = None;
        if let Some(mut child) = inner.child.take() {
            #[cfg(unix)]
            {
                // Kill the entire process group (Python backend + MLX worker)
                let pid = child.id() as i32;
                unsafe {
                    libc::killpg(pid, libc::SIGTERM);
                }
                // Give it a moment to clean up
                thread::sleep(Duration::from_millis(500));
                unsafe {
                    libc::killpg(pid, libc::SIGKILL);
                }
            }
            #[cfg(windows)]
            {
                // On Windows, child.kill() only kills the parent Python
                // process, not its children (MLX worker, etc.).  Use
                // `taskkill /T` to terminate the entire process tree.
                //
                // Previously this ignored taskkill's exit code, which made
                // the "Restart Backend" button hang on machines where
                // taskkill returned non-zero (race with process exit, UAC
                // elevation mismatch, etc.) — child.wait() below would
                // then block forever holding the BackendManager mutex, and
                // subsequent runtime_info() calls deadlocked the UI.
                let pid = child.id();
                let taskkill_ok = match std::process::Command::new("taskkill")
                    .args(["/F", "/T", "/PID", &pid.to_string()])
                    .creation_flags(0x08000000) // CREATE_NO_WINDOW
                    .output()
                {
                    Ok(out) => out.status.success(),
                    Err(_) => false,
                };
                if !taskkill_ok {
                    // Fall back to TerminateProcess on the parent. Any
                    // grandchildren may leak, but the port-release poll in
                    // restart_backend_sidecar covers the subsequent respawn.
                    let _ = child.kill();
                }
            }
            #[cfg(not(any(unix, windows)))]
            {
                let _ = child.kill();
            }
            // Bounded wait: try_wait in a loop so a hung child can't deadlock
            // the shutdown path. std::process::Child::wait has no timeout.
            let wait_deadline = Instant::now() + Duration::from_secs(3);
            loop {
                match child.try_wait() {
                    Ok(Some(_)) => break,
                    Ok(None) => {
                        if Instant::now() >= wait_deadline {
                            break;
                        }
                        thread::sleep(Duration::from_millis(50));
                    }
                    Err(_) => break,
                }
            }
        } else if let Some((port, api_token)) = attached_backend {
            let effective_token = api_token.or_else(|| fetch_backend_api_token(port));
            let _ = request_backend_shutdown(port, effective_token.as_deref());
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::ManagedBackend;
    use std::net::TcpListener;
    use std::process::Child;
    use std::sync::Mutex;

    fn manager(child: Option<Child>, note: Option<&str>, soft: bool, port: u16) -> BackendManager {
        let mut inner = ManagedBackend::default();
        inner.child = child;
        inner.info.port = port;
        inner.info.managed_by_tauri = true;
        inner.info.process_running = true;
        inner.info.startup_error = note.map(str::to_string);
        inner.startup_note_soft = soft;
        BackendManager { inner: Mutex::new(inner) }
    }

    fn exited_child(code: i32) -> Child {
        let mut child = Command::new("sh")
            .args(["-c", &format!("exit {code}")])
            .spawn()
            .expect("spawn sh");
        child.wait().expect("wait for sh");
        child
    }

    fn stop(manager: &BackendManager) {
        if let Some(mut child) = manager.inner.lock().expect("lock").child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    #[test]
    fn exited_message_names_the_status_and_appends_the_log_tail() {
        let status = exited_child(3).try_wait().expect("try_wait").expect("exit status");
        let bare = exited_message(status, "");
        assert!(bare.contains("exit status: 3"), "{bare}");
        assert!(bare.ends_with('.'), "{bare}");
        assert!(exited_message(status, "boom").ends_with(" boom"));
    }

    #[test]
    fn sidecar_running_follows_the_child() {
        assert!(!manager(None, None, false, 1).sidecar_running());
        assert!(!manager(Some(exited_child(0)), None, false, 1).sidecar_running());
        let alive = manager(Some(Command::new("sleep").arg("30").spawn().expect("spawn sleep")), None, false, 1);
        assert!(alive.sidecar_running());
        stop(&alive);
    }

    #[test]
    fn runtime_info_replaces_a_soft_note_with_the_crash() {
        let manager = manager(Some(exited_child(1)), Some("slow start"), true, 1);
        let info = manager.runtime_info();
        assert!(!info.process_running);
        let message = info.startup_error.expect("a crash message");
        assert!(message.contains("exited with status"), "{message}");
        assert!(!manager.inner.lock().expect("lock").startup_note_soft);
    }

    #[test]
    fn runtime_info_keeps_a_real_failure_over_a_later_poll() {
        let manager = manager(Some(exited_child(1)), Some("Failed to start the backend sidecar: denied"), false, 1);
        let info = manager.runtime_info();
        assert_eq!(info.startup_error.as_deref(), Some("Failed to start the backend sidecar: denied"));
    }

    #[test]
    fn runtime_info_clears_a_soft_note_once_the_port_answers() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind");
        let port = listener.local_addr().expect("local addr").port();
        let manager = manager(
            Some(Command::new("sleep").arg("30").spawn().expect("spawn sleep")),
            Some("slow start"),
            true,
            port,
        );
        // A cached token keeps runtime_info from asking the listener for one.
        manager.inner.lock().expect("lock").info.api_token = Some("token".to_string());
        let info = manager.runtime_info();
        stop(&manager);
        drop(listener);
        assert!(info.started);
        assert_eq!(info.startup_error, None);
    }

    #[test]
    fn runtime_info_keeps_a_soft_note_while_the_port_is_silent() {
        let free = TcpListener::bind(("127.0.0.1", 0)).expect("bind").local_addr().expect("local addr").port();
        let manager = manager(
            Some(Command::new("sleep").arg("30").spawn().expect("spawn sleep")),
            Some("slow start"),
            true,
            free,
        );
        let info = manager.runtime_info();
        stop(&manager);
        assert!(info.process_running);
        assert!(!info.started);
        assert_eq!(info.startup_error.as_deref(), Some("slow start"));
    }
}
