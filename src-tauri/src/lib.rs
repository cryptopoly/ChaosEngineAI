use serde::{Deserialize, Serialize};
use std::{
    env,
    ffi::OsString,
    fs,
    fs::OpenOptions,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::Mutex,
    thread,
    time::{Duration, Instant},
};
#[cfg(unix)]
use std::os::unix::process::CommandExt;
#[cfg(windows)]
use std::os::windows::process::CommandExt;
use tauri::{AppHandle, Manager, State};
use tar::Archive;

const DEFAULT_BACKEND_PORT: u16 = 8876;
const BACKEND_START_TIMEOUT: Duration = Duration::from_secs(12);
const BACKEND_POLL_INTERVAL: Duration = Duration::from_millis(200);

// Desktop settings + port selection — see settings.rs.
mod settings;
use settings::{
    saved_allow_remote_connections, saved_backend_port, saved_hf_cache_path,
    select_backend_port, selected_bind_host,
};

// HTTP probe + lifecycle helpers — see probe.rs.
mod probe;
use probe::{
    fetch_backend_api_token, port_responding, probe_chaosengine_backend,
    request_backend_shutdown, wait_for_port,
};

// Managed-backend lease persistence — see lease.rs.
mod lease;
use lease::{
    cleanup_stale_managed_backend, clear_managed_backend_lease,
    write_managed_backend_lease, ManagedBackendLease,
};

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct BackendRuntimeInfo {
    api_base: String,
    api_token: Option<String>,
    port: u16,
    managed_by_tauri: bool,
    process_running: bool,
    started: bool,
    startup_error: Option<String>,
    workspace_root: Option<String>,
    python_executable: Option<String>,
    log_path: Option<String>,
    launcher_mode: String,
}

impl Default for BackendRuntimeInfo {
    fn default() -> Self {
        Self {
            api_base: format!("http://127.0.0.1:{DEFAULT_BACKEND_PORT}"),
            api_token: None,
            port: DEFAULT_BACKEND_PORT,
            managed_by_tauri: false,
            process_running: false,
            started: false,
            startup_error: None,
            workspace_root: None,
            python_executable: None,
            log_path: None,
            launcher_mode: "unknown".to_string(),
        }
    }
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct EmbeddedRuntimeManifest {
    mode: Option<String>,
    backend_root: String,
    python_binary: String,
    python_home: String,
    python_path: Vec<String>,
    library_path_entries: Vec<String>,
    path_entries: Vec<String>,
    llama_server: Option<String>,
    llama_server_turbo: Option<String>,
    llama_cli: Option<String>,
    sd_cpp: Option<String>,
    // ``"3.12"`` etc. — used to namespace the persistent extras dir so
    // wheels compiled for Python X.Y don't get loaded by a different X.Z.
    python_version: Option<String>,
}


#[derive(Clone)]
struct EmbeddedRuntime {
    backend_root: PathBuf,
    python_binary: PathBuf,
    python_home: PathBuf,
    python_path: Vec<PathBuf>,
    library_path_entries: Vec<PathBuf>,
    path_entries: Vec<PathBuf>,
    llama_server: Option<PathBuf>,
    llama_server_turbo: Option<PathBuf>,
    llama_cli: Option<PathBuf>,
    sd_cpp: Option<PathBuf>,
    python_version: Option<String>,
}



#[derive(Default)]
struct ManagedBackend {
    info: BackendRuntimeInfo,
    child: Option<Child>,
}

#[derive(Default)]
struct BackendManager {
    inner: Mutex<ManagedBackend>,
}

#[tauri::command]
fn app_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[tauri::command]
fn backend_runtime_info(state: State<'_, BackendManager>) -> BackendRuntimeInfo {
    state.runtime_info()
}

#[tauri::command]
fn stop_backend_sidecar(state: State<'_, BackendManager>) -> BackendRuntimeInfo {
    state.shutdown();
    state.runtime_info()
}

#[tauri::command]
fn restart_backend_sidecar(app: AppHandle, state: State<'_, BackendManager>) -> BackendRuntimeInfo {
    let port = {
        let inner = state.inner.lock().expect("backend lock poisoned");
        inner.info.port
    };
    state.shutdown();
    // Wait for the OS to actually release the port before re-spawning.
    // A fixed sleep is unreliable on slow hardware — poll until the port
    // stops responding (or bail after 5 seconds).
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if !port_responding(port) {
            break;
        }
        thread::sleep(Duration::from_millis(150));
    }
    state.bootstrap(&app);
    state.runtime_info()
}

#[derive(serde::Serialize)]
struct RebuildLlamaCppResult {
    ok: bool,
    exit_code: Option<i32>,
    output: String,
}

#[tauri::command]
fn rebuild_llama_cpp() -> RebuildLlamaCppResult {
    // Run scripts/update-llama-cpp.sh from the workspace root and return
    // stdout+stderr. Blocking: the frontend shows a "rebuilding..." state
    // until this returns. Typical wall time is 2-5 min.
    //
    // IMPORTANT: we spawn the build in its OWN process group via setsid()
    // so that if the user later kills the Tauri dev session (or the
    // ChaosEngineAI process dies), the running cmake build is NOT killed
    // by SIGTERM propagation. Tauri child processes normally inherit the
    // parent process group, and `./scripts/kill-dev.sh` / Ctrl-C would
    // tear the build down mid-compile and leave the binary outdated.
    let workspace_root = source_workspace_root();
    let script = workspace_root.join("scripts").join("update-llama-cpp.sh");
    if !script.exists() {
        return RebuildLlamaCppResult {
            ok: false,
            exit_code: None,
            output: format!("update script not found at {}", script.display()),
        };
    }
    let mut command = Command::new("bash");
    command
        .arg(script.as_os_str())
        .current_dir(&workspace_root);
    #[cfg(unix)]
    unsafe {
        command.pre_exec(|| {
            libc::setsid();
            Ok(())
        });
    }
    let output = command.output();
    match output {
        Ok(out) => {
            let mut combined = String::new();
            combined.push_str(&String::from_utf8_lossy(&out.stdout));
            if !out.stderr.is_empty() {
                combined.push_str("\n--- stderr ---\n");
                combined.push_str(&String::from_utf8_lossy(&out.stderr));
            }
            RebuildLlamaCppResult {
                ok: out.status.success(),
                exit_code: out.status.code(),
                output: combined,
            }
        }
        Err(err) => RebuildLlamaCppResult {
            ok: false,
            exit_code: None,
            output: format!("failed to spawn update script: {err}"),
        },
    }
}

impl BackendManager {
    fn bootstrap(&self, app: &AppHandle) {
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
            inner.info.startup_error = port_warning;
            port = inner.info.port;

            let workspace_root = if let Some(runtime) = embedded_runtime.as_ref() {
                runtime.backend_root.clone()
            } else {
                match resolve_workspace_root(app) {
                    Some(path) => path,
                    None => {
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
                        let _ = windows_job::assign_to_kill_on_close_job(&child);
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
                    inner.info.startup_error = Some(format!("Failed to start the backend sidecar: {error}"));
                    return;
                }
            }

            log_path = log_candidate;
        }

        let started = wait_for_port(port, BACKEND_START_TIMEOUT, BACKEND_POLL_INTERVAL);

        let mut inner = self.inner.lock().expect("backend lock poisoned");
        inner.info.started = started;
        if started {
            inner.info.api_token = fetch_backend_api_token(port);
            return;
        }

        let detail = read_log_tail(&log_path);
        inner.info.startup_error = Some(if detail.is_empty() {
            "The backend sidecar did not become ready in time.".to_string()
        } else {
            format!("The backend sidecar did not become ready in time. {detail}")
        });
    }

    fn runtime_info(&self) -> BackendRuntimeInfo {
        let mut inner = self.inner.lock().expect("backend lock poisoned");
        let log_path = inner.info.log_path.clone().map(PathBuf::from);

        if let Some(child) = inner.child.as_mut() {
            match child.try_wait() {
                Ok(Some(status)) => {
                    inner.info.process_running = false;
                    inner.info.started = false;
                    if inner.info.startup_error.is_none() {
                        let tail = log_path
                            .as_ref()
                            .map(|path| read_log_tail(path))
                            .unwrap_or_default();
                        inner.info.startup_error = Some(if tail.is_empty() {
                            format!("The backend sidecar exited with status {status}.")
                        } else {
                            format!("The backend sidecar exited with status {status}. {tail}")
                        });
                    }
                }
                Ok(None) => {
                    inner.info.process_running = true;
                    inner.info.started = port_responding(inner.info.port);
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

    fn shutdown(&self) {
        let mut inner = self.inner.lock().expect("backend lock poisoned");
        let attached_backend = if inner.child.is_none() && inner.info.managed_by_tauri && inner.info.started {
            Some((inner.info.port, inner.info.api_token.clone()))
        } else {
            None
        };
        inner.info.process_running = false;
        inner.info.started = false;
        inner.info.startup_error = None;
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

fn apply_embedded_runtime_env(command: &mut Command, runtime: &EmbeddedRuntime) {
    command
        .env("PYTHONHOME", runtime.python_home.as_os_str())
        .env("PYTHONNOUSERSITE", "1")
        .env("CHAOSENGINE_EMBEDDED_RUNTIME", "1");

    // Insert the user-local extras dir after the app backend but before
    // bundled site-packages. Runtime-installed packages (CUDA torch,
    // diffusers, etc.) still shadow bundled third-party wheels, while
    // app-owned adapter modules in backend/ keep priority over same-named
    // upstream packages installed into extras.
    //
    // The extras dir lives outside the ephemeral %TEMP% runtime extraction
    // so it survives app updates — the installer re-extracts the bundled
    // runtime from scratch on each launch, but never touches the extras tree.
    // (CHAOSENGINE_EXTRAS_SITE_PACKAGES is already set by the caller so
    // the backend can target it for pip --target installs.)
    let extras_dir = chaosengine_extras_site_packages_for_python(
        &runtime.python_binary,
        runtime.python_version.as_deref(),
    )
    .filter(|path| path.is_dir());
    let mut python_path_entries: Vec<PathBuf> = Vec::with_capacity(runtime.python_path.len() + 1);
    if let Some(first) = runtime.python_path.first() {
        python_path_entries.push(first.clone());
    }
    if let Some(extras) = extras_dir.as_ref() {
        python_path_entries.push(extras.clone());
    }
    python_path_entries.extend(runtime.python_path.iter().skip(1).cloned());
    if let Some(python_path) = join_paths(&python_path_entries) {
        command.env("PYTHONPATH", python_path);
    }
    if let Some(path_value) = prepend_env_paths("PATH", &runtime.path_entries) {
        command.env("PATH", path_value);
    }

    apply_library_path(command, "DYLD_LIBRARY_PATH", &runtime.library_path_entries);
    apply_library_path(
        command,
        "DYLD_FALLBACK_LIBRARY_PATH",
        &runtime.library_path_entries,
    );
    apply_library_path(command, "LD_LIBRARY_PATH", &runtime.library_path_entries);

    if let Some(cert_bundle) = resolve_cert_bundle(runtime) {
        command.env("SSL_CERT_FILE", cert_bundle.as_os_str());
    }
}

/// Persistent user-local site-packages directory. Survives app updates,
/// so CUDA torch / diffusers installed once stays installed forever.
///
/// Path is namespaced by Python ``major.minor`` (``cp312``, ``cp311``)
/// because compiled C-extensions are ABI-incompatible across Python
/// versions. A pydantic_core wheel built for cp311 will fail to import
/// on cp312 and stall app launch — see the rc.4 boot crash that drove
/// this scheme.
///
/// - Windows: ``%LOCALAPPDATA%\ChaosEngineAI\extras\cp{tag}\site-packages``
/// - macOS:   ``~/Library/Application Support/ChaosEngineAI/extras/cp{tag}/site-packages``
/// - Linux:   ``$XDG_DATA_HOME/ChaosEngineAI/extras/cp{tag}/site-packages``
///            (fallback ``~/.local/share/...``)
///
/// Returns ``None`` if we can't resolve a home directory at all (headless
/// environments). Callers treat that as "no extras available".
fn chaosengine_extras_root() -> Option<PathBuf> {
    // The extras tree lives OUTSIDE the Tauri install directory so it
    // survives uninstall + reinstall cycles — re-downloading the 2.5 GB
    // GPU bundle on every desktop upgrade is unacceptable. The Windows
    // NSIS installer is told to leave this path alone via the empty
    // hooks in ``src-tauri/installer.nsh``; if anyone changes either
    // side the other MUST be kept in sync.
    let base = if cfg!(windows) {
        env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .or_else(|| env::var_os("APPDATA").map(PathBuf::from))
    } else if cfg!(target_os = "macos") {
        env::var_os("HOME")
            .map(|home| PathBuf::from(home).join("Library").join("Application Support"))
    } else {
        env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".local").join("share")))
    }?;
    Some(base.join("ChaosEngineAI").join("extras"))
}

fn python_version_tag(raw: &str) -> Option<String> {
    // Accept "3.12", "3.12.7", "cpython-3.12.7+...", etc. Extract major.minor.
    let mut parts = raw.split(|c: char| !c.is_ascii_digit() && c != '.');
    let candidate = parts.find(|chunk| chunk.contains('.'))?;
    let mut iter = candidate.split('.');
    let major = iter.next()?.parse::<u32>().ok()?;
    let minor = iter.next()?.parse::<u32>().ok()?;
    Some(format!("cp{major}{minor}"))
}

fn detect_python_version_tag(python: &Path) -> Option<String> {
    let output = Command::new(python)
        .args([
            "-c",
            "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let raw = String::from_utf8(output.stdout).ok()?;
    python_version_tag(raw.trim())
}

fn chaosengine_extras_site_packages_for(tag: &str) -> Option<PathBuf> {
    Some(chaosengine_extras_root()?.join(tag).join("site-packages"))
}

fn chaosengine_extras_site_packages_for_python(python: &Path, hint: Option<&str>) -> Option<PathBuf> {
    let tag = hint
        .and_then(python_version_tag)
        .or_else(|| detect_python_version_tag(python))?;
    chaosengine_extras_site_packages_for(&tag)
}

fn ensure_extras_site_packages_for_python(python: &Path, hint: Option<&str>) -> Option<PathBuf> {
    let path = chaosengine_extras_site_packages_for_python(python, hint)?;
    match fs::create_dir_all(&path) {
        Ok(_) => Some(path),
        Err(error) => {
            debug_embedded(format!(
                "failed to create extras dir {}: {error}",
                path.display(),
            ));
            None
        }
    }
}

fn resolve_cert_bundle(runtime: &EmbeddedRuntime) -> Option<PathBuf> {
    runtime
        .python_path
        .iter()
        .map(|base| base.join("certifi").join("cacert.pem"))
        .find(|path| path.exists())
}

fn apply_library_path(command: &mut Command, variable: &str, entries: &[PathBuf]) {
    if let Some(value) = prepend_env_paths(variable, entries) {
        command.env(variable, value);
    }
}

fn join_paths(entries: &[PathBuf]) -> Option<OsString> {
    if entries.is_empty() {
        return None;
    }
    env::join_paths(entries).ok()
}

fn prepend_env_paths(variable: &str, entries: &[PathBuf]) -> Option<OsString> {
    if entries.is_empty() {
        return env::var_os(variable);
    }
    let mut combined = entries.to_vec();
    if let Some(existing) = env::var_os(variable) {
        combined.extend(env::split_paths(&existing));
    }
    env::join_paths(combined).ok()
}

fn source_workspace_root() -> PathBuf {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    root.canonicalize().unwrap_or(root)
}

fn current_platform_tag() -> String {
    let platform = match env::consts::OS {
        "macos" => "darwin",
        other => other,
    };
    format!("{platform}-{}", env::consts::ARCH)
}

/// Short fingerprint of the manifest content used as an extraction-dir
/// suffix. DefaultHasher is not cryptographic and not stable across Rust
/// versions, but we only need within-process stability: same input →
/// same dir name for this running binary.
fn manifest_fingerprint(manifest_payload: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    manifest_payload.hash(&mut hasher);
    // 8 hex chars is ~4B values — plenty for the handful of manifest
    // revisions a single install ever sees, and short enough to keep
    // MAX_PATH headroom on Windows for deep torch/lib paths.
    format!("{:08x}", hasher.finish() as u32)
}

/// Best-effort cleanup of the pre-0.6.2 unsuffixed extraction path
/// (``chaosengine-embedded-runtime/<platform>/``). The new layout uses a
/// manifest-hash suffix, so the old path is unambiguously stale after
/// a 0.6.2+ install. Ignoring rmtree failures is fine — TEMP gets
/// cleaned by the OS periodically, and leaving a dead directory in
/// place doesn't affect correctness.
fn cleanup_legacy_extraction_root() {
    let legacy = env::temp_dir()
        .join("chaosengine-embedded-runtime")
        .join(current_platform_tag());
    if legacy.exists() {
        let _ = fs::remove_dir_all(&legacy);
    }
}

fn embedded_debug_enabled() -> bool {
    env::var_os("CHAOSENGINE_DEBUG_EMBEDDED").is_some()
}

fn debug_embedded(message: impl AsRef<str>) {
    if embedded_debug_enabled() {
        eprintln!("[embedded-runtime] {}", message.as_ref());
    }
}

fn embedded_resource_roots(app: &AppHandle) -> Vec<PathBuf> {
    let mut roots = Vec::new();

    if let Ok(resource_dir) = app.path().resource_dir() {
        roots.push(resource_dir.join("embedded"));
    }

    let dev_resources = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("resources")
        .join("embedded");
    if !roots.iter().any(|candidate| candidate == &dev_resources) {
        roots.push(dev_resources);
    }

    roots
}

fn resolve_embedded_runtime(app: &AppHandle) -> Option<EmbeddedRuntime> {
    let tag = current_platform_tag();
    let candidates = embedded_resource_roots(app);
    for candidate in &candidates {
        debug_embedded(format!("checking resource root {}", candidate.display()));
    }
    let (manifest_path, archive_path) = candidates
        .into_iter()
        .map(|root| {
            (
                root.join(format!("runtime-{tag}.manifest.json")),
                root.join(format!("runtime-{tag}.tar.gz")),
            )
        })
        .find(|(manifest_path, archive_path)| manifest_path.exists() && archive_path.exists())?;
    debug_embedded(format!(
        "using manifest {} and archive {}",
        manifest_path.display(),
        archive_path.display()
    ));

    let manifest_payload = match fs::read_to_string(&manifest_path) {
        Ok(payload) => payload,
        Err(error) => {
            debug_embedded(format!(
                "failed to read manifest {}: {error}",
                manifest_path.display()
            ));
            return None;
        }
    };
    let manifest: EmbeddedRuntimeManifest = match serde_json::from_str(&manifest_payload) {
        Ok(parsed) => parsed,
        Err(error) => {
            debug_embedded(format!(
                "failed to parse manifest {}: {error}",
                manifest_path.display()
            ));
            return None;
        }
    };
    if manifest.mode.as_deref() == Some("development") {
        let source_root = source_workspace_root();
        if source_root.join("backend_service").join("app.py").exists() {
            debug_embedded(format!(
                "development embedded runtime detected; preferring source workspace {}",
                source_root.display()
            ));
            return None;
        }
    }
    let extracted_root = match ensure_embedded_runtime_extracted(app, &archive_path, &manifest_path) {
        Ok(path) => path,
        Err(error) => {
            debug_embedded(error);
            return None;
        }
    };
    debug_embedded(format!("extracted runtime to {}", extracted_root.display()));

    let runtime = EmbeddedRuntime {
        backend_root: extracted_root.join(&manifest.backend_root),
        python_binary: extracted_root.join(&manifest.python_binary),
        python_home: extracted_root.join(&manifest.python_home),
        python_path: manifest
            .python_path
            .iter()
            .map(|entry| extracted_root.join(entry))
            .collect(),
        library_path_entries: manifest
            .library_path_entries
            .iter()
            .map(|entry| extracted_root.join(entry))
            .collect(),
        path_entries: manifest
            .path_entries
            .iter()
            .map(|entry| extracted_root.join(entry))
            .collect(),
        llama_server: manifest
            .llama_server
            .as_ref()
            .map(|entry| extracted_root.join(entry)),
        llama_server_turbo: manifest
            .llama_server_turbo
            .as_ref()
            .map(|entry| extracted_root.join(entry)),
        llama_cli: manifest.llama_cli.as_ref().map(|entry| extracted_root.join(entry)),
        sd_cpp: manifest.sd_cpp.as_ref().map(|entry| extracted_root.join(entry)),
        python_version: manifest.python_version.clone(),
    };

    if runtime.backend_root.exists() && runtime.python_binary.exists() && runtime.python_home.exists() {
        debug_embedded("embedded runtime passed file existence checks");
        Some(runtime)
    } else {
        debug_embedded("embedded runtime failed file existence checks");
        None
    }
}

fn ensure_embedded_runtime_extracted(
    _app: &AppHandle,
    archive_path: &Path,
    manifest_path: &Path,
) -> Result<PathBuf, String> {
    let manifest_payload = fs::read_to_string(manifest_path)
        .map_err(|error| format!("failed to read manifest {}: {error}", manifest_path.display()))?;

    // Key the extraction directory by a hash of the manifest content.
    // Why: the old code used a fixed path per platform and rmtree'd it
    // on manifest change, which was silently failing on Windows when
    // torch/lib/*.dll or llama-server.exe was still held open by a
    // prior session (or Windows Defender's lingering scan lock). The
    // rmtree swallowed the error, unpack then wrote into a dirty dir,
    // and users ended up with an empty backend/ + a fresh manifest.json
    // claiming the extraction was current. Observed on the user's box:
    // backend/ directory present but empty, backend_service missing.
    //
    // By hashing the manifest, each unique build lands in its own
    // directory. No path is ever overwritten — fresh extraction is
    // always to a fresh dir. Old directories become orphans in %TEMP%
    // but Windows / macOS clean TEMP periodically, and the legacy
    // cleanup below handles the unsuffixed dir from earlier versions.
    //
    // Hash is u64 via DefaultHasher, first 8 hex chars. That's 4B
    // possible values — collision chance for the handful of manifest
    // versions a user accumulates is negligible. Short keeps Windows
    // MAX_PATH headroom for deep ``site-packages`` paths.
    let fingerprint = manifest_fingerprint(&manifest_payload);
    let extraction_root = env::temp_dir()
        .join("chaosengine-embedded-runtime")
        .join(format!("{}-{}", current_platform_tag(), fingerprint));
    let extracted_manifest = extraction_root.join("manifest.json");

    if extracted_manifest.exists()
        && fs::read_to_string(&extracted_manifest)
            .ok()
            .as_deref()
            == Some(manifest_payload.as_str())
    {
        return Ok(extraction_root);
    }

    // Defence in depth: if this specific fingerprint dir somehow exists
    // in a partial state (e.g. prior unpack failed), nuke it. With
    // manifest-hash keying this should only happen when we crashed
    // mid-unpack, which is far rarer than the old rmtree-race case.
    if extraction_root.exists() {
        fs::remove_dir_all(&extraction_root).map_err(|error| {
            format!(
                "failed to clear partial extraction {}: {error}. \
                 Close ChaosEngineAI fully and try again, or delete \
                 the directory manually.",
                extraction_root.display(),
            )
        })?;
    }
    fs::create_dir_all(&extraction_root).map_err(|error| {
        format!(
            "failed to create extraction root {}: {error}",
            extraction_root.display()
        )
    })?;

    let archive_file = fs::File::open(archive_path)
        .map_err(|error| format!("failed to open archive {}: {error}", archive_path.display()))?;
    let archive_reader = flate2::read::GzDecoder::new(archive_file);
    let mut archive = Archive::new(archive_reader);
    archive.set_unpack_xattrs(false);
    archive.set_preserve_permissions(false);
    archive.set_preserve_ownerships(false);
    archive.set_preserve_mtime(false);
    if let Err(error) = archive.unpack(&extraction_root) {
        let _ = fs::remove_dir_all(&extraction_root);
        return Err(format!(
            "failed to unpack archive {} into {}: {error}",
            archive_path.display(),
            extraction_root.display()
        ));
    }

    if !extracted_manifest.exists() {
        fs::write(&extracted_manifest, &manifest_payload).map_err(|error| {
            format!(
                "failed to write extracted manifest {}: {error}",
                extracted_manifest.display()
            )
        })?;
    }

    Ok(extraction_root)
}

fn legacy_resource_python_root(app: &AppHandle) -> Option<PathBuf> {
    app.path()
        .resource_dir()
        .ok()
        .map(|path| path.join("python"))
        .filter(|path| path.join("backend_service").join("app.py").exists())
}

fn resolve_workspace_root(app: &AppHandle) -> Option<PathBuf> {
    if let Some(value) = env::var_os("CHAOSENGINE_BACKEND_ROOT") {
        let path = PathBuf::from(value);
        if path.join("backend_service").join("app.py").exists() {
            return Some(path.canonicalize().unwrap_or(path));
        }
    }

    if let Some(resource_path) = legacy_resource_python_root(app) {
        return Some(resource_path);
    }

    let source_root = source_workspace_root();
    if source_root.join("backend_service").join("app.py").exists() {
        return Some(source_root);
    }

    None
}

fn resolve_python_executable(workspace_root: &Path) -> Option<PathBuf> {
    if let Some(value) = env::var_os("CHAOSENGINE_MLX_PYTHON") {
        if let Some(path) = resolve_candidate(value) {
            return Some(path);
        }
    }

    let candidates = vec![
        // Windows
        workspace_root.join(".venv").join("Scripts").join("python.exe"),
        workspace_root.join("Scripts").join("python.exe"),
        // Unix
        workspace_root.join(".venv").join("bin").join("python"),
        workspace_root.join(".venv").join("bin").join("python3"),
        workspace_root.join("bin").join("python3"),
        workspace_root.join("bin").join("python"),
    ];

    for candidate in candidates {
        if candidate.exists() {
            return Some(candidate);
        }
    }

    find_in_path(&["python3", "python"])
}

fn resolve_llama_server(_workspace_root: &Path) -> Option<PathBuf> {
    if let Some(value) = env::var_os("CHAOSENGINE_LLAMA_SERVER") {
        if let Some(path) = resolve_candidate(value) {
            return Some(path);
        }
    }

    find_in_path(&["llama-server"])
}

fn resolve_llama_server_turbo(_workspace_root: &Path) -> Option<PathBuf> {
    if let Some(value) = env::var_os("CHAOSENGINE_LLAMA_SERVER_TURBO") {
        if let Some(path) = resolve_candidate(value) {
            return Some(path);
        }
    }

    // Check ~/.chaosengine/bin/ first (ChaosEngineAI-managed installs),
    // then fall back to PATH.
    if let Ok(home) = env::var("HOME") {
        let managed = PathBuf::from(home).join(".chaosengine").join("bin").join("llama-server-turbo");
        if managed.exists() {
            return Some(managed);
        }
    }

    find_in_path(&["llama-server-turbo"])
}

fn resolve_llama_cli(_workspace_root: &Path) -> Option<PathBuf> {
    if let Some(value) = env::var_os("CHAOSENGINE_LLAMA_CLI") {
        if let Some(path) = resolve_candidate(value) {
            return Some(path);
        }
    }

    find_in_path(&["llama-cli"])
}

fn resolve_sd_cpp(_workspace_root: &Path) -> Option<PathBuf> {
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

fn resolve_candidate(value: impl Into<PathBuf>) -> Option<PathBuf> {
    let candidate = value.into();
    if candidate.exists() {
        return Some(candidate);
    }
    if candidate.components().count() == 1 {
        return find_in_path(&[candidate.to_string_lossy().as_ref()]);
    }
    None
}

fn find_in_path(names: &[&str]) -> Option<PathBuf> {
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

fn open_log_file(path: &Path) -> Option<std::fs::File> {
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .ok()
}

fn read_log_tail(path: &Path) -> String {
    let payload = fs::read_to_string(path).unwrap_or_default();
    let mut lines = payload.lines().rev().take(30).collect::<Vec<_>>();
    lines.reverse();
    lines.join("\n")
}


// Orphan-process cleanup at app launch — see orphans.rs.
mod orphans;
use orphans::cleanup_orphaned_backends;


// Windows-only Job Object orphan prevention.
// See windows_job.rs for the full explanation + safety notes.
#[cfg(windows)]
mod windows_job;


#[cfg(unix)]
static SIGNAL_RECEIVED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[tauri::command]
async fn pick_directory(app: AppHandle) -> Option<String> {
    use tauri_plugin_dialog::DialogExt;
    let (tx, rx) = std::sync::mpsc::channel();
    app.dialog()
        .file()
        .set_title("Choose data directory")
        .pick_folder(move |path| {
            let result = path.and_then(|p| p.into_path().ok().map(|pb| pb.to_string_lossy().into_owned()));
            let _ = tx.send(result);
        });
    rx.recv().ok().flatten()
}

pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_process::init())
        .setup(|app| {
            // Register the manager in its default (not-yet-started) state
            // immediately so the frontend can query runtime_info without
            // racing the bootstrap. The frontend already renders a
            // "Connecting..." state when started=false.
            app.manage(BackendManager::default());

            // The Python sidecar serves /api/* — without it the desktop
            // app is dead. Always bootstrap on launch, regardless of the
            // `autoStartServer` setting (which controls whether the
            // OpenAI inference server *inside* the backend auto-starts,
            // not whether the backend process itself runs).
            //
            // Bootstrap runs on a background thread so the Tauri event
            // loop stays responsive — otherwise the window paints but
            // can't service events for up to ~15s (runtime extraction +
            // Python startup + port-wait), which macOS surfaces as the
            // spinning beachball and looks like a crash.
            let bootstrap_handle = app.handle().clone();
            thread::spawn(move || {
                let state = bootstrap_handle.state::<BackendManager>();
                state.bootstrap(&bootstrap_handle);
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            app_version,
            backend_runtime_info,
            stop_backend_sidecar,
            restart_backend_sidecar,
            rebuild_llama_cpp,
            pick_directory
        ])
        .build(tauri::generate_context!())
        .expect("error while running ChaosEngineAI");

    // Translate Unix terminal signals (Ctrl-C in `npm run tauri:dev`, SIGTERM
    // from a parent process, etc.) into a clean Tauri exit so the
    // RunEvent::Exit handler below fires and tears down the Python sidecar
    // process group. Without this the Tauri event loop dies on signal and
    // the spawned backend gets orphaned.
    #[cfg(unix)]
    {
        let signal_handle = app.handle().clone();
        std::thread::spawn(move || {
            extern "C" fn forward(_sig: libc::c_int) {
                // Re-raise default disposition is unsafe inside this handler;
                // we just set a flag via a self-pipe-less approach: write to
                // a global AtomicBool that the polling thread reads.
                SIGNAL_RECEIVED.store(true, std::sync::atomic::Ordering::SeqCst);
            }
            unsafe {
                libc::signal(libc::SIGINT, forward as libc::sighandler_t);
                libc::signal(libc::SIGTERM, forward as libc::sighandler_t);
                libc::signal(libc::SIGHUP, forward as libc::sighandler_t);
            }
            loop {
                std::thread::sleep(std::time::Duration::from_millis(150));
                if SIGNAL_RECEIVED.load(std::sync::atomic::Ordering::SeqCst) {
                    signal_handle.exit(0);
                    break;
                }
            }
        });
    }

    app.run(|app_handle, event| {
        match event {
            tauri::RunEvent::WindowEvent {
                event: tauri::WindowEvent::CloseRequested { api, .. },
                ..
            } => {
                api.prevent_close();
                app_handle.exit(0);
            }
            tauri::RunEvent::Exit => {
                app_handle.state::<BackendManager>().shutdown();
            }
            _ => {}
        }
    });
}
