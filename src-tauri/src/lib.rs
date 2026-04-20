use serde::{Deserialize, Serialize};
use std::{
    env,
    ffi::OsString,
    fs,
    fs::OpenOptions,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
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
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SavedDesktopSettings {
    preferred_server_port: Option<u16>,
    allow_remote_connections: Option<bool>,
    // `auto_start_server` was previously read here to gate Python sidecar
    // bootstrap. The sidecar now always starts (it's required for /api/*),
    // and that toggle only controls the inference engine inside the backend.
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
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct ManagedBackendLease {
    pid: u32,
    port: u16,
}

#[derive(Default)]
struct ExistingBackendProbe {
    workspace_root: Option<String>,
    python_executable: Option<String>,
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
            } else {
                if let Some(llama_server) = resolve_llama_server(&workspace_root) {
                    command.env("CHAOSENGINE_LLAMA_SERVER", llama_server.as_os_str());
                }
                if let Some(llama_server_turbo) = resolve_llama_server_turbo(&workspace_root) {
                    command.env("CHAOSENGINE_LLAMA_SERVER_TURBO", llama_server_turbo.as_os_str());
                }
                if let Some(llama_cli) = resolve_llama_cli(&workspace_root) {
                    command.env("CHAOSENGINE_LLAMA_CLI", llama_cli.as_os_str());
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
            #[cfg(unix)]
            unsafe {
                command.pre_exec(|| {
                    libc::setsid();
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

        let started = wait_for_port(port, BACKEND_START_TIMEOUT);

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

    if let Some(python_path) = join_paths(&runtime.python_path) {
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
    let extraction_root = env::temp_dir()
        .join("chaosengine-embedded-runtime")
        .join(current_platform_tag());
    let extracted_manifest = extraction_root.join("manifest.json");

    if extracted_manifest.exists()
        && fs::read_to_string(&extracted_manifest)
            .ok()
            .as_deref()
            == Some(manifest_payload.as_str())
    {
        return Ok(extraction_root);
    }

    if extraction_root.exists() {
        let _ = fs::remove_dir_all(&extraction_root);
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

fn settings_path() -> Option<PathBuf> {
    let base = if cfg!(windows) {
        env::var_os("APPDATA").map(PathBuf::from)
    } else {
        env::var_os("HOME").map(PathBuf::from)
    };
    base.map(|dir| dir.join(".chaosengine").join("settings.json"))
}

fn saved_backend_port() -> Option<u16> {
    let path = settings_path()?;
    let payload = fs::read_to_string(path).ok()?;
    let settings: SavedDesktopSettings = serde_json::from_str(&payload).ok()?;
    settings
        .preferred_server_port
        .filter(|port| (1024..=65535).contains(port))
}

fn saved_allow_remote_connections() -> Option<bool> {
    let path = settings_path()?;
    let payload = fs::read_to_string(path).ok()?;
    let settings: SavedDesktopSettings = serde_json::from_str(&payload).ok()?;
    settings.allow_remote_connections
}

fn selected_bind_host(allow_remote_connections: bool) -> &'static str {
    if allow_remote_connections {
        "0.0.0.0"
    } else {
        "127.0.0.1"
    }
}

/// Try to bind the preferred port; fall back to an OS-assigned port if busy.
/// Returns `(port, warning)` — `warning` is set when the preferred port was
/// unavailable so the caller can surface it to the user.
fn select_backend_port(preferred: u16, allow_remote_connections: bool) -> (u16, Option<String>) {
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

fn port_responding(port: u16) -> bool {
    TcpStream::connect(("127.0.0.1", port)).is_ok()
}

fn wait_for_port(port: u16, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    // Phase 1: wait for TCP port to accept connections (fast check).
    while Instant::now() < deadline {
        if port_responding(port) {
            break;
        }
        thread::sleep(BACKEND_POLL_INTERVAL);
    }
    // Phase 2: wait for /api/health to return {"status": "ok"}.
    // The port may be open (uvicorn bound) before FastAPI is ready to serve.
    while Instant::now() < deadline {
        if probe_chaosengine_backend(port).is_some() {
            return true;
        }
        thread::sleep(BACKEND_POLL_INTERVAL);
    }
    false
}

fn backend_http_json(
    method: &str,
    port: u16,
    path: &str,
    api_token: Option<&str>,
) -> Option<serde_json::Value> {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).ok()?;
    let _ = stream.set_read_timeout(Some(Duration::from_millis(1200)));
    let _ = stream.set_write_timeout(Some(Duration::from_millis(1200)));
    let auth_header = api_token
        .filter(|token| !token.is_empty())
        .map(|token| format!("Authorization: Bearer {token}\r\n"))
        .unwrap_or_default();
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\nAccept: application/json\r\n{auth_header}Content-Length: 0\r\n\r\n"
    );
    stream.write_all(request.as_bytes()).ok()?;
    let mut response = String::new();
    stream.read_to_string(&mut response).ok()?;
    let (_, body) = response.split_once("\r\n\r\n")?;
    serde_json::from_str(body).ok()
}

fn probe_chaosengine_backend(port: u16) -> Option<ExistingBackendProbe> {
    let payload = backend_http_json("GET", port, "/api/health", None)?;
    if payload.get("status").and_then(|value| value.as_str()) != Some("ok") {
        return None;
    }
    Some(ExistingBackendProbe {
        workspace_root: payload
            .get("workspaceRoot")
            .and_then(|value| value.as_str())
            .map(|value| value.to_string()),
        python_executable: payload
            .get("nativeBackends")
            .and_then(|value| value.get("pythonExecutable"))
            .and_then(|value| value.as_str())
            .map(|value| value.to_string()),
    })
}

fn fetch_backend_api_token(port: u16) -> Option<String> {
    backend_http_json("GET", port, "/api/auth/session", None)?
        .get("apiToken")
        .and_then(|value| value.as_str())
        .map(|value| value.to_string())
}

fn request_backend_shutdown(port: u16, api_token: Option<&str>) -> bool {
    let _ = backend_http_json("POST", port, "/api/server/shutdown", api_token);
    let deadline = Instant::now() + Duration::from_secs(3);
    while Instant::now() < deadline {
        if !port_responding(port) {
            return true;
        }
        thread::sleep(Duration::from_millis(150));
    }
    !port_responding(port)
}

fn managed_backend_lease_path(app: &AppHandle) -> Option<PathBuf> {
    app.path().app_data_dir().ok().map(|path| path.join("managed-backend.json"))
}

fn write_managed_backend_lease(app: &AppHandle, lease: &ManagedBackendLease) {
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

fn read_managed_backend_lease(app: &AppHandle) -> Option<ManagedBackendLease> {
    let path = managed_backend_lease_path(app)?;
    let payload = fs::read(path).ok()?;
    serde_json::from_slice(&payload).ok()
}

fn clear_managed_backend_lease(app: &AppHandle) {
    if let Some(path) = managed_backend_lease_path(app) {
        let _ = fs::remove_file(path);
    }
}

fn cleanup_stale_managed_backend(app: &AppHandle) {
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

#[cfg(unix)]
fn cleanup_orphaned_backends() {
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
        if ppid != 1 || !command.contains("backend_service.app") {
            continue;
        }
        terminate_process_group(pid);
    }
}

#[cfg(windows)]
fn cleanup_orphaned_backends() {
    // Use WMIC to find orphaned backend_service.app processes whose parent
    // no longer exists.  On Windows, unlike Unix, orphaned children keep
    // their original PPID — so we check whether the parent PID is still
    // alive via tasklist.
    let output = match Command::new("wmic")
        .args(["process", "where", "commandline like '%backend_service.app%'", "get", "processid,parentprocessid", "/format:csv"])
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
        // Check if parent is still running
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
fn cleanup_orphaned_backends() {}

#[cfg(unix)]
fn terminate_process_group(pid: i32) {
    unsafe {
        libc::killpg(pid, libc::SIGTERM);
    }
    thread::sleep(Duration::from_millis(300));
    unsafe {
        libc::killpg(pid, libc::SIGKILL);
    }
}

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
