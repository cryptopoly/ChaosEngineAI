use serde::Serialize;
use std::{
    env, fs,
    fs::OpenOptions,
    path::Path,
    process::{Child, Command},
    sync::Mutex,
    thread,
    time::{Duration, Instant},
};
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use tauri::{AppHandle, Manager, State};

pub(crate) const DEFAULT_BACKEND_PORT: u16 = 8876;
pub(crate) const BACKEND_START_TIMEOUT: Duration = Duration::from_secs(12);
pub(crate) const BACKEND_POLL_INTERVAL: Duration = Duration::from_millis(200);

mod settings;
mod binaries;
mod env_setup;
mod runtime;
use runtime::source_workspace_root;
mod probe;
use probe::port_responding;
mod lease;

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

// BackendManager impl — sidecar lifecycle (bootstrap / shutdown / probe).
mod backend;

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
