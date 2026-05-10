//! Embedded Python runtime extraction + env-setup helpers.
//!
//! Owns the EmbeddedRuntimeManifest + EmbeddedRuntime structs that
//! describe the bundled Python sidecar, plus all the helpers that
//! locate the bundled tar, extract it on first launch, namespace the
//! persistent user-local extras dir by Python ABI tag, and apply the
//! resulting library/path/PYTHONPATH env vars to the spawned Command.
//!
//! Extracted from `src-tauri/src/lib.rs` as part of the v0.8.0
//! Phase 3-3 refactor.

use serde::Deserialize;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tauri::{AppHandle, Manager};
use tar::Archive;

use crate::binaries::{find_in_path, resolve_candidate};
use crate::env_setup::{apply_library_path, join_paths, prepend_env_paths};


#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbeddedRuntimeManifest {
    pub mode: Option<String>,
    pub backend_root: String,
    pub python_binary: String,
    pub python_home: String,
    pub python_path: Vec<String>,
    pub library_path_entries: Vec<String>,
    pub path_entries: Vec<String>,
    pub llama_server: Option<String>,
    pub llama_server_turbo: Option<String>,
    pub llama_cli: Option<String>,
    pub sd_cpp: Option<String>,
    // ``"3.12"`` etc. — used to namespace the persistent extras dir so
    // wheels compiled for Python X.Y don't get loaded by a different X.Z.
    pub python_version: Option<String>,
}

#[derive(Clone)]
pub struct EmbeddedRuntime {
    pub backend_root: PathBuf,
    pub python_binary: PathBuf,
    pub python_home: PathBuf,
    pub python_path: Vec<PathBuf>,
    pub library_path_entries: Vec<PathBuf>,
    pub path_entries: Vec<PathBuf>,
    pub llama_server: Option<PathBuf>,
    pub llama_server_turbo: Option<PathBuf>,
    pub llama_cli: Option<PathBuf>,
    pub sd_cpp: Option<PathBuf>,
    pub python_version: Option<String>,
}

pub fn apply_embedded_runtime_env(command: &mut Command, runtime: &EmbeddedRuntime) {
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
pub fn chaosengine_extras_root() -> Option<PathBuf> {
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

pub fn python_version_tag(raw: &str) -> Option<String> {
    // Accept "3.12", "3.12.7", "cpython-3.12.7+...", etc. Extract major.minor.
    let mut parts = raw.split(|c: char| !c.is_ascii_digit() && c != '.');
    let candidate = parts.find(|chunk| chunk.contains('.'))?;
    let mut iter = candidate.split('.');
    let major = iter.next()?.parse::<u32>().ok()?;
    let minor = iter.next()?.parse::<u32>().ok()?;
    Some(format!("cp{major}{minor}"))
}

pub fn detect_python_version_tag(python: &Path) -> Option<String> {
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

pub fn chaosengine_extras_site_packages_for(tag: &str) -> Option<PathBuf> {
    Some(chaosengine_extras_root()?.join(tag).join("site-packages"))
}

pub fn chaosengine_extras_site_packages_for_python(python: &Path, hint: Option<&str>) -> Option<PathBuf> {
    let tag = hint
        .and_then(python_version_tag)
        .or_else(|| detect_python_version_tag(python))?;
    chaosengine_extras_site_packages_for(&tag)
}

pub fn ensure_extras_site_packages_for_python(python: &Path, hint: Option<&str>) -> Option<PathBuf> {
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

pub fn resolve_cert_bundle(runtime: &EmbeddedRuntime) -> Option<PathBuf> {
    runtime
        .python_path
        .iter()
        .map(|base| base.join("certifi").join("cacert.pem"))
        .find(|path| path.exists())
}

pub fn source_workspace_root() -> PathBuf {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    root.canonicalize().unwrap_or(root)
}

pub fn current_platform_tag() -> String {
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
pub fn manifest_fingerprint(manifest_payload: &str) -> String {
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
pub fn cleanup_legacy_extraction_root() {
    let legacy = env::temp_dir()
        .join("chaosengine-embedded-runtime")
        .join(current_platform_tag());
    if legacy.exists() {
        let _ = fs::remove_dir_all(&legacy);
    }
}

pub fn embedded_debug_enabled() -> bool {
    env::var_os("CHAOSENGINE_DEBUG_EMBEDDED").is_some()
}

pub fn debug_embedded(message: impl AsRef<str>) {
    if embedded_debug_enabled() {
        eprintln!("[embedded-runtime] {}", message.as_ref());
    }
}

pub fn embedded_resource_roots(app: &AppHandle) -> Vec<PathBuf> {
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

pub fn resolve_embedded_runtime(app: &AppHandle) -> Option<EmbeddedRuntime> {
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

pub fn ensure_embedded_runtime_extracted(
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

pub fn legacy_resource_python_root(app: &AppHandle) -> Option<PathBuf> {
    app.path()
        .resource_dir()
        .ok()
        .map(|path| path.join("python"))
        .filter(|path| path.join("backend_service").join("app.py").exists())
}

pub fn resolve_workspace_root(app: &AppHandle) -> Option<PathBuf> {
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

pub fn resolve_python_executable(workspace_root: &Path) -> Option<PathBuf> {
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
