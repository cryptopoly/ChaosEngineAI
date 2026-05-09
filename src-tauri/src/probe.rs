//! HTTP probe + lifecycle helpers for an external/embedded backend.
//!
//! Talks to the Python sidecar over plain TCP (no reqwest/hyper —
//! Tauri's app shell already drags in a heavy dependency tree, and
//! these endpoints are simple enough to hand-craft):
//!
//! - `port_responding` / `wait_for_port` — TCP-only liveness probe
//!   used both for spawned-backend startup and pre-spawn collision
//!   checks.
//! - `backend_http_json` — minimal HTTP/1.1 request that posts an
//!   optional Bearer token + parses the JSON body.
//! - `probe_chaosengine_backend` — query `/api/health`, return
//!   `ExistingBackendProbe` (workspace root + python executable) when
//!   the response shape matches our backend.
//! - `fetch_backend_api_token` — pull a fresh token from
//!   `/api/auth/session`.
//! - `request_backend_shutdown` — POST `/api/server/shutdown` and
//!   wait up to 3 s for the port to free.
//!
//! Backend poll interval is owned by lib.rs; we accept it as a
//! parameter rather than re-defining the constant here.
//!
//! Extracted from `lib.rs` as part of the v0.8.0 refactor.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::thread;
use std::time::{Duration, Instant};

#[derive(Default)]
pub struct ExistingBackendProbe {
    pub workspace_root: Option<String>,
    pub python_executable: Option<String>,
}

pub fn port_responding(port: u16) -> bool {
    TcpStream::connect(("127.0.0.1", port)).is_ok()
}

pub fn wait_for_port(port: u16, timeout: Duration, poll_interval: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    // Phase 1: wait for TCP port to accept connections (fast check).
    while Instant::now() < deadline {
        if port_responding(port) {
            break;
        }
        thread::sleep(poll_interval);
    }
    // Phase 2: wait for /api/health to return {"status": "ok"}.
    // The port may be open (uvicorn bound) before FastAPI is ready to serve.
    while Instant::now() < deadline {
        if probe_chaosengine_backend(port).is_some() {
            return true;
        }
        thread::sleep(poll_interval);
    }
    false
}

pub fn backend_http_json(
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

pub fn probe_chaosengine_backend(port: u16) -> Option<ExistingBackendProbe> {
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

pub fn fetch_backend_api_token(port: u16) -> Option<String> {
    backend_http_json("GET", port, "/api/auth/session", None)?
        .get("apiToken")
        .and_then(|value| value.as_str())
        .map(|value| value.to_string())
}

pub fn request_backend_shutdown(port: u16, api_token: Option<&str>) -> bool {
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
