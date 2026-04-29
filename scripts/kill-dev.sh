#!/usr/bin/env bash
# Kill any stragglers from a previous `npm run tauri:dev` session.
# Use this if Cmd-Q'ing the app left vite, the python sidecar, or an mlx
# worker subprocess running in the background.
#
# Usage:  ./scripts/kill-dev.sh

set -u

kill_match() {
  local label="$1"
  local pattern="$2"
  local pids
  pids=$(pgrep -f "$pattern" || true)
  if [[ -z "$pids" ]]; then
    echo "  $label: none"
    return
  fi
  echo "  $label: killing $(echo "$pids" | tr '\n' ' ')"
  # SIGTERM first, then SIGKILL after a short grace period
  echo "$pids" | xargs -r kill 2>/dev/null || true
  sleep 0.5
  pids=$(pgrep -f "$pattern" || true)
  if [[ -n "$pids" ]]; then
    echo "$pids" | xargs -r kill -9 2>/dev/null || true
  fi
}

echo "Stopping ChaosEngineAI dev stragglers..."
kill_match "ChaosEngineAI binary"  "target/debug/ChaosEngineAI"
kill_match "tauri dev"             "node .*tauri.*dev"
kill_match "vite"                  "node .*vite"
kill_match "python backend"        "backend_service.app"
kill_match "mlx_worker"            "backend_service.mlx_worker"
# llama-server orphans — only target ChaosEngineAI-spawned copies, leave
# instances spawned by other tools (e.g. system-installed binaries) alone.
kill_match "llama-server (chaos embedded)" "chaosengine-embedded-runtime.*llama-server"
kill_match "llama-server (local build)"    "llama.cpp/build/bin/llama-server"

# Free the dev port if anything is still squatting on it
for port in 5174 5173; do
  pid=$(lsof -nP -iTCP:$port -sTCP:LISTEN -t 2>/dev/null || true)
  if [[ -n "$pid" ]]; then
    echo "  port $port held by pid $pid → killing"
    kill -9 "$pid" 2>/dev/null || true
  fi
done

echo "Done."
