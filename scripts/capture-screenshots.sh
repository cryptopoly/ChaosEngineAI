#!/usr/bin/env bash
# Capture README screenshots of the ChaosEngineAI desktop app.
#
# This must be run interactively from a Terminal that has Screen Recording AND
# Accessibility permissions (System Settings → Privacy & Security). Sandboxed
# Claude sessions cannot grant these permissions, so this script exists for the
# human to run.
#
# Usage:  ./scripts/capture-screenshots.sh
# Make sure ChaosEngineAI is running and ideally has 1-2 models warm.

set -euo pipefail

OUT="$(cd "$(dirname "$0")/.." && pwd)/docs/screenshots"
mkdir -p "$OUT"

PAGES=(
  "Dashboard:dashboard"
  "Discover:discover"
  "My Models:my-models"
  "Server:server"
  "Chat:chat"
  "Benchmarks:benchmarks"
  "History:benchmark-history"
  "Conversion:conversion"
  "Logs:logs"
  "Settings:settings"
)

bring_to_front() {
  osascript -e 'tell application "System Events" to tell process "ChaosEngineAI" to set frontmost to true' >/dev/null
}

window_id() {
  # Try GetWindowID if installed, else fall back to Quartz via python.
  if command -v GetWindowID >/dev/null 2>&1; then
    GetWindowID ChaosEngineAI ChaosEngineAI 2>/dev/null || true
  fi
}

click_sidebar() {
  local label="$1"
  osascript <<EOF >/dev/null 2>&1 || true
tell application "System Events"
  tell process "ChaosEngineAI"
    set frontmost to true
    delay 0.2
    try
      click (first button whose name is "$label") of window 1
    on error
      try
        click (first UI element whose name is "$label") of window 1
      end try
    end try
  end tell
end tell
EOF
}

capture() {
  local out="$1"
  local wid
  wid="$(window_id)"
  if [[ -n "$wid" ]]; then
    screencapture -l "$wid" -o -x "$out"
  else
    # Fall back to interactive window capture (user clicks the window).
    echo "  (no window id — click the ChaosEngineAI window when the cursor changes)"
    screencapture -w -o -x "$out"
  fi
}

bring_to_front
echo "ChaosEngineAI: capturing 10 screenshots → $OUT"
echo

for entry in "${PAGES[@]}"; do
  label="${entry%%:*}"
  slug="${entry##*:}"
  out="$OUT/$slug.png"
  echo "→ $label  ($slug.png)"
  click_sidebar "$label"
  sleep 1.2
  capture "$out"
  if [[ -f "$out" ]]; then
    size=$(stat -f%z "$out" 2>/dev/null || echo 0)
    echo "  saved ($size bytes)"
  else
    echo "  FAILED"
  fi
  echo
done

if command -v pngquant >/dev/null 2>&1; then
  echo "Optimising PNGs with pngquant..."
  pngquant --quality=80-95 --skip-if-larger --ext .png --force "$OUT"/*.png || true
fi

echo "Done."
ls -lh "$OUT"
