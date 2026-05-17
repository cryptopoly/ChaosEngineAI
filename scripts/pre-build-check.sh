#!/usr/bin/env bash
# Pre-build quality gate for ChaosEngineAI.
#
# Runs all automated checks before a release or PR. Can be invoked
# manually or wired into CI.
#
# Usage:  ./scripts/pre-build-check.sh
#
# Exit code 0 = all checks passed, non-zero = at least one failed.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PASS=0
FAIL=0
WARN=0
RESULTS=()

pass() { PASS=$((PASS + 1)); RESULTS+=("  PASS  $1"); }
fail() { FAIL=$((FAIL + 1)); RESULTS+=("  FAIL  $1"); }
warn() { WARN=$((WARN + 1)); RESULTS+=("  WARN  $1"); }

echo "=== ChaosEngineAI Pre-Build Checks ==="
echo

# ------------------------------------------------------------------
# 1. Python tests
# ------------------------------------------------------------------
echo "[1/9] Python tests..."
if .venv/bin/python -m pytest tests/ -q --tb=line 2>&1 | tail -3; then
  pass "Python tests"
else
  fail "Python tests — see output above"
fi
echo

# ------------------------------------------------------------------
# 2. TypeScript tests
# ------------------------------------------------------------------
echo "[2/9] TypeScript tests..."
if npm test 2>&1 | tail -5; then
  pass "TypeScript tests"
else
  fail "TypeScript tests — see output above"
fi
echo

# ------------------------------------------------------------------
# 3. TypeScript type checking
# ------------------------------------------------------------------
echo "[3/9] TypeScript type checking..."
if npx tsc --noEmit 2>&1; then
  pass "TypeScript types"
else
  fail "TypeScript type errors — see output above"
fi
echo

# ------------------------------------------------------------------
# 4. Licence notices
# ------------------------------------------------------------------
echo "[4/9] Licence notices..."
if [[ -f "THIRD_PARTY_NOTICES.md" ]] && [[ -s "THIRD_PARTY_NOTICES.md" ]]; then
  # Check that key dependencies are mentioned
  missing=""
  # FU-030 dropped vendored ChaosEngine; FU-028 added mtplx — keep this
  # list in sync with THIRD_PARTY_NOTICES.md section headings so any new
  # runtime dep is flagged at build time if it's missing notice attribution.
  for dep in "llama.cpp" "llama-cpp-turboquant" "dflash-mlx" "turboquant" "mtplx" "mlx-video"; do
    if ! grep -qi "$dep" THIRD_PARTY_NOTICES.md; then
      missing="$missing $dep"
    fi
  done
  if [[ -z "$missing" ]]; then
    pass "THIRD_PARTY_NOTICES.md — all key deps listed"
  else
    warn "THIRD_PARTY_NOTICES.md — missing:$missing"
  fi
else
  fail "THIRD_PARTY_NOTICES.md missing or empty"
fi
echo

# ------------------------------------------------------------------
# 5. Cache strategy validation
# ------------------------------------------------------------------
echo "[5/9] Cache strategy validation..."
CACHE_CHECK=$(.venv/bin/python -c "
from cache_compression import registry
registry.discover()
valid = {'f32','f16','bf16','q8_0','q4_0','q4_1','iq4_nl','q5_0','q5_1'}
nat = registry.get('native')
for bits in (0,):
    flags = nat.llama_cpp_cache_flags(bits)
    for i, f in enumerate(flags):
        if f.startswith('--cache-type-') and i+1 < len(flags):
            if flags[i+1] not in valid:
                print(f'INVALID: Native emits {flags[i+1]}')
tq = registry.get('turboquant')
if tq.required_llama_binary() != 'turbo': print('INVALID: TurboQuant not routing to turbo')
# FU-030 (2026-05-10): legacy ids must coerce to turboquant via the
# alias map. Assert the wiring works in CI rather than at runtime.
for legacy_id in ('rotorquant', 'chaosengine'):
    if registry.resolve_legacy_id(legacy_id) != 'turboquant':
        print(f'INVALID: legacy id {legacy_id} did not coerce to turboquant')
    if registry.get(legacy_id) is None:
        print(f'INVALID: legacy id {legacy_id} did not resolve via registry.get')
print('OK')
" 2>&1)
if echo "$CACHE_CHECK" | grep -q "INVALID"; then
  fail "Cache strategy validation: $CACHE_CHECK"
else
  pass "Cache strategy validation"
fi
echo

# ------------------------------------------------------------------
# 6. Upstream dependency update check
# ------------------------------------------------------------------
echo "[6/9] Upstream dependency check..."

# Turbo fork
TURBO_VERSION_FILE="$HOME/.chaosengine/bin/llama-server-turbo.version"
if [[ -f "$TURBO_VERSION_FILE" ]]; then
  LOCAL_COMMIT=$(head -1 "$TURBO_VERSION_FILE")
  REMOTE_COMMIT=$(git ls-remote https://github.com/TheTom/llama-cpp-turboquant.git refs/heads/feature/turboquant-kv-cache 2>/dev/null | cut -f1)
  if [[ -n "$REMOTE_COMMIT" ]] && [[ "$LOCAL_COMMIT" != "$REMOTE_COMMIT" ]]; then
    warn "llama-server-turbo update available (local: ${LOCAL_COMMIT:0:12}, remote: ${REMOTE_COMMIT:0:12})"
  else
    pass "llama-server-turbo — up to date"
  fi
else
  warn "llama-server-turbo — not installed (run scripts/build-llama-turbo.sh)"
fi

# FU-030 dropped vendor/ChaosEngine; the staleness probe that lived
# here used to walk vendor/ChaosEngine/.git and warn on commits behind
# upstream. Removed alongside the vendored package.

# FU-033: dflash-mlx pin sync between pyproject.toml and stage-runtime.mjs.
# Mirrors the assert in scripts/pre-build-check.mjs — see that file for
# the full rationale (the two pins drifted in May 2026 and shipped an
# old binary in release builds).
PYPROJECT_PIN=$(grep -E 'dflash-mlx\.git@[a-f0-9]+' pyproject.toml | head -1 | sed -E 's/.*dflash-mlx\.git@([a-f0-9]+).*/\1/')
STAGE_PIN=$(grep -E 'dflash-mlx\.git@[a-f0-9]+' scripts/stage-runtime.mjs | head -1 | sed -E 's/.*dflash-mlx\.git@([a-f0-9]+).*/\1/')
if [[ -z "$PYPROJECT_PIN" || -z "$STAGE_PIN" ]]; then
  warn "dflash-mlx pin sync — could not extract commit hashes from both files"
elif [[ "$PYPROJECT_PIN" != "$STAGE_PIN" ]]; then
  fail "dflash-mlx pin drift — pyproject.toml=${PYPROJECT_PIN:0:12} stage-runtime.mjs=${STAGE_PIN:0:12}. Sync both to the same commit."
else
  pass "dflash-mlx pin sync (${PYPROJECT_PIN:0:12})"
fi

# App version sync across the 4 manifests. v0.9.0 release shipped with
# pyproject.toml at 0.8.0 because nothing enforced cross-file sync.
PKG_VERSION=$(grep -E '"version"' package.json | head -1 | sed -E 's/.*"version"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/')
PY_VERSION=$(grep -E '^[[:space:]]*version[[:space:]]*=' pyproject.toml | head -1 | sed -E 's/.*"([^"]+)".*/\1/')
CARGO_VERSION=$(grep -E '^[[:space:]]*version[[:space:]]*=' src-tauri/Cargo.toml | head -1 | sed -E 's/.*"([^"]+)".*/\1/')
TAURI_VERSION=$(grep -E '"version"' src-tauri/tauri.conf.json | head -1 | sed -E 's/.*"version"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/')
if [[ -z "$PKG_VERSION" || -z "$PY_VERSION" || -z "$CARGO_VERSION" || -z "$TAURI_VERSION" ]]; then
  warn "app version sync — could not extract version from one or more manifests"
elif [[ "$PKG_VERSION" != "$PY_VERSION" || "$PKG_VERSION" != "$CARGO_VERSION" || "$PKG_VERSION" != "$TAURI_VERSION" ]]; then
  fail "app version drift — package.json=$PKG_VERSION pyproject.toml=$PY_VERSION Cargo.toml=$CARGO_VERSION tauri.conf.json=$TAURI_VERSION. Bump all four to the same string."
else
  pass "app version sync ($PKG_VERSION)"
fi
echo

# ------------------------------------------------------------------
# 7. Binary availability
# ------------------------------------------------------------------
echo "[7/9] Binary availability..."
if command -v llama-server &>/dev/null || [[ -f "/opt/homebrew/bin/llama-server" ]]; then
  pass "llama-server (standard) — found"
else
  warn "llama-server (standard) — not found"
fi

if [[ -x "$HOME/.chaosengine/bin/llama-server-turbo" ]]; then
  pass "llama-server-turbo — found"
else
  warn "llama-server-turbo — not found (TurboQuant GGUF will fall back to f16)"
fi

# FU-047: detect when the resolved llama-server is too old to support
# --spec-type draft-mtp (llama.cpp PR #22673 merged 2026-05-16). The
# capability probe in inference/capabilities.py keys ggufMtpAvailable
# off the presence of --spec-type, but the older Apr-12 homebrew bottle
# advertises --spec-type without the draft-mtp value. Surface that gap
# at build time so a release doesn't ship MTP-GGUF catalog entries that
# the bundled binary can't actually load.
LLAMA_SERVER_BIN=$(command -v llama-server 2>/dev/null || echo "/opt/homebrew/bin/llama-server")
if [[ -x "$LLAMA_SERVER_BIN" ]]; then
  if "$LLAMA_SERVER_BIN" --help 2>&1 | grep -A1 -- "--spec-type" | grep -q "draft-mtp"; then
    pass "llama-server supports --spec-type draft-mtp (FU-047)"
  else
    warn "llama-server lacks draft-mtp support — upgrade llama.cpp (brew upgrade llama.cpp or rebuild from master ≥ 2026-05-16) to ship GGUF MTP speculative decoding"
  fi
fi
echo

# ------------------------------------------------------------------
# 8. i18n locale validation (FU-042)
# ------------------------------------------------------------------
# Runs the JS-side parity + ICU compile + orphan checker.  Warn-only
# by default per FU-042 §Coverage Gate; pass --strict to upgrade the
# threshold to fail-on-merge once translations stabilise.
echo "[8/9] i18n locale validation..."
if node scripts/i18n-validate.mjs 2>&1 | tail -15; then
  pass "i18n locale catalogs"
else
  fail "i18n locale catalogs — see output above"
fi
echo

# ------------------------------------------------------------------
# 9. E2E smoke (CLI + backend round-trip across every major surface)
# ------------------------------------------------------------------
# Runs the phased E2E driver in --smoke mode: Phases 0,3,4,5,6,7
# (skips the heavy chat sweep + compare to stay under ~90s wall).
# Skip if no backend is running — pre-build doesn't spawn one.
# Full suite (./scripts/e2e_test_suite.py) is required for releases;
# see docs/E2E_TESTING.md.
echo "[9/9] E2E smoke (CLI + backend)..."
if curl -s -m 2 http://127.0.0.1:8876/api/health > /dev/null 2>&1; then
  if ./scripts/e2e_test_suite.py --smoke 2>&1 | tail -12; then
    pass "E2E smoke — all phases green"
  else
    fail "E2E smoke — see report in ~/.chaosengine/test-results/"
  fi
else
  warn "E2E smoke skipped — backend not running on :8876. Start ./scripts/chaosengine-cli serve then re-run."
fi
echo

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
echo "=== Summary ==="
for r in "${RESULTS[@]}"; do
  echo "$r"
done
echo
echo "  $PASS passed, $FAIL failed, $WARN warnings"
echo

if [[ $FAIL -gt 0 ]]; then
  echo "BUILD BLOCKED — fix failures above before shipping."
  exit 1
else
  echo "All gates passed."
  exit 0
fi
