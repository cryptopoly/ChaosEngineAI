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
echo "[1/7] Python tests..."
if .venv/bin/python -m pytest tests/ -q --tb=line 2>&1 | tail -3; then
  pass "Python tests"
else
  fail "Python tests — see output above"
fi
echo

# ------------------------------------------------------------------
# 2. TypeScript tests
# ------------------------------------------------------------------
echo "[2/7] TypeScript tests..."
if npm test 2>&1 | tail -5; then
  pass "TypeScript tests"
else
  fail "TypeScript tests — see output above"
fi
echo

# ------------------------------------------------------------------
# 3. TypeScript type checking
# ------------------------------------------------------------------
echo "[3/7] TypeScript type checking..."
if npx tsc --noEmit 2>&1; then
  pass "TypeScript types"
else
  fail "TypeScript type errors — see output above"
fi
echo

# ------------------------------------------------------------------
# 4. Licence notices
# ------------------------------------------------------------------
echo "[4/7] Licence notices..."
if [[ -f "THIRD_PARTY_NOTICES.md" ]] && [[ -s "THIRD_PARTY_NOTICES.md" ]]; then
  # Check that key dependencies are mentioned
  missing=""
  for dep in "llama.cpp" "llama-cpp-turboquant" "dflash-mlx" "turboquant" "ChaosEngine"; do
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
echo "[5/7] Cache strategy validation..."
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
echo "[6/7] Upstream dependency check..."

# Turbo fork
TURBO_VERSION_FILE="$HOME/.chaosengine/bin/llama-server-turbo.version"
if [[ -f "$TURBO_VERSION_FILE" ]]; then
  LOCAL_COMMIT=$(head -1 "$TURBO_VERSION_FILE")
  REMOTE_COMMIT=$(git ls-remote https://github.com/johndpope/llama-cpp-turboquant.git refs/heads/feature/planarquant-kv-cache 2>/dev/null | cut -f1)
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
echo

# ------------------------------------------------------------------
# 7. Binary availability
# ------------------------------------------------------------------
echo "[7/7] Binary availability..."
if command -v llama-server &>/dev/null || [[ -f "/opt/homebrew/bin/llama-server" ]]; then
  pass "llama-server (standard) — found"
else
  warn "llama-server (standard) — not found"
fi

if [[ -x "$HOME/.chaosengine/bin/llama-server-turbo" ]]; then
  pass "llama-server-turbo — found"
else
  warn "llama-server-turbo — not found (RotorQuant/TurboQuant GGUF will fall back to f16)"
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
