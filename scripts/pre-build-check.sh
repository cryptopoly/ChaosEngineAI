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
from compression import registry
registry.discover()
valid = {'f32','f16','bf16','q8_0','q4_0','q4_1','iq4_nl','q5_0','q5_1'}
ce = registry.get('chaosengine')
for bits in (2,3,4,5,6,8):
    flags = ce.llama_cpp_cache_flags(bits)
    for i, f in enumerate(flags):
        if f.startswith('--cache-type-') and i+1 < len(flags):
            if flags[i+1] not in valid:
                print(f'INVALID: ChaosEngine {bits}-bit emits {flags[i+1]}')
rq = registry.get('rotorquant')
tq = registry.get('turboquant')
if rq.required_llama_binary() != 'turbo': print('INVALID: RotorQuant not routing to turbo')
if tq.required_llama_binary() != 'turbo': print('INVALID: TurboQuant not routing to turbo')
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

# ChaosEngine submodule
if [[ -d "vendor/ChaosEngine/.git" ]]; then
  CE_BEHIND=$(git -C vendor/ChaosEngine rev-list HEAD..origin/main --count 2>/dev/null || echo "?")
  if [[ "$CE_BEHIND" == "0" ]]; then
    pass "vendor/ChaosEngine — up to date"
  elif [[ "$CE_BEHIND" == "?" ]]; then
    warn "vendor/ChaosEngine — could not check (fetch first)"
  else
    warn "vendor/ChaosEngine — $CE_BEHIND commits behind upstream"
  fi
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
