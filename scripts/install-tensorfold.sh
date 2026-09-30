#!/usr/bin/env bash
# Install TensorFold into an isolated venv at ~/.chaosengine/tensorfold-venv/.
#
# TensorFold (https://github.com/ashhart/TensorFold, MIT) is a local
# OpenAI-compatible server whose per-family kernels verify speculative drafts
# exactly, so its speedups cost no output quality. The engine lane in
# backend_service/inference/tensorfold_engine.py spawns it from this venv.
#
# Why a separate venv: TensorFold pins mlx >=0.32.2,<0.32.4 and mlx-lm
# >=0.31.3,<0.32, which can differ from the versions the app itself runs.
#
# Apple Silicon only (the Mac engine); requires native arm64 Python 3.11-3.13.
#
# Structured progress lines, parsed by routes/setup/tensorfold.py:
#   PHASE:<name>   emitted before each phase starts
#   OK             emitted on clean exit
#   FAIL:<msg>     emitted before a non-zero exit

set -euo pipefail

# The packaged app exports variables for its embedded Python runtime; a venv
# interpreter that inherited them would import the app's packages instead of
# its own.
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP VIRTUAL_ENV \
    DYLD_LIBRARY_PATH DYLD_FALLBACK_LIBRARY_PATH DYLD_INSERT_LIBRARIES || true

VENV_DIR="${HOME}/.chaosengine/tensorfold-venv"
BIN_DIR="${HOME}/.chaosengine/bin"
VERSION_FILE="${BIN_DIR}/tensorfold.version"

# Pinned to v0.5.0. Bump together with the facts recorded in
# backend_service/inference/_tensorfold.py (model families, tested repos,
# drafters) after re-reading TensorFold's families/*/__init__.py. Override with
# TENSORFOLD_REF=<commit|tag> to try a newer build.
TENSORFOLD_REF="${TENSORFOLD_REF:-9cd52ab4daba68ddd09be89be8f23ad43175e821}"
# TENSORFOLD_SOURCE overrides where pip installs from (a checkout or an
# internal mirror); the archive URL is the default.
TENSORFOLD_SOURCE="${TENSORFOLD_SOURCE:-https://github.com/ashhart/TensorFold/archive/${TENSORFOLD_REF}.tar.gz}"

log() { echo "$*"; }
phase() { echo "PHASE:$1"; }
fail() { echo "FAIL:$*"; exit 1; }

# ---------------------------------------------------------------------------
# Preflight: macOS on Apple Silicon, and a usable Python
# ---------------------------------------------------------------------------

phase "preflight"

if [[ "$(uname -s)" != "Darwin" ]]; then
    fail "TensorFold's Mac engine needs macOS on Apple Silicon."
fi

# Succeeds when $1 is native arm64 Python 3.11-3.13 (mlx wheels; 3.14 has none yet).
python_ok() {
    "$1" -c "import platform, sys; sys.exit(0 if platform.machine() == 'arm64' and (3, 11) <= sys.version_info[:2] < (3, 14) else 1)" \
        >/dev/null 2>&1
}

pick_python() {
    local candidate
    for candidate in "${PYTHON:-}" \
        python3.13 python3.12 python3.11 python3 \
        /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11 /opt/homebrew/bin/python3 \
        /usr/local/bin/python3.13 /usr/local/bin/python3.12 /usr/local/bin/python3.11 /usr/local/bin/python3; do
        [[ -n "${candidate}" ]] || continue
        command -v "${candidate}" >/dev/null 2>&1 || continue
        if python_ok "${candidate}"; then
            command -v "${candidate}"
            return 0
        fi
    done
    return 1
}

if ! PYTHON_BIN="$(pick_python)"; then
    fail "TensorFold needs native arm64 Python 3.11, 3.12 or 3.13 (not found; not under Rosetta). Install one with: brew install python@3.12"
fi

PY_VER="$("${PYTHON_BIN}" -c "import sys; print('%d.%d' % sys.version_info[:2])")"
log "Python ${PY_VER} (arm64) at ${PYTHON_BIN}: OK"
mkdir -p "${BIN_DIR}"

# A half-finished install must never read as installed.
rm -f "${VERSION_FILE}"

# ---------------------------------------------------------------------------
# Isolated venv
# ---------------------------------------------------------------------------

phase "creating-venv"

if [[ -d "${VENV_DIR}" ]]; then
    log "Removing existing venv at ${VENV_DIR}"
    rm -rf "${VENV_DIR}"
fi

log "Creating venv at ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
log "Upgrading pip"
"${VENV_DIR}/bin/pip" install --quiet --upgrade pip

# ---------------------------------------------------------------------------
# Install (pulls the pinned mlx / mlx-lm it was tested against)
# ---------------------------------------------------------------------------

phase "installing"

log "Installing TensorFold @ ${TENSORFOLD_REF}"
PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1 \
    "${VENV_DIR}/bin/pip" install --upgrade "${TENSORFOLD_SOURCE}"

# ---------------------------------------------------------------------------
# Verify: imports, then the CLI itself
# ---------------------------------------------------------------------------

phase "verifying"

TENSORFOLD_VERSION="$("${VENV_DIR}/bin/pip" show tensorfold 2>/dev/null \
    | grep -i "^Version:" | awk '{print $2}' || true)"
TENSORFOLD_VERSION="${TENSORFOLD_VERSION:-unknown}"

if ! "${VENV_DIR}/bin/python" -c "import tensorfold, mlx.core, mlx_lm" >/dev/null 2>&1; then
    fail "TensorFold import check failed (tensorfold / mlx / mlx-lm): the install may be incomplete."
fi

if ! "${VENV_DIR}/bin/tensorfold" --version >/dev/null 2>&1; then
    fail "The tensorfold command does not run after install."
fi

log "TensorFold ${TENSORFOLD_VERSION} import + CLI verified"

# ---------------------------------------------------------------------------
# Version file (its presence is what marks the install usable)
# ---------------------------------------------------------------------------

{
    echo "${TENSORFOLD_VERSION}"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "${TENSORFOLD_REF}"
} > "${VERSION_FILE}"

log "Version file written to ${VERSION_FILE}"
echo "OK"
