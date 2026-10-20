#!/usr/bin/env bash
# Install mflux into an isolated venv at ~/.chaosengine/mflux-venv/.
#
# mflux (https://github.com/filipstrand/mflux, MIT) runs image models natively
# on MLX: Qwen-Image-2.1, FLUX.1 / FLUX.2, Z-Image and more. The engine in
# backend_service/image_runtime/mflux_engine.py spawns its mflux-generate-*
# commands from this venv, one process per image.
#
# Why a separate venv: mflux 0.20 pins mlx >=0.32,<0.33, torch >=2.13 and
# pillow >=12.3, which is newer than the stack the app itself runs (mlx 0.31).
# Installing it next to the app's packages would replace mlx for every MLX lane.
#
# Apple Silicon only; requires native arm64 Python 3.11-3.13.
#
# Structured progress lines, parsed by routes/setup/mflux.py:
#   PHASE:<name>   emitted before each phase starts
#   OK             emitted on clean exit
#   FAIL:<msg>     emitted before a non-zero exit

set -euo pipefail

# The packaged app exports variables for its embedded Python runtime; a venv
# interpreter that inherited them would import the app's packages instead of
# its own.
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP VIRTUAL_ENV \
    DYLD_LIBRARY_PATH DYLD_FALLBACK_LIBRARY_PATH DYLD_INSERT_LIBRARIES || true

VENV_DIR="${HOME}/.chaosengine/mflux-venv"
BIN_DIR="${HOME}/.chaosengine/bin"
VERSION_FILE="${BIN_DIR}/mflux.version"

# Pinned: the command names and flags in the engine's family table were read
# from this release. Bump together with that table. Override with
# MFLUX_VERSION=<version> to try a newer build.
MFLUX_VERSION="${MFLUX_VERSION:-0.20.0}"
# MFLUX_SOURCE overrides where pip installs from (a wheelhouse path or an
# internal mirror); the pinned PyPI release is the default.
MFLUX_SOURCE="${MFLUX_SOURCE:-mflux==${MFLUX_VERSION}}"

log() { echo "$*"; }
phase() { echo "PHASE:$1"; }
fail() { echo "FAIL:$*"; exit 1; }

phase "preflight"

if [[ "$(uname -s)" != "Darwin" ]]; then
    fail "mflux runs on macOS with Apple Silicon only."
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
    fail "mflux needs native arm64 Python 3.11, 3.12 or 3.13 (not found; not under Rosetta). Install one with: brew install python@3.12"
fi

PY_VER="$("${PYTHON_BIN}" -c "import sys; print('%d.%d' % sys.version_info[:2])")"
log "Python ${PY_VER} (arm64) at ${PYTHON_BIN}: OK"
mkdir -p "${BIN_DIR}"

# A half-finished install must never read as installed.
rm -f "${VERSION_FILE}"

phase "creating-venv"

if [[ -d "${VENV_DIR}" ]]; then
    log "Removing existing venv at ${VENV_DIR}"
    rm -rf "${VENV_DIR}"
fi

log "Creating venv at ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
log "Upgrading pip"
"${VENV_DIR}/bin/pip" install --quiet --upgrade pip

phase "installing"

log "Installing mflux ${MFLUX_VERSION} (pulls mlx, torch and transformers; about 1.3 GB)"
PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1 \
    "${VENV_DIR}/bin/pip" install --upgrade "${MFLUX_SOURCE}"

phase "verifying"

INSTALLED_VERSION="$("${VENV_DIR}/bin/pip" show mflux 2>/dev/null \
    | grep -i "^Version:" | awk '{print $2}' || true)"
INSTALLED_VERSION="${INSTALLED_VERSION:-unknown}"

if ! "${VENV_DIR}/bin/python" -c "import mflux, mlx.core" >/dev/null 2>&1; then
    fail "mflux import check failed (mflux / mlx): the install may be incomplete."
fi

if ! "${VENV_DIR}/bin/mflux-generate" --help >/dev/null 2>&1; then
    fail "The mflux-generate command does not run after install."
fi

log "mflux ${INSTALLED_VERSION} import + CLI verified"

# The version file's presence is what marks the install usable.
{
    echo "${INSTALLED_VERSION}"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "${MFLUX_VERSION}"
} > "${VERSION_FILE}"

log "Version file written to ${VERSION_FILE}"
echo "OK"
