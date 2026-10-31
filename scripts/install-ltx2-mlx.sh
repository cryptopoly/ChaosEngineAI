#!/usr/bin/env bash
# Install ltx-2-mlx into an isolated venv at ~/.chaosengine/ltx2-venv/.
#
# ltx-2-mlx (https://github.com/dgrauet/ltx-2-mlx, MIT) is a pure MLX port of
# Lightricks LTX-2.3 / 2.5: text, image and audio to video with stereo sound, on
# the Apple GPU without PyTorch. The engine in
# backend_service/ltx2_mlx_runtime.py spawns its `ltx-2-mlx` command from this
# venv, one process per clip.
#
# Why a separate venv: the package pins its own mlx / mlx-lm floors and the
# weights are loaded by its own code; keeping it apart leaves the app's MLX
# stack untouched. It is not on PyPI, so it installs from a pinned commit.
#
# Apple Silicon only; requires native arm64 Python 3.11-3.13.
#
# Structured progress lines, parsed by routes/setup/ltx2.py:
#   PHASE:<name>   emitted before each phase starts
#   OK             emitted on clean exit
#   FAIL:<msg>     emitted before a non-zero exit

set -euo pipefail

# The packaged app exports variables for its embedded Python runtime; a venv
# interpreter that inherited them would import the app's packages instead of
# its own.
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP VIRTUAL_ENV \
    DYLD_LIBRARY_PATH DYLD_FALLBACK_LIBRARY_PATH DYLD_INSERT_LIBRARIES || true

VENV_DIR="${HOME}/.chaosengine/ltx2-venv"
BIN_DIR="${HOME}/.chaosengine/bin"
VERSION_FILE="${BIN_DIR}/ltx2-mlx.version"

# Pinned: the command line and weight file names in the engine were read from
# this commit (v0.18.0). Bump together with backend_service/ltx2_mlx_runtime.py.
LTX2_MLX_REF="${LTX2_MLX_REF:-895201df379b079c2154738bcf92005b7a371923}"
LTX2_MLX_REPO="${LTX2_MLX_REPO:-https://github.com/dgrauet/ltx-2-mlx.git}"

log() { echo "$*"; }
phase() { echo "PHASE:$1"; }
fail() { echo "FAIL:$*"; exit 1; }

phase "preflight"

if [[ "$(uname -s)" != "Darwin" ]]; then
    fail "ltx-2-mlx runs on macOS with Apple Silicon only."
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
    fail "ltx-2-mlx needs native arm64 Python 3.11, 3.12 or 3.13 (not found; not under Rosetta). Install one with: brew install python@3.12"
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

log "Installing ltx-2-mlx at ${LTX2_MLX_REF:0:8} (mlx, mlx-lm and transformers; about 1 GB)"
# Both packages in one command so the pipelines package resolves ltx-core-mlx
# from the same commit instead of looking for it on PyPI.
PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1 \
    "${VENV_DIR}/bin/pip" install --upgrade \
    "ltx-core-mlx @ git+${LTX2_MLX_REPO}@${LTX2_MLX_REF}#subdirectory=packages/ltx-core-mlx" \
    "ltx-pipelines-mlx @ git+${LTX2_MLX_REPO}@${LTX2_MLX_REF}#subdirectory=packages/ltx-pipelines-mlx"

phase "verifying"

INSTALLED_VERSION="$("${VENV_DIR}/bin/pip" show ltx-pipelines-mlx 2>/dev/null \
    | grep -i "^Version:" | awk '{print $2}' || true)"
INSTALLED_VERSION="${INSTALLED_VERSION:-unknown}"

if ! "${VENV_DIR}/bin/python" -c "import ltx_core_mlx, ltx_pipelines_mlx, mlx.core" >/dev/null 2>&1; then
    fail "ltx-2-mlx import check failed (ltx_pipelines_mlx / mlx): the install may be incomplete."
fi

if ! "${VENV_DIR}/bin/ltx-2-mlx" generate --help >/dev/null 2>&1; then
    fail "The ltx-2-mlx command does not run after install."
fi

log "ltx-2-mlx ${INSTALLED_VERSION} import + CLI verified"

# The version file's presence is what marks the install usable.
{
    echo "${INSTALLED_VERSION}"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "${LTX2_MLX_REF}"
} > "${VERSION_FILE}"

log "Version file written to ${VERSION_FILE}"
echo "OK"
