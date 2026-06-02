#!/usr/bin/env bash
# Install MTPLX into an isolated venv at ~/.chaosengine/mtplx-venv/.
#
# Requires native arm64 Python 3.10+ (MTPLX's forked mlx won't build under
# Rosetta). The script prints structured progress lines so the backend job
# worker can parse phase transitions:
#
#   PHASE:<name>   — emitted before each phase starts
#   OK             — emitted on clean exit
#   FAIL:<msg>     — emitted before a non-zero exit
#
# The backend worker (routes/setup/mtplx.py) reads these markers to drive
# the InstallLogPanel phases without scraping pip output.

set -euo pipefail

VENV_DIR="${HOME}/.chaosengine/mtplx-venv"
BIN_DIR="${HOME}/.chaosengine/bin"
VERSION_FILE="${BIN_DIR}/mtplx.version"
MTPLX_PACKAGE="mtplx"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() { echo "$*"; }
phase() { echo "PHASE:$1"; }
fail() { echo "FAIL:$*"; exit 1; }

# ---------------------------------------------------------------------------
# Preflight — verify native arm64 Python 3.10+
# ---------------------------------------------------------------------------

phase "preflight"

PYTHON="${PYTHON:-python3}"

ARCH=$(${PYTHON} -c "import platform; print(platform.machine())" 2>/dev/null || true)
if [[ "${ARCH}" != "arm64" ]]; then
    fail "MTPLX requires native arm64 Python (got: ${ARCH:-unknown}). Make sure you are not running under Rosetta."
fi

PY_VER=$(${PYTHON} -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>/dev/null || true)
PY_MAJ=$(echo "${PY_VER}" | cut -d. -f1)
PY_MIN=$(echo "${PY_VER}" | cut -d. -f2)
if [[ "${PY_MAJ}" -lt 3 ]] || { [[ "${PY_MAJ}" -eq 3 ]] && [[ "${PY_MIN}" -lt 10 ]]; }; then
    fail "MTPLX requires Python 3.10+ (got: ${PY_VER})"
fi

log "Python ${PY_VER} (arm64) — OK"
mkdir -p "${BIN_DIR}"

# ---------------------------------------------------------------------------
# Create isolated venv
# ---------------------------------------------------------------------------

phase "creating-venv"

if [[ -d "${VENV_DIR}" ]]; then
    log "Removing existing venv at ${VENV_DIR}"
    rm -rf "${VENV_DIR}"
fi

log "Creating venv at ${VENV_DIR}"
${PYTHON} -m venv "${VENV_DIR}"
log "Upgrading pip"
"${VENV_DIR}/bin/pip" install --quiet --upgrade pip

# ---------------------------------------------------------------------------
# Install MTPLX (pulls in its mlx fork automatically)
# ---------------------------------------------------------------------------

phase "installing"

log "Installing ${MTPLX_PACKAGE}"
"${VENV_DIR}/bin/pip" install --upgrade "${MTPLX_PACKAGE}"

# ---------------------------------------------------------------------------
# Verify: import check + extract version
# ---------------------------------------------------------------------------

phase "verifying"

MTPLX_VERSION=$("${VENV_DIR}/bin/pip" show mtplx 2>/dev/null \
    | grep -i "^Version:" | awk '{print $2}' || echo "unknown")

IMPORT_OK=$("${VENV_DIR}/bin/python" -c "import mtplx; print('ok')" 2>/dev/null || echo "fail")
if [[ "${IMPORT_OK}" != "ok" ]]; then
    fail "MTPLX import check failed — installation may be incomplete"
fi

# FU-077: ``import mtplx`` succeeds even when the *server* deps (numpy,
# safetensors, uvicorn, fastapi, pydantic, mlx-lm, ...) are missing,
# because they're imported lazily by ``mtplx.server.openai`` — not at
# package top level. A truncated ``pip install`` therefore passed the
# old verify but produced a venv whose ``mtplx quickstart`` server died
# at startup with ModuleNotFoundError, silently falling back to DFlash.
# Import the server module so an incomplete install fails loudly here.
SERVER_OK=$("${VENV_DIR}/bin/python" -c "import mtplx.server.openai; print('ok')" 2>/dev/null || echo "fail")
if [[ "${SERVER_OK}" != "ok" ]]; then
    log "MTPLX server module import failed — retrying full dependency install"
    "${VENV_DIR}/bin/pip" install --upgrade --upgrade-strategy eager "${MTPLX_PACKAGE}"
    SERVER_OK=$("${VENV_DIR}/bin/python" -c "import mtplx.server.openai; print('ok')" 2>/dev/null || echo "fail")
    if [[ "${SERVER_OK}" != "ok" ]]; then
        fail "MTPLX server import check failed after retry — server deps incomplete (numpy / safetensors / uvicorn / fastapi / mlx-lm)"
    fi
fi

log "MTPLX ${MTPLX_VERSION} import + server module verified"

# ---------------------------------------------------------------------------
# Write version file
# ---------------------------------------------------------------------------

{
    echo "${MTPLX_VERSION}"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${VERSION_FILE}"

log "Version file written to ${VERSION_FILE}"
echo "OK"
