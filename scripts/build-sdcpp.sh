#!/usr/bin/env bash
# Build the ``sd`` binary from leejet/stable-diffusion.cpp (FU-008).
#
# Cross-platform diffusion runtime: SD 1.x/2.x/XL, FLUX.1/2, Wan 2.1 / 2.2
# video, Qwen Image, Z-Image. Wired into ChaosEngineAI as a subprocess
# engine via ``backend_service/sdcpp_video_runtime.py``. Mirrors the
# llama-server-turbo build script pattern so the desktop installer can
# trigger it the same way.
#
# Usage:
#   ./scripts/build-sdcpp.sh
#
# Environment variables:
#   SDCPP_DIR            Source checkout dir  (default: /tmp/stable-diffusion.cpp)
#   CHAOSENGINE_BIN_DIR  Install destination  (default: ~/.chaosengine/bin)
#   SDCPP_BRANCH         Git branch to build  (default: master)
#   SDCPP_JOBS           Parallel build jobs  (default: $(nproc) or sysctl)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SDCPP_REPO="https://github.com/leejet/stable-diffusion.cpp.git"
SDCPP_BRANCH="${SDCPP_BRANCH:-master}"
SDCPP_DIR="${SDCPP_DIR:-/tmp/stable-diffusion.cpp}"
INSTALL_DIR="${CHAOSENGINE_BIN_DIR:-$HOME/.chaosengine/bin}"

# Detect parallel jobs (matches build-llama-turbo.sh)
if command -v nproc &>/dev/null; then
  JOBS="${SDCPP_JOBS:-$(nproc)}"
elif command -v sysctl &>/dev/null; then
  JOBS="${SDCPP_JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
else
  JOBS="${SDCPP_JOBS:-4}"
fi

echo "==> stable-diffusion.cpp builder"
echo "    repo:     $SDCPP_REPO"
echo "    branch:   $SDCPP_BRANCH"
echo "    source:   $SDCPP_DIR"
echo "    install:  $INSTALL_DIR"
echo "    jobs:     $JOBS"
echo

# Clone or update the source checkout — sd.cpp uses git submodules for
# ggml, so always pass --recurse-submodules / --recursive.
if [[ -d "$SDCPP_DIR/.git" ]]; then
  echo "==> updating existing checkout"
  cd "$SDCPP_DIR"
  git fetch --all --prune
  git checkout "$SDCPP_BRANCH"
  git reset --hard "origin/$SDCPP_BRANCH"
  git submodule update --init --recursive
else
  echo "==> cloning $SDCPP_REPO (branch: $SDCPP_BRANCH)"
  git clone --recursive --branch "$SDCPP_BRANCH" "$SDCPP_REPO" "$SDCPP_DIR"
  cd "$SDCPP_DIR"
fi

# Platform-specific CMake flags
# -DBUILD_SHARED_LIBS=OFF — match build-llama-turbo.sh: produce a
# self-contained binary so dyld doesn't need rpath-resolved .dylibs.
CMAKE_FLAGS=(-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF)
case "$(uname -s)" in
  Darwin)
    CMAKE_FLAGS+=(-DSD_METAL=ON)
    ;;
  Linux)
    if command -v nvcc &>/dev/null; then
      CMAKE_FLAGS+=(-DSD_CUBLAS=ON)
    fi
    ;;
esac

echo "==> cmake configure"
cmake -B build "${CMAKE_FLAGS[@]}"

echo "==> building sd-cli binary"
# Upstream renamed the CLI target ``sd`` → ``sd-cli`` around master-590
# (2026-04). Build the new target; install with the legacy ``sd`` name
# so the runtime resolver in ``sdcpp_video_runtime.py`` and
# ``scripts/stage-runtime.mjs`` keep working without a path rename.
cmake --build build --config Release -j "$JOBS" --target sd-cli

echo "==> installing to $INSTALL_DIR"
mkdir -p "$INSTALL_DIR"
cp build/bin/sd-cli "$INSTALL_DIR/sd"
chmod +x "$INSTALL_DIR/sd"

# Version tracking — mirrors build-llama-turbo.sh shape so the same
# update detection logic applies.
VERSION_FILE="$INSTALL_DIR/sd.version"
{
  git rev-parse HEAD
  echo "$SDCPP_BRANCH"
  date -u +"%Y-%m-%dT%H:%M:%SZ"
} > "$VERSION_FILE"
echo "==> version tracked in $VERSION_FILE"

echo
echo "==> build complete"
echo "sd installed to $INSTALL_DIR/sd"
echo "ChaosEngineAI will auto-detect it on next video generate request."
echo "Restart the app if it is currently running."
