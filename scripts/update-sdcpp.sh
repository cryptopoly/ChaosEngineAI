#!/usr/bin/env bash
# Update the ``sd`` binary from leejet/stable-diffusion.cpp.
#
# Companion to ``build-sdcpp.sh`` — fetches the latest commit on the
# tracked branch and rebuilds in place. Mirrors update-llama-turbo.sh.
#
# Usage:  ./scripts/update-sdcpp.sh
#
# Override the source dir with SDCPP_DIR if the checkout lives somewhere
# other than /tmp/stable-diffusion.cpp.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SDCPP_BRANCH="${SDCPP_BRANCH:-master}"
SDCPP_DIR="${SDCPP_DIR:-/tmp/stable-diffusion.cpp}"
INSTALL_DIR="${CHAOSENGINE_BIN_DIR:-$HOME/.chaosengine/bin}"
VERSION_FILE="$INSTALL_DIR/sd.version"

if command -v nproc &>/dev/null; then
  JOBS="${SDCPP_JOBS:-$(nproc)}"
elif command -v sysctl &>/dev/null; then
  JOBS="${SDCPP_JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
else
  JOBS="${SDCPP_JOBS:-4}"
fi

if [[ ! -d "$SDCPP_DIR/.git" ]]; then
  echo "No existing checkout at $SDCPP_DIR — running full build instead."
  exec "$SCRIPT_DIR/build-sdcpp.sh"
fi

cd "$SDCPP_DIR"

if [[ -f "$VERSION_FILE" ]]; then
  CURRENT_COMMIT=$(head -1 "$VERSION_FILE")
  echo "Current installed commit: $CURRENT_COMMIT"
else
  CURRENT_COMMIT=""
  echo "No version file found — will rebuild regardless."
fi

echo "==> fetching latest changes"
git fetch --all --prune

echo "==> checking out $SDCPP_BRANCH"
git checkout "$SDCPP_BRANCH"

REMOTE_COMMIT=$(git rev-parse "origin/$SDCPP_BRANCH")
echo "Remote HEAD: $REMOTE_COMMIT"

if [[ "$CURRENT_COMMIT" == "$REMOTE_COMMIT" ]]; then
  echo
  echo "Already up to date. No rebuild needed."
  exit 0
fi

echo "==> resetting to origin/$SDCPP_BRANCH"
git reset --hard "origin/$SDCPP_BRANCH"
git submodule update --init --recursive

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

echo "==> rebuilding sd-cli binary"
# Target renamed upstream; install with legacy ``sd`` name so downstream
# resolvers don't need a rename. See build-sdcpp.sh for context.
cmake --build build --config Release -j "$JOBS" --target sd-cli

echo "==> installing to $INSTALL_DIR"
mkdir -p "$INSTALL_DIR"
cp build/bin/sd-cli "$INSTALL_DIR/sd"
chmod +x "$INSTALL_DIR/sd"

{
  git rev-parse HEAD
  echo "$SDCPP_BRANCH"
  date -u +"%Y-%m-%dT%H:%M:%SZ"
} > "$VERSION_FILE"

echo
echo "==> update complete"
echo "Updated from ${CURRENT_COMMIT:0:12} to $(git rev-parse --short HEAD)"
echo "Restart ChaosEngineAI to pick up the new binary."
