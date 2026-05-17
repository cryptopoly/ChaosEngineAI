#!/usr/bin/env bash
# One-shot installer: fetch latest llama.cpp Linux release into
# ~/.chaosengine/bin so the WSL dev backend has a usable llama-server
# binary.
set -euo pipefail

INSTALL_DIR="${HOME}/.chaosengine/bin"
mkdir -p "$INSTALL_DIR"

cat >/tmp/find_llamacpp.py <<'EOF'
import json, sys, urllib.request
url = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
with urllib.request.urlopen(url) as r:
    data = json.load(r)
tag = data["tag_name"]
# Match the plain ubuntu-x64 tar.gz (CPU build — vulkan/sycl/openvino/
# rocm/cuda variants need their respective runtime; the dev runner only
# needs llama-server's HTTP path, not GPU acceleration).
target = None
for a in data["assets"]:
    n = a["name"].lower()
    if n.startswith(f"llama-{tag.lower()}-bin-ubuntu-x64.tar.gz"):
        target = a["browser_download_url"]
        break
if not target:
    print(f"ERROR: no plain ubuntu-x64 asset in {tag}")
    sys.exit(1)
print(f"{tag} {target}")
EOF

RESULT=$(python3 /tmp/find_llamacpp.py)
TAG=$(echo "$RESULT" | awk '{print $1}')
URL=$(echo "$RESULT" | awk '{print $2}')
echo "Downloading $TAG from $URL ..."

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

curl -fsSL "$URL" -o "$TMPDIR/llamacpp.tar.gz"
tar -xzf "$TMPDIR/llamacpp.tar.gz" -C "$TMPDIR"

# Find the llama-server binary in the extracted tree.
SERVER_BIN=$(find "$TMPDIR" -name 'llama-server' -type f 2>/dev/null | head -1)
if [ -z "$SERVER_BIN" ]; then
    echo "llama-server binary not found in tarball"
    find "$TMPDIR" -type f | head -20
    exit 1
fi
cp "$SERVER_BIN" "$INSTALL_DIR/llama-server"
chmod +x "$INSTALL_DIR/llama-server"
echo "$TAG" > "$INSTALL_DIR/llama-server.version"

# Bundle shared libraries the binary depends on. The Linux release
# layout has libllama.so / libggml*.so alongside the binary.
find "$TMPDIR" -name '*.so*' -exec cp {} "$INSTALL_DIR/" \; 2>/dev/null || true

echo ""
echo "Installed to: $INSTALL_DIR/llama-server"
echo "Version:      $TAG"
echo ""

# Smoke-test the install.
"$INSTALL_DIR/llama-server" --version 2>&1 | head -3 || true
