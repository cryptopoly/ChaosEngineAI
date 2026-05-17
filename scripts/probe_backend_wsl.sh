#!/usr/bin/env bash
# Quick probe: does backend_service.app import + can it bind a port?
set -e
cd /home/dan/ChaosEngineAI
echo "=== importing backend_service.app ==="
.venv/bin/python -c "import backend_service.app as a; print('OK, main:', callable(a.main))"
echo "=== running main with --help ==="
.venv/bin/python -m backend_service.app --help 2>&1 | head -10 || echo "main failed exit $?"
echo "=== running for 3 seconds ==="
timeout 3s .venv/bin/python -m backend_service.app --port 8877 2>&1 | head -30 || true
echo "=== probe done ==="
