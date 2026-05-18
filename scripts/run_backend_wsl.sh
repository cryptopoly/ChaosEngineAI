#!/usr/bin/env bash
# Launch the dev backend inside WSL on port 8877 with auth disabled
# (test-runs don't need the bearer-token gate). Avoids the host's
# 8876 (which the Windows-side ChaosEngineAI binds — WSL2 mirrors
# loopback so the port collision is real).
set -euo pipefail

cd /home/dan/ChaosEngineAI
export CHAOSENGINE_LLAMA_SERVER="$HOME/.chaosengine/bin/llama-server"
export CHAOSENGINE_PORT=8877
export CHAOSENGINE_HOST=127.0.0.1
export CHAOSENGINE_REQUIRE_AUTH=0

nohup .venv/bin/python -m backend_service.app \
    > /tmp/backend_wsl.log 2>&1 &
echo "PID=$!"
disown
