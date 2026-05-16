# Automation

`chaosengine-cli` is designed to be composable with shell pipelines, `jq`,
CI runners, and orchestration tools. Here are the patterns we use
ourselves and recommend.

## Exit code contract

| Exit | Meaning |
|---|---|
| **0** | Success — call returned 2xx and parsed cleanly. |
| **non-zero** | Backend error (4xx / 5xx), network failure, or invalid arguments. The body of the error response is written to stderr; stdout is empty. |

Specifically for the E2E suite (`scripts/e2e_test_suite.py`):

| Exit | Meaning |
|---|---|
| **0** | Every phase passed or was correctly skipped. |
| **1** | At least one phase had a `fail` check (regression). |
| **2** | Backend was not reachable; suite could not run. |

## Output formats

By default the CLI writes structured JSON to stdout. Specific subcommands
add helpful surfacing:

- **`prompt`** — supports `--stream` (raw token text to stdout) and
  `--metrics` (appends a metrics line at the end). Without `--stream`,
  the full response is one JSON object.
- **`image-progress` / `video-progress`** — single snapshot by default;
  pipe through `watch` for a polling view.
- **`server-logs`** — streams Server Sent Events to stdout until SIGINT.

Set `JQ` style filters with `jq` for grep-like extraction:

```bash
./scripts/chaosengine-cli status | jq '.runtime | {state, model, engine}'
```

## Headless backend

For CI you usually want the backend running headlessly:

```bash
nohup ./scripts/chaosengine-cli serve > /var/log/chaosengine.log 2>&1 &
echo $! > /var/run/chaosengine.pid

# Wait for readiness
for _ in $(seq 1 60); do
    ./scripts/chaosengine-cli health 2>/dev/null && break
    sleep 1
done
```

A systemd unit is a cleaner alternative on Linux:

```ini
# /etc/systemd/system/chaosengine.service
[Unit]
Description=ChaosEngineAI backend
After=network.target

[Service]
Type=simple
User=chaosengine
WorkingDirectory=/opt/ChaosEngineAI
ExecStart=/opt/ChaosEngineAI/scripts/chaosengine-cli serve
Restart=on-failure
Environment=CHAOSENGINE_HOST=127.0.0.1
Environment=CHAOSENGINE_PORT=8876

[Install]
WantedBy=multi-user.target
```

## CI gate (GitHub Actions example)

```yaml
- name: Install
  run: |
      python3 -m venv .venv
      .venv/bin/pip install -e .

- name: Start backend
  run: |
      nohup ./scripts/chaosengine-cli serve > backend.log 2>&1 &
      for _ in $(seq 1 60); do
          ./scripts/chaosengine-cli health 2>/dev/null && break
          sleep 1
      done

- name: E2E smoke
  run: ./scripts/e2e_test_suite.py --smoke

- name: Shutdown
  if: always()
  run: ./scripts/chaosengine-cli server-shutdown || true

- name: Upload backend log
  if: always()
  uses: actions/upload-artifact@v4
  with:
      name: backend-log
      path: backend.log
```

The pre-build check (`scripts/pre-build-check.sh`) already gates the E2E
smoke as phase 9 of 9.

## Retry + back-off

The CLI doesn't retry by default — every call is one-shot HTTP. For
flaky workloads (e.g. a model that occasionally crashes mid-load),
wrap the call in a retry loop:

```bash
for try in 1 2 3; do
    if ./scripts/chaosengine-cli load "$MODEL" --spec; then
        break
    fi
    echo "Load attempt $try failed, retrying..." >&2
    sleep $((try * 5))
done
```

## Long-running installers

`mtplx-install`, `longlive-install`, `wan-install`, and `gpu-bundle-install`
are background jobs. The POST endpoint kicks them off and returns
immediately with a job ID; the GET endpoint streams progress.

```bash
JOB=$(./scripts/chaosengine-cli mtplx-install | jq -r '.jobId')

while :; do
    snap=$(./scripts/chaosengine-cli call GET /api/setup/install-mtplx/status)
    state=$(echo "$snap" | jq -r '.state')
    echo "$snap" | jq -r '.phase'
    [[ "$state" == "complete" || "$state" == "failed" ]] && break
    sleep 2
done
```

## Orphan cleanup

The backend tracks subprocess children (MLX worker, `llama-server`,
MTPLX). On clean shutdown they're killed; on a crash they may leak. The
diagnostics snapshot reports `recentOrphanedWorkers` — Phase 7 of the
E2E suite asserts it's empty. To check from a script:

```bash
./scripts/chaosengine-cli diagnostics-snapshot \
    | jq '.recentOrphanedWorkers | length == 0' \
    | grep -q true || echo "Orphans detected"
```

## See also

- [Pre-build check](../testing/pre-build-check.md) — the canonical CI gate.
- [E2E testing](../testing/e2e-testing.md) — the suite this CLI drives.
