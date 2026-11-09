# RTX 4090 test progress — v0.10.0

Windows native test run of `release/v0.10.0` on branch `test/rtx4090`. One line of evidence per item, marked PASS / FAIL / SKIP. Plan: [rtx4090-test.md](../plans/rtx4090-test.md).

## Machine

| Field | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 |
| Driver | 591.86 |
| VRAM | 24564 MiB (24 GB) |
| RAM | 63.8 GB |
| Windows build | 10.0.26300 (26300) |
| Python (venv) | 3.12.10 (3.11 not installed; 3.12 is in the test matrix) |
| Free disk C: | 81.5 GB at start |
| CUDA (torch) | torch 2.11.0+cu128, torchvision 0.26.0+cu128, `cuda.is_available()` True, GPU matmul OK |

## 0. Setup

| Item | Result | Evidence |
|---|---|---|
| Fetch + check out `release/v0.10.0`, branch `test/rtx4090` | PASS | `git checkout -b test/rtx4090` from `release/v0.10.0` (up to date with origin); on `test/rtx4090`. |
| Record GPU, driver, VRAM, RAM, Windows build, Python, free disk | PASS | See Machine table above. |
| Create venv + install per README (CUDA extras, `-c release-constraints.txt`) | PASS | `pip install -e .[desktop,images,dev,ffmpeg] -c release-constraints.txt` (exit 0). Deviations: (1) Python 3.11 not on box → used 3.12.10. (2) PyPI torch is CPU-only on Windows; constraints pin torch==2.11.0, which has no cu124 wheel (cu124 tops at 2.6.0) → reinstalled `torch==2.11.0 torchvision==0.26.0` from the cu128 index (`--no-deps`), matching the Setup route's cu124→cu126→cu128 fallback. vLLM/mlx/triattention extras skipped (no Windows wheels). |
| Backend starts, `/api/health` answers | PASS | `python -m backend_service.app` → `http://127.0.0.1:8876/api/health` returns `{"status":"ok","appVersion":"0.10.0",...,"ggufAvailable":true,"mlxAvailable":false}`. llama-server resolved; mlx false as expected on Windows. |

## 1. Automated checks (Windows native)

| Item | Result | Evidence |
|---|---|---|
| `pytest tests/ -q` | _pending_ | |
| `npm test` | _pending_ | |
| `npx tsc --noEmit` | _pending_ | |
| `npm run i18n:validate` | _pending_ | |
| `e2e_test_suite.py --smoke` | _pending_ | |
| `e2e_test_suite.py` (full) | _pending_ | |
