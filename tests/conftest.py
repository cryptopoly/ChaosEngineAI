"""pytest collection-time hook to make tests resolve against the
installed ChaosEngineAI app's extras dir.

The dev ``.venv`` deliberately ships **without** torch / diffusers /
mlx / vllm / nunchaku / sageattention / triattention. Those packages
live in the persistent extras directory the desktop app populates via
``/api/setup/install-gpu-bundle`` and friends — the same path the
production embedded runtime puts on ``PYTHONPATH`` at backend launch.

Importing torch in the dev venv would fork the install state from
what real users run. So instead of asking developers to ``pip install``
into ``.venv``, this conftest reuses the production extras dir at
collection time so:

  - Tests that touch ``torch`` / ``diffusers`` etc. resolve them
    against the same wheels the user's actual app uses.
  - "No custom test setup" — open the app, run ``pytest``, you're
    testing the production install.
  - A torch upgrade landing via the in-app installer is reflected in
    the next pytest run automatically.
  - CI boxes without the extras dir get a silent no-op: tests that
    require torch will still fail in the same place they did before
    (the import line), but tests that don't need it run normally.

The append-vs-prepend decision is delegated to
``ensure_extras_on_sys_path`` — repo-local shims (notably the
``turboquant_mlx`` adapter that wraps the upstream
``turboquant-mlx-full`` install) must keep import authority over the
raw upstream packages, so the helper appends rather than prepends.

This is a pytest-native conftest, not a fixture. The side effect runs
once when pytest collects ``tests/``, before any test module imports.
"""

from __future__ import annotations

import os
import sys

# We import the helper through ``backend_service`` so the editable
# install of this repo (``pip install -e .``) is what provides the
# import path. No special bootstrap needed — pytest's rootdir handling
# already finds ``backend_service`` via the installed package.
from backend_service.runtime_paths import ensure_extras_on_sys_path


_INSERTED = ensure_extras_on_sys_path()


# Surface what we wired in via ``-v -s`` so CI logs and local
# debugging make it obvious which extras dir the run pulled from.
# Silent in the default ``-q`` output so it doesn't add noise.
if _INSERTED and os.environ.get("CHAOSENGINE_TEST_TRACE_EXTRAS"):
    print(
        f"[conftest] appended extras to sys.path: {[str(p) for p in _INSERTED]}",
        file=sys.stderr,
        flush=True,
    )
