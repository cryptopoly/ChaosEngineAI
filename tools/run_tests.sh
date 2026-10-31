#!/bin/zsh
# Hub checks for ChaosEngineAI. The product's own tests are listed in docs/hub/testing.md
# (pytest, vitest, tsc, E2E, pre-build-check); pass --full to run the quick ones too.
# Rules: tests never touch the user's real settings or data; one PASS/FAIL line per test;
# non-zero exit if any failed.
emulate -L zsh
cd "${0:A:h}/.."
fails=0
run() {  # run <name> <command...>
  local name=$1; shift
  if "$@" > /tmp/test_$name.log 2>&1; then echo "PASS $name"; else echo "FAIL $name (log /tmp/test_$name.log)"; fails=$((fails + 1)); fi
}
run hub_tracker_current python3 tools/hub_tracker.py --check
run hub_links python3 tools/build_hub.py --check --strict
if [[ "$1" == "--full" ]]; then
  run pytest .venv/bin/python -m pytest tests/ -q
  run vitest npm test --silent
  run tsc npx tsc --noEmit
fi
echo "$fails failed"
exit $fails
