# Adding a feature

The discoverability invariant: **every feature ships with at least one
E2E check**. The pre-build gate enforces this — phase 9 runs the E2E
smoke suite, and a feature without a check is invisible to that smoke.

This page is the checklist to follow when landing a feature, big or
small.

## 1. Design the change

State your assumptions explicitly. If multiple interpretations exist,
present them — don't pick silently. If a simpler approach exists, say
so. If something is unclear, stop and ask before writing code.

For multi-step work, write a brief plan and verify each step:

```
1. Add Pydantic model field        → verify: tests/test_payloads pass
2. Wire field through controller   → verify: pytest tests/test_inference passes
3. Add CLI flag                    → verify: scripts/chaosengine-cli call ... returns 200
4. Add E2E check                   → verify: e2e_test_suite --phases <N> green
5. Update docs                     → verify: mkdocs build --strict
```

## 2. Implement surgically

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it in the PR — don't
  delete it.

The test: every changed line should trace directly to the user's
request.

## 3. Add tests

Pick the right level:

- **Pure-function helper** → `pytest` unit test.
- **FastAPI route shape** → `pytest tests/test_backend_service.py`
  with `FakeRuntime`.
- **Engine adapter** → `pytest tests/test_inference.py` with mocked
  subprocess.
- **Cache strategy contract** → `pytest tests/test_cache_strategies.py`.
- **Frontend logic** → `vitest` test next to the component / hook.
- **Real-engine behaviour** → E2E check in the right phase.

If a change touches `inference/controller.py` or an engine
implementation, add **both** a unit test (mocked) **and** an E2E
check (real subprocess).

See [Adding checks](../testing/adding-checks.md) for the E2E pattern.

## 4. Update the catalog / registry

If you're adding a model family:

- Add the canonical repo + community quants to
  `backend_service/catalog/text_models.py` (or `image_models.py` /
  `video_models.py`).
- If a DFlash drafter exists, add it to `dflash/__init__.py`'s
  `DRAFT_MODEL_MAP` and community variants to `_ALIASES`.
- If the model has MTP heads, add it to
  `backend_service/inference/_mtp.py`'s `MTP_MODEL_MAP` and aliases.
- Pin unit tests against the new mappings.

## 5. Update the docs

If the feature is user-visible:

- Add a paragraph to the relevant `usage/*.md` page.
- If it's a CLI flag, mention it in `cli/overview.md` and the
  [reference](../cli/reference.md).
- If it's a runtime knob, mention it in `features/*.md`.

If the feature changes the architecture (a new engine, a new
subprocess, a new dependency):

- Update `architecture/inference-engines.md` or whichever architecture
  page applies.
- Add the dependency to `THIRD_PARTY_NOTICES.md` and
  `reference/third-party-deps.md`.
- If it ships a new environment variable, add it to
  `reference/env-vars.md`.

Then build the docs locally:

```bash
.venv/bin/mkdocs build --strict
```

`--strict` fails on broken links, missing pages, and orphan files.

## 6. Run the full check

```bash
./scripts/pre-build-check.sh
```

This runs:

1. Python tests
2. TypeScript tests
3. TypeScript type-check
4. Licence notices
5. Cache strategy validation
6. Upstream dependency check
7. Binary availability
8. i18n locale validation
9. E2E smoke

Everything must be green before merging.

## 7. Write a commit message

```
docs: short summary of what changed

Longer explanation if needed. Use bullet points for multi-part
changes. Reference issues or PRs by number.
```

Conventional prefixes: `feat:`, `fix:`, `docs:`, `refactor:`,
`perf:`, `test:`, `chore:`, `ci:`. No `Co-Authored-By` lines.

## 8. Open the PR

The PR description should answer:

- **What changed.** Files touched + the high-level intent.
- **Why.** What problem this solves; reference any tracking issue.
- **How to test.** A few CLI commands or click-paths a reviewer can
  follow.
- **Risks.** What could break; what's covered by tests vs left to
  manual QA.

Smaller, focused PRs are easier to review than mega-PRs. If a change
touches > 10 files outside the one feature, ask whether it should be
split.

## See also

- [Coding guidelines](coding-guidelines.md)
- [Adding checks](../testing/adding-checks.md)
- [Development setup](development-setup.md)
- [Pre-build check](../testing/pre-build-check.md)
