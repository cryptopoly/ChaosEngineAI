# ChaosEngineAI macOS Release

This document covers the repeatable macOS release path for ChaosEngineAI's native desktop bundle.

## Outputs

- `desktop/releases/macos/ChaosEngineAI.app`
- `desktop/releases/macos/ChaosEngineAI_<version>_aarch64.dmg`

## Recommended command

```bash
cd desktop
npm run release:macos
```

The release scripts automatically read `desktop/.env` and `desktop/.env.local` if present.

The release script performs these steps:

1. Builds the production frontend and native Tauri app bundle.
2. Stages the embedded Python, MLX, and llama.cpp runtime archive.
3. Signs embedded runtime binaries when a signing identity is configured.
4. Copies the built `.app` into `desktop/releases/macos/`.
5. Signs and verifies the `.app` when a signing identity is configured.
6. Creates a distribution DMG with an `Applications` shortcut.
7. Signs, notarizes, staples, and validates the DMG when Apple credentials are configured.

## Signing configuration

Set one of these before running `npm run release:macos`:

- `CHAOSENGINE_APPLE_SIGNING_IDENTITY`
- `APPLE_SIGNING_IDENTITY`

Typical value:

```bash
export CHAOSENGINE_APPLE_SIGNING_IDENTITY="Developer ID Application: Your Company, Inc. (TEAMID1234)"
```

## Notarization configuration

The script supports three authentication styles, in this order:

1. Keychain profile
2. App Store Connect API key
3. Apple ID + app-specific password

### Option 1: keychain profile

Recommended for local release work.

```bash
xcrun notarytool store-credentials chaosengine-notary \
  --apple-id "you@example.com" \
  --team-id "TEAMID1234" \
  --password "app-specific-password"

export CHAOSENGINE_NOTARY_KEYCHAIN_PROFILE="chaosengine-notary"
```

### Option 2: App Store Connect API key

```bash
export CHAOSENGINE_APPLE_API_KEY_ID="ABC123DEFG"
export CHAOSENGINE_APPLE_API_KEY_PATH="/absolute/path/AuthKey_ABC123DEFG.p8"
export CHAOSENGINE_APPLE_API_ISSUER="00000000-0000-0000-0000-000000000000"
```

The script also accepts Tauri-compatible aliases:

- `APPLE_API_KEY`
- `APPLE_API_KEY_PATH`
- `APPLE_API_ISSUER`

### Option 3: Apple ID credentials

```bash
export CHAOSENGINE_APPLE_ID="you@example.com"
export CHAOSENGINE_APPLE_PASSWORD="app-specific-password"
export CHAOSENGINE_APPLE_TEAM_ID="TEAMID1234"
```

## Useful flags

- `--skip-sign`: build unsigned artifacts
- `--skip-notarize`: sign artifacts but skip notarization
- `CHAOSENGINE_SKIP_NOTARIZE=1`: env alternative to `--skip-notarize`

Examples:

```bash
cd desktop
npm run release:macos -- --skip-sign
```

```bash
cd desktop
CHAOSENGINE_SKIP_NOTARIZE=1 npm run release:macos
```

## Release checklist

- Confirm `python3 -m unittest tests/test_backend_service.py` passes.
- Confirm `cd desktop && npm test` passes.
- Confirm `cd desktop && npm run build` passes.
- Confirm `cd desktop/src-tauri && cargo check` passes.
- Run `cd desktop && npm run release:macos`.
- Verify the signed app with `codesign --verify --deep --strict --verbose=2 desktop/releases/macos/ChaosEngineAI.app`.
- Verify Gatekeeper with `spctl --assess --type execute -vv desktop/releases/macos/ChaosEngineAI.app`.
- If notarized, verify stapling with `xcrun stapler validate -v desktop/releases/macos/ChaosEngineAI_<version>_aarch64.dmg`.
- Launch the packaged app and smoke:
  - `GET /api/health`
  - MLX model load and generate
  - GGUF model load and generate
  - GGUF or HF to MLX conversion
- Archive the final `.app` and `.dmg` plus release notes.

## Notes

- The Codex sandbox can block Metal or DMG tooling even when the packaged app works normally on macOS. Final release validation should always use the packaged app outside the sandbox.
- ChaosEngineAI's embedded runtime is delivered as an archive and unpacked on first launch. The release staging step signs embedded Mach-O payloads before that archive is bundled.
