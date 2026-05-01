; Tauri 2 NSIS installer hooks for the Windows ChaosEngineAI bundle.
;
; Tauri's default NSIS template installs the app under
; %LOCALAPPDATA%\<identifier>\ and the uninstaller removes that tree on
; uninstall. The GPU runtime bundle (torch + diffusers + transformers,
; ~2.5 GB) is intentionally written to a sibling directory:
;
;     %LOCALAPPDATA%\ChaosEngineAI\extras\cp{major}{minor}\site-packages
;
; The path is namespaced by Python ABI tag (commit 24518af, v0.7.0-rc.5)
; so a runtime upgrade that changes Python minor versions cannot shadow
; the wheels from the previous tag.
;
; CRITICAL: this directory MUST survive an uninstall + reinstall cycle.
; Re-downloading 2.5 GB of CUDA wheels every time the user upgrades the
; desktop app is unacceptable, both for users on slow links and for the
; PyPI mirrors that serve the bundle.
;
; The hooks below are intentionally empty as a guardrail. If anyone
; later adds custom uninstall behaviour:
;
;   1. NEVER ``RMDir /r "$LOCALAPPDATA\ChaosEngineAI\extras"`` here.
;   2. Test that ``setup.py:_extras_site_packages()`` resolves the same
;      path before AND after a clean uninstall + reinstall on Windows.
;   3. Mirror any change in ``src-tauri/src/lib.rs::chaosengine_extras_root``.

!macro NSIS_HOOK_PREINSTALL
  ; Reserved — currently a no-op. See contract above before adding code.
!macroend

!macro NSIS_HOOK_POSTINSTALL
  ; Reserved — currently a no-op. See contract above before adding code.
!macroend

!macro NSIS_HOOK_PREUNINSTALL
  ; Reserved — currently a no-op. See contract above before adding code.
!macroend

!macro NSIS_HOOK_POSTUNINSTALL
  ; Reserved — currently a no-op. The persistent GPU runtime tree at
  ; %LOCALAPPDATA%\ChaosEngineAI\extras MUST be left intact so an
  ; immediate reinstall can pick it up without re-downloading 2.5 GB.
  ; See contract above before adding code.
!macroend
