"""FU-042 — FastAPI i18n middleware.

Negotiates a request locale from the ``Accept-Language`` header (set
by the frontend on every fetch), the persisted ``settings.locale``,
and the platform default.  Exposes a ``request.state.t`` callable that
returns translated strings backed by ``babel.support.Translations``.

Per CLAUDE.md §"Adding New Dependencies" + §"Performance" guidelines
``babel`` is lazy-imported — the module top only does ``functools``
+ ``pathlib`` so importing this module doesn't drag Babel into the
mlx_worker / vllm subprocesses that have no need for it.

Coverage: backend translation surface is **user-facing strings only**
— ``HTTPException`` details, ``runtimeNote`` payloads, install action
descriptions.  Log lines stay English-only (devs diagnose) per
FU-042 §Q4 user decision.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Awaitable, Callable, Optional

from fastapi import Request
from starlette.types import ASGIApp

LOCALE_DIR = Path(__file__).resolve().parent / "locales"
DOMAIN = "messages"

SUPPORTED_LOCALES: tuple[str, ...] = (
    "en",
    "zh-CN",
    "zh-TW",
    "ja",
    "de",
    "ru",
    "ko",
    "fr",
    "es",
    "pt-BR",
)

DEFAULT_LOCALE = "en"


def _normalise(raw: Optional[str]) -> Optional[str]:
    """Map a raw BCP-47 tag onto our supported set, or return None."""
    if not raw:
        return None
    lower = raw.strip().lower()
    if not lower:
        return None
    # Direct match (case-insensitive)
    for tag in SUPPORTED_LOCALES:
        if tag.lower() == lower:
            return tag
    # Traditional Chinese family
    if lower in {"zh-tw", "zh-hk", "zh-mo"} or lower.startswith("zh-hant"):
        return "zh-TW"
    if lower in {"zh", "zh-cn", "zh-sg"} or lower.startswith("zh-hans"):
        return "zh-CN"
    if lower in {"pt", "pt-br", "pt-pt"}:
        return "pt-BR"
    if lower == "en" or lower.startswith("en-"):
        return "en"
    # Region-stripped match
    base = lower.split("-", 1)[0]
    for tag in SUPPORTED_LOCALES:
        if tag.lower() == base:
            return tag
    return None


def parse_accept_language(header: Optional[str]) -> list[tuple[str, float]]:
    """Parse RFC-7231 ``Accept-Language`` into ``[(tag, qvalue), ...]``
    sorted high→low.  No external dep — Babel has its own parser but
    we keep this lightweight so the negotiation path stays cheap even
    when Babel isn't imported yet."""
    if not header:
        return []
    out: list[tuple[str, float]] = []
    for raw_chunk in header.split(","):
        chunk = raw_chunk.strip()
        if not chunk:
            continue
        if ";" in chunk:
            tag, _, params = chunk.partition(";")
            tag = tag.strip()
            qvalue = 1.0
            for param in params.split(";"):
                param = param.strip()
                if param.lower().startswith("q="):
                    try:
                        qvalue = float(param[2:])
                    except ValueError:
                        qvalue = 0.0
        else:
            tag = chunk
            qvalue = 1.0
        if tag:
            out.append((tag, qvalue))
    out.sort(key=lambda item: item[1], reverse=True)
    return out


def negotiate_locale(
    accept_language: Optional[str],
    override: Optional[str] = None,
) -> str:
    """Pick the best locale for this request.

    Priority chain (first hit wins):
      1. Explicit ``override`` (e.g. ``settings.locale`` from ChaosEngineState).
      2. ``Accept-Language`` qvalue-ordered candidates.
      3. ``DEFAULT_LOCALE`` (``"en"``).
    """
    override_norm = _normalise(override)
    if override_norm:
        return override_norm
    for raw_tag, _qvalue in parse_accept_language(accept_language):
        match = _normalise(raw_tag)
        if match:
            return match
    return DEFAULT_LOCALE


@functools.lru_cache(maxsize=len(SUPPORTED_LOCALES) + 4)
def _load_translations(locale: str):
    """Cache ``babel.support.Translations`` per locale.  Returns the
    English ``NullTranslations`` for missing catalogs — fallback to the
    ``en`` source string preserves the FU-042 "ship en-only, follow-up
    fills" workflow without breaking user-facing surfaces."""
    # Lazy import: keeps Babel out of the import chain for callers
    # that never localize (mlx_worker, vllm, ddtree, etc.).
    from babel.support import NullTranslations, Translations

    if locale == DEFAULT_LOCALE:
        return NullTranslations()
    try:
        return Translations.load(str(LOCALE_DIR), [locale, DEFAULT_LOCALE], DOMAIN)
    except Exception:
        return NullTranslations()


def translator_for(locale: str) -> Callable[[str], str]:
    """Return a ``t(message)`` callable for the given locale.

    Usage in routes::

        @router.post("/api/foo")
        async def foo(request: Request):
            t = request.state.t
            raise HTTPException(400, t("Not enough disk space"))

    The English message string is also the gettext key — keeps `.po`
    file authoring simple and lets the source string survive even if
    the catalog is missing.
    """
    trans = _load_translations(locale)
    return trans.gettext


def ntranslator_for(locale: str) -> Callable[[str, str, int], str]:
    """Pluralized form: ``tn(singular, plural, n)``."""
    trans = _load_translations(locale)
    return trans.ngettext


class I18nMiddleware:
    """Pure-ASGI middleware that stashes the negotiated locale +
    translation callables on ``request.state``.  Reads
    ``app.state.chaosengine.settings.locale`` when present for the
    per-user override (set via the Settings tab → Language dropdown).
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        accept_language = headers.get("accept-language")
        override = None
        app = scope.get("app")
        try:
            override = app.state.chaosengine.settings.locale  # type: ignore[attr-defined]
        except AttributeError:
            override = None
        # ``"system"`` sentinel means "no explicit override; auto-detect".
        if override == "system":
            override = None
        locale = negotiate_locale(accept_language, override)

        scope.setdefault("state", {})
        scope["state"]["locale"] = locale
        scope["state"]["t"] = translator_for(locale)
        scope["state"]["tn"] = ntranslator_for(locale)
        await self.app(scope, receive, send)


def install(app) -> None:
    """Register the middleware on a FastAPI app.  Idempotent."""
    if getattr(app.state, "_i18n_installed", False):
        return
    app.add_middleware(I18nMiddleware)
    app.state._i18n_installed = True


def localized_detail(request, raw_message: str, fallback_key: Optional[str] = None) -> dict[str, str]:
    """Build the localized error envelope used by user-facing HTTP error
    responses.

    Returns a dict of shape ``{"message": <english>, "localized":
    <translated or english>, "errorKey": <optional canonical key>}``.
    Routes call this in their ``raise HTTPException(detail=...)`` site
    so the React layer can prefer ``localized`` when present and fall
    back to ``message`` if the catalog hasn't been authored yet.

    Example::

        @router.post("/api/chat/sessions/{session_id}/delve")
        def delve(request: Request, ...):
            try:
                ...
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=localized_detail(request, str(exc)),
                ) from exc

    The frontend ``api`` layer's error-handling path reads
    ``detail.localized`` when the response body is a JSON object with
    that shape; otherwise it falls back to the legacy plain-string
    ``detail`` format so older routes keep working.
    """
    locale = getattr(request.state, "locale", DEFAULT_LOCALE)
    t = getattr(request.state, "t", None)
    localized = raw_message
    if t is not None and fallback_key is not None:
        candidate = t(fallback_key)
        if candidate and candidate != fallback_key:
            localized = candidate
    elif t is not None:
        candidate = t(raw_message)
        if candidate and candidate != raw_message:
            localized = candidate
    envelope: dict[str, str] = {
        "message": raw_message,
        "localized": localized,
        "locale": locale,
    }
    if fallback_key is not None:
        envelope["errorKey"] = fallback_key
    return envelope


__all__ = [
    "DEFAULT_LOCALE",
    "I18nMiddleware",
    "SUPPORTED_LOCALES",
    "install",
    "localized_detail",
    "negotiate_locale",
    "ntranslator_for",
    "parse_accept_language",
    "translator_for",
]
