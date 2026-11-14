"""Territory statement routes: status, confirm, withdraw (D-76).

Confirm and withdraw are accepted only from the app's own window on this computer: a call with an agent token,
through ``/mcp`` or from another machine is refused, so no agent can tick the statement for the user.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend_service import licence_gate

router = APIRouter()

UI_ONLY_MESSAGE = "The territory statement can only be confirmed in the ChaosEngineAI window on this computer."


class LicenceGateConfirmRequest(BaseModel):
    key: str = Field(min_length=1, max_length=64)
    confirmed: bool
    sentence: str = Field(min_length=1, max_length=1000)


class LicenceGateWithdrawRequest(BaseModel):
    key: str = Field(min_length=1, max_length=64)


def _ui_only(request: Request) -> None:
    from backend_service.voice_clone_safety import request_origin

    if request_origin(request) != "ui":
        raise HTTPException(status_code=403, detail=UI_ONLY_MESSAGE)


@router.get("/api/licence-gate")
def licence_gate_status() -> dict[str, Any]:
    return licence_gate.status()


@router.post("/api/licence-gate/confirm")
def confirm_licence_gate(request: Request, body: LicenceGateConfirmRequest) -> dict[str, Any]:
    _ui_only(request)
    if body.confirmed is not True:
        raise HTTPException(status_code=400, detail="Tick the statement to confirm it.")
    from backend_service.app import app_version

    try:
        licence_gate.confirm(body.key, sentence=body.sentence, app_version=app_version)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown territory-limited model '{body.key}'.") from None
    except licence_gate.LicenceGateError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return licence_gate.status()


@router.post("/api/licence-gate/withdraw")
def withdraw_licence_gate(request: Request, body: LicenceGateWithdrawRequest) -> dict[str, Any]:
    _ui_only(request)
    try:
        licence_gate.withdraw(body.key)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown territory-limited model '{body.key}'.") from None
    return licence_gate.status()
