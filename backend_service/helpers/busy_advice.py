"""Tell the user when the machine is busy before they start a chat or load a model.

A heavy job (image, video, music, voice, Storyboard batch) already holds the GPU and a large share
of memory. Chat is not in the heavy-job queue, so it starts anyway; it then shares the GPU and every
token is slower. This module turns the running jobs, the footprint of the model about to be used and
the memory figures into one of three verdicts:

* ``ok``     nothing running and the model fits.
* ``slower`` a job is running: the work will share the GPU and be slower, but it fits.
* ``risky``  the estimated combined peak is over 90% of RAM: the message says what to unload or wait for.

It never refuses. Hard refusals stay with ``memory_gate``. The combined peak is an estimate: the
larger of memory in use now and the running jobs' own peaks, plus the new model's footprint.
"""

from __future__ import annotations

from typing import Any

RISKY_SHARE = 0.9

# Peak memory of a running job when the job row carries no figure of its own (``peakGb``). Rough
# catalog medians for Apple Silicon; only used when it is larger than what the machine already uses.
DEFAULT_JOB_PEAK_GB = {"image": 12.0, "video": 24.0, "music": 10.0, "voice": 4.0}

_KIND_NAMES = {"image": "image", "video": "video", "music": "music", "voice": "voice"}
_SOURCE_NAMES = {"ui": "Studio", "agent": "agent", "storyboard": "Storyboard"}


def _one(value: float) -> str:
    return f"{value:.1f}"


def describe_job(job: dict[str, Any]) -> str:
    """'video job "Harbour: s2 clip" (Storyboard)'."""
    kind = _KIND_NAMES.get(str(job.get("kind") or ""), str(job.get("kind") or "heavy"))
    label = str(job.get("label") or "").strip()
    source = _SOURCE_NAMES.get(str(job.get("source") or ""), str(job.get("source") or ""))
    text = f"{kind} job"
    if label:
        text += f' "{label}"'
    if source:
        text += f" ({source})"
    return text


def job_peak_gb(job: dict[str, Any]) -> float:
    value = job.get("peakGb")
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    return DEFAULT_JOB_PEAK_GB.get(str(job.get("kind") or ""), 8.0)


def busy_advice(
    running_jobs: list[dict[str, Any]],
    chat_model_footprint_gb: float | None,
    total_gb: float,
    available_gb: float,
) -> dict[str, Any]:
    """The verdict for starting chat work now. ``chat_model_footprint_gb`` is 0 or None for a resident model."""
    footprint = float(chat_model_footprint_gb) if chat_model_footprint_gb and chat_model_footprint_gb > 0 else 0.0
    in_use = max(0.0, total_gb - available_gb) if total_gb > 0 else 0.0
    job_peaks = sum(job_peak_gb(job) for job in running_jobs)
    combined = footprint + max(in_use, job_peaks)
    jobs = [{"kind": j.get("kind"), "label": j.get("label") or "", "source": j.get("source") or ""} for j in running_jobs]
    base: dict[str, Any] = {
        "jobs": jobs,
        "combinedPeakGb": round(combined, 1),
        "totalGb": round(total_gb, 1),
        "availableGb": round(available_gb, 1),
        "footprintGb": round(footprint, 1),
    }
    named = " and ".join(describe_job(j) for j in running_jobs)
    risky = total_gb > 0 and combined > RISKY_SHARE * total_gb
    if risky:
        what = f"a {named} is running" if len(running_jobs) == 1 else f"{named} are running"
        lead = (
            f"Estimated peak {_one(combined)} GB of {_one(total_gb)} GB RAM"
            f" ({_one(footprint)} GB for this model plus {_one(combined - footprint)} GB already in use)"
            if footprint
            else f"Estimated peak {_one(combined)} GB of {_one(total_gb)} GB RAM"
        )
        if running_jobs:
            advice = (
                f"{lead}: {what}. Wait for it to finish or cancel it, "
                "or unload a model you are not using, before continuing."
            )
        else:
            advice = f"{lead}. Unload a model you are not using or close other apps before continuing."
        return {"level": "risky", "message": advice, **base}
    if running_jobs:
        plural = "" if len(running_jobs) == 1 else "s"
        message = (
            f"The machine is busy: {named} {'is' if len(running_jobs) == 1 else 'are'} running"
            f" and uses about {_one(max(in_use, job_peaks))} GB of {_one(total_gb)} GB RAM. "
            f"Chat shares the GPU with the job{plural}, so expect slower tokens until it finishes."
        )
        return {"level": "slower", "message": message, **base}
    return {"level": "ok", "message": "", **base}


def running_jobs_now() -> list[dict[str, Any]]:
    """The running heavy job, if any (read-only; waiting jobs do not use memory yet)."""
    from backend_service.helpers.job_queue import HEAVY_JOBS

    running = HEAVY_JOBS.snapshot().get("running")
    return [running] if running else []


def current_advice(chat_model_footprint_gb: float | None = None) -> dict[str, Any] | None:
    """Advice from the live queue and memory; None when it cannot be read. Never raises."""
    try:
        import psutil

        from backend_service.helpers.memory_gate import available_memory_gb

        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024 ** 3)
        available = available_memory_gb(memory.total, memory.available)
        return busy_advice(running_jobs_now(), chat_model_footprint_gb, total_gb, available)
    except Exception:  # noqa: BLE001 - advice must never block a chat
        return None
