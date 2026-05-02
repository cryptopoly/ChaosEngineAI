"""Phase 3.5: cross-platform per-turn perf telemetry snapshot.

Captures a small bundle of system-side metrics (CPU %, GPU %,
thermal state, available memory) at chat-turn finalisation time so
the frontend can render a compact perf strip below each assistant
response without making a separate round-trip.

Backed by:
- macOS: psutil + pmset thermal probe (already used by the watchdog
  stack — Phase 2.0.5-I)
- Linux: psutil + best-effort GPU sampler. Thermal stays None
  because there's no portable read; future iteration could surface
  /sys/class/thermal_zone* readings.
- Windows: psutil + best-effort NVML / pdh.dll counter (deferred —
  returns None for now).

Best-effort everywhere: any sampler error falls through to None
fields so the UI degrades gracefully.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class PerfTelemetry:
    cpuPercent: float | None = None
    gpuPercent: float | None = None
    thermalState: str | None = None
    availableMemoryGb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def is_empty(self) -> bool:
        return all(
            v is None for v in (
                self.cpuPercent,
                self.gpuPercent,
                self.thermalState,
                self.availableMemoryGb,
            )
        )


def snapshot_perf_telemetry() -> PerfTelemetry:
    """Sample current host telemetry. Always returns a PerfTelemetry —
    fields default to None when the underlying probe fails. Cheap to
    call: no subprocess fork unless thermal is read on Darwin (which
    re-uses the watchdog's pmset call).
    """
    telemetry = PerfTelemetry()

    # CPU + memory via psutil — universally available.
    try:
        import psutil  # noqa: WPS433 — local import keeps boot lean

        # interval=None = non-blocking sample using the rolling baseline
        # psutil maintains since import. First call returns 0; subsequent
        # calls reflect the delta since the last sample. The chat path
        # has been running long enough that the baseline is warm.
        telemetry.cpuPercent = round(psutil.cpu_percent(interval=None), 1)
        vm = psutil.virtual_memory()
        telemetry.availableMemoryGb = round(vm.available / (1024 ** 3), 2)
    except Exception:
        # Any psutil failure → leave as None. Telemetry strip will
        # render only the fields that are present.
        pass

    # Thermal — Darwin only today, re-uses Phase 2.0.5-I sampler.
    try:
        from backend_service.helpers.thermal import read_thermal_state

        telemetry.thermalState = read_thermal_state()
    except Exception:
        pass

    # GPU utilisation — best-effort, falls back to None on platforms
    # without a known sampler. The dashboard's _detect_gpu_utilization
    # already covers macOS Metal + NVML, so re-use it.
    try:
        from backend_service.helpers.system import _detect_gpu_utilization

        telemetry.gpuPercent = _detect_gpu_utilization()
    except Exception:
        pass

    return telemetry
