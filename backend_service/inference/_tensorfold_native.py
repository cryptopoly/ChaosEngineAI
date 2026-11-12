"""TensorFold 1.x native engine: install layout, capability gate, model registry.

TensorFold 1.x (https://github.com/ashhart/TensorFold, Apache-2.0) is a single
native binary (``tensorfold-native serve``) with CUDA and Metal back ends. This
module is pure data plus small probes (no heavy imports) so the installer
route, the controller, the engine and the system-stats payload share one view:

* where the binary lives (``find_binary``) and which release archive belongs to
  this platform (``release_asset``);
* whether *this* machine can run it (``evaluate_gate``): GPU compute
  capability and free VRAM from ``nvidia-smi``, driver floor, and the binary's
  own ``capabilities --json`` (chip, back ends, families, serve flags);
* which models it can serve and what they need (``NATIVE_MODELS``,
  ``estimate_vram_gb``, ``supported_models``).

The gate never names a card. It reads numbers and the binary's own verdict, so a
12 GB card, a 4090, a 5090 and a DGX Spark are each judged on what they report.
The Mac Python engine (0.6.6, ``_tensorfold.py``) is a separate lane and is not
touched here.
"""

from __future__ import annotations

import json
import os
import platform as _platform
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

NATIVE_VERSION = "1.2.7"
RELEASE_BASE = "https://github.com/ashhart/TensorFold/releases/download"

# Platform key -> archive name. Windows has no release archive (1.2.7 ships
# Linux x86_64 / aarch64 and macOS arm64 only): its binary is a source build
# the user adopts, or a self-hosted archive named by the two env vars below.
_ASSETS: dict[str, str] = {
    "linux-x86_64": "tensorfold-{version}-linux-x86_64.tar.gz",
    "linux-aarch64": "tensorfold-{version}-linux-aarch64.tar.gz",
    "macos-arm64": "tensorfold-{version}-macos-arm64.tar.gz",
}
WINDOWS_ARCHIVE_URL_ENV = "CHAOSENGINE_TENSORFOLD_WINDOWS_URL"
WINDOWS_ARCHIVE_SHA_ENV = "CHAOSENGINE_TENSORFOLD_WINDOWS_SHA256"
BINARY_ENV = "CHAOSENGINE_TENSORFOLD_NATIVE"

# The driver floor the server enforces at startup (CUDA 13.0 libcuda).
MIN_DRIVER_MAJOR = 580
# Compute capabilities the release fatbins carry images for. Other Ampere or
# newer cards are "unqualified": offered with a warning, and confirmed by the
# binary's own chip report once installed.
RELEASE_IMAGE_CC = frozenset({"8.0", "8.6", "12.0", "12.1"})
MIN_COMPUTE_CAPABILITY = (8, 0)

_BINARY_NAMES = ("tensorfold-native.exe", "tensorfold-native")


def uses_native_engine() -> bool:
    """True off the Mac: the CUDA lane runs the 1.x native binary, not the Python engine."""
    return sys.platform != "darwin"


def install_root() -> Path:
    return Path.home() / ".chaosengine" / "tensorfold-native"


def install_record_path() -> Path:
    return install_root() / "install.json"


def platform_key(system: str | None = None, machine: str | None = None) -> str | None:
    """``windows-x86_64`` / ``linux-x86_64`` / ``linux-aarch64`` / ``macos-arm64``, or ``None``."""
    system = (system or sys.platform).lower()
    machine = (machine or _platform.machine()).lower()
    arch = {"amd64": "x86_64", "x86_64": "x86_64", "arm64": "arm64", "aarch64": "aarch64"}.get(machine)
    if arch is None:
        return None
    if system.startswith("win"):
        return "windows-x86_64" if arch == "x86_64" else None
    if system.startswith("linux"):
        return {"x86_64": "linux-x86_64", "aarch64": "linux-aarch64", "arm64": "linux-aarch64"}.get(arch)
    if system == "darwin":
        return "macos-arm64" if arch == "arm64" else None
    return None


@dataclass(frozen=True)
class ReleaseAsset:
    platform: str
    name: str
    url: str
    sha256_url: str | None
    sha256: str | None  # set when the digest comes from the environment (Windows)


def release_asset(key: str | None, *, version: str = NATIVE_VERSION, environ: dict[str, str] | None = None) -> ReleaseAsset | None:
    """The archive to download for ``key``, or ``None`` when none exists."""
    env = os.environ if environ is None else environ
    if key in _ASSETS:
        name = _ASSETS[key].format(version=version)
        url = f"{RELEASE_BASE}/v{version}/{name}"
        return ReleaseAsset(key, name, url, f"{url}.sha256", None)
    if key == "windows-x86_64":
        url = (env.get(WINDOWS_ARCHIVE_URL_ENV) or "").strip()
        digest = (env.get(WINDOWS_ARCHIVE_SHA_ENV) or "").strip().lower()
        if url.startswith("https://") and re.fullmatch(r"[0-9a-f]{64}", digest):
            return ReleaseAsset(key, url.rsplit("/", 1)[-1] or "tensorfold-windows.zip", url, None, digest)
    return None


def parse_sha256_file(text: str, name: str | None = None) -> str | None:
    """The digest from a ``.sha256`` file (``<hex>  <name>`` or a bare hex line)."""
    for line in text.splitlines():
        parts = line.strip().split()
        if not parts or not re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]):
            continue
        if name is None or len(parts) == 1 or parts[-1].lstrip("*").endswith(name):
            return parts[0].lower()
    return None


def read_install_record() -> dict[str, Any] | None:
    try:
        data = json.loads(install_record_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def find_binary(environ: dict[str, str] | None = None) -> Path | None:
    """The ``tensorfold-native`` binary: env override first, then the install record."""
    env = os.environ if environ is None else environ
    override = (env.get(BINARY_ENV) or "").strip()
    if override:
        candidate = Path(override).expanduser()
        return candidate if candidate.is_file() else None
    record = read_install_record()
    if record and isinstance(record.get("binary"), str):
        candidate = install_root() / record["binary"]
        try:
            resolved = candidate.resolve()
            resolved.relative_to(install_root().resolve())
        except (OSError, ValueError):
            return None
        return resolved if resolved.is_file() else None
    return None


def locate_binary_in(directory: Path) -> Path | None:
    """The binary inside an unpacked archive (searched, so layout changes don't matter)."""
    for name in _BINARY_NAMES:
        for found in sorted(directory.rglob(name)):
            if found.is_file():
                return found
    return None


# ---------------------------------------------------------------------------
# GPU and binary probes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GpuInfo:
    name: str
    compute_capability: str  # "8.9"
    driver_version: str  # "621.14"
    total_mb: int
    free_mb: int
    cuda_version: str | None = None  # the driver's CUDA level, from the nvidia-smi header

    @property
    def cc_tuple(self) -> tuple[int, int] | None:
        match = re.fullmatch(r"(\d+)\.(\d+)", self.compute_capability.strip())
        return (int(match.group(1)), int(match.group(2))) if match else None

    @property
    def driver_major(self) -> int | None:
        match = re.match(r"(\d+)", self.driver_version.strip())
        return int(match.group(1)) if match else None


def parse_nvidia_smi(csv_text: str, header_text: str = "") -> GpuInfo | None:
    """First GPU row of ``nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,memory.free``."""
    for line in csv_text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            total = int(float(parts[3].split()[0]))
            free = int(float(parts[4].split()[0]))
        except (ValueError, IndexError):
            continue
        cuda = re.search(r"CUDA Version:\s*([0-9.]+)", header_text)
        return GpuInfo(parts[0], parts[1], parts[2], total, free, cuda.group(1) if cuda else None)
    return None


def read_gpu(timeout: float = 10.0) -> GpuInfo | None:
    """Query ``nvidia-smi`` (list-form subprocess). ``None`` when there is no NVIDIA driver."""
    query = "name,compute_cap,driver_version,memory.total,memory.free"
    try:
        csv = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
        if csv.returncode != 0:
            return None
        header = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_nvidia_smi(csv.stdout, header.stdout if header.returncode == 0 else "")


@dataclass(frozen=True)
class NativeCapabilities:
    version: str | None
    chip: str | None
    backends: tuple[str, ...]
    families: dict[str, tuple[str, ...]]
    flags: frozenset[str]

    def supports_flag(self, flag: str) -> bool:
        return flag in self.flags


def parse_capabilities(text: str) -> NativeCapabilities | None:
    """Parse ``tensorfold-native capabilities --json`` output."""
    try:
        data = json.loads(text)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    families = {
        str(key): tuple(str(v) for v in value)
        for key, value in (data.get("families") or {}).items()
        if isinstance(value, list)
    }
    serve = data.get("serve") if isinstance(data.get("serve"), dict) else {}
    flags = serve.get("flags") if isinstance(serve.get("flags"), dict) else {}
    return NativeCapabilities(
        version=str(data["version"]) if data.get("version") else None,
        chip=str(data["chip"]) if data.get("chip") else None,
        backends=tuple(str(b) for b in (data.get("backends") or [])),
        families=families,
        flags=frozenset(str(k) for k in flags),
    )


def probe_capabilities(binary: Path, timeout: float = 30.0) -> NativeCapabilities | None:
    try:
        result = subprocess.run(
            [str(binary), "capabilities", "--json"],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_capabilities(result.stdout) if result.returncode == 0 else None


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NativeModel:
    key: str
    title: str
    repos: tuple[str, ...]
    # ``config.json`` ``model_type`` and the weight format the family needs, as
    # ``capabilities --json`` ``families`` names them.
    family: str
    weight_format: str
    drafter: str | None = None
    vision: bool = False
    # Measured (27B) or documented total VRAM in GB at ``ctx_ref`` tokens, one
    # stream, with the drafter when there is one, and the KV share per ``ctx_ref``.
    vram_gb: float = 0.0
    kv_gb_per_ref: float = 0.0
    drafter_gb: float = 0.0
    ctx_ref: int = 8192
    measured: bool = False
    note: str = ""


NATIVE_MODELS: tuple[NativeModel, ...] = (
    NativeModel(
        key="qwen3_8_27b",
        title="Qwen3.8-27B + DFlash2",
        repos=("TensorFold/Qwen3.8-27B-MLX-4bit", "Vontra/Qwen3.8-27B-MLX-4bit"),
        family="qwen3_5",
        weight_format="mlx-q4g64",
        drafter="z-lab/Qwen3.8-27B-DFlash2",
        vision=True,
        # RTX 4090 run: 18.8 GB on the card at 8192 tokens with the drafter
        # (16.06 GiB weights, 1.46 GiB KV per stream), 47-49 tok/s plain,
        # 147-151 drafted on prose, about 300 on code.
        vram_gb=18.8,
        kv_gb_per_ref=1.57,
        drafter_gb=3.84,
        measured=True,
    ),
    NativeModel(
        key="qwen3_5_2b",
        title="Qwen3.5-2B",
        repos=("mlx-community/Qwen3.5-2B-MLX-4bit",),
        family="qwen3_5",
        weight_format="mlx-q4g64",
        vram_gb=3.1,  # 1.58 GiB model + 0.17 GiB KV at 8192 + runtime; 457-461 tok/s on a 4090
        kv_gb_per_ref=0.18,
        measured=True,
    ),
    NativeModel(
        key="nemotron_lightning",
        title="Nemotron 3.5 Lightning 30B-A3B",
        repos=("TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit",),
        family="nemotron_h",
        weight_format="mlx-q4g64",
        vram_gb=19.5,  # estimate: ~17 GB 4-bit weights + KV; the README lists RTX 3090 as checked
        kv_gb_per_ref=0.4,
        note="Documented on RTX 3090 and GB10; size is an estimate.",
    ),
    NativeModel(
        key="nemotron_bolt",
        title="Nemotron 3.5 Bolt 74B-A9B (24 GB checkpoint)",
        repos=("nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB",),
        family="nemotron_h",
        weight_format="modelopt-nvfp4",
        vram_gb=26.0,  # estimate: the 24 GB checkpoint plus KV; checked on an RTX PRO 6000
        kv_gb_per_ref=0.6,
        note="Needs more than 24 GB of free VRAM.",
    ),
    NativeModel(
        # CUDA since 1.2.5 (GB10 checked), from the MLX 4-bit export only; no
        # drafter, no prompt cache on CUDA. The figure is the binary's own
        # `info` memory floor: 12.65 GiB of weights plus its 2.00 GiB margin.
        key="lfm2_5_24b_a2b",
        title="LFM2.5-24B-A2B",
        repos=("LiquidAI/LFM2.5-24B-A2B-MLX-4bit",),
        family="lfm2_moe",
        weight_format="mlx-q4g64",
        vram_gb=15.7,
        note=(
            "Size is TensorFold's own memory floor (weights plus a 2 GiB margin); KV growth with the window "
            "is not documented or measured here. Checked upstream on a GB10, not on a discrete card."
        ),
    ),
)


def estimate_vram_gb(model: NativeModel, context: int, *, drafts: bool = True) -> float:
    """VRAM (GB) one stream needs: measured total at the reference window, rescaled for ``context``."""
    context = max(int(context or model.ctx_ref), 512)
    kv_delta = model.kv_gb_per_ref * (context / model.ctx_ref - 1.0)
    total = model.vram_gb + kv_delta
    if model.drafter and not drafts:
        total -= model.drafter_gb
    return round(max(total, 0.5), 1)


def match_native_model(*repos: str | None) -> NativeModel | None:
    wanted = {repo.lower() for repo in repos if repo}
    for model in NATIVE_MODELS:
        if wanted & {repo.lower() for repo in model.repos}:
            return model
    return None


def supported_models(caps: NativeCapabilities | None, *, free_gb: float | None = None) -> list[dict[str, Any]]:
    """The registry joined with the binary's own family list, never hard-coded per card."""
    rows: list[dict[str, Any]] = []
    for model in NATIVE_MODELS:
        formats = caps.families.get(model.family, ()) if caps else ()
        servable = bool(caps) and model.weight_format in formats
        need = estimate_vram_gb(model, model.ctx_ref)
        rows.append(
            {
                "key": model.key,
                "title": model.title,
                "repo": model.repos[0],
                "drafter": model.drafter,
                "vision": model.vision,
                "servable": servable,
                "vramGb": need,
                "fits": None if free_gb is None else need <= free_gb,
                "measured": model.measured,
                "note": model.note,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateResult:
    # "ready" (binary confirms CUDA on this card), "unqualified" (offered, not
    # confirmed), "needs-binary" (hardware fine, nothing installed yet that
    # runs here) or "refused".
    status: str
    message: str
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def offered(self) -> bool:
        return self.status != "refused"

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "offered": self.offered, "message": self.message, "reasons": list(self.reasons)}


def evaluate_gate(
    gpu: GpuInfo | None,
    caps: NativeCapabilities | None,
    *,
    key: str | None,
    asset_available: bool,
    binary_present: bool,
    smallest_model_gb: float | None = None,
) -> GateResult:
    """Decide whether the native engine is offered on this machine, and say why not.

    Apple Silicon goes through the Metal build and needs no GPU probe. Everywhere
    else the engine is CUDA: a visible NVIDIA GPU, an R580+ driver, Ampere or
    newer, and, once a binary exists, the binary's own report of a CUDA chip.
    """
    if key == "macos-arm64":
        if caps is not None and "metal" not in caps.backends:
            return GateResult("refused", "This TensorFold build does not report the Metal back end.")
        return GateResult("ready" if caps else "needs-binary", "Apple Silicon runs the Metal build.")
    if key is None:
        return GateResult("refused", f"TensorFold ships no build for {sys.platform}/{_platform.machine()}.")
    if gpu is None:
        return GateResult("refused", "No NVIDIA GPU or driver found (nvidia-smi is missing or reports nothing). TensorFold's CUDA engine needs one.")

    reasons: list[str] = []
    major = gpu.driver_major
    if major is None or major < MIN_DRIVER_MAJOR:
        return GateResult(
            "refused",
            f"NVIDIA driver {gpu.driver_version} is older than R{MIN_DRIVER_MAJOR}; TensorFold needs a driver with CUDA 13.0 "
            f"(R{MIN_DRIVER_MAJOR} or newer). Update the driver and try again.",
        )
    cc = gpu.cc_tuple
    if cc is None:
        return GateResult("refused", f"Could not read the compute capability of {gpu.name}; not offering TensorFold on an unknown architecture.")
    if cc < MIN_COMPUTE_CAPABILITY:
        return GateResult(
            "refused",
            f"{gpu.name} is compute capability {gpu.compute_capability}; TensorFold's CUDA kernels need Ampere (8.0) or newer.",
        )
    free_gb = gpu.free_mb / 1000.0
    if smallest_model_gb is not None and gpu.total_mb / 1000.0 < smallest_model_gb:
        return GateResult(
            "refused",
            f"{gpu.name} has {gpu.total_mb / 1000:.0f} GB of VRAM; the smallest TensorFold model needs about {smallest_model_gb:.0f} GB.",
        )
    if smallest_model_gb is not None and free_gb < smallest_model_gb:
        reasons.append(f"Only {free_gb:.1f} GB of VRAM is free right now; close other GPU programs before loading.")

    if caps is not None:
        if "cuda" not in caps.backends or not (caps.chip or "").startswith("nvidia"):
            return GateResult(
                "refused",
                f"This TensorFold binary reports chip '{caps.chip}' and back ends {list(caps.backends)}, not CUDA on this card. "
                "It was likely built for another GPU architecture; rebuild or install one that matches.",
            )
        return GateResult("ready", f"TensorFold reports {caps.chip} with CUDA on {gpu.name}.", tuple(reasons))

    qualified = gpu.compute_capability in RELEASE_IMAGE_CC
    if not binary_present and not asset_available:
        return GateResult(
            "needs-binary",
            f"{gpu.name} (sm_{cc[0]}{cc[1]}) can run TensorFold, but there is no release archive for this platform. "
            f"Build tensorfold-native from source for this card and install it from the Setup page, or set {BINARY_ENV}.",
            tuple(reasons),
        )
    if not qualified:
        reasons.append(
            f"The release archive carries kernel images for sm_80, 86, 120 and 121 only; sm_{cc[0]}{cc[1]} is unconfirmed until the "
            "installed binary reports a CUDA chip."
        )
        return GateResult("unqualified", f"{gpu.name} is not in TensorFold's release image set; install and check.", tuple(reasons))
    return GateResult("ready" if binary_present else "needs-binary", f"{gpu.name} matches a TensorFold release image.", tuple(reasons))
