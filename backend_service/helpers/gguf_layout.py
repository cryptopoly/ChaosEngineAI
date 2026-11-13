"""Where a GGUF's weights live once llama.cpp loads it on a discrete GPU, read from the file's tensor table.

llama.cpp keeps the input embedding tables in system memory (one row is read per token): ``token_embd`` and, for
Gemma 3n / Gemma 4 style models such as Nemotron 3.5 Bolt, the large ``per_layer_token_embd``. A model with a tied
output head gets a second copy of ``token_embd`` on the GPU for the output projection. So the GPU holds
``file - input embeddings (+ token_embd when tied)``. Measured on an RTX 4090 (b11981): Nemotron 3.5 Bolt E4B
Q4_K_M, a 5.74 GiB file, put 3536 MiB in system memory and 3023 MiB on the card; Qwen3-0.6B Q8_0 158 / 604 MiB.
Both match this split to the MiB. The compute buffer is dominated by the output logits for one micro-batch
(``n_vocab x 512 x 4`` bytes): 299 MiB for Qwen3's 152K vocabulary, 582 MiB for Bolt's 262K.

Only the header is read (no tensor data); sizes come from the gaps between tensor offsets, so no table of
quantisation block sizes is needed. Split files (``-00001-of-0000N``) are summed.
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from backend_service.helpers.model_arch import _GGUF_ARRAY, _GGUF_SCALARS, _GGUF_STRING

# Tensors llama.cpp places on the CPU for every model (the input layer).
_INPUT_TENSORS = ("token_embd.weight", "per_layer_token_embd.weight")
_UBATCH = 512
_LOGIT_BYTES = 4
# Measured compute buffers were 1.0x and 1.1x the logits size; the rest is the attention scratch.
_COMPUTE_MARGIN = 1.1
_SPLIT_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


@dataclass(frozen=True)
class GgufLayout:
    total_bytes: int
    gpu_bytes: int
    host_bytes: int
    vocab: int
    tied_output: bool

    def compute_buffer_bytes(self) -> int:
        """llama.cpp's GPU compute buffer, about the logits of one 512-token micro-batch."""
        return int(self.vocab * _UBATCH * _LOGIT_BYTES * _COMPUTE_MARGIN)


def _skip_value(handle, value_type: int) -> None:
    if value_type in _GGUF_SCALARS:
        handle.seek(struct.calcsize(_GGUF_SCALARS[value_type]), 1)
    elif value_type == _GGUF_STRING:
        (length,) = struct.unpack("<Q", handle.read(8))
        handle.seek(length, 1)
    elif value_type == _GGUF_ARRAY:
        (item_type,) = struct.unpack("<I", handle.read(4))
        (count,) = struct.unpack("<Q", handle.read(8))
        if item_type == _GGUF_STRING:
            for _ in range(count):
                (length,) = struct.unpack("<Q", handle.read(8))
                handle.seek(length, 1)
        elif item_type in _GGUF_SCALARS:
            handle.seek(struct.calcsize(_GGUF_SCALARS[item_type]) * count, 1)
        else:
            raise ValueError(f"unsupported GGUF array type {item_type}")
    else:
        raise ValueError(f"unsupported GGUF value type {value_type}")


def _tensor_table(path: Path) -> tuple[dict[str, int], dict[str, list[int]]] | None:
    """``({name: bytes}, {name: dims})`` for one GGUF file, from its header only."""
    try:
        file_size = path.stat().st_size
        with open(path, "rb") as handle:
            if handle.read(4) != b"GGUF":
                return None
            (version,) = struct.unpack("<I", handle.read(4))
            if version < 2:
                return None
            n_tensors, kv_count = struct.unpack("<QQ", handle.read(16))
            alignment = 32
            for _ in range(kv_count):
                (length,) = struct.unpack("<Q", handle.read(8))
                name = handle.read(length).decode("utf-8", "replace")
                (value_type,) = struct.unpack("<I", handle.read(4))
                if name == "general.alignment" and value_type == 4:
                    (alignment,) = struct.unpack("<I", handle.read(4))
                    continue
                _skip_value(handle, value_type)
            entries: list[tuple[str, list[int], int]] = []
            for _ in range(n_tensors):
                (length,) = struct.unpack("<Q", handle.read(8))
                name = handle.read(length).decode("utf-8", "replace")
                (n_dims,) = struct.unpack("<I", handle.read(4))
                dims = list(struct.unpack(f"<{n_dims}Q", handle.read(8 * n_dims)))
                _type, offset = struct.unpack("<IQ", handle.read(12))
                entries.append((name, dims, offset))
            header_end = handle.tell()
    except (OSError, struct.error, ValueError, UnicodeDecodeError):
        return None
    data_start = (header_end + alignment - 1) // alignment * alignment
    data_size = file_size - data_start
    ordered = sorted(entries, key=lambda e: e[2])
    sizes: dict[str, int] = {}
    for index, (name, _dims, offset) in enumerate(ordered):
        end = ordered[index + 1][2] if index + 1 < len(ordered) else data_size
        sizes[name] = max(0, end - offset)
    return sizes, {name: dims for name, dims, _ in entries}


def _shards(path: Path) -> list[Path]:
    match = _SPLIT_RE.search(path.name)
    if not match:
        return [path]
    count = int(match.group(2))
    stem = path.name[: match.start()]
    shards = [path.with_name(f"{stem}-{i:05d}-of-{count:05d}.gguf") for i in range(1, count + 1)]
    return shards if all(s.is_file() for s in shards) else [path]


@lru_cache(maxsize=64)
def _layout_cached(path_str: str, _mtime: float) -> GgufLayout | None:
    sizes: dict[str, int] = {}
    dims: dict[str, list[int]] = {}
    for shard in _shards(Path(path_str)):
        table = _tensor_table(shard)
        if table is None:
            return None
        sizes.update(table[0])
        dims.update(table[1])
    if not sizes:
        return None
    total = sum(sizes.values())
    host = sum(sizes.get(name, 0) for name in _INPUT_TENSORS)
    tied = "output.weight" not in sizes
    gpu = total - host + (sizes.get("token_embd.weight", 0) if tied else 0)
    embd_dims = dims.get("token_embd.weight") or []
    vocab = int(embd_dims[1]) if len(embd_dims) >= 2 else 0
    return GgufLayout(total_bytes=total, gpu_bytes=gpu, host_bytes=host, vocab=vocab, tied_output=tied)


def gguf_layout(path: str | Path) -> GgufLayout | None:
    """The GPU / system-memory split of a ``.gguf`` file, or None when it cannot be read. Never raises."""
    target = Path(path).expanduser().absolute()
    try:
        if target.suffix.lower() != ".gguf" or not target.is_file():
            return None
        return _layout_cached(str(target), target.stat().st_mtime)
    except OSError:
        return None
