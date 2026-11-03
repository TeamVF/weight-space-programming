"""Read and write small weight tensors.

JSON is always available. `.npz` needs the optional `weights` extra
(`numpy`). `.pt` / `.pth` need the optional `torch` extra. Neither library
is imported unless that format is requested.

The text format is what a tool-free prompt can carry:

    tensor W1 shape 2 2
    0.250 -0.500
    0.000 0.125

Author: wanghaobo
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

_HEADER = re.compile(r"^tensor\s+(\S+)\s+shape((?:\s+\d+)+)\s*$")
_FORMAT = "wsp-weights-v1"


def toy_tensors() -> dict[str, list[Any]]:
    """A stand-in checkpoint. Not a network from the blog."""
    return {
        "W1": [[0.25, -0.5], [0.0, 0.125]],
        "b1": [0.0, 0.0],
    }


def count_parameters(tensors: Mapping[str, Any]) -> int:
    return sum(_numel(value) for value in tensors.values())


def to_nested_lists(value: Any) -> Any:
    """Convert a tensor, array, or nested sequence into JSON-ready lists."""
    if isinstance(value, bool) or isinstance(value, (str, bytes)):
        raise TypeError(f"unsupported weight value: {type(value).__name__}")
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
    if hasattr(value, "tolist") and not isinstance(value, (list, tuple)):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [to_nested_lists(item) for item in value]
    return _number(value)


def save_weights(path: str | Path, tensors: Mapping[str, Any]) -> Path:
    """Write `tensors` to `.json`, `.npz`, or `.pt`/`.pth` based on the suffix."""
    destination = Path(path)
    suffix = destination.suffix.lower()
    destination.parent.mkdir(parents=True, exist_ok=True)
    nested = {name: to_nested_lists(value) for name, value in tensors.items()}
    if suffix == ".json":
        payload = {"format": _FORMAT, "tensors": nested}
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return destination
    if suffix == ".npz":
        numpy = _optional_import("numpy", 'pip install -e ".[weights]"')
        arrays = {name: numpy.asarray(value, dtype=numpy.float32) for name, value in nested.items()}
        numpy.savez(destination, **arrays)
        return destination
    if suffix in {".pt", ".pth"}:
        torch = _optional_import("torch", 'pip install -e ".[torch]"')
        blobs = {name: torch.tensor(value) for name, value in nested.items()}
        torch.save(blobs, destination)
        return destination
    raise ValueError(f"unsupported weight suffix {suffix!r}; use .json, .npz, .pt, or .pth")


def load_weights(path: str | Path) -> dict[str, Any]:
    """Load tensors. JSON returns lists; npz returns arrays; pt returns tensors."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "tensors" in payload:
            tensors = payload["tensors"]
        else:
            tensors = payload
        if not isinstance(tensors, dict):
            raise ValueError(f"weight file must contain a tensor map: {source}")
        return tensors
    if suffix == ".npz":
        numpy = _optional_import("numpy", 'pip install -e ".[weights]"')
        with numpy.load(source) as archive:
            return {name: archive[name] for name in archive.files}
    if suffix in {".pt", ".pth"}:
        torch = _optional_import("torch", 'pip install -e ".[torch]"')
        blobs = torch.load(source, map_location="cpu", weights_only=True)
        if not isinstance(blobs, dict):
            raise ValueError(f"torch checkpoint must be a dict of tensors: {source}")
        return blobs
    raise ValueError(f"unsupported weight suffix {suffix!r}")


def format_weights_text(tensors: Mapping[str, Any], *, ndigits: int = 3) -> str:
    """Render 1D or 2D tensors as plain text for a tool-free prompt."""
    blocks: list[str] = []
    for name, value in tensors.items():
        nested = to_nested_lists(value)
        shape = _shape(nested)
        if len(shape) > 2:
            raise ValueError(f"{name} has rank {len(shape)}; text format supports rank 1 or 2")
        header = "tensor " + name + " shape " + " ".join(str(dim) for dim in shape)
        if len(shape) == 1:
            body = _format_row(nested, ndigits)
        else:
            body = "\n".join(_format_row(row, ndigits) for row in nested)
        blocks.append(header + "\n" + body)
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def parse_weights_text(text: str) -> dict[str, list[Any]]:
    """Parse `format_weights_text` output. Prose lines outside a block are ignored."""
    tensors: dict[str, list[Any]] = {}
    name: str | None = None
    shape: tuple[int, ...] | None = None
    rows: list[list[float]] = []

    def flush() -> None:
        nonlocal name, shape, rows
        if name is None or shape is None:
            return
        tensors[name] = _assemble(name, shape, rows)
        name, shape, rows = None, None, []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        header = _HEADER.match(line)
        if header:
            flush()
            name = header.group(1)
            shape = tuple(int(part) for part in header.group(2).split())
            continue
        if name is not None and _is_numeric_row(line):
            rows.append([float(part) for part in line.split()])
            continue
        flush()
    flush()
    return tensors


def _assemble(name: str, shape: tuple[int, ...], rows: list[list[float]]) -> list[Any]:
    if len(shape) == 1:
        if len(rows) != 1 or len(rows[0]) != shape[0]:
            raise ValueError(f"{name}: expected shape {shape}, got {len(rows)} rows")
        return rows[0]
    if len(shape) == 2:
        if len(rows) != shape[0] or any(len(row) != shape[1] for row in rows):
            raise ValueError(f"{name}: expected shape {shape}")
        return rows
    raise ValueError(f"{name}: text format supports rank 1 or 2, got {shape}")


def _shape(value: Any) -> tuple[int, ...]:
    if isinstance(value, list):
        if not value or not isinstance(value[0], list):
            return (len(value),)
        return (len(value),) + _shape(value[0])
    raise TypeError("expected a nested list")


def _numel(value: Any) -> int:
    if hasattr(value, "numel") and callable(value.numel):
        return int(value.numel())
    if hasattr(value, "size") and not isinstance(value, (list, tuple, str)):
        return int(value.size)
    nested = value if isinstance(value, list) else to_nested_lists(value)
    return _count_nested(nested)


def _count_nested(value: Any) -> int:
    if isinstance(value, list):
        return sum(_count_nested(item) for item in value)
    return 1


def _format_row(row: list[Any], ndigits: int) -> str:
    return " ".join(f"{_number(item):.{ndigits}f}" for item in row)


def _is_numeric_row(line: str) -> bool:
    parts = line.split()
    if not parts:
        return False
    try:
        for part in parts:
            float(part)
    except ValueError:
        return False
    return True


def _number(value: Any) -> int | float:
    if isinstance(value, bool):
        raise TypeError("bool is not a weight")
    if hasattr(value, "item") and not isinstance(value, (int, float)):
        value = value.item()
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("weight must be finite")
        return value
    raise TypeError(f"unsupported weight scalar: {type(value).__name__}")


def _optional_import(module: str, hint: str) -> Any:
    try:
        return __import__(module)
    except ImportError as exc:
        raise ImportError(f"{module} is optional. Install it with: {hint}") from exc
