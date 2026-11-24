"""Server ↔ viewer protocol (spec §293-298, §426).

Binary messages are a tiny self-describing container:

    b"GOB1" | uint32 header_length | JSON header | padding to 4 | raw arrays

The header lists ``{"kind", "meta", "arrays": [{name, dtype, shape, offset, nbytes}]}``;
offsets are relative to the start of the raw section and 4-byte aligned, so
the browser can create typed-array views without copying.
"""

from __future__ import annotations

import json
import struct

import numpy as np

from ..version import PROTOCOL_VERSION

MAGIC = b"GOB1"
DTYPES = {"float32": "f4", "int32": "i4", "uint32": "u4", "int16": "i2", "uint16": "u2", "uint8": "u1",
          "int8": "i1"}


def encode(kind: str, arrays: dict, meta: dict | None = None) -> bytes:
    specs, chunks, off = [], [], 0
    for name, arr in arrays.items():
        a = np.ascontiguousarray(arr)
        if a.dtype == np.float64:
            a = a.astype(np.float32)
        if a.dtype == np.int64:
            a = a.astype(np.int32)
        if a.dtype == np.float16:
            a = a.astype(np.float32)
        if a.dtype == np.bool_:
            a = a.astype(np.uint8)
        dt = str(a.dtype)
        if dt not in DTYPES:
            raise TypeError(f"unsupported dtype {dt} for {name}")
        b = a.astype(a.dtype.newbyteorder("<"), copy=False).tobytes()
        pad = (-len(b)) % 4
        specs.append({"name": name, "dtype": dt, "shape": list(a.shape), "offset": off, "nbytes": len(b)})
        chunks.append(b + b"\0" * pad)
        off += len(b) + pad
    header = json.dumps({"kind": kind, "protocol": PROTOCOL_VERSION, "meta": meta or {}, "arrays": specs},
                        separators=(",", ":"), default=_default).encode("utf-8")
    hpad = (-(8 + len(header))) % 4
    return MAGIC + struct.pack("<I", len(header) + hpad) + header + b" " * hpad + b"".join(chunks)


def decode(buf: bytes) -> tuple[str, dict, dict]:
    if buf[:4] != MAGIC:
        raise ValueError("bad magic")
    (hl,) = struct.unpack("<I", buf[4:8])
    header = json.loads(buf[8:8 + hl].decode("utf-8"))
    base = 8 + hl
    arrays = {}
    for s in header["arrays"]:
        raw = buf[base + s["offset"]: base + s["offset"] + s["nbytes"]]
        arrays[s["name"]] = np.frombuffer(raw, dtype=np.dtype(s["dtype"]).newbyteorder("<")).reshape(s["shape"])
    return header["kind"], header["meta"], arrays


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def json_safe(obj):
    """Recursively convert numpy / torch scalars for JSON responses."""
    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if torch is not None and torch.is_tensor(obj):
        return json_safe(obj.detach().cpu().numpy())
    return obj
