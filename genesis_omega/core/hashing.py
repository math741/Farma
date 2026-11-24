"""Reproducible digests of simulation state (spec §541-542)."""

from __future__ import annotations

import hashlib

import numpy as np


def tensor_digest(*tensors) -> str:
    """SHA-256 over the raw bytes of tensors/arrays (bitwise identity)."""
    h = hashlib.sha256()
    for t in tensors:
        if t is None:
            h.update(b"<none>")
            continue
        if hasattr(t, "detach"):
            a = t.detach().cpu().contiguous().numpy()
        else:
            a = np.ascontiguousarray(t)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def file_sha256(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
