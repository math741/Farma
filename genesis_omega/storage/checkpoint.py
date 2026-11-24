"""CheckpointEngine + StorageEngine (spec §256-263, §410-411, §422-430, §553-565, §618).

Layout of a cosmos on persistent storage (e.g. Google Drive)::

    <root>/<cosmos-label>/
        cosmos.json                 identity, seed, versions (never rewritten silently)
        auth.json                   viewer / admin tokens (private)
        checkpoints/current.pt      + current.json  (sha256, tick, versions)
        checkpoints/previous.pt     + previous.json
        checkpoints/history/ckpt_u<tick>.pt (+ .json)
        events/events.jsonl         append-only history
        discoveries/<label>.json    discovery packages; frames_<event>.npz frozen windows
        forks/<fork-label>/...      fork timelines (same layout)
        reports/  failures/  telemetry/

Writes are atomic: the payload is serialised to fast local scratch, fsynced,
copied next to its destination under a temporary name, fsynced, and moved
into place with os.replace. Rotation keeps the previous checkpoint. Loading
verifies the sha256 sidecar and falls back current → previous → history.
A cosmos directory without a valid checkpoint is never silently replaced.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import time
from dataclasses import dataclass

import torch

from ..core.hashing import file_sha256
from ..core.logging import get_logger
from ..version import CHECKPOINT_SCHEMA, PHYSICS_VERSION, version_info

log = get_logger("storage")


class NoValidCheckpoint(RuntimeError):
    pass


class PhysicsVersionMismatch(RuntimeError):
    pass


@dataclass
class StorageLayout:
    root: str
    label: str
    scratch: str

    @property
    def base(self) -> str:
        return os.path.join(self.root, self.label)

    def path(self, *parts) -> str:
        return os.path.join(self.base, *parts)

    def ensure(self) -> None:
        for d in ("checkpoints/history", "events", "discoveries", "forks", "reports", "failures", "telemetry"):
            os.makedirs(self.path(d), exist_ok=True)
        os.makedirs(self.scratch, exist_ok=True)

    def fork_layout(self, fork_label: str) -> "StorageLayout":
        return StorageLayout(os.path.join(self.base, "forks"), fork_label, os.path.join(self.scratch, fork_label))

    def exists(self) -> bool:
        return os.path.exists(self.path("cosmos.json"))


def _fsync_file(path: str) -> None:
    with open(path, "rb") as fh:
        os.fsync(fh.fileno())


def atomic_write_bytes(path: str, data: bytes) -> None:
    tmp = f"{path}.tmp-{os.getpid()}-{time.time_ns()}"
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: str, obj: dict) -> None:
    atomic_write_bytes(path, json.dumps(obj, indent=1, default=str).encode("utf-8"))


def write_identity(layout: StorageLayout, identity: dict) -> None:
    p = layout.path("cosmos.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            old = json.load(fh)
        if old.get("seed") != identity.get("seed") or old.get("label") != identity.get("label"):
            raise RuntimeError("cosmos.json identity mismatch; refusing to overwrite another cosmos")
        return
    atomic_write_json(p, identity)


def read_identity(layout: StorageLayout) -> dict:
    with open(layout.path("cosmos.json"), encoding="utf-8") as fh:
        return json.load(fh)


def save_checkpoint(layout: StorageLayout, state: dict, meta: dict) -> dict:
    """Atomically write a checkpoint and rotate current → previous."""
    layout.ensure()
    t0 = time.time()
    meta = dict(meta)
    meta.update({"schema": CHECKPOINT_SCHEMA, "versions": version_info(), "saved_at": time.time()})
    buf = io.BytesIO()
    torch.save({"meta": meta, "state": state}, buf)
    payload = buf.getvalue()
    scratch_tmp = os.path.join(layout.scratch, f"ckpt-{time.time_ns()}.pt")
    with open(scratch_tmp, "wb") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    digest = file_sha256(scratch_tmp)
    cdir = layout.path("checkpoints")
    dst_tmp = os.path.join(cdir, f"incoming-{time.time_ns()}.pt")
    shutil.copyfile(scratch_tmp, dst_tmp)
    _fsync_file(dst_tmp)
    os.remove(scratch_tmp)
    cur, prev = os.path.join(cdir, "current.pt"), os.path.join(cdir, "previous.pt")
    if os.path.exists(cur):
        os.replace(cur, prev)
        if os.path.exists(cur[:-3] + ".json"):
            os.replace(cur[:-3] + ".json", prev[:-3] + ".json")
    os.replace(dst_tmp, cur)
    side = {"sha256": digest, "bytes": len(payload), "meta": meta}
    atomic_write_json(cur[:-3] + ".json", side)
    log.info("checkpoint saved", extra={"fields": {"utick": meta.get("utick"), "mb": round(len(payload) / 1e6, 2),
                                                    "seconds": round(time.time() - t0, 2)}})
    return side


def snapshot_history(layout: StorageLayout, keep: int) -> str | None:
    cdir = layout.path("checkpoints")
    cur = os.path.join(cdir, "current.pt")
    if not os.path.exists(cur):
        return None
    with open(cur[:-3] + ".json", encoding="utf-8") as fh:
        side = json.load(fh)
    name = f"ckpt_u{int(side['meta'].get('utick', 0)):012d}"
    hdir = os.path.join(cdir, "history")
    os.makedirs(hdir, exist_ok=True)
    dst = os.path.join(hdir, name + ".pt")
    tmp = dst + ".tmp"
    shutil.copyfile(cur, tmp)
    _fsync_file(tmp)
    os.replace(tmp, dst)
    atomic_write_json(os.path.join(hdir, name + ".json"), side)
    snaps = sorted(f for f in os.listdir(hdir) if f.endswith(".pt"))
    for old in snaps[:-keep] if keep > 0 else []:
        os.remove(os.path.join(hdir, old))
        j = os.path.join(hdir, old[:-3] + ".json")
        if os.path.exists(j):
            os.remove(j)
    return dst


def _candidates(layout: StorageLayout) -> list[str]:
    cdir = layout.path("checkpoints")
    out = [os.path.join(cdir, "current.pt"), os.path.join(cdir, "previous.pt")]
    hdir = os.path.join(cdir, "history")
    if os.path.isdir(hdir):
        out += [os.path.join(hdir, f) for f in sorted(os.listdir(hdir), reverse=True) if f.endswith(".pt")]
    return [p for p in out if os.path.exists(p)]


def load_latest(layout: StorageLayout, map_location="cpu") -> tuple[dict, dict, str]:
    """Load the newest checkpoint whose sha256 matches its sidecar."""
    errors = []
    for path in _candidates(layout):
        side_path = path[:-3] + ".json"
        try:
            with open(side_path, encoding="utf-8") as fh:
                side = json.load(fh)
            if file_sha256(path) != side["sha256"]:
                errors.append(f"{os.path.basename(path)}: sha256 mismatch")
                continue
            blob = torch.load(path, map_location=map_location, weights_only=False)
            meta = blob["meta"]
            state = migrate(blob["state"], meta)
            return state, meta, path
        except Exception as e:  # corrupt / partial file: try the next candidate
            errors.append(f"{os.path.basename(path)}: {e!r}")
    raise NoValidCheckpoint("; ".join(errors) or "no checkpoint files")


# ------------------------------------------------------------ migrations
def _m0_to_1(state: dict, meta: dict) -> dict:
    """Schema 0 → 1: worlds gained the `acc` accumulators and blocked-material sets."""
    for w in state.get("worlds", []):
        w.setdefault("acc", torch.zeros(4))
        w.setdefault("blocked_materials", [])
        w.setdefault("param_scale", {})
    return state


MIGRATIONS = {0: _m0_to_1}


def migrate(state: dict, meta: dict) -> dict:
    v = int(meta.get("schema", 0))
    while v < CHECKPOINT_SCHEMA:
        fn = MIGRATIONS.get(v)
        if fn is None:
            raise RuntimeError(f"no migration from schema {v}")
        state = fn(state, meta)
        v += 1
        meta.setdefault("migrations", []).append(v)
    phys = meta.get("versions", {}).get("physics", PHYSICS_VERSION)
    if phys != PHYSICS_VERSION and not meta.get("allow_physics_change"):
        raise PhysicsVersionMismatch(f"checkpoint physics {phys} != engine physics {PHYSICS_VERSION}; "
                                     "continuing would silently change the science — start a fork with an "
                                     "explicit physics migration instead")
    return state


def archive_cosmos(layout: StorageLayout) -> str:
    """Explicit reset: move the old cosmos aside (never deleted)."""
    dst = os.path.join(layout.root, "archive", f"{layout.label}_{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(layout.base, dst)
    return dst
