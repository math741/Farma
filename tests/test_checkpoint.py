"""Ω0 persistence: atomic checkpoints, verified loading with fallback, versions, migrations, archives."""

from __future__ import annotations

import json
import os

import pytest
import torch

from genesis_omega.storage import checkpoint as ck
from genesis_omega.storage.checkpoint import (NoValidCheckpoint, PhysicsVersionMismatch, StorageLayout,
                                              archive_cosmos, load_latest, read_identity, save_checkpoint,
                                              snapshot_history, write_identity)
from genesis_omega.version import CHECKPOINT_SCHEMA, PHYSICS_VERSION


def _state(v: int) -> dict:
    return {"utick": v, "worlds": [{"x": torch.full((3,), float(v))}]}


def test_save_and_load_roundtrip(storage):
    side = save_checkpoint(storage, _state(1), {"utick": 1})
    assert len(side["sha256"]) == 64 and side["meta"]["schema"] == CHECKPOINT_SCHEMA
    state, meta, path = load_latest(storage)
    assert state["utick"] == 1 and torch.equal(state["worlds"][0]["x"], torch.ones(3))
    assert path.endswith("current.pt") and meta["versions"]["physics"] == PHYSICS_VERSION


def test_rotation_keeps_previous(storage):
    save_checkpoint(storage, _state(1), {"utick": 1})
    save_checkpoint(storage, _state(2), {"utick": 2})
    cdir = storage.path("checkpoints")
    assert os.path.exists(os.path.join(cdir, "previous.pt"))
    with open(os.path.join(cdir, "previous.json"), encoding="utf-8") as fh:
        assert json.load(fh)["meta"]["utick"] == 1
    assert not [f for f in os.listdir(cdir) if f.startswith("incoming") or ".tmp" in f]   # nothing left half-written


def test_corrupt_current_falls_back_to_previous(storage):
    save_checkpoint(storage, _state(1), {"utick": 1})
    save_checkpoint(storage, _state(2), {"utick": 2})
    cur = storage.path("checkpoints", "current.pt")
    with open(cur, "r+b") as fh:                 # simulate a torn write / bit rot
        fh.seek(100)
        fh.write(b"\0\0\0\0garbage")
    state, _, path = load_latest(storage)
    assert state["utick"] == 1 and path.endswith("previous.pt")


def test_history_snapshots_rotate(storage):
    for k in range(5):
        save_checkpoint(storage, _state(k), {"utick": k})
        snapshot_history(storage, keep=3)
    hist = sorted(f for f in os.listdir(storage.path("checkpoints", "history")) if f.endswith(".pt"))
    assert hist == ["ckpt_u000000000002.pt", "ckpt_u000000000003.pt", "ckpt_u000000000004.pt"]
    # with current and previous destroyed, the newest history snapshot is used
    os.remove(storage.path("checkpoints", "current.pt"))
    os.remove(storage.path("checkpoints", "previous.pt"))
    state, _, path = load_latest(storage)
    assert state["utick"] == 4 and "history" in path


def test_no_valid_checkpoint_is_an_error_not_a_reset(storage):
    with pytest.raises(NoValidCheckpoint):
        load_latest(storage)


def test_physics_version_mismatch_is_refused(storage):
    save_checkpoint(storage, _state(1), {"utick": 1})
    blob = torch.load(storage.path("checkpoints", "current.pt"), weights_only=False)
    blob["meta"]["versions"]["physics"] = "phys-0"
    with pytest.raises(PhysicsVersionMismatch):
        ck.migrate(blob["state"], blob["meta"])
    blob["meta"]["allow_physics_change"] = True        # an explicit, recorded migration is allowed
    ck.migrate(blob["state"], blob["meta"])


def test_schema_migration_0_to_1():
    state = {"worlds": [{"tick": 3}]}
    meta = {"schema": 0, "versions": {"physics": PHYSICS_VERSION}}
    out = ck.migrate(state, meta)
    w = out["worlds"][0]
    assert torch.equal(w["acc"], torch.zeros(4)) and w["blocked_materials"] == [] and w["param_scale"] == {}
    assert meta["migrations"] == [1]


def test_identity_is_never_overwritten(storage):
    write_identity(storage, {"label": "test-cosmos", "seed": 1})
    write_identity(storage, {"label": "test-cosmos", "seed": 1})      # idempotent
    assert read_identity(storage)["seed"] == 1
    with pytest.raises(RuntimeError):
        write_identity(storage, {"label": "test-cosmos", "seed": 2})


def test_explicit_reset_archives_instead_of_deleting(tmp_path):
    lay = StorageLayout(str(tmp_path / "root"), "C1", str(tmp_path / "scratch"))
    lay.ensure()
    write_identity(lay, {"label": "C1", "seed": 1})
    dst = archive_cosmos(lay)
    assert not os.path.exists(lay.base) and os.path.exists(os.path.join(dst, "cosmos.json"))


def test_universe_checkpoint_roundtrip_is_bit_exact(storage):
    from conftest import small_cfg
    from genesis_omega.runtime.universe import Universe, cpu_state

    u = Universe.create(small_cfg(seed=31), storage, run_controls=False)
    for _ in range(40):
        u.step_round()
    save_checkpoint(storage, cpu_state(u), {"utick": u.utick})
    state, _, _ = load_latest(storage)
    v = Universe.from_state(small_cfg(seed=31), storage, state)
    assert v.utick == u.utick and v.world_digest(v.worlds[0]) == u.world_digest(u.worlds[0])
    for _ in range(25):                      # the restored cosmos continues exactly as the original would
        u.step_round()
        v.step_round()
    assert v.world_digest(v.worlds[0]) == u.world_digest(u.worlds[0])
    assert v.events.events[-1].id <= v.ids.peek("event")
