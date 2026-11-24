"""Ω0/Ω5-Ω7 runtime: the engine room, recovery, forks, replay, multiple worlds, promotion, expeditions, tuning."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from conftest import small_cfg
from genesis_omega.runtime.runner import Runner, write_report
from genesis_omega.runtime.universe import Universe, cpu_state


def _wait(cond, timeout=60.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.05)
    return False


# ------------------------------------------------------------------ engine room
def test_runner_runs_publishes_and_checkpoints(storage):
    u = Universe.create(small_cfg(seed=61, checkpoint_interval_s=0.5), storage, run_controls=False)
    r = Runner(u)
    r.start()
    try:
        assert _wait(lambda: r.rounds > 30)
        assert _wait(lambda: r.hub.snapshot()["seq"] > 0)
        snap = r.hub.snapshot()
        assert 0 in snap["agents"] and 0 in snap["fields"] and snap["status"]["cosmos"] == u.label
        assert _wait(lambda: os.path.exists(storage.path("checkpoints", "current.pt")))
        assert r.call(lambda rr: rr.u.utick) > 0               # commands run inside the sim thread
    finally:
        r.stop(checkpoint=True)
    assert not r.thread.is_alive()
    assert r.telemetry()["checkpoints"] >= 1
    assert write_report(r) is not None


def test_crash_is_recorded_and_recovered(storage):
    u = Universe.create(small_cfg(seed=62, keyframe_every=20), storage, run_controls=False)
    r = Runner(u)
    r.start()
    try:
        assert _wait(lambda: len(u.keyframes) > 0)
        r.fault_injection = "crash"
        assert _wait(lambda: any(e.type == "TIMELINE_ROLLBACK" for e in r.u.events.events))
        assert r.halted is None
        assert any(e.get("kind") == "crash" for e in r.errors)
        assert os.listdir(storage.path("failures"))           # the failure is on disk, not hidden
        n = r.rounds
        assert _wait(lambda: r.rounds > n + 10)                 # and the cosmos keeps running
    finally:
        r.stop(checkpoint=False)


def test_oom_sheds_science_batch_then_fidelity(storage):
    u = Universe.create(small_cfg(seed=63, keyframe_every=20), storage, run_controls=False)
    r = Runner(u)
    batch0 = u.scientist.lab_batch
    r.start()
    try:
        assert _wait(lambda: len(u.keyframes) > 0)
        r.fault_injection = "oom"
        assert _wait(lambda: any(e.type == "RESOURCE_LIMIT" for e in r.u.events.events))
        assert r.u.scientist.lab_batch == max(8, batch0 // 2)
        births0 = r.u.worlds[0].max_births
        r.fault_injection = "oom"
        assert _wait(lambda: sum(e.type == "RESOURCE_LIMIT" for e in r.u.events.events) >= 2)
        assert r.u.worlds[0].max_births < births0              # active fidelity reduced, and logged
    finally:
        r.stop(checkpoint=False)


def test_resume_never_resets(tmp_path):
    from genesis_omega.runtime.launcher import open_or_create

    root, scratch = str(tmp_path / "root"), str(tmp_path / "scratch")
    u, lay, info = open_or_create(root, scratch, "cpu-test", 5, False, "cpu", None, tune=False)
    assert not info["resumed"]
    for _ in range(30):
        u.step_round()
    from genesis_omega.storage.checkpoint import save_checkpoint
    save_checkpoint(lay, cpu_state(u), {"utick": u.utick})
    u.events.close()
    v, lay2, info2 = open_or_create(root, scratch, "cpu-test", None, False, "cpu", None, tune=False)
    assert info2["resumed"] and v.label == u.label and v.utick == 30
    assert any(e.type == "SESSION_RESUMED" for e in v.events.events)
    v.events.close()
    # a cosmos whose checkpoints are all unreadable is an error — never a silent new history
    for f in os.listdir(lay.path("checkpoints")):
        p = lay.path("checkpoints", f)
        if f.endswith(".pt"):
            with open(p, "wb") as fh:
                fh.write(b"broken")
    from genesis_omega.storage.checkpoint import NoValidCheckpoint
    with pytest.raises(NoValidCheckpoint):
        open_or_create(root, scratch, "cpu-test", None, False, "cpu", None, tune=False)


# ------------------------------------------------------------------ timelines
def test_forks_never_write_into_the_prime_timeline():
    u = Universe.create(small_cfg(seed=64), None, run_controls=False)
    twin = Universe.create(small_cfg(seed=64), None, run_controls=False)
    for _ in range(20):
        u.step_round()
        twin.step_round()
    f = u.fork("drought", "test", salt=5)
    f.intervene({"world": 0, "steps": [{"op": "perturb", "param": "growth_rate", "value": 0.25},
                                       {"op": "cull", "fraction": 0.5}]})
    galaxy_before = u.galaxy.f0_biomass.copy()
    for _ in range(int(u.cfg.galaxy_every) + 5):
        f.step_round()
    assert np.array_equal(np.nan_to_num(u.galaxy.f0_biomass, nan=-1), np.nan_to_num(galaxy_before, nan=-1))
    assert u.levels is not f.levels and u.construction is not f.construction
    for _ in range(30):
        u.step_round()
        twin.step_round()
    assert u.world_digest(u.worlds[0]) == twin.world_digest(twin.worlds[0])   # prime unaffected by the fork
    assert f.timeline == "F-0001" and f.parent == "PRIME"
    assert any(e.type == "FORK_CREATED" for e in u.events.events)
    assert all(e.epistemic == "EXTERNAL" for e in f.events.events if e.type == "EXTERNAL_INTERVENTION")
    with pytest.raises(PermissionError):
        u.intervene({"world": 0, "steps": [{"op": "cull", "fraction": 0.1}]})


def test_same_salt_forks_are_identical_and_salts_diverge():
    u = Universe.create(small_cfg(seed=65), None, run_controls=False)
    for _ in range(10):
        u.step_round()
    a, b, c = u.fork(salt=1), u.fork(salt=1), u.fork(salt=2)
    for _ in range(25):
        for f in (a, b, c):
            f.step_round()
    assert a.world_digest(a.worlds[0]) == b.world_digest(b.worlds[0])
    assert a.world_digest(a.worlds[0]) != c.world_digest(c.worlds[0])


def test_deterministic_replay_between_keyframes():
    u = Universe.create(small_cfg(seed=66, keyframe_every=25), None, run_controls=False)
    for _ in range(75):
        u.step_round()
    kfs = list(u.keyframes)
    assert len(kfs) >= 2
    res = u.verify_replay(kfs[0]["utick"], kfs[-1]["utick"], kfs[-1]["digests"])
    assert res["identical"], res
    r = u.replay_copy(kfs[0]["utick"])
    assert r.timeline.startswith("REPLAY@") and r.storage is None


# ------------------------------------------------------------------ several worlds
def test_multiple_f4_worlds_have_different_pressures():
    u = Universe.create(small_cfg(seed=67, active_worlds=2), None, run_controls=False)
    if len(u.worlds) < 2:
        pytest.skip("this galaxy has a single life-bearing rocky world")
    a, b = u.worlds
    assert a.planet["star"] != b.planet["star"]
    assert a.xenos.to_dict() != b.xenos.to_dict()
    assert b.agents.N < a.agents.N                              # secondary worlds get a smaller share
    for _ in range(10):
        u.step_round()
    assert a.tick == b.tick == 10


def test_promotion_from_f0_to_f4_is_external():
    u = Universe.create(small_cfg(seed=68), None, run_controls=False)
    g = u.galaxy
    rows = [int(r) for r in np.where(~np.isnan(g.f0_biomass))[0] if int(r) != u.worlds[0].planet["row"]]
    if not rows:
        pytest.skip("no other F0 biosphere in this galaxy")
    n0 = len(u.worlds)
    w = u.promote(rows[0])
    assert len(u.worlds) == n0 + 1 and w.agents.count() > 0
    ev = [e for e in u.events.events if e.type == "FIDELITY_PROMOTION"]
    assert ev and ev[-1].epistemic == "EXTERNAL"
    with pytest.raises(ValueError):
        u.promote(rows[0])                                      # already simulated at F4


def test_expedition_transplants_lineages_into_a_marked_fork():
    u = Universe.create(small_cfg(seed=69, active_worlds=2), None, run_controls=False)
    if len(u.worlds) < 2:
        pytest.skip("this galaxy has a single life-bearing rocky world")
    f = u.fork("expedition", "test")
    target = f.worlds[1]
    n0 = target.agents.count()
    f.intervene({"world": 1, "steps": [{"op": "expedition", "source_world": 0, "count": 6, "lat": 0, "lon": 0}]},
                source_worlds=u.worlds)
    assert target.agents.count() == min(target.agents.N, n0 + 6)
    assert u.worlds[1].agents.count() == n0                     # the prime world is untouched
    assert f.worlds[1].interventions[-1]["op"] == "expedition"


# ------------------------------------------------------------------ tuning
def test_autotune_measures_and_only_changes_capacity():
    from genesis_omega.runtime.autotune import autotune

    cfg = small_cfg(seed=70)
    out, report = autotune(cfg, min_tps=1.0, ticks=5)
    assert out.world.n_max in {c["n_max"] for c in report["candidates"]}
    assert report["candidates"][0]["tps"] > 0
    assert out.seed == cfg.seed and out.world.hidden == cfg.world.hidden    # the laws are never tuned
