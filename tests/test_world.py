"""Ω0-Ω5 world: terrain, climate, evolution, QD, social graph, culture lineage, levels, events, fidelity."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from conftest import SEED, small_cfg
from genesis_omega.history.events import Event, EventLog, Pantheon, PANTHEON_KEYS
from genesis_omega.life.evolution import LineageStore, QDArchive, SeedBank, SpeciesTracker
from genesis_omega.life.interface import ST
from genesis_omega.science import levels as levels_mod
from genesis_omega.society.social import SocialGraph, T_GIVE, T_KIN, T_PROX, T_STRIKE
from genesis_omega.society.technology import cultural_lineage
from genesis_omega.world.geometry import SphereGrid
from genesis_omega.world.materials import N_BASE
from genesis_omega.world.terrain import generate_terrain


# ------------------------------------------------------------------ terrain
@pytest.mark.parametrize("water", [0.0, 0.3, 0.7])
def test_terrain_ocean_fraction_and_deposits(water):
    g = SphereGrid(64, 32)
    t = generate_terrain(g, 1234, water, N_BASE)
    frac = float((t.ocean.float() * g.area).sum())
    assert abs(frac - water) < 0.06
    assert bool((t.elev_km[t.ocean] < 0).all()) if water > 0 else not bool(t.ocean.any())
    assert float(t.elev_km.max()) <= 5.5 + 1e-4 and float(t.elev_km.min()) >= -4.5 - 1e-4
    assert t.deposits.shape == (N_BASE, 32, 64)
    assert float(t.deposits[:, t.ocean].abs().sum()) == 0.0          # no ground deposits under the sea
    assert bool(((t.fertility >= 0) & (t.fertility <= 1)).all())


def test_sphere_geometry_motion_stays_on_sphere():
    g = SphereGrid(32, 16)
    gen = torch.Generator().manual_seed(0)
    p = torch.nn.functional.normalize(torch.randn(100, 3, generator=gen), dim=1)
    h = g.random_tangent(p, gen)
    p2, h2 = g.move(p, h, torch.full((100,), 0.3))
    assert torch.allclose(p2.norm(dim=1), torch.ones(100), atol=1e-5)
    assert torch.allclose((p2 * h2).sum(1), torch.zeros(100), atol=1e-5)    # heading stays tangent
    assert torch.allclose(g.angular_distance(p, p2), torch.full((100,), 0.3), atol=1e-4)
    # the conservative Laplacian conserves the area-weighted integral
    f = torch.rand(16, 32, generator=gen)
    lap = g.laplacian(f)
    assert abs(float((lap * g.coslat).sum())) < 1e-3


# ------------------------------------------------------------------ climate
def test_climate_day_night_and_seasons(universe):
    w = universe.worlds[0]
    c = w.climate
    ins = c.insolation(w.grid, w.tick)
    assert float(ins.min()) == 0.0 and float(ins.max()) > 0      # a night side and a day side
    lat0, _ = c.subsolar_point(0)
    lat_q, _ = c.subsolar_point(w.xenos.year_ticks // 4)
    assert abs(lat_q - lat0) > 0.5 * w.xenos.tilt_deg * 0.5      # seasons move the subsolar latitude
    assert abs(w.calibration["achieved"] - w.calibration["target"]) < 15.0
    for f in (w.temp, w.precip, w.cloud):
        assert bool(torch.isfinite(f).all())
    assert bool(((w.cloud >= 0) & (w.cloud <= 1)).all())


# ------------------------------------------------------------------ the tick
def test_world_steps_with_finite_state(universe):
    w = universe.worlds[0]
    A = w.agents
    act = A.active
    assert A.count() > 0
    assert torch.allclose(A.pos[act].norm(dim=1), torch.ones(int(act.sum())), atol=1e-4)
    assert torch.allclose((A.pos[act] * A.head[act]).sum(1), torch.zeros(int(act.sum())), atol=1e-4)
    for name in ("energy", "integrity", "tbody", "x", "W"):
        assert bool(torch.isfinite(getattr(A, name)[act]).all()), name
    for name in ("R", "temp", "light", "sound", "scent", "store", "compaction", "struct_h", "shelter"):
        assert bool(torch.isfinite(getattr(w, name)).all()), name
    assert int(A.id[act].unique().numel()) == int(act.sum())        # persistent ids are unique
    s = w.summary()
    assert s["population"] == A.count() and s["fidelity"] == "F4"


def test_world_is_deterministic():
    from genesis_omega.runtime.universe import Universe

    digests = []
    for _ in range(2):
        u = Universe.create(small_cfg(seed=SEED + 7), None, run_controls=False)
        for _ in range(60):
            u.step_round()
        digests.append(u.world_digest(u.worlds[0]))
    assert digests[0] == digests[1]


def test_reproduction_and_generations(universe):
    w = universe.worlds[0]
    assert w.counters["births"] > 0
    A = w.agents
    assert int(A.generation[A.active].max()) >= 1
    kids = A.parent[A.active]
    assert bool((kids >= -1).all())
    rec = w.lineage.record(int(A.id[A.active][0]))
    assert rec is not None and rec["death"] == -1


def test_seed_bank_germinates_a_depleted_world(fresh_world):
    w = fresh_world
    A = w.agents
    adults = torch.nonzero(A.active).flatten()
    w.bank.deposit(A.genome[adults[:20]], A.pos[adults[:20]],
                   torch.stack([A.lineage[adults[:20]], A.species[adults[:20]], A.generation[adults[:20]],
                                torch.zeros(20, dtype=torch.long)], 1))
    A.energy[A.active] = -1.0                  # everyone dies this tick (starvation)
    w.step()
    assert A.count() == 0
    for _ in range(3000):
        w.step()
        if w.counters["germinations"]:
            break
    assert w.counters["germinations"] > 0 and A.count() > 0


def test_species_tracker_discovers_species():
    sp = SpeciesTracker(50, 1, threshold=0.5)
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.05, (20, 32)).astype(np.float32)
    b = (rng.normal(0, 0.05, (20, 32)) + 3.0).astype(np.float32)
    proj = np.concatenate([a, b])
    out, events = sp.update(proj, np.full(40, -1), tick=10)
    assert len(set(out[:20].tolist())) == 1 and len(set(out[20:].tolist())) == 1 and out[0] != out[20]
    assert sum(1 for k, _ in events if k == "new") == 2
    out2, events2 = sp.update(proj[:20], out[:20], tick=20)
    assert ("extinct", int(out[20])) in events2                    # the second cluster vanished
    st = sp.state_dict()
    sp2 = SpeciesTracker(50, 1, threshold=0.5)
    sp2.load_state_dict(st)
    assert sp2.table() == sp.table()


def test_qd_archive_fills_niches():
    qd = QDArchive(10, "cpu")
    stats = torch.zeros(6, len(ST))
    stats[:, ST["ticks"]] = 100
    stats[0, ST["water_ticks"]] = 100                           # aquatic
    stats[1, ST["builds"]] = 50                                 # builder
    stats[2, ST["strikes"]] = 50                                # striker
    mass = torch.tensor([1.0, 5.0, 10.0, 1.0, 1.0, 14.0])
    d = QDArchive.descriptors(stats, mass, ST, 0.9)
    new = qd.ingest(torch.randn(6, 10), d, torch.arange(6).float(), torch.zeros(6, 4, dtype=torch.long), 1)
    assert len(new) >= 3 and qd.coverage() > 0
    dominants = {e["dominant"] for e in qd.elites()}
    assert {"build", "strike"} <= dominants
    assert qd.nov_count == 1
    qd2 = QDArchive(10, "cpu")
    qd2.load_state_dict(qd.state_dict())
    assert qd2.coverage() == qd.coverage()


def test_lineage_store():
    ls = LineageStore(5)
    ls.add_births(np.array([1, 2, 3]), np.array([-1, 1, 2]), np.array([1, 1, 1]), np.array([1, 1, 1]),
                  np.array([0, 1, 2]), 5, np.array([0, 1, 2]))
    ls.add_deaths(np.array([2]), np.array([4]), np.array([1]), 9)
    assert ls.record(2)["cause"] == "predation" and ls.record(2)["death"] == 9
    assert [a["id"] for a in ls.ancestors(3)] == [2, 1]
    assert ls.children(1) == [2]
    ls.add_births(np.arange(10, 14), np.full(4, 3), np.ones(4), np.ones(4), np.ones(4), 10, np.arange(4))
    assert ls.record(1) is None                                 # ring table: the oldest records rotate out


def test_seed_bank_ring():
    sb = SeedBank(4, 3, "cpu")
    sb.deposit(torch.ones(6, 3), torch.zeros(6, 3), torch.zeros(6, 4, dtype=torch.long))
    assert sb.count() == 4 and sb.ptr == 2


# ------------------------------------------------------------------ society
def test_social_graph_groups_and_persistence():
    sg = SocialGraph()
    a = np.array([1, 1, 2, 4, 4, 5, 1])
    b = np.array([2, 3, 3, 5, 6, 6, 4])
    t = np.array([T_GIVE, T_GIVE, T_GIVE, T_KIN, T_KIN, T_KIN, T_STRIKE])
    sg.ingest(a, b, t, 10)
    sg.ingest(np.array([1]), np.array([1]), np.array([T_PROX]), 11)        # self-interaction ignored
    groups = sg.groups({1, 2, 3, 4, 5, 6}, min_weight=2.0)
    assert groups[1] == groups[2] == groups[3]
    assert groups[4] == groups[5] == groups[6] != groups[1]   # a strike is not a tie
    assert 4 not in sg.groups({1, 2, 3, 4, 5, 6}, min_weight=3.0)   # kin ties alone are weak
    stats = sg.group_stats({1, 2, 3, 4, 5, 6})
    assert stats["totals"]["give"] == 3 and stats["edges"] == 7
    nb = sg.neighbors(1)
    assert {r["id"] for r in nb} == {2, 3, 4}
    sg2 = SocialGraph()
    sg2.load_state_dict(sg.state_dict())
    assert sg2.edges.keys() == sg.edges.keys()


def test_cultural_lineage_classification():
    class W:
        pass

    w = W()
    w.social = SocialGraph()
    w.social.ingest(np.array([10]), np.array([20]), np.array([T_GIVE]), 1)
    m = N_BASE + 1
    # (tick, agent, material, cell, lineage, species, first)
    w.production_log = [(5, 10, m, 0, 100, 1, 1), (9, 20, m, 0, 200, 1, 1),   # contact with earlier producer
                        (12, 30, m, 0, 100, 1, 1),                             # same matriline as producer 10
                        (15, 40, m, 0, 400, 1, 1)]                             # neither: independent
    out = cultural_lineage(w)
    assert out["totals"] == {"social": 1, "kin": 1, "independent": 1}
    assert out["materials"][m]["producers"] == 4


# ------------------------------------------------------------------ levels / events / fidelity
def test_levels_are_evidence_assessments(universe):
    u = universe
    w = u.worlds[0]
    lv = levels_mod.assess(w, u.chronicles[0], [], u.social_stats.get(0, {}), None)
    assert {"organism", "affect", "civilization", "science", "observer"} <= set(lv)
    assert len(lv["civilization"]) == len(levels_mod.CIVILIZATION)
    flat = json_dump(lv).lower()
    assert "conscious" in lv["note"].lower() and "consciousness level" not in flat
    assert lv["civilization"][9]["status"] == levels_mod.NOT_IMPLEMENTED       # no spaceflight physics


def json_dump(o):
    import json
    return json.dumps(o, default=str)


def test_events_are_logged_and_reloadable(tmp_path):
    p = str(tmp_path / "events" / "events.jsonl")
    log = EventLog(p)
    for i in range(1, 6):
        log.append(Event(id=i, type="TEST", title=f"e{i}", utick=i))
    log.close()
    assert [e["id"] for e in log.since(2)] == [3, 4, 5]
    assert log.get(4).title == "e4"
    with open(p, "a", encoding="utf-8") as fh:
        fh.write('{"id": 6, "type": "TO')                    # torn final line after a crash
    fresh = EventLog(p)
    assert fresh.load_jsonl(max_id=4) == 4                  # events after the checkpoint are moved aside
    assert [e.id for e in fresh.events] == [1, 2, 3, 4]
    assert fresh.discarded == 2
    moved = [f for f in tmp_path.joinpath("events").iterdir() if "discarded" in f.name]
    assert len(moved) == 1 and "e5" in moved[0].read_text()
    again = EventLog(p)
    assert again.load_jsonl() == 4


def test_pantheon_keeps_firsts():
    pt = Pantheon()
    e1 = Event(id=1, type="X", title="first", utick=1)
    e2 = Event(id=2, type="X", title="second", utick=2)
    assert pt.offer("first_predation", e1) and not pt.offer("first_predation", e2)
    rows = pt.table()
    assert len(rows) == len(PANTHEON_KEYS)
    assert next(r for r in rows if r["key"] == "first_predation")["title"] == "first"


def test_chronicle_emits_world_events(universe):
    types = {e.type for e in universe.events.events}
    assert {"COSMOS_BORN", "WORLD_FORMED", "LIFE_SEEDED"} <= types
    for e in universe.events.events:
        assert e.epistemic in {"SIMULATED", "PROGRAMMED", "EMERGENT", "MEASURED", "INFERRED", "HYPOTHESIS",
                               "EXTERNAL"}


def test_fidelity_levels(universe):
    g = universe.galaxy
    f0 = ~np.isnan(g.f0_biomass)
    assert f0.sum() >= 1                                     # F0: aggregate biospheres elsewhere
    w = universe.worlds[0]
    assert w.fidelity == "F4" and w.agents.count() > 0       # F4: individuals
    focus = w.focus
    assert focus.k == w.cfg.focus_agents
    j = int(np.argmax(focus.valid.sum(1)))
    ser = focus.series(j)                                    # F5: full internal recording
    assert ser["x"].shape[1] == w.layout.hidden and ser["u"].shape[0] == ser["x"].shape[0] > 0
    assert np.all(np.diff(ser["ticks"]) > 0)


def test_ring_buffer_and_frames(universe):
    w = universe.worlds[0]
    frames = w.ring.window()
    assert frames and frames[-1]["tick"] <= w.tick
    f = next(f for f in reversed(frames) if f["n"] > 0)
    for k in ("id", "lat", "lon", "species", "bits", "energy", "emit", "turn", "thrust", "recv", "cell"):
        assert len(f[k]) == f["n"], k
    assert f["recv"].shape[1] == w.cfg.emit_bands


def test_metrics_vocabulary_matches_dsl(universe):
    from genesis_omega.science.experiments import METRICS

    m = universe.worlds[0].metrics()
    assert set(m) == set(METRICS)
    assert m["population"] == universe.worlds[0].agents.count()
    assert math.isfinite(m["eaten"]) and m["eaten"] > 0
