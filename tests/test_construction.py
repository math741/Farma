"""Ω4 matter: hidden chemistry, construction physics, settlement detectors, technology graph."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import SEED, force_outputs, small_cfg
from genesis_omega.society.construction import ConstructionDetector, label_periodic, shelter_effect
from genesis_omega.society.technology import convergence, technology_graph
from genesis_omega.world.materials import (BAND_EDGES, IDENTITY_TOL, N_BANDS, N_BASE, N_PROPS, Chemistry,
                                           MaterialRegistry)


# ------------------------------------------------------------------ chemistry
def test_chemistry_is_a_deterministic_seeded_law():
    a, b = Chemistry(11), Chemistry(11)
    assert a.digest() == b.digest() and torch.equal(a.base_props, b.base_props)
    assert Chemistry(12).digest() != a.digest()
    pa = a.base_props[[0, 1, 2]]
    pb = a.base_props[[3, 4, 5]]
    band = torch.tensor([0, 1, 3])
    p1, r1 = a.react(pa, pb, band)
    p2, r2 = b.react(pa, pb, band)
    assert torch.equal(p1, p2) and torch.equal(r1, r2)
    assert bool(((p1 > 0) & (p1 < 1)).all()) and p1.shape == (3, N_PROPS)


def test_identical_reagents_do_not_react():
    ch = Chemistry(3)
    p = ch.base_props[:1].repeat(N_BANDS, 1)
    _, reacted = ch.react(p, p, torch.arange(N_BANDS))
    assert not bool(reacted.any())


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6])
def test_reaction_rate_is_bounded_for_every_cosmos(seed):
    """Roughly a third of reagent pairs react — never none, never all (open-ended but not explosive)."""
    ch = Chemistry(seed)
    g = torch.Generator().manual_seed(seed)
    pa, pb = torch.rand(4000, N_PROPS, generator=g), torch.rand(4000, N_PROPS, generator=g)
    band = torch.randint(0, N_BANDS, (4000,), generator=g)
    _, reacted = ch.react(pa, pb, band)
    assert 0.1 < float(reacted.float().mean()) < 0.55


def test_temperature_bands():
    ch = Chemistry(1)
    t = torch.tensor([-30.0, 0.5, 25.0, 41.0, 80.0])
    bands = ch.band_of(t)
    assert bands.tolist() == [0, 1, 2, 3, 3] and N_BANDS == len(BAND_EDGES) + 1


def test_registry_identifies_and_registers_products():
    ch = Chemistry(7)
    reg = MaterialRegistry(ch, 64, "cpu")
    assert reg.n == N_BASE and all(r.base for r in reg.records)
    ids, d = reg.identify(reg.props[:N_BASE] + 0.2 * IDENTITY_TOL)
    assert ids.tolist() == list(range(N_BASE))                           # within tolerance → same material
    pa, pb = ch.base_props[[0, 0, 1]], ch.base_props[[1, 1, 2]]
    prod, _ = ch.react(pa, pb, torch.tensor([1, 1, 2]))
    rec = torch.tensor([[0, 1, 1], [0, 1, 1], [1, 2, 2]])
    ids, new = reg.register_products(prod, rec, 42, torch.tensor([5, 6, 7]), torch.tensor([10, 11, 12]),
                                     torch.tensor([1, 1, 2]), torch.tensor([3, 3, 4]))
    assert ids[0] == ids[1]                                              # same recipe → same material
    assert len(new) == len(set(ids.tolist()))
    r = reg.records[int(ids[0])]
    assert r.depth == 1 and r.recipe == (0, 1, 1) and r.first_agent == 5 and r.first_tick == 42
    # a product of a product is one step deeper
    p2, _ = ch.react(reg.props[ids[:1]], reg.props[[3]], torch.tensor([0]))
    ids2, _ = reg.register_products(p2, torch.tensor([[int(ids[0]), 3, 0]]), 50, torch.tensor([1]),
                                    torch.tensor([1]), torch.tensor([1]))
    assert reg.records[int(ids2[0])].depth in (0, 2) or int(ids2[0]) < N_BASE
    st = reg.state_dict()
    reg2 = MaterialRegistry(ch, 64, "cpu")
    reg2.load_state_dict(st)
    assert reg2.n == reg.n and [x.to_dict() for x in reg2.records] == [x.to_dict() for x in reg.records]
    with pytest.raises(RuntimeError):
        MaterialRegistry(Chemistry(8), 64, "cpu").load_state_dict(st)     # another law: refuse


def test_registry_saturation_is_flagged():
    ch = Chemistry(9)
    reg = MaterialRegistry(ch, N_BASE + 1, "cpu")
    prod = torch.tensor([[0.01, 0.99, 0.01, 0.99, 0.01], [0.99, 0.01, 0.99, 0.01, 0.99]])
    ids, new = reg.register_products(prod, torch.tensor([[0, 1, 0], [1, 2, 0]]), 1, torch.zeros(2).long(),
                                     torch.zeros(2).long(), torch.zeros(2).long())
    assert len(new) == 1 and reg.saturated and reg.n == N_BASE + 1
    assert 0 <= int(ids[1]) < reg.n


# ------------------------------------------------------------------ construction physics
@pytest.fixture(scope="module")
def builder_world():
    from genesis_omega.runtime.universe import Universe

    u = Universe.create(small_cfg(seed=SEED + 3), None, run_controls=False)
    w = u.worlds[0]
    for k in range(700):
        if k % 50 == 0:
            force_outputs(w, ("dig", "process", "build", "store"), 3.0)
            force_outputs(w, ("strike", "reproduce"), -3.0)
            force_outputs(w, ("reproduce",), 1.0)
        u.step_round()
    return u


def test_digging_building_processing_happen(builder_world):
    w = builder_world.worlds[0]
    assert w.counters["builds"] > 0 and w.counters["processes"] > 0
    assert int((w.struct_h > 0.05).sum()) > 0
    assert float(w.struct_h.max()) <= w.phys.max_struct_height * 1.3 * w.xenos.struct_scale + 1e-4
    assert float(w.acc[2]) > 0                                        # matter placed into structures
    assert w.materials.built[: w.materials.n].sum() > 0


def test_structures_give_shelter_and_decay(builder_world):
    w = builder_world.worlds[0]
    eff = shelter_effect(w)
    assert eff["mean_near_structures"] >= eff["mean"]
    h0 = w.struct_h.clone()
    keep = w.struct_h > 0.05
    w2 = w.fork(1)                                                    # decay in a copy, never in the original
    w2.agents.active[:] = False
    for _ in range(100):
        w2.step()
    assert float(w2.struct_h[keep].sum()) < float(h0[keep].sum())
    assert torch.equal(w.struct_h, h0)


def test_walls_block_movement(fresh_world):
    w = fresh_world
    A = w.agents
    s = int(torch.nonzero(A.active).flatten()[0])
    _, _, cell = w.grid.cell_of(A.pos[s:s + 1])
    ahead = w.grid.offset_points(A.pos[s:s + 1], A.head[s:s + 1], (0.0,), 0.6 * w.grid.dang)[:, 0]
    _, _, c2 = w.grid.cell_of(ahead)
    if int(c2) == int(cell):
        pytest.skip("organism not near a cell edge")
    w.struct_h.view(-1)[c2] = 3.0
    force_outputs(w, ("thrust",), 6.0)
    for _ in range(3):
        w.step()
        _, _, now = w.grid.cell_of(A.pos[s:s + 1])
        assert int(now) != int(c2) or not bool(A.active[s])


def test_storage_deposits_and_withdrawals(builder_world):
    w = builder_world.worlds[0]
    assert w.counters["stores"] > 0
    assert float(w.store_in.sum()) > 0


def test_trails_from_compaction(builder_world):
    w = builder_world.worlds[0]
    assert float(w.compaction.max()) > 0


# ------------------------------------------------------------------ detectors
class _FakeWorld:
    def __init__(self, H=24, W=48):
        from genesis_omega.world.physics import WorldPhysics

        self.phys = WorldPhysics()
        self.tick = 10_000

        class X:
            day_ticks = 100
            year_days = 30
        self.xenos = X()
        z = lambda: torch.zeros(H, W)  # noqa: E731
        self.struct_h, self.struct_age, self.occupancy, self.cold_occupancy = z(), z(), z(), z()
        self.warm_occupancy, self.compaction, self.store_in, self.store_out = z(), z(), z(), z()
        self.shelter, self.last_visit = z(), z()
        self.struct_builder = torch.full((H, W), -1, dtype=torch.long)


def test_house_like_detector():
    fw = _FakeWorld()
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if (di, dj) != (0, 0):
                fw.struct_h[10 + di, 20 + dj] = 1.5
                fw.struct_age[10 + di, 20 + dj] = 1000
    fw.cold_occupancy[10, 20] = 0.5
    fw.warm_occupancy[10, 20] = 0.05
    fw.occupancy[10, 20] = 0.55
    out = ConstructionDetector(fw).analyze()
    assert len(out["house_like"]) == 1 and out["house_like"][0]["cold_warm_ratio"] > 1.5
    fw.cold_occupancy[10, 20] = 0.05                    # used equally in warm and cold → not house-like
    fw.warm_occupancy[10, 20] = 0.5
    assert ConstructionDetector(fw).analyze()["house_like"] == []


def test_storage_village_trail_and_ruin_detectors():
    fw = _FakeWorld()
    fw.store_in[5, 5], fw.store_out[5, 5], fw.shelter[5, 5] = 2.0, 1.0, 0.5
    fw.struct_h[12:16, 30:34] = 0.6
    fw.struct_age[12:16, 30:34] = 1000
    fw.occupancy[12:16, 30:34] = 0.5
    fw.last_visit[12:16, 30:34] = fw.tick
    fw.compaction[2, 0:12] = 0.8
    out = ConstructionDetector(fw).analyze()
    assert out["storage_like_cells"] == 1
    assert len(out["village_like"]) == 1 and out["ruins"] == []
    assert out["trails"] and out["trails"][0]["cells"] >= 8
    fw.last_visit[:] = 0                                  # nobody comes back for > 10 days
    fw.occupancy[:] = 0
    out = ConstructionDetector(fw).analyze()
    assert len(out["ruins"]) == 1 and out["village_like"] == []
    assert all(not all(c.values()) for c in out["city_criteria"])


def test_labels_wrap_in_longitude():
    m = np.zeros((6, 10), bool)
    m[2, 0] = m[2, 9] = True
    lab, n = label_periodic(m)
    assert n == 1 and lab[2, 0] == lab[2, 9]


# ------------------------------------------------------------------ technology
def test_technology_graph_from_real_events(builder_world):
    w = builder_world.worlds[0]
    tg = technology_graph(w)
    assert len(tg["nodes"]) == w.materials.n
    assert tg["prop_names"] == ["density", "hardness", "insulation", "cohesion", "reactivity"]
    for e in tg["edges"]:
        assert 0 <= e["from"] < w.materials.n and e["to"] >= N_BASE
    if w.materials.n > N_BASE:
        assert tg["n_products"] == w.materials.n - N_BASE and tg["max_depth"] >= 1
        assert any(e.type == "NOVEL_MATERIAL" for e in builder_world.events.events)


def test_convergence_across_worlds_with_the_same_law():
    class W:
        def __init__(self, i, reg):
            self.index, self.materials = i, reg

    ch = Chemistry(5)
    r1, r2 = MaterialRegistry(ch, 32, "cpu"), MaterialRegistry(ch, 32, "cpu")
    prod, _ = ch.react(ch.base_props[[0]], ch.base_props[[1]], torch.tensor([2]))
    z = torch.zeros(1).long()
    r1.register_products(prod, torch.tensor([[0, 1, 2]]), 1, z, z, z)
    r2.register_products(prod, torch.tensor([[0, 1, 2]]), 9, z, z, z)
    found = convergence([W(0, r1), W(1, r2)])
    assert found and found[0]["material_a"] == N_BASE and found[0]["distance"] < 1e-6
