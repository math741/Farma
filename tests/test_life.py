"""Ω1-Ω2 organisms: genome layout, development → body, body-derived capabilities, signal substrate."""

from __future__ import annotations

import math

import pytest
import torch

from genesis_omega.life.development import Development, NODE_PROPS, maturity
from genesis_omega.life.genome import (GenomeLayout, PHYS_GENES, crossover, decode_physiology, genome_distance,
                                       mutate, proto_genome)
from genesis_omega.life.interface import (BODY_CAPS, CAP, INPUT_NAMES, N_INPUTS, N_OUTPUTS, OUTPUT_NAMES)
from genesis_omega.life.substrate import SubstrateParams, substrate_step

LAYOUT = GenomeLayout(12, 16, 24)


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def _population(n=16, seed=0):
    g = _gen(seed)
    protos = [proto_genome(LAYOUT, g, "cpu") for _ in range(n)]
    return torch.cat(protos)


def test_genome_layout_is_consistent():
    total = sum(b - a for a, b, _ in LAYOUT.slices.values())
    assert total == LAYOUT.size
    bounds = LAYOUT.segment_bounds()
    assert bounds[0][0] == 0 and bounds[-1][1] == LAYOUT.size
    assert all(b0[1] == b1[0] for b0, b1 in zip(bounds, bounds[1:]))     # contiguous crossover blocks
    d = LAYOUT.to_dict()
    assert d["size"] == LAYOUT.size and "br_w" in d["segments"]
    assert N_INPUTS == len(INPUT_NAMES) and N_OUTPUTS == len(OUTPUT_NAMES)


def test_variation_operators():
    G = _population(8)
    g = _gen(1)
    child = mutate(LAYOUT, G, g)
    assert child.shape == G.shape and not torch.equal(child, G)
    a_ls = LAYOUT.slices["phys"][0] + PHYS_GENES.index("log_sigma")
    assert torch.all(child[:, a_ls] <= math.log(0.25) + 1e-6) and torch.all(child[:, a_ls] >= math.log(0.004) - 1e-6)
    a, b = G[:4], G[4:]
    x = crossover(LAYOUT, a, b, _gen(2))
    from_a = (x == a) | (x == b)
    assert bool(from_a.all())                    # every gene comes from one of the parents
    for s, e in LAYOUT.segment_bounds():         # blocks are inherited whole
        blk = x[:, s:e]
        assert bool(((blk == a[:, s:e]).all(1) | (blk == b[:, s:e]).all(1)).all())
    assert torch.allclose(genome_distance(G, G), torch.zeros(8))


def test_physiology_decoding_ranges():
    ph = decode_physiology(LAYOUT, _population(32), lifespan_days=3.0, day_ticks=120)
    assert torch.all((ph["t_tol"] >= 4) & (ph["t_tol"] <= 18))
    assert torch.all(ph["lifespan"] > 0) and torch.all((ph["invest"] > 0.2) & (ph["invest"] < 0.65))
    assert torch.all((ph["sigma"] >= 0.004) & (ph["sigma"] <= 0.25))


def test_development_is_deterministic_and_genome_dependent():
    dev = Development(LAYOUT, 4, 5, "cpu", 4)
    G = _population(12)
    b1 = dev.develop(G, LAYOUT.hidden)
    b2 = dev.develop(G.clone(), LAYOUT.hidden)
    assert torch.equal(b1["caps"], b2["caps"])
    assert torch.equal(b1["alive"], b2["alive"])
    # different genomes grow different bodies
    assert len({tuple(c.tolist()) for c in b1["caps"].round(decimals=4)}) > 1
    # a body is never empty: the seed node always exists
    assert torch.all(b1["alive"].flatten(1).sum(1) >= 1)


def test_body_capabilities_are_bounded_and_physical():
    dev = Development(LAYOUT, 4, 5, "cpu", 4)
    b = dev.develop(_population(24, seed=3), LAYOUT.hidden)
    caps = b["caps"]
    for name in ("land_speed", "water_speed", "turn", "grasp", "strike", "armor", "insulation", "emit_power",
                 "bite", "aquatic"):
        v = caps[:, CAP[name]]
        assert torch.all((v >= 0) & (v <= 1.0001)), name
    assert torch.all(caps[:, CAP["mass"]] > 0)
    assert torch.all(caps[:, CAP["cells"]] >= 1)
    u = caps[:, CAP["units"]]
    assert torch.all((u >= max(4, LAYOUT.hidden // 6)) & (u <= LAYOUT.hidden))
    tot = b["emit_profile"].sum(1)
    assert bool((((tot - 1).abs() < 1e-4) | (tot == 0)).all())      # a spectrum (or silence)
    assert b["sens"].shape == (24, 4, 4) and torch.all((b["sens"] >= 0) & (b["sens"] <= 1))
    assert len(BODY_CAPS) == caps.shape[1]


def test_body_form_does_part_of_the_control():
    """A single-node body (no growth) is slower and senses less than a grown one."""
    dev = Development(LAYOUT, 4, 5, "cpu", 4)
    grown = _population(1, seed=5)
    seed_only = grown.clone()
    a, b, _ = LAYOUT.slices["dev_b2"]
    seed_only[:, a] = -3.0                       # the "alive" channel decays: no division
    a1, b1, _ = LAYOUT.slices["dev_w2"]
    seed_only[:, a1:b1] = 0.0
    bg = dev.develop(grown, LAYOUT.hidden)
    bs = dev.develop(seed_only, LAYOUT.hidden)
    assert float(bs["caps"][0, CAP["cells"]]) == 1.0
    assert float(bg["caps"][0, CAP["cells"]]) > 1.0
    assert float(bs["caps"][0, CAP["mass"]]) < float(bg["caps"][0, CAP["mass"]])


def test_body_export_matches_lattice():
    dev = Development(LAYOUT, 4, 5, "cpu", 4)
    G = _population(1, seed=9)
    ex = dev.export_body(G[0], LAYOUT.hidden)
    b = dev.develop(G, LAYOUT.hidden)
    assert len(ex["voxels"]) == int(b["alive"].sum())
    assert all(len(v) == 3 + len(NODE_PROPS) for v in ex["voxels"])
    assert ex["props"] == list(NODE_PROPS)


def test_maturity_curve():
    life = torch.tensor([100.0, 100.0, 100.0])
    m = maturity(torch.tensor([0.0, 9.0, 1000.0]), life, 0.18)
    assert m[0] == pytest.approx(0.4) and 0.4 < m[1] < 1.0 and m[2] == pytest.approx(1.0)


def _params(n=6, units=None):
    G = _population(n, seed=11)
    u = torch.full((n,), float(LAYOUT.hidden)) if units is None else units
    return G, SubstrateParams(LAYOUT, G, u)


def test_substrate_shapes_and_masking():
    n = 6
    units = torch.tensor([4.0, 6, 8, 10, 12, 12])
    G, p = _params(n, units)
    W = p.W0.clone()
    x = torch.zeros(n, LAYOUT.hidden)
    m = torch.zeros(n)
    u = torch.randn(n, N_INPUTS, generator=_gen(3))
    W2, x2, y, m2, dW = substrate_step(p, W, x, m, u, torch.randn(n, LAYOUT.hidden, generator=_gen(4)))
    assert x2.shape == (n, LAYOUT.hidden) and y.shape == (n, N_OUTPUTS) and m2.shape == (n,)
    for i, k in enumerate(units.long().tolist()):
        assert torch.all(x2[i, k:] == 0)       # inactive substrate units are masked by the body
    assert torch.all(m2.abs() <= 1)


def test_plasticity_and_ablations():
    n = 4
    G, p = _params(n)
    W = p.W0.clone()
    x = torch.rand(n, LAYOUT.hidden) - 0.5
    m = torch.full((n,), 0.8)
    u = torch.rand(n, N_INPUTS)
    W_off, *_ = substrate_step(p, W, x, m, u, None, plastic=False)
    assert torch.equal(W_off, W)                                  # frozen substrate: weights unchanged
    W_on, *_ = substrate_step(p, W, x, m, u, None, plastic=True)
    assert not torch.equal(W_on, W)                               # the evolvable rule changes weights
    # recurrence ablation: the new state no longer depends on the previous state
    _, xa, *_ = substrate_step(p, W, x, m, u, None, plastic=False, recurrent_scale=0.0)
    _, xb, *_ = substrate_step(p, W, torch.zeros_like(x), m, u, None, plastic=False, recurrent_scale=0.0)
    assert torch.allclose(xa, xb)
    # clamping a direction removes the state's projection on it
    d = torch.zeros(n, LAYOUT.hidden)
    d[:, 2] = 1.0
    _, xc, *_ = substrate_step(p, W, x, m, u, None, plastic=False, clamp_dir=d)
    assert torch.allclose(xc[:, 2], torch.zeros(n), atol=1e-6)


def test_consolidation_bounds_weights():
    n = 3
    G, p = _params(n)
    W = p.W0.clone() * 10
    x = torch.rand(n, LAYOUT.hidden)
    W2, *_ = substrate_step(p, W, x, torch.ones(n), torch.rand(n, N_INPUTS), None, plastic=True, consolidate=True)
    assert float(W2.abs().max()) <= 4.0 + 1e-6
