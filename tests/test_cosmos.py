"""Ω0 cosmos: galaxy, stellar populations, planetary systems, orbits, F0 biospheres."""

from __future__ import annotations

import math

import numpy as np
import pytest

from genesis_omega.cosmos import stars as st
from genesis_omega.cosmos.galaxy import Galaxy, KIND_ROCKY, KIND_GAS, KIND_ICE, mean_surface_temp


@pytest.fixture(scope="module")
def galaxy():
    return Galaxy.generate(1234, 800)


def test_generation_is_deterministic():
    a, b = Galaxy.generate(77, 300), Galaxy.generate(77, 300)
    assert a.static_digest() == b.static_digest()
    assert np.array_equal(a.planets["a"], b.planets["a"])
    assert Galaxy.generate(78, 300).static_digest() != a.static_digest()


def test_spiral_disk_and_bulge(galaxy):
    r = np.linalg.norm(galaxy.pos[:, :2], axis=1)
    assert galaxy.pos.shape == (800, 3)
    assert r.max() <= 16.5                     # disk truncation (kpc)
    assert np.abs(galaxy.pos[:, 2]).mean() < 1.0  # thin disk
    assert galaxy.smbh_mass > 1e6              # central supermassive black hole


def test_imf_and_stellar_stages(galaxy):
    m0 = st.sample_kroupa(np.random.default_rng(0), 20000)
    assert m0.min() >= 0.08 and m0.max() <= 40.0
    assert np.median(m0) < 0.6                 # low-mass stars dominate a Kroupa IMF
    stages = np.bincount(galaxy.stage, minlength=5)
    assert stages[st.STAGE_MAIN] > 0.6 * galaxy.n_stars
    assert stages[st.STAGE_WHITE_DWARF] > 0    # remnants exist in an old population
    assert np.all(galaxy.lum[galaxy.stage == st.STAGE_BLACK_HOLE] == 0)
    lum, rad, teff, life = st.main_sequence(np.array([1.0]))
    assert abs(lum[0] - 1.0) < 1e-9 and abs(teff[0] - st.T_SUN) < 1.0 and abs(life[0] - 10.0) < 1e-9
    assert st.spectral_class(5772) == "G" and st.spectral_class(3000) == "M" and st.spectral_class(0) == "BH"


def test_planetary_systems(galaxy):
    p = galaxy.planets
    assert galaxy.n_planets == int(galaxy.planet_counts.sum())
    for s in range(0, galaxy.n_stars, 37):
        rows = list(galaxy.planet_rows(s))
        assert all(int(p["star"][r]) == s for r in rows)
        a = [p["a"][r] for r in rows]
        assert a == sorted(a)                  # orbits ordered outward
    kinds = set(np.unique(p["kind"]).tolist())
    assert {KIND_ROCKY, KIND_GAS} <= kinds | {KIND_ICE}
    giants = p["kind"] != KIND_ROCKY
    assert np.all(p["mass"][giants] >= 8) and np.all(p["moons"][giants] >= 1)
    assert np.all((p["habitability"] >= 0) & (p["habitability"] <= 1))
    assert np.all(p["habitability"][giants] == 0)
    lo, hi = st.habitable_zone(np.array([1.0]))
    assert 0.8 < lo[0] < 1.0 and 1.3 < hi[0] < 1.5
    # the energy-balance equilibrium of an Earth analogue is temperate (Budyko's A already
    # contains Earth's greenhouse effect, so the extra greenhouse term is 0 here)
    assert 10 < mean_surface_temp(1.0, 0.3, 0.0) < 25
    assert mean_surface_temp(1.0, 0.3, 20.0) > mean_surface_temp(1.0, 0.3, 0.0)
    d = galaxy.system_dict(int(p["star"][0]))
    assert d["planets"] and d["planets"][0]["id"].startswith("P-")


def test_focus_world_choice(galaxy):
    rows = galaxy.choose_focus_worlds(2)
    assert 1 <= len(rows) <= 2
    p = galaxy.planets
    for r in rows:
        assert p["kind"][r] == KIND_ROCKY and p["life"][r] == 1
    assert len({int(p["star"][r]) for r in rows}) == len(rows)   # independent systems


def test_orbits_are_keplerian(galaxy):
    p = galaxy.planets
    row = int(np.argmax(p["e"]))
    a, e, per = float(p["a"][row]), float(p["e"][row]), float(p["period"][row])
    for t in np.linspace(0, per, 9):
        x, y, z = galaxy.orbit_position(row, t)
        r = math.hypot(x, y)
        assert a * (1 - e) - 1e-6 <= r <= a * (1 + e) + 1e-6
    p0 = galaxy.orbit_position(row, 0.3)
    p1 = galaxy.orbit_position(row, 0.3 + per)
    assert np.allclose(p0, p1, atol=1e-6)        # periodic
    # Kepler's third law (solar units): P² = a³ / M
    m_star = float(galaxy.mass[int(p["star"][row])])
    assert abs(per ** 2 - a ** 3 / max(m_star, 0.05)) < 1e-6 * max(1.0, per ** 2)


def test_f0_biospheres_and_galactic_history():
    g = Galaxy.generate(99, 600)
    live = np.where(~np.isnan(g.f0_biomass))[0]
    assert live.size > 0, "some unfocused planets carry F0 aggregate life"
    excluded = int(live[0])
    before = g.f0_biomass[excluded]
    events = []
    for k in range(40):
        events += g.step_f0(5000.0, k, {excluded})
    assert g.f0_biomass[excluded] == before            # F4 worlds are not touched by the F0 model
    b = g.f0_biomass[~np.isnan(g.f0_biomass)]
    assert np.all((b >= 0) & (b <= 1))
    assert g.time_years == pytest.approx(40 * 5000.0)
    assert {e["type"] for e in events} <= {"F0_IMPACT", "F0_STERILIZED"}


def test_state_roundtrip_and_digest_guard():
    g = Galaxy.generate(5, 300)
    g.step_f0(1000.0, 1, set())
    sd = g.state_dict()
    h = Galaxy.generate(5, 300)
    h.load_state_dict(sd)
    assert np.allclose(np.nan_to_num(h.f0_biomass, nan=-1), np.nan_to_num(g.f0_biomass, nan=-1))
    other = Galaxy.generate(6, 300)
    with pytest.raises(RuntimeError):
        other.load_state_dict(sd)        # world generation changed: never a silent change
