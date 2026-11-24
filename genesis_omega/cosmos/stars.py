"""Stellar population synthesis (spec §6, §108-109).

Stars are drawn from a Kroupa initial mass function and given main-sequence
luminosity, radius, effective temperature and lifetime from standard
power-law relations. Stars older than their main-sequence lifetime become
giants, white dwarfs, neutron stars or black holes. No nuclear physics is
simulated — these are population-level relations (PROGRAMMED).
"""

from __future__ import annotations

import numpy as np

STAGE_MAIN = 0
STAGE_GIANT = 1
STAGE_WHITE_DWARF = 2
STAGE_NEUTRON = 3
STAGE_BLACK_HOLE = 4
STAGE_NAMES = {0: "main sequence", 1: "giant", 2: "white dwarf", 3: "neutron star", 4: "black hole"}

T_SUN = 5772.0


def sample_kroupa(rng: np.random.Generator, n: int, m_min=0.08, m_break=0.5, m_max=40.0) -> np.ndarray:
    """Inverse-CDF sampling of the two-segment Kroupa IMF (α=1.3 below 0.5 M☉, 2.3 above)."""
    a1, a2 = 1.3, 2.3

    def seg_integral(lo, hi, a, norm):
        return norm * (hi ** (1 - a) - lo ** (1 - a)) / (1 - a)

    # continuity at the break: k1 m^-a1 = k2 m^-a2 at m_break
    k1 = 1.0
    k2 = k1 * m_break ** (a2 - a1)
    i1 = seg_integral(m_min, m_break, a1, k1)
    i2 = seg_integral(m_break, m_max, a2, k2)
    p1 = i1 / (i1 + i2)
    u = rng.random(n)
    seg = rng.random(n) < p1
    out = np.empty(n)
    # segment 1
    lo, hi = m_min ** (1 - a1), m_break ** (1 - a1)
    out[seg] = (lo + u[seg] * (hi - lo)) ** (1 / (1 - a1))
    lo, hi = m_break ** (1 - a2), m_max ** (1 - a2)
    out[~seg] = (lo + u[~seg] * (hi - lo)) ** (1 / (1 - a2))
    return out


def main_sequence(mass: np.ndarray):
    """Luminosity [L☉], radius [R☉], effective temperature [K], lifetime [Gyr]."""
    m = np.asarray(mass, dtype=np.float64)
    lum = np.where(m < 0.43, 0.23 * m**2.3,
          np.where(m < 2.0, m**4.0,
          np.where(m < 55.0, 1.4 * m**3.5, 32000.0 * m)))
    rad = np.where(m < 1.0, m**0.8, m**0.57)
    teff = T_SUN * (lum / rad**2) ** 0.25
    life = 10.0 * m ** -2.5
    return lum, rad, teff, life


def evolve(rng: np.random.Generator, mass, age_gyr):
    """Apply post-main-sequence evolution given an age. Returns arrays."""
    lum, rad, teff, life = main_sequence(mass)
    stage = np.full(mass.shape, STAGE_MAIN, dtype=np.int8)
    giant = (age_gyr > life) & (age_gyr < life * 1.12) & (mass < 25)
    dead = age_gyr >= np.where(mass < 25, life * 1.12, life * 1.02)
    stage[giant] = STAGE_GIANT
    lum = lum.copy(); rad = rad.copy(); teff = teff.copy()
    g = np.where(giant)[0]
    if g.size:
        rad[g] = rad[g] * rng.uniform(10, 80, g.size)
        teff[g] = rng.uniform(3300, 4800, g.size)
        lum[g] = rad[g] ** 2 * (teff[g] / T_SUN) ** 4
    d = np.where(dead)[0]
    if d.size:
        m = mass[d]
        wd = m < 8
        ns = (m >= 8) & (m < 25)
        bh = m >= 25
        stage[d[wd]] = STAGE_WHITE_DWARF
        stage[d[ns]] = STAGE_NEUTRON
        stage[d[bh]] = STAGE_BLACK_HOLE
        cool = np.clip(age_gyr[d] - life[d], 0.01, None)
        teff[d[wd]] = np.clip(25000 * cool[wd] ** -0.35, 4000, 60000)
        rad[d[wd]] = 0.012
        lum[d[wd]] = rad[d[wd]] ** 2 * (teff[d[wd]] / T_SUN) ** 4
        teff[d[ns]] = 1e6
        rad[d[ns]] = 1.6e-5
        lum[d[ns]] = 1e-3
        teff[d[bh]] = 0.0
        rad[d[bh]] = 0.0
        lum[d[bh]] = 0.0
        mass = mass.copy()
        mass[d[wd]] = np.clip(0.5 + 0.08 * m[wd], 0.5, 1.3)
        mass[d[ns]] = 1.4
        mass[d[bh]] = np.clip(m[bh] * 0.3, 5, 30)
    return mass, lum, rad, teff, life, stage


def habitable_zone(lum):
    """Conservative habitable zone (AU) from stellar luminosity."""
    lum = np.maximum(np.asarray(lum, dtype=np.float64), 1e-9)
    return np.sqrt(lum / 1.1), np.sqrt(lum / 0.53)


def spectral_class(teff: float) -> str:
    if teff <= 0:
        return "BH"
    for cls, lo in (("O", 30000), ("B", 10000), ("A", 7500), ("F", 6000), ("G", 5200), ("K", 3700)):
        if teff >= lo:
            return cls
    return "M"
