"""Laws of the artificial universe (PROGRAMMED, versioned by PHYSICS_VERSION).

Every constant that shapes what organisms experience lives here, so a change
to any of them is visible, hashed into checkpoints, and requires a new
physics version (spec §430: never silently change science).

:class:`XenosProfile` derives per-planet variations (gravity, atmosphere,
light, day/year length) from the planet itself, so different worlds impose
genuinely different pressures (spec §44-46).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, asdict

from ..version import PHYSICS_VERSION


@dataclass(frozen=True)
class WorldPhysics:
    # --- climate (energy-balance model, W m^-2, days) ---
    c_land: float = 12.0
    c_ocean: float = 90.0
    d_heat: float = 0.65
    albedo_land: float = 0.25
    albedo_ocean: float = 0.07
    albedo_ice: float = 0.62
    albedo_cloud: float = 0.22
    lapse_rate: float = 6.5            # °C per km
    evap_ocean: float = 0.9
    evap_land: float = 0.12
    qsat0: float = 0.6
    qsat_k: float = 0.065
    precip_rate: float = 0.45
    moist_mix: float = 6.0            # per day, horizontal moisture mixing
    # --- primary production (resource field) ---
    growth_rate: float = 0.012         # per tick at optimum
    plant_t_opt: float = 22.0
    plant_t_width: float = 15.0
    k_land: float = 2.0
    k_ocean: float = 1.0
    trample: float = 0.6
    refuge: float = 0.08              # fraction of capacity grazers cannot remove
    reseed: float = 2e-5
    # --- organisms ---
    max_speed: float = 0.9             # cells per tick at full thrust
    max_turn: float = 0.7              # radians per tick
    metab_mass: float = 0.005            # × mass^0.75 (Kleiber-like scaling)
    metab_unit: float = 0.00025        # per active substrate unit
    move_cost: float = 0.006
    emit_cost: float = 0.02
    dig_cost: float = 0.008
    build_cost: float = 0.006
    process_cost: float = 0.05
    strike_cost: float = 0.02
    mark_cost: float = 0.012
    give_frac: float = 0.15
    store_frac: float = 0.12
    bite: float = 0.22                 # resource units per tick at full capacity
    eat_eff: float = 3.2               # energy per resource unit
    energy_cap_base: float = 6.0
    energy_cap_storage: float = 18.0
    thermal_damage: float = 0.0008     # integrity per °C beyond tolerance per tick
    thermal_rate: float = 0.06
    tolerance_cost: float = 0.00015
    repair_rate: float = 0.003
    repair_cost: float = 0.004
    senescence: float = 0.002
    lifespan_days: float = 3.0
    maturity_frac: float = 0.18
    repro_threshold: float = 0.55
    repro_overhead: float = 0.15
    body_cost: float = 0.35           # energy per unit of offspring body mass
    strike_damage: float = 0.25
    kill_transfer: float = 0.6
    sense_dist: float = 1.5            # cells
    sensor_noise: float = 0.03
    # --- matter / construction ---
    dig_amount: float = 0.5
    wall_height: float = 1.0
    max_struct_height: float = 4.0
    struct_decay: float = 0.0004
    store_decay: float = 0.002
    compaction_gain: float = 0.02
    compaction_decay: float = 0.0008
    compaction_speed: float = 0.5
    mark_gain: float = 0.3
    mark_decay: float = 0.0005
    tool_gain: float = 0.8
    # --- signal fields ---
    emit_gain: float = 1.0
    scent_decay: float = 0.7
    # --- seed bank (dormant propagules: a declared law) ---
    bank_deposit_prob: float = 0.08
    bank_germination: float = 0.002   # per entry per tick when population is depleted

    def digest(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob + PHYSICS_VERSION.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["version"] = PHYSICS_VERSION
        return d


@dataclass
class XenosProfile:
    """Planet-derived modifiers. Different stars, atmospheres and gravities give
    different sensory and energetic regimes, so lineages are not reskins."""

    planet_id: str
    gravity: float
    albedo: float
    pressure: float
    star_teff: float
    flux_rel: float
    greenhouse: float
    water: float
    tilt_deg: float
    eccentricity: float
    day_ticks: int
    year_days: int
    move_scale: float
    sound_decay: float
    light_scale: float
    struct_scale: float
    star_color: tuple

    @property
    def year_ticks(self) -> int:
        return self.day_ticks * self.year_days

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_planet(planet: dict, star: dict, base_day_ticks: int) -> "XenosProfile":
        g = float(planet["gravity"])
        p = float(planet["pressure"])
        teff = float(star["teff"])
        flux = float(star["luminosity"]) / float(planet["a"]) ** 2
        day = int(round(base_day_ticks * min(max(float(planet["day_factor"]), 0.5), 2.0) / 8) * 8)
        year_days = int(min(max(round(40 * math.sqrt(max(float(planet["period"]), 0.05))), 24), 96))
        return XenosProfile(
            planet_id=str(planet["id"]),
            gravity=g,
            albedo=float(planet["albedo"]),
            pressure=p,
            star_teff=teff,
            flux_rel=flux,
            greenhouse=float(planet["greenhouse"]),
            water=float(planet["water"]),
            tilt_deg=float(planet["tilt"]),
            eccentricity=float(planet["e"]),
            day_ticks=max(day, 48),
            year_days=year_days,
            move_scale=float(min(max(g, 0.2), 4.0) ** 0.6),
            sound_decay=float(0.35 + 0.45 * p / (p + 1.0)),
            light_scale=float(min(max(teff / 5800.0, 0.4), 1.3)),
            struct_scale=float(1.0 / math.sqrt(min(max(g, 0.2), 4.0))),
            star_color=_blackbody_rgb(teff),
        )


def _blackbody_rgb(teff: float) -> tuple:
    """Approximate sRGB of a blackbody (Tanner Helland fit), 0-1 floats."""
    t = max(1000.0, min(40000.0, teff)) / 100.0
    if t <= 66:
        r = 255.0
        g = 99.4708025861 * math.log(t) - 161.1195681661
        b = 0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307
    else:
        r = 329.698727446 * (t - 60) ** -0.1332047592
        g = 288.1221695283 * (t - 60) ** -0.0755148492
        b = 255.0
    return tuple(round(min(max(c, 0.0), 255.0) / 255.0, 4) for c in (r, g, b))
