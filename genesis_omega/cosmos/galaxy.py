"""GalaxyEngine + StarSystemEngine (spec §5-8, §86, §108-110, §330).

The galaxy is generated deterministically from the cosmos seed: a spiral disk
plus bulge, stars from a Kroupa IMF, stellar stages from age, a central
supermassive black hole, planetary systems with Keplerian orbits, and
equilibrium climates for every planet. Life-bearing worlds that are not
actively simulated run an F0 *population-field* model (aggregate biomass);
they contain no individuals until promoted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..core.rng import np_rng, derive_seed
from ..core.ids import star_label, planet_label
from . import stars as st

KIND_ROCKY, KIND_GAS, KIND_ICE = 0, 1, 2
KIND_NAMES = {0: "rocky", 1: "gas giant", 2: "ice giant"}

# Budyko-Sellers linearised outgoing radiation (W m^-2, W m^-2 °C^-1)
EBM_A = 203.3
EBM_B = 2.09
S_EARTH = 1361.0


def mean_surface_temp(flux_rel, albedo, greenhouse):
    """Global-mean equilibrium surface temperature [°C] of the EBM."""
    absorbed = (1.0 - albedo) * (S_EARTH / 4.0) * flux_rel
    return (absorbed - (EBM_A - greenhouse)) / EBM_B


@dataclass
class Galaxy:
    seed: int
    n_stars: int
    # stars
    pos: np.ndarray = field(repr=False, default=None)        # kpc (N,3)
    mass: np.ndarray = field(repr=False, default=None)
    lum: np.ndarray = field(repr=False, default=None)
    rad: np.ndarray = field(repr=False, default=None)
    teff: np.ndarray = field(repr=False, default=None)
    age: np.ndarray = field(repr=False, default=None)
    life_ms: np.ndarray = field(repr=False, default=None)
    stage: np.ndarray = field(repr=False, default=None)
    belt_au: np.ndarray = field(repr=False, default=None)
    # planets (columnar)
    planets: dict = field(repr=False, default_factory=dict)
    planet_offsets: np.ndarray = field(repr=False, default=None)  # star -> first planet row
    planet_counts: np.ndarray = field(repr=False, default=None)
    # dynamic F0 state
    f0_biomass: np.ndarray = field(repr=False, default=None)      # per planet row (nan = sterile)
    time_years: float = 0.0
    smbh_mass: float = 4.0e6

    # ------------------------------------------------------------------ build
    @staticmethod
    def generate(seed: int, n_stars: int) -> "Galaxy":
        g = Galaxy(seed=seed, n_stars=n_stars)
        rng = np_rng(seed, "galaxy")
        n_bulge = int(n_stars * 0.14)
        n_disk = n_stars - n_bulge
        n_arms = int(rng.integers(2, 5))
        pitch = math.radians(rng.uniform(11, 20))
        r_scale = 3.4
        r = np.minimum(rng.exponential(r_scale, n_disk) + 0.6, 16.0)
        arm = rng.integers(0, n_arms, n_disk)
        theta0 = arm * (2 * math.pi / n_arms)
        theta = theta0 + np.log(r) / math.tan(pitch) + rng.normal(0, 0.28, n_disk) * (1.2 - r / 18)
        # some inter-arm stars
        inter = rng.random(n_disk) < 0.25
        theta[inter] = rng.uniform(0, 2 * math.pi, inter.sum())
        z = rng.normal(0, 0.12 + 0.02 * r, n_disk)
        disk = np.stack([r * np.cos(theta), r * np.sin(theta), z], 1)
        br = np.abs(rng.normal(0, 1.1, n_bulge)) + 0.05
        bv = rng.normal(0, 1, (n_bulge, 3))
        bv /= np.linalg.norm(bv, axis=1, keepdims=True)
        bulge = bv * br[:, None] * np.array([1.0, 1.0, 0.6])
        g.pos = np.concatenate([disk, bulge]).astype(np.float32)
        mass0 = st.sample_kroupa(rng, n_stars)
        age = np.concatenate([rng.uniform(0.05, 10.0, n_disk), rng.uniform(6.0, 12.0, n_bulge)])
        mass, lum, rad, teff, life, stage = st.evolve(rng, mass0, age)
        g.mass, g.lum, g.rad, g.teff = mass, lum, rad, teff
        g.age, g.life_ms, g.stage = age, life, stage
        g._generate_systems()
        g._init_f0()
        return g

    def _generate_systems(self) -> None:
        cols = {k: [] for k in (
            "star", "index", "kind", "a", "e", "mass", "radius", "gravity", "period", "tilt", "day_factor",
            "albedo", "pressure", "greenhouse", "water", "t_mean", "habitability", "life", "moons", "rings",
            "m0", "omega", "incl")}
        offsets = np.zeros(self.n_stars, dtype=np.int64)
        counts = np.zeros(self.n_stars, dtype=np.int64)
        belts = np.zeros(self.n_stars, dtype=np.float32)
        row = 0
        for s in range(self.n_stars):
            rng = np_rng(self.seed, "system", s)
            offsets[s] = row
            stage = int(self.stage[s])
            teff = float(self.teff[s])
            lum = float(self.lum[s])
            m_star = float(self.mass[s])
            if stage in (st.STAGE_NEUTRON, st.STAGE_BLACK_HOLE):
                lam = 0.15
            elif stage == st.STAGE_WHITE_DWARF:
                lam = 0.6
            elif teff < 3700:
                lam = 2.6
            else:
                lam = 3.6
            n = int(min(rng.poisson(lam), 9))
            snow = 2.7 * math.sqrt(max(lum, 1e-4)) if stage == st.STAGE_MAIN else 2.7 * math.sqrt(max(lum, 1e-4))
            a = rng.uniform(0.04, 0.35) * max(m_star, 0.1) ** 0.4
            prev_kind = -1
            for k in range(n):
                if k:
                    a *= rng.uniform(1.45, 2.3)
                if a > 60:
                    break
                if a < snow:
                    kind = KIND_ROCKY
                    pm = float(np.exp(rng.normal(-0.2, 0.9)))
                    pm = min(max(pm, 0.03), 12.0)
                    pr = pm ** 0.27
                else:
                    if rng.random() < 0.58:
                        kind = KIND_GAS
                        pm = float(np.exp(rng.uniform(math.log(25), math.log(1500))))
                        pr = 11.0 * (pm / 318.0) ** -0.04
                    else:
                        kind = KIND_ICE
                        pm = float(rng.uniform(8, 30))
                        pr = 3.9 * (pm / 17.0) ** 0.3
                if prev_kind == KIND_ROCKY and kind != KIND_ROCKY and belts[s] == 0 and rng.random() < 0.6:
                    belts[s] = a / rng.uniform(1.3, 1.8)
                prev_kind = kind
                grav = pm / pr**2
                period = math.sqrt(a**3 / max(m_star, 0.05))
                flux = lum / a**2
                if kind == KIND_ROCKY:
                    if grav > 0.35:
                        pressure = float(pm ** 1.3 * math.exp(rng.normal(0, 1.2)))
                    else:
                        pressure = float(0.01 * math.exp(rng.normal(0, 1.0)))
                    t_eq_guess = mean_surface_temp(flux, 0.3, 0)
                    if t_eq_guess > 120 and grav < 1.5:
                        pressure *= 0.05  # stripped
                    greenhouse = 25.0 * math.log10(max(pressure, 1e-4)) * (1 if pressure < 3 else 1.6)
                    greenhouse = float(min(max(greenhouse, -90.0), 160.0))
                    water = float(np.clip(rng.beta(2.2, 1.8), 0, 0.97)) if pressure > 0.03 else 0.0
                    albedo = float(np.clip(0.18 + 0.25 * rng.random(), 0.1, 0.6))
                    t_mean = mean_surface_temp(flux, albedo, greenhouse)
                    if t_mean > 100:
                        water = 0.0
                    rings = 0
                    moons = int(rng.integers(0, 3))
                else:
                    pressure = 1000.0
                    greenhouse = 0.0
                    water = 0.0
                    albedo = float(rng.uniform(0.3, 0.55))
                    t_mean = mean_surface_temp(flux, albedo, 0)
                    rings = int(rng.random() < (0.45 if kind == KIND_GAS else 0.2))
                    moons = int(rng.integers(1, 9))
                hab = self._habitability(kind, t_mean, water, grav, pressure, stage, teff)
                life = int(kind == KIND_ROCKY and rng.random() < 0.3 * hab ** 2)
                vals = dict(star=s, index=k, kind=kind, a=a, e=float(min(rng.rayleigh(0.05), 0.5)), mass=pm,
                            radius=pr, gravity=grav, period=period, tilt=float(abs(rng.normal(18, 12)) % 60),
                            day_factor=float(np.exp(rng.normal(0, 0.35))), albedo=albedo, pressure=pressure,
                            greenhouse=greenhouse, water=water, t_mean=t_mean, habitability=hab, life=life,
                            moons=moons, rings=rings, m0=float(rng.uniform(0, 2 * math.pi)),
                            omega=float(rng.uniform(0, 2 * math.pi)), incl=float(rng.normal(0, 0.03)))
                for kk, vv in vals.items():
                    cols[kk].append(vv)
                row += 1
                counts[s] += 1
        dtypes = {"star": np.int32, "index": np.int16, "kind": np.int8, "life": np.int8, "moons": np.int8,
                  "rings": np.int8}
        self.planets = {k: np.asarray(v, dtype=dtypes.get(k, np.float64)) for k, v in cols.items()}
        self.planet_offsets = offsets
        self.planet_counts = counts
        self.belt_au = belts

    @staticmethod
    def _habitability(kind, t_mean, water, grav, pressure, stage, teff) -> float:
        if kind != KIND_ROCKY or stage not in (st.STAGE_MAIN,):
            return 0.0
        h_t = math.exp(-((t_mean - 14.0) / 24.0) ** 2)
        h_w = min(1.0, water / 0.25) * (1.0 if water < 0.96 else 0.4)
        h_g = 1.0 if 0.3 <= grav <= 3.0 else 0.15
        h_p = 1.0 if 0.05 <= pressure <= 25 else 0.1
        h_s = 1.0 if 3000 <= teff <= 7400 else 0.25
        return float(h_t * h_w * h_g * h_p * h_s)

    def _init_f0(self) -> None:
        n = len(self.planets["star"])
        b = np.full(n, np.nan)
        life = self.planets["life"].astype(bool)
        b[life] = 0.05 + 0.5 * self.planets["habitability"][life]
        self.f0_biomass = b

    # ------------------------------------------------------------- queries
    @property
    def n_planets(self) -> int:
        return len(self.planets["star"])

    def planet_rows(self, star: int) -> range:
        o = int(self.planet_offsets[star])
        return range(o, o + int(self.planet_counts[star]))

    def planet_dict(self, row: int) -> dict:
        p = {k: (v[row].item() if hasattr(v[row], "item") else v[row]) for k, v in self.planets.items()}
        p["row"] = int(row)
        p["id"] = planet_label(int(p["star"]), int(p["index"]))
        p["kind_name"] = KIND_NAMES[int(p["kind"])]
        bm = self.f0_biomass[row]
        p["f0_biomass"] = None if np.isnan(bm) else float(bm)
        return p

    def star_dict(self, s: int) -> dict:
        hz = st.habitable_zone(self.lum[s])
        return {
            "index": int(s), "id": star_label(s), "pos": [float(x) for x in self.pos[s]],
            "mass": float(self.mass[s]), "luminosity": float(self.lum[s]), "radius": float(self.rad[s]),
            "teff": float(self.teff[s]), "age_gyr": float(self.age[s]), "stage": st.STAGE_NAMES[int(self.stage[s])],
            "class": st.spectral_class(float(self.teff[s])), "hz_au": [float(hz[0]), float(hz[1])],
            "belt_au": float(self.belt_au[s]), "n_planets": int(self.planet_counts[s]),
        }

    def system_dict(self, s: int) -> dict:
        d = self.star_dict(s)
        d["planets"] = [self.planet_dict(r) for r in self.planet_rows(s)]
        return d

    def choose_focus_worlds(self, k: int) -> list[int]:
        """Pick planet rows for F4 simulation: the best life-bearing rocky worlds
        around G/K main-sequence stars, one per system (independent histories)."""
        p = self.planets
        cand = np.where((p["life"] == 1) & (p["kind"] == KIND_ROCKY))[0]
        score = p["habitability"][cand].copy()
        teff = self.teff[p["star"][cand]]
        score *= np.where((teff > 3900) & (teff < 6400), 1.0, 0.75)
        order = cand[np.argsort(-score, kind="stable")]
        chosen, systems = [], set()
        for r in order:
            if int(p["star"][r]) in systems:
                continue
            chosen.append(int(r))
            systems.add(int(p["star"][r]))
            if len(chosen) >= k:
                break
        if not chosen:
            # No life-bearing rocky planet arose from the generator: seed the most
            # habitable one. This is a PROGRAMMED initial condition and is logged.
            rocky = np.where(p["kind"] == KIND_ROCKY)[0]
            best = int(rocky[np.argmax(p["habitability"][rocky])])
            p["life"][best] = 1
            self.f0_biomass[best] = 0.1
            chosen = [best]
        return chosen

    # ------------------------------------------------------------- dynamics
    def step_f0(self, dt_years: float, tick: int, exclude_rows: set[int]) -> list[dict]:
        """Advance F0 aggregate biospheres. Returns galactic events (dicts)."""
        events = []
        if dt_years <= 0:
            return events
        rng = np_rng(self.seed, "f0", tick)
        b = self.f0_biomass
        live = np.where(~np.isnan(b))[0]
        live = np.array([r for r in live if r not in exclude_rows], dtype=np.int64)
        if live.size == 0:
            self.time_years += dt_years
            return events
        hab = self.planets["habitability"][live]
        k = np.clip(hab, 0.05, 1.0)
        r = 0.02 + 0.08 * hab
        old = b[live].copy()
        noise = rng.normal(0, 0.01 * math.sqrt(dt_years), live.size)
        new = old + dt_years * r * old * (1 - old / k) + noise * old
        # rare impacts (aggregate catastrophe model)
        p_imp = 1 - math.exp(-2e-5 * dt_years)
        hit = rng.random(live.size) < p_imp
        new[hit] *= rng.uniform(0.05, 0.6, hit.sum())
        new = np.clip(new, 0.0, 1.0)
        for i in np.where(hit)[0]:
            row = int(live[i])
            events.append({"type": "F0_IMPACT", "planet_row": row, "before": float(old[i]), "after": float(new[i])})
        died = new < 1e-3
        for i in np.where(died)[0]:
            events.append({"type": "F0_STERILIZED", "planet_row": int(live[i]), "before": float(old[i])})
        new[died] = np.nan
        b[live] = new
        self.time_years += dt_years
        return events

    # -------------------------------------------------------- serialisation
    def static_digest(self) -> str:
        import hashlib
        h = hashlib.sha256()
        h.update(self.pos.tobytes())
        h.update(self.mass.tobytes())
        h.update(self.planets["a"].tobytes())
        return h.hexdigest()[:16]

    def state_dict(self) -> dict:
        return {"seed": self.seed, "n_stars": self.n_stars, "time_years": self.time_years,
                "f0_biomass": self.f0_biomass.copy(), "life": self.planets["life"].copy(),
                "digest": self.static_digest()}

    def load_state_dict(self, d: dict) -> None:
        if d.get("digest") != self.static_digest():
            raise RuntimeError("galaxy static digest mismatch: world generation changed; refusing silent change")
        self.time_years = float(d["time_years"])
        self.f0_biomass = np.asarray(d["f0_biomass"], dtype=np.float64).copy()
        self.planets["life"] = np.asarray(d["life"], dtype=np.int8).copy()

    # -------------------------------------------------------------- orbits
    def orbit_position(self, row: int, time_years: float | None = None) -> tuple[float, float, float]:
        """Keplerian position (AU) of planet ``row`` at galaxy time."""
        t = self.time_years if time_years is None else time_years
        p = self.planets
        a, e, per = float(p["a"][row]), float(p["e"][row]), float(p["period"][row])
        m = (float(p["m0"][row]) + 2 * math.pi * t / max(per, 1e-6)) % (2 * math.pi)
        E = m
        for _ in range(8):
            E -= (E - e * math.sin(E) - m) / (1 - e * math.cos(E))
        x = a * (math.cos(E) - e)
        y = a * math.sqrt(1 - e * e) * math.sin(E)
        w = float(p["omega"][row])
        xr = x * math.cos(w) - y * math.sin(w)
        yr = x * math.sin(w) + y * math.cos(w)
        return xr, yr, xr * float(p["incl"][row])


def galaxy_seed(cosmos_seed: int) -> int:
    return derive_seed(cosmos_seed, "galaxy-seed")
