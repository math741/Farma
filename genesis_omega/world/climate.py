"""EnvironmentEngine — climate (spec §5, §8, §270).

A diurnal + seasonal energy-balance model (Budyko–Sellers radiation,
conservative diffusion, land/ocean heat capacity, ice-albedo and cloud
feedback) and a bulk moisture cycle run on a coarse grid (slow clock). Fine
fields are derived by periodic bilinear upsampling and an elevation lapse
rate. Clouds shown by the viewer are this humidity field — not decoration.
"""

from __future__ import annotations

import math

import torch

from ..cosmos.galaxy import EBM_A, EBM_B, S_EARTH
from .geometry import SphereGrid
from .physics import WorldPhysics, XenosProfile
from .terrain import Terrain


class Climate:
    def __init__(self, grid: SphereGrid, div: int, terrain: Terrain, xenos: XenosProfile, phys: WorldPhysics):
        self.fine = grid
        self.div = div
        self.cg = SphereGrid(grid.W // div, grid.H // div, grid.device)
        self.xenos = xenos
        self.phys = phys
        land = (~terrain.ocean).float()
        self.land = SphereGrid.downsample(land, div)
        self.elev_pos = terrain.elev_km.clamp(min=0)
        self.C = phys.c_land * self.land + phys.c_ocean * (1 - self.land)
        Hc = self.cg.H
        self.Dg = phys.d_heat * (Hc / math.pi) ** 2
        self.albedo_offset = 0.0
        self._reset_state(15.0)

    def _reset_state(self, t_mean: float) -> None:
        lat = self.cg.lat.view(-1, 1)
        t0 = t_mean + 14.0 - 32.0 * torch.sin(lat) ** 2 + 0.0 * self.land
        self.T = t0.expand_as(self.land).clone()
        self.q = torch.full_like(self.T, 0.6)
        self.precip_mean = torch.full_like(self.T, 0.3)
        self.cloud = torch.full_like(self.T, 0.4)
        self.S_mean = torch.full_like(self.T, 300.0)

    def calibrate(self, t_target: float, start_tick: int, climate_every: int, iters: int = 6) -> dict:
        """Choose the bulk-albedo offset so the warm-branch equilibrium of this
        detailed model matches the generator's global-mean temperature. If the
        target cannot be reached (e.g. a snowball branch), the world keeps the
        closest state found — it is not forced."""
        S_glob = S_EARTH / 4.0 * self.xenos.flux_rel
        hist = []
        for _ in range(iters):
            self._reset_state(t_target)
            self.spin_up(start_tick, 1.2, climate_every)
            tg = self.global_mean_temperature()
            hist.append((round(self.albedo_offset, 4), round(tg, 2)))
            err = tg - t_target
            if abs(err) < 0.75:
                break
            self.albedo_offset = float(min(max(self.albedo_offset + err * EBM_B / S_glob, -0.25), 0.35))
        return {"target": t_target, "achieved": self.global_mean_temperature(), "iterations": hist,
                "albedo_offset": self.albedo_offset}

    # ------------------------------------------------------------- astronomy
    def solar(self, tick: int):
        x = self.xenos
        yt = max(x.year_ticks, 1)
        theta = 2 * math.pi * ((tick % yt) / yt)
        decl = math.radians(x.tilt_deg) * math.sin(theta)
        frac = (tick % x.day_ticks) / x.day_ticks
        lon_s = math.pi - 2 * math.pi * frac
        dist = 1.0 + 2.0 * x.eccentricity * math.cos(theta)
        return decl, lon_s, dist, theta

    def insolation(self, grid: SphereGrid, tick: int) -> torch.Tensor:
        decl, lon_s, dist, _ = self.solar(tick)
        cosdl = grid.coslon * math.cos(lon_s) + grid.sinlon * math.sin(lon_s)
        cosz = grid.sinlat * math.sin(decl) + grid.coslat * math.cos(decl) * cosdl
        return S_EARTH * self.xenos.flux_rel * dist * cosz.clamp(min=0)

    def subsolar_point(self, tick: int) -> tuple[float, float]:
        decl, lon_s, _, _ = self.solar(tick)
        lon = (lon_s + math.pi) % (2 * math.pi) - math.pi
        return math.degrees(decl), math.degrees(lon)

    # ------------------------------------------------------------------ step
    def daily_mean_insolation(self, grid: SphereGrid, tick: int) -> torch.Tensor:
        decl, _, dist, _ = self.solar(tick)
        sl, cl = grid.sinlat, grid.coslat
        x = (-(sl / cl.clamp_min(1e-6)) * math.tan(decl)).clamp(-1, 1)
        h0 = torch.acos(x)
        q = (h0 * sl * math.sin(decl) + cl * math.cos(decl) * torch.sin(h0)) / math.pi
        return (S_EARTH * self.xenos.flux_rel * dist * q).clamp(min=0).expand(grid.H, grid.W)

    def step(self, tick: int, dt_days: float, daily_mean: bool = False) -> None:
        p = self.phys
        S = self.daily_mean_insolation(self.cg, tick) if daily_mean else self.insolation(self.cg, tick)
        ice = torch.sigmoid(-(self.T + 2.0) / 2.0)
        alb_surf = (p.albedo_land * self.land + p.albedo_ocean * (1 - self.land)) * (1 - ice) + p.albedo_ice * ice
        alb = (alb_surf + p.albedo_cloud * self.cloud + self.albedo_offset).clamp(0.02, 0.9)
        absorbed = S * (1 - alb)
        A_eff = EBM_A - self.xenos.greenhouse
        kx_max = 6.0
        stab = self.Dg * dt_days * (2 + 2 * kx_max) / float(self.C.min())
        n_sub = max(1, int(math.ceil(stab / 0.8)))
        h = dt_days / n_sub
        for _ in range(n_sub):
            out = A_eff + EBM_B * self.T
            self.T = self.T + h / self.C * (absorbed - out + self.Dg * self.cg.laplacian(self.T, kx_max))
        # moisture
        T = self.T
        evap = p.evap_ocean * (1 - self.land) * ((T + 10) / 35).clamp(0, 1.5) + \
            p.evap_land * self.land * ((T + 5) / 35).clamp(0, 1.0)
        qs = p.qsat0 * torch.exp(p.qsat_k * T)
        precip = p.precip_rate * (self.q - qs).clamp(min=0) + 0.05 * self.q
        self.q = (self.q + dt_days * (evap - precip)).clamp(min=0)
        mix = min(1.0, p.moist_mix * dt_days)
        self.q = (1 - mix) * self.q + mix * self.cg.blur(self.q)
        a = min(1.0, dt_days / 10.0)
        self.precip_mean = (1 - a) * self.precip_mean + a * precip
        self.cloud = (self.q / qs).clamp(0, 1) ** 1.5 * (self.q / 0.8).clamp(0, 1)
        self.S_mean = (1 - a) * self.S_mean + a * S

    def spin_up(self, start_tick: int, years: float, climate_every: int) -> None:
        """Reach the seasonal equilibrium with daily-mean steps, then settle the
        diurnal cycle over the final day with full-resolution steps."""
        x = self.xenos
        n_days = int(years * x.year_days)
        for d in range(n_days):
            tick = start_tick - (n_days - d) * x.day_ticks
            self.step(tick, 1.0, daily_mean=True)
        dt = climate_every / x.day_ticks
        for k in range(x.day_ticks // climate_every):
            self.step(start_tick - x.day_ticks + k * climate_every, dt)

    # ----------------------------------------------------------- fine fields
    def fine_temperature(self) -> torch.Tensor:
        up = self.fine.upsample(self.T, self.fine.H, self.fine.W)
        return up - self.phys.lapse_rate * self.elev_pos

    def fine_precip(self) -> torch.Tensor:
        return self.fine.upsample(self.precip_mean, self.fine.H, self.fine.W).clamp(min=0)

    def fine_cloud(self) -> torch.Tensor:
        return self.fine.upsample(self.cloud, self.fine.H, self.fine.W).clamp(0, 1)

    def global_mean_temperature(self) -> float:
        return float((self.T * self.cg.area).sum())

    def state_dict(self) -> dict:
        return {"T": self.T, "q": self.q, "precip_mean": self.precip_mean, "cloud": self.cloud,
                "S_mean": self.S_mean, "albedo_offset": torch.tensor(self.albedo_offset)}

    def load_state_dict(self, d: dict) -> None:
        for k in ("T", "q", "precip_mean", "cloud", "S_mean"):
            setattr(self, k, d[k].to(self.cg.device))
        self.albedo_offset = float(d["albedo_offset"])
