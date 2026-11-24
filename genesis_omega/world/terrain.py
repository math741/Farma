"""TerrainEngine: planetary relief, oceans, soils and mineral deposits (spec §8, §15).

Relief comes from domain-warped fractal + ridged gradient noise sampled on the
sphere (no seams). Sea level is placed so the ocean covers the planet's water
fraction by area. Deposits of the base materials follow terrain affinities;
their physical properties are hidden laws drawn from the physics seed — the
generator gives them no names.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .geometry import SphereGrid
from .noise import GradientNoise3D


@dataclass
class Terrain:
    elev_km: torch.Tensor      # (H,W)
    ocean: torch.Tensor        # (H,W) bool
    slope: torch.Tensor        # (H,W) km per cell
    fertility: torch.Tensor    # (H,W) 0..1 (land soil / marine productivity)
    deposits: torch.Tensor     # (M,H,W) material units in the ground
    nat_shelter: torch.Tensor  # (H,W) 0..0.45

    def state_dict(self) -> dict:
        return {"elev_km": self.elev_km, "ocean": self.ocean, "slope": self.slope, "fertility": self.fertility,
                "deposits": self.deposits, "nat_shelter": self.nat_shelter}

    @staticmethod
    def from_state(d: dict) -> "Terrain":
        return Terrain(**{k: d[k] for k in ("elev_km", "ocean", "slope", "fertility", "deposits", "nat_shelter")})


def _smoothstep(e0, e1, x):
    t = ((x - e0) / (e1 - e0)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def weighted_quantile(values: torch.Tensor, weights: torch.Tensor, q: float) -> float:
    v = values.flatten()
    w = weights.flatten()
    order = torch.argsort(v)
    cw = torch.cumsum(w[order], 0)
    cw = cw / cw[-1]
    idx = int(torch.searchsorted(cw, torch.tensor([q], device=cw.device, dtype=cw.dtype)).clamp(0, v.numel() - 1))
    return float(v[order][idx])


def generate_terrain(grid: SphereGrid, seed: int, water_fraction: float, n_materials: int) -> Terrain:
    dev = grid.device
    nz = GradientNoise3D(seed, dev)
    nz2 = GradientNoise3D(seed + 1, dev)
    nz3 = GradientNoise3D(seed + 2, dev)
    p = grid.xyz
    warp = torch.stack([nz.fbm(p * 1.4 + 3.1, 3), nz2.fbm(p * 1.4 + 7.9, 3), nz3.fbm(p * 1.4 + 1.7, 3)], -1)
    q = p + 0.38 * warp
    continent = nz.fbm(q * 1.05, octaves=5)
    mountains = nz2.ridged(q * 2.7, octaves=4)
    detail = nz3.fbm(p * 9.0, octaves=3)
    raw = continent + 0.55 * mountains * _smoothstep(-0.05, 0.35, continent) + 0.07 * detail
    if water_fraction > 0.005:
        sea = weighted_quantile(raw, grid.area, min(water_fraction, 0.985))
    else:
        sea = float(raw.min()) - 1e-3
    hi, lo = float(raw.max()), float(raw.min())
    land_t = ((raw - sea) / max(hi - sea, 1e-6)).clamp(min=0)
    sea_t = ((sea - raw) / max(sea - lo, 1e-6)).clamp(min=0)
    elev = torch.where(raw >= sea, 5.5 * land_t.pow(1.25), -4.5 * sea_t.pow(0.55))
    ocean = raw < sea
    nb = grid.neighbors8(elev)
    slope = torch.stack([(n - elev).abs() for n in nb]).amax(0)
    relief = torch.stack([(n - elev).clamp(0, 1.0) for n in nb]).mean(0)
    nat_shelter = torch.where(ocean, torch.zeros_like(elev), (relief * 0.8).clamp(0, 0.45))
    n01 = 0.5 + 0.5 * nz3.fbm(p * 4.0 + 11.0, 3)
    soil = (1 - (elev / 4.5).clamp(0, 1)) * (1 - (slope / 1.5).clamp(0, 0.7)) * (0.6 + 0.4 * n01)
    marine = 0.3 + 0.7 * torch.exp(-elev.abs() / 1.5)
    fertility = torch.where(ocean, marine, soil).clamp(0, 1)
    # deposits: terrain affinities x patchiness
    aff = []
    e_pos = elev.clamp(min=0)
    for m in range(n_materials):
        patch = (0.5 + 0.5 * GradientNoise3D(seed + 100 + m, dev).fbm(p * (3.0 + m), 3)).clamp(0, 1)
        kind = m % 6
        if kind == 0:
            w = torch.exp(-e_pos / 0.4)
        elif kind == 1:
            w = (e_pos / 2.0).clamp(0, 1) + 0.5 * (slope / 1.0).clamp(0, 1)
        elif kind == 2:
            w = torch.full_like(elev, 0.6)
        elif kind == 3:
            w = ((patch - 0.62) / 0.38).clamp(0, 1) * 3.0
        elif kind == 4:
            w = torch.exp(-((e_pos - 1.0) / 0.8) ** 2)
        else:
            w = (mountains - 0.55).clamp(min=0) * 4.0 * (e_pos > 0.3).float()
        aff.append(w * (0.35 + patch))
    deposits = torch.stack(aff) * 30.0
    deposits = torch.where(ocean.unsqueeze(0), torch.zeros_like(deposits), deposits)
    return Terrain(elev_km=elev.float(), ocean=ocean, slope=slope.float(), fertility=fertility.float(),
                   deposits=deposits.float(), nat_shelter=nat_shelter.float())
