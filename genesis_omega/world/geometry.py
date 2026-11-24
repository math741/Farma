"""Spherical grid geometry (spec §9, §613).

Agents live on the unit sphere as 3D unit vectors with tangent headings, so
there is no pole singularity in motion. Fields live on an equirectangular
latitude/longitude grid with conservative (flux-form) diffusion. Any other
geometry only needs to provide the same small interface.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class SphereGrid:
    def __init__(self, width: int, height: int, device="cpu"):
        self.W = int(width)
        self.H = int(height)
        self.device = torch.device(device)
        H, W = self.H, self.W
        lat = math.pi / 2 - (torch.arange(H, dtype=torch.float64) + 0.5) * math.pi / H
        lon = -math.pi + (torch.arange(W, dtype=torch.float64) + 0.5) * 2 * math.pi / W
        self.lat = lat.float().to(self.device)
        self.lon = lon.float().to(self.device)
        cl = torch.cos(lat)
        self.coslat = cl.float().to(self.device).view(H, 1)
        self.sinlat = torch.sin(lat).float().to(self.device).view(H, 1)
        self.coslon = torch.cos(lon).float().to(self.device).view(1, W)
        self.sinlon = torch.sin(lon).float().to(self.device).view(1, W)
        x = (cl.view(H, 1) * torch.cos(lon).view(1, W))
        y = (cl.view(H, 1) * torch.sin(lon).view(1, W))
        z = torch.sin(lat).view(H, 1).expand(H, W)
        self.xyz = torch.stack([x, y, z], -1).float().to(self.device)  # (H,W,3)
        area = cl.view(H, 1).expand(H, W)
        self.area = (area / area.sum()).float().to(self.device)
        # interface cosines for meridional fluxes (between row i and i+1)
        lat_if = math.pi / 2 - (torch.arange(1, H, dtype=torch.float64)) * math.pi / H
        self.cos_if = torch.cos(lat_if).float().to(self.device).view(H - 1, 1)
        self.dang = 2 * math.pi / W  # angular width of an equatorial cell

    # ---------------------------------------------------------------- cells
    def latlon(self, p: torch.Tensor):
        lat = torch.asin(p[..., 2].clamp(-1.0, 1.0))
        lon = torch.atan2(p[..., 1], p[..., 0])
        return lat, lon

    def cell_of(self, p: torch.Tensor):
        lat, lon = self.latlon(p)
        i = ((math.pi / 2 - lat) / math.pi * self.H).floor().long().clamp_(0, self.H - 1)
        j = ((lon + math.pi) / (2 * math.pi) * self.W).floor().long() % self.W
        return i, j, i * self.W + j

    def cell_center(self, flat: torch.Tensor) -> torch.Tensor:
        return self.xyz.view(-1, 3)[flat]

    # --------------------------------------------------------------- motion
    @staticmethod
    def _norm(v: torch.Tensor) -> torch.Tensor:
        return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def move(self, p, h, ang):
        """Advance along the great circle through p in direction h by ``ang`` radians."""
        ang = ang.unsqueeze(-1)
        c, s = torch.cos(ang), torch.sin(ang)
        p2 = self._norm(p * c + h * s)
        h2 = h * c - p * s
        h2 = h2 - (h2 * p2).sum(-1, keepdim=True) * p2
        return p2, self._norm(h2)

    @staticmethod
    def left(p, h):
        return torch.cross(p, h, dim=-1)

    def turn(self, p, h, theta):
        theta = theta.unsqueeze(-1)
        h2 = h * torch.cos(theta) + self.left(p, h) * torch.sin(theta)
        h2 = h2 - (h2 * p).sum(-1, keepdim=True) * p
        return self._norm(h2)

    def offset_points(self, p, h, phis, ang):
        """Points at angular distance ``ang`` in directions rotated by ``phis`` from h.
        Returns (N, len(phis), 3)."""
        lft = self.left(p, h)
        out = []
        for phi in phis:
            d = h * math.cos(phi) + lft * math.sin(phi)
            q = p * math.cos(ang) + d * math.sin(ang)
            out.append(q)
        return self._norm(torch.stack(out, 1))

    def random_tangent(self, p, gen):
        v = torch.randn(p.shape, generator=gen, device=p.device)
        v = v - (v * p).sum(-1, keepdim=True) * p
        return self._norm(v)

    def random_points_near(self, center, n, spread_ang, gen):
        """n points scattered around unit vector ``center`` (3,) within ~spread_ang."""
        c = center.view(1, 3).expand(n, 3)
        h = self.random_tangent(c, gen)
        ang = torch.rand(n, generator=gen, device=center.device) * spread_ang
        p, _ = self.move(c, h, ang)
        return p

    # --------------------------------------------------------------- fields
    def laplacian(self, f: torch.Tensor, kx_max: float = 6.0) -> torch.Tensor:
        """Conservative spherical Laplacian (grid units), longitude-periodic, no flux at poles."""
        inv_c = 1.0 / self.coslat
        kx = (inv_c * inv_c).clamp(max=kx_max)
        ew = (torch.roll(f, -1, -1) + torch.roll(f, 1, -1) - 2 * f) * kx
        d = f[..., 1:, :] - f[..., :-1, :]
        flux = d * self.cos_if  # (…,H-1,W)
        zeros = torch.zeros_like(f[..., :1, :])
        up = torch.cat([zeros, flux], -2)      # flux from row i-1 into i (as +)
        down = torch.cat([flux, zeros], -2)    # flux from row i+1 into i
        ns = (down - up) * inv_c
        return ew + ns

    def blur(self, f: torch.Tensor) -> torch.Tensor:
        """Separable [1,2,1]/4 blur, periodic in longitude, replicate at poles."""
        g = 0.25 * torch.roll(f, 1, -1) + 0.5 * f + 0.25 * torch.roll(f, -1, -1)
        north = torch.cat([g[..., :1, :], g[..., :-1, :]], -2)
        south = torch.cat([g[..., 1:, :], g[..., -1:, :]], -2)
        return 0.25 * north + 0.5 * g + 0.25 * south

    def neighbors8(self, f: torch.Tensor):
        """List of the 8 neighbour-shifted copies of f (periodic lon, replicate lat)."""
        out = []
        for di in (-1, 0, 1):
            if di == -1:
                g = torch.cat([f[..., :1, :], f[..., :-1, :]], -2)
            elif di == 1:
                g = torch.cat([f[..., 1:, :], f[..., -1:, :]], -2)
            else:
                g = f
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                out.append(torch.roll(g, dj, -1) if dj else g)
        return out

    def upsample(self, coarse: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Bilinear upsampling with periodic longitude. coarse: (C,h,w) or (h,w)."""
        squeeze = coarse.dim() == 2
        c = coarse.unsqueeze(0) if squeeze else coarse
        h, w = c.shape[-2:]
        fx = width // w
        padded = torch.cat([c[..., -1:], c, c[..., :1]], -1).unsqueeze(0)
        up = F.interpolate(padded, size=(height, (w + 2) * fx), mode="bilinear", align_corners=False)[0]
        up = up[..., fx:fx + width]
        return up[0] if squeeze else up

    @staticmethod
    def downsample(fine: torch.Tensor, factor: int) -> torch.Tensor:
        squeeze = fine.dim() == 2
        f = fine.unsqueeze(0) if squeeze else fine
        out = F.avg_pool2d(f.unsqueeze(0), factor)[0]
        return out[0] if squeeze else out

    def angular_distance(self, a, b):
        return torch.acos((a * b).sum(-1).clamp(-1.0, 1.0))
