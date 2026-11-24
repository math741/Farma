"""Vectorised 3D gradient noise (Perlin-style) for planetary terrain."""

from __future__ import annotations

import torch


class GradientNoise3D:
    def __init__(self, seed: int, device="cpu"):
        g = torch.Generator().manual_seed(int(seed) & ((1 << 63) - 1))
        self.perm = torch.randperm(256, generator=g).to(device)
        grads = torch.randn(256, 3, generator=g)
        self.grads = (grads / grads.norm(dim=1, keepdim=True)).to(device)
        self.device = device

    def _hash(self, ix, iy, iz):
        p = self.perm
        return p[(p[(p[ix & 255] + iy) & 255] + iz) & 255]

    def __call__(self, pts: torch.Tensor) -> torch.Tensor:
        """pts: (..., 3) float → noise in roughly [-1, 1]."""
        shape = pts.shape[:-1]
        x = pts.reshape(-1, 3)
        x0 = torch.floor(x)
        f = x - x0
        i0 = x0.long()
        u = f * f * f * (f * (f * 6 - 15) + 10)
        acc = torch.zeros(x.shape[0], device=x.device)
        for dx in (0, 1):
            wx = u[:, 0] if dx else 1 - u[:, 0]
            for dy in (0, 1):
                wy = u[:, 1] if dy else 1 - u[:, 1]
                for dz in (0, 1):
                    wz = u[:, 2] if dz else 1 - u[:, 2]
                    h = self._hash(i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz)
                    g = self.grads[h]
                    d = f - torch.tensor([dx, dy, dz], dtype=f.dtype, device=f.device)
                    acc = acc + wx * wy * wz * (g * d).sum(1)
        return (acc * 1.6).reshape(shape)

    def fbm(self, pts, octaves=5, lacunarity=2.0, gain=0.5, offset=0.0):
        amp, freq, total, norm = 1.0, 1.0, 0.0, 0.0
        for o in range(octaves):
            total = total + amp * self(pts * freq + offset + 17.3 * o)
            norm += amp
            amp *= gain
            freq *= lacunarity
        return total / norm

    def ridged(self, pts, octaves=4, lacunarity=2.1, gain=0.55):
        amp, freq, total, norm = 1.0, 1.0, 0.0, 0.0
        for o in range(octaves):
            n = 1.0 - self(pts * freq + 41.7 * o).abs()
            total = total + amp * n * n
            norm += amp
            amp *= gain
            freq *= lacunarity
        return total / norm
