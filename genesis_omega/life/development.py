"""DevelopmentEngine + BodyEngine (spec §16-22, §28, §45).

Each organism starts as a single seed node in a D×D×D lattice. Every node runs
the same genome-encoded regulatory network on its own state and on local
gradients of its neighbours' states (morphogen fields); nodes divide into
empty neighbours, differentiate, or die. After development each node carries
generic physical properties (rigidity, contractility, sensitivity to several
modalities, conductivity, adhesion, storage, insulation, resonance).
No tissue types exist: "muscle-like" or "sensor-like" are only ever external
descriptions of measured function.

Body *capabilities* (speed on land / in water, turning, grasp, strike, armor,
insulation, storage, emission spectrum, directional sensor gains, substrate
size) are then computed from the geometry and properties of the grown lattice
— the body's form does part of the control (morphological computation).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .genome import GenomeLayout
from .interface import CAP, N_CAPS

#: state channel meaning (generic physical properties, sigmoid-decoded)
NODE_PROPS = ("rigidity", "contractility", "photosens", "chemosens", "mechanosens", "thermosens", "conductivity",
              "adhesion", "storage", "insulation", "resonance")
P = {n: i + 1 for i, n in enumerate(NODE_PROPS)}  # channel 0 is "alive"


class Development:
    def __init__(self, layout: GenomeLayout, size: int, steps: int, device="cpu", emit_bands: int = 4):
        self.layout = layout
        self.D = int(size)
        self.steps = int(steps)
        self.C = layout.dev_channels
        self.K = emit_bands
        self.device = torch.device(device)
        D = self.D
        c = (D - 1) / 2.0
        r = torch.arange(D, dtype=torch.float32, device=self.device) - c
        zz, yy, xx = torch.meshgrid(r, r, r, indexing="ij")
        self.xx, self.yy, self.zz = xx, yy, zz  # lattice coordinates (z, y, x) order in tensors
        # depthwise perception kernels: identity + central differences along x, y, z
        k = torch.zeros(4, 3, 3, 3)
        k[0, 1, 1, 1] = 1.0
        k[1, 1, 1, 2], k[1, 1, 1, 0] = 0.5, -0.5   # d/dx
        k[2, 1, 2, 1], k[2, 1, 0, 1] = 0.5, -0.5   # d/dy
        k[3, 2, 1, 1], k[3, 0, 1, 1] = 0.5, -0.5   # d/dz
        self.kernel = k.repeat(self.C, 1, 1, 1).unsqueeze(1).to(self.device)  # (4C,1,3,3,3)
        six = torch.zeros(1, 1, 3, 3, 3)
        for (a, b, d) in ((0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1), (1, 1, 0), (1, 1, 2)):
            six[0, 0, a, b, d] = 1.0
        self.six = six.to(self.device)

    # --------------------------------------------------------------- grow
    def grow(self, genomes: torch.Tensor) -> torch.Tensor:
        """Run development. Returns final lattice state (B, C, D, D, D)."""
        B = genomes.shape[0]
        D, C = self.D, self.C
        L = self.layout
        w1 = L.get(genomes, "dev_w1")  # (B, Hd, 4C)
        b1 = L.get(genomes, "dev_b1")
        w2 = L.get(genomes, "dev_w2")  # (B, C, Hd)
        b2 = L.get(genomes, "dev_b2")
        s = torch.zeros(B, C, D, D, D, device=genomes.device)
        m = D // 2
        s[:, 0, m, m, m] = 1.0
        s[:, C - 4:, m, m, m] = 1.0  # morphogen sources at the seed
        for _ in range(self.steps):
            alive_pre = F.max_pool3d(s[:, :1], 3, 1, 1) > 0.1
            perc = F.conv3d(s, self.kernel, padding=1, groups=C)          # (B, 4C, D,D,D)
            pf = perc.flatten(2).transpose(1, 2)                          # (B, N, 4C)
            h = torch.relu(torch.bmm(pf, w1.transpose(1, 2)) + b1.unsqueeze(1))
            ds = torch.bmm(h, w2.transpose(1, 2)) + b2.unsqueeze(1)       # (B, N, C)
            ds = ds.transpose(1, 2).reshape(B, C, D, D, D)
            s = (s + ds * alive_pre.float()).clamp(-3.0, 3.0)
            alive_post = F.max_pool3d(s[:, :1], 3, 1, 1) > 0.1
            s = s * (alive_pre & alive_post).float()
        return s

    # --------------------------------------------------------------- body
    def body(self, s: torch.Tensor, hidden: int) -> dict:
        """Derive capabilities from the grown lattice."""
        B = s.shape[0]
        A = (s[:, 0] > 0.1).float()                                   # (B,D,D,D)
        # ensure the seed node always exists (a body is at least one node)
        m = self.D // 2
        A[:, m, m, m] = 1.0
        prop = torch.sigmoid(s[:, 1:1 + len(NODE_PROPS)])              # (B,11,D,D,D)
        pr = {n: prop[:, i] for i, n in enumerate(NODE_PROPS)}
        n_cells = A.flatten(1).sum(1)
        mass = (A * (0.5 + pr["rigidity"])).flatten(1).sum(1) / 10.0
        xx, yy, zz = self.xx, self.yy, self.zz
        w = A / n_cells.view(B, 1, 1, 1)
        cx = (w * xx).flatten(1).sum(1).view(B, 1, 1, 1)
        cy = (w * yy).flatten(1).sum(1).view(B, 1, 1, 1)
        cz = (w * zz).flatten(1).sum(1).view(B, 1, 1, 1)
        nb = F.conv3d(A.unsqueeze(1), self.six, padding=1)[:, 0]
        surface = A * (nb < 5.5).float()
        n_surf = surface.flatten(1).sum(1).clamp_min(1.0)
        vx, vy = xx - cx, yy - cy
        vn = torch.sqrt(vx * vx + vy * vy).clamp_min(1e-6)
        centre = (vn < 0.25).float()
        wdir = [
            torch.where(centre > 0, torch.full_like(vx, 0.25), (vx / vn).clamp(min=0)),   # front +x
            torch.where(centre > 0, torch.full_like(vx, 0.25), (-vy / vn).clamp(min=0)),  # right -y
            torch.where(centre > 0, torch.full_like(vx, 0.25), (-vx / vn).clamp(min=0)),  # back
            torch.where(centre > 0, torch.full_like(vx, 0.25), (vy / vn).clamp(min=0)),   # left +y
        ]
        sens_keys = ("photosens", "chemosens", "mechanosens", "thermosens")
        sens = torch.zeros(B, 4, 4, device=s.device)
        for d in range(4):
            for mi, key in enumerate(sens_keys):
                tot = (surface * wdir[d] * pr[key]).flatten(1).sum(1)
                sens[:, d, mi] = 1 - torch.exp(-tot / 6.0)
        dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2)
        zmin = torch.where(A > 0, zz, torch.full_like(zz, 1e3)).flatten(1).amin(1).view(B, 1, 1, 1)
        ground = A * (zz <= zmin + 0.5).float()
        fin = A * (nb <= 2.0).float()
        contr = pr["contractility"]
        land_drive = (ground * contr * (0.3 + dist)).flatten(1).sum(1)
        water_drive = (fin * contr * (0.3 + dist)).flatten(1).sum(1) + 0.3 * (A * contr).flatten(1).sum(1) / 4
        load = 2.0 + 6.0 * mass
        land_speed = 0.05 + 0.95 * (1 - torch.exp(-land_drive / load))
        water_speed = 0.05 + 0.95 * (1 - torch.exp(-water_drive / load))
        turn = 0.08 + 0.92 * (1 - torch.exp(-(A * contr * (yy - cy).abs()).flatten(1).sum(1) / load))
        front = surface * ((xx - cx) > 0.4).float()
        grasp = 1 - torch.exp(-(front * pr["adhesion"] * contr).flatten(1).sum(1) / 8.0)
        strike = 1 - torch.exp(-(front * pr["rigidity"] * contr).flatten(1).sum(1) / 8.0)
        armor = (surface * pr["rigidity"]).flatten(1).sum(1) / n_surf
        insul = (surface * pr["insulation"]).flatten(1).sum(1) / n_surf * (n_cells / n_surf / 1.6).clamp(0.3, 1.0)
        storage = 1.5 * (1 - torch.exp(-(A * pr["storage"]).flatten(1).sum(1) / 40.0))
        reso = A * pr["resonance"]
        emit_power = 1 - torch.exp(-reso.flatten(1).sum(1) / 25.0)
        band = (pr["rigidity"] * self.K).floor().clamp(0, self.K - 1).long()
        prof = torch.zeros(B, self.K, device=s.device)
        prof.scatter_add_(1, band.flatten(1), reso.flatten(1))
        prof = prof / prof.sum(1, keepdim=True).clamp_min(1e-6)
        bite = 0.25 + 0.75 * (1 - torch.exp(-(front * (contr + pr["adhesion"])).flatten(1).sum(1) / 12.0))
        conduct = (A * pr["conductivity"]).flatten(1).sum(1)
        h_min = max(4, hidden // 6)
        units = (h_min + (hidden - h_min) * (1 - torch.exp(-conduct / 30.0))).round().clamp(h_min, hidden)
        caps = torch.zeros(B, N_CAPS, device=s.device)
        caps[:, CAP["mass"]] = mass
        caps[:, CAP["land_speed"]] = land_speed
        caps[:, CAP["water_speed"]] = water_speed
        caps[:, CAP["turn"]] = turn
        caps[:, CAP["grasp"]] = grasp
        caps[:, CAP["strike"]] = strike
        caps[:, CAP["armor"]] = armor
        caps[:, CAP["insulation"]] = insul
        caps[:, CAP["storage"]] = storage
        caps[:, CAP["emit_power"]] = emit_power
        caps[:, CAP["bite"]] = bite
        caps[:, CAP["cells"]] = n_cells
        caps[:, CAP["units"]] = units
        caps[:, CAP["aquatic"]] = water_speed / (land_speed + water_speed)
        return {"caps": caps, "sens": sens, "emit_profile": prof, "alive": A, "props": prop}

    def develop(self, genomes: torch.Tensor, hidden: int) -> dict:
        return self.body(self.grow(genomes), hidden)

    # ------------------------------------------------------------- export
    def export_body(self, genome: torch.Tensor, hidden: int) -> dict:
        """Voxel export of one organism for the viewer (real grown lattice)."""
        b = self.develop(genome.view(1, -1), hidden)
        A = b["alive"][0]
        pr = b["props"][0]
        idx = torch.nonzero(A > 0)
        vox = []
        for z, y, x in idx.tolist():
            p = pr[:, z, y, x].tolist()
            vox.append([x - self.D // 2, y - self.D // 2, z - self.D // 2] + [round(v, 3) for v in p])
        return {"size": self.D, "props": list(NODE_PROPS), "voxels": vox,
                "caps": [round(v, 4) for v in b["caps"][0].tolist()],
                "sens": [[round(v, 3) for v in row] for row in b["sens"][0].tolist()],
                "emit_profile": [round(v, 3) for v in b["emit_profile"][0].tolist()]}


def maturity(age: torch.Tensor, lifespan: torch.Tensor, frac: float) -> torch.Tensor:
    """Juveniles grow into their capabilities (0.4 → 1.0 over the maturation period)."""
    return 0.4 + 0.6 * (age / (lifespan * frac).clamp_min(1.0)).clamp(0, 1)
