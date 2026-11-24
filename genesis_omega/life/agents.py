"""AgentEngine state container (spec §11-12, §561).

All individuals of a world live in fixed-size slot tensors (static shapes for
GPU efficiency); ``active`` marks occupied slots. Persistent agent IDs are
never reused. Every individual in an F4 world is grown from its genome by the
DevelopmentEngine and controlled only through the world interface.
"""

from __future__ import annotations

import torch

from .development import Development
from .genome import GenomeLayout, decode_physiology
from .interface import N_INPUTS, N_OUTPUTS, N_CAPS, N_STATS, CAP
from .substrate import SubstrateParams

TENSORS = ("active", "id", "parent", "lineage", "generation", "species", "birth", "pos", "head", "speed", "energy",
           "integrity", "tbody", "age", "genome", "W", "x", "m", "dW", "caps", "sens", "emit_prof", "carried_mat",
           "carried_amt", "last_damage", "last_gift", "bump", "stats", "made", "last_y", "last_u", "partner_id")


class Agents:
    def __init__(self, n_max: int, layout: GenomeLayout, dev: Development, m_max: int, emit_bands: int,
                 lifespan_days: float, day_ticks: int, device):
        self.N = n_max
        self.layout = layout
        self.dev = dev
        self.device = torch.device(device)
        self.lifespan_days = lifespan_days
        self.day_ticks = day_ticks
        N, H, G = n_max, layout.hidden, layout.size
        d = self.device
        f32 = dict(device=d, dtype=torch.float32)
        self.active = torch.zeros(N, dtype=torch.bool, device=d)
        self.id = torch.full((N,), -1, dtype=torch.long, device=d)
        self.parent = torch.full((N,), -1, dtype=torch.long, device=d)
        self.lineage = torch.full((N,), -1, dtype=torch.long, device=d)
        self.generation = torch.zeros(N, dtype=torch.long, device=d)
        self.species = torch.zeros(N, dtype=torch.long, device=d)
        self.birth = torch.zeros(N, dtype=torch.long, device=d)
        self.pos = torch.zeros(N, 3, **f32)
        self.pos[:, 2] = 1.0
        self.head = torch.zeros(N, 3, **f32)
        self.head[:, 0] = 1.0
        self.speed = torch.zeros(N, **f32)
        self.energy = torch.zeros(N, **f32)
        self.integrity = torch.zeros(N, **f32)
        self.tbody = torch.zeros(N, **f32)
        self.age = torch.zeros(N, **f32)
        self.genome = torch.zeros(N, G, **f32)
        self.W = torch.zeros(N, H, N_INPUTS + H, **f32)
        self.x = torch.zeros(N, H, **f32)
        self.m = torch.zeros(N, **f32)
        self.dW = torch.zeros(N, **f32)
        self.caps = torch.zeros(N, N_CAPS, **f32)
        self.caps[:, CAP["mass"]] = 0.1
        self.caps[:, CAP["units"]] = 4
        self.sens = torch.zeros(N, 4, 4, **f32)
        self.emit_prof = torch.zeros(N, emit_bands, **f32)
        self.carried_mat = torch.full((N,), -1, dtype=torch.long, device=d)
        self.carried_amt = torch.zeros(N, **f32)
        self.last_damage = torch.zeros(N, **f32)
        self.last_gift = torch.zeros(N, **f32)
        self.bump = torch.zeros(N, **f32)
        self.stats = torch.zeros(N, N_STATS, **f32)
        self.made = torch.zeros(N, m_max, dtype=torch.bool, device=d)
        self.last_y = torch.zeros(N, N_OUTPUTS, **f32)
        self.last_u = torch.zeros(N, N_INPUTS, **f32)
        self.partner_id = torch.full((N,), -1, dtype=torch.long, device=d)
        self._params: SubstrateParams | None = None
        self._phys: dict | None = None

    # ------------------------------------------------------------ derived
    def invalidate(self) -> None:
        self._params = None
        self._phys = None

    @property
    def params(self) -> SubstrateParams:
        if self._params is None:
            self._params = SubstrateParams(self.layout, self.genome, self.caps[:, CAP["units"]])
        return self._params

    @property
    def phys(self) -> dict:
        if self._phys is None:
            self._phys = decode_physiology(self.layout, self.genome, self.lifespan_days, self.day_ticks)
        return self._phys

    def count(self) -> int:
        return int(self.active.sum())

    def free_slots(self, k: int) -> torch.Tensor:
        free = torch.nonzero(~self.active).flatten()
        return free[:k]

    # -------------------------------------------------------------- spawn
    def spawn(self, slots: torch.Tensor, genomes: torch.Tensor, pos: torch.Tensor, head: torch.Tensor,
              energy: torch.Tensor, parent: torch.Tensor, lineage: torch.Tensor, generation: torch.Tensor,
              species: torch.Tensor, tick: int, first_id: int) -> torch.Tensor:
        """Grow bodies for new organisms and place them in ``slots``. Returns new IDs."""
        n = slots.numel()
        if n == 0:
            return torch.zeros(0, dtype=torch.long, device=self.device)
        body = self.dev.develop(genomes, self.layout.hidden)
        ids = torch.arange(first_id, first_id + n, dtype=torch.long, device=self.device)
        self.active[slots] = True
        self.id[slots] = ids
        self.parent[slots] = parent
        self.lineage[slots] = torch.where(lineage >= 0, lineage, ids)
        self.generation[slots] = generation
        self.species[slots] = species
        self.birth[slots] = tick
        self.pos[slots] = pos
        self.head[slots] = head
        self.speed[slots] = 0
        self.energy[slots] = energy
        self.integrity[slots] = 1.0
        self.age[slots] = 0
        self.genome[slots] = genomes
        self.caps[slots] = body["caps"]
        self.sens[slots] = body["sens"]
        self.emit_prof[slots] = body["emit_profile"]
        self.W[slots] = self.layout.get(genomes, "br_w")
        self.x[slots] = 0
        self.m[slots] = 0
        self.dW[slots] = 0
        self.carried_mat[slots] = -1
        self.carried_amt[slots] = 0
        self.last_damage[slots] = 0
        self.last_gift[slots] = 0
        self.bump[slots] = 0
        self.stats[slots] = 0
        self.made[slots] = False
        self.last_y[slots] = 0
        self.last_u[slots] = 0
        self.partner_id[slots] = -1
        self.invalidate()
        ph = self.phys
        self.tbody[slots] = ph["t_opt"][slots]
        return ids

    # ------------------------------------------------------------ persist
    def state_dict(self) -> dict:
        return {k: getattr(self, k) for k in TENSORS}

    def load_state_dict(self, d: dict) -> None:
        for k in TENSORS:
            v = d[k]
            cur = getattr(self, k)
            if v.shape != cur.shape:
                raise ValueError(f"agent tensor {k} shape {tuple(v.shape)} != {tuple(cur.shape)}")
            setattr(self, k, v.to(self.device).clone())
        self.invalidate()

    def slot_of(self, agent_id: int) -> int:
        hit = torch.nonzero(self.active & (self.id == int(agent_id))).flatten()
        return int(hit[0]) if hit.numel() else -1
