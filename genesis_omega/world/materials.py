"""MaterialEngine — open-ended hidden chemistry (spec §15, §61-63, §70-71, §129-131).

Matter in this universe is a point in a small space of generic physical
properties (all in [0, 1])::

    density     — weight per unit carried (load, slows carriers, heavy tools)
    hardness    — tool effectiveness, resistance of structures to erosion
    insulation  — how well walls built from it keep heat in (shelter)
    cohesion    — how tall a stable pile / wall can be stacked
    reactivity  — erosion by rain; how readily it takes part in reactions

The ground holds ``N_BASE`` base materials whose properties are hidden laws
drawn from the cosmos seed — the generator gives them no names.

:class:`Chemistry` is the *universal* reaction law of one cosmos (PROGRAMMED,
seeded, versioned by ``PHYSICS_VERSION``): processing a carried material
against the material of the cell (ground deposit or structure) at the local
temperature band either does nothing or yields a product whose properties
are a fixed, deterministic, nonlinear function of the reagents. Products can
react again, so chains of any depth are possible (open-ended). Every world of
a cosmos shares the same law, so independent invention of the same material
in different worlds (convergence) is meaningful.

:class:`MaterialRegistry` is the per-world record of every material that has
actually been produced: properties, recipe, depth in the production chain,
who made it first, where and when, how often it was made, by how many
distinct producers and how much of it went into structures. Organisms never
see any of this — they only feel the physical properties.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, asdict, field

import torch

from ..core.rng import derive_seed
from ..version import PHYSICS_VERSION

PROP_NAMES = ("density", "hardness", "insulation", "cohesion", "reactivity")
N_PROPS = len(PROP_NAMES)
#: base materials present as ground deposits (terrain affinity kind = id % 6)
N_BASE = 6
#: temperature bands of the reaction law (°C edges): frozen | cool | warm | hot
BAND_EDGES = (0.0, 20.0, 40.0)
N_BANDS = len(BAND_EDGES) + 1
#: two products closer than this (L2 in property space) are the same material
IDENTITY_TOL = 0.06
#: random-feature resolution of the law (smooth on the scale of ~0.15 property units)
_FEATURES = 64
_FREQ = 6.0
#: reaction threshold on the standardised affinity (≈ 30 % of distinct reagent pairs react)
_REACT_THRESHOLD = 0.55
#: how strongly products inherit their reagents' properties (0 = not at all)
_INHERIT = 0.45
_LAWS = ("base_props", "freq", "phase", "band_phase", "amp", "inherit_sign", "edges")


class Chemistry:
    """The hidden reaction law of one cosmos. Immutable after construction.

    Each of the ``N_PROPS + 1`` outputs (product properties + reaction affinity)
    is a fixed smooth random function of (carried, substrate, band) built from
    random Fourier features, so its values are ≈ standard normal wherever
    reagents lie in property space: the reaction rate and the spread of products
    do not drift as chains deepen, and nothing is hand-designed per material.
    Products map through the normal CDF (≈ uniform in [0, 1]) and partially
    inherit their reagents' properties.
    """

    def __init__(self, seed: int, device="cpu"):
        self.seed = int(seed)
        self.device = torch.device(device)
        g = torch.Generator().manual_seed(derive_seed(self.seed, "chemistry", PHYSICS_VERSION))
        # base materials: well spread in property space (rejection of near-duplicates)
        base = []
        while len(base) < N_BASE:
            cand = 0.08 + 0.84 * torch.rand(N_PROPS, generator=g, dtype=torch.float64)
            if all(float((cand - b).norm()) > 0.35 for b in base):
                base.append(cand)
        self.base_props = torch.stack(base).float()
        n_out, d_in = N_PROPS + 1, 2 * N_PROPS
        self.freq = torch.randn(n_out, _FEATURES, d_in, generator=g) * _FREQ
        self.phase = torch.rand(n_out, _FEATURES, generator=g) * (2 * math.pi)
        self.band_phase = torch.rand(N_BANDS, n_out, _FEATURES, generator=g) * (2 * math.pi)
        self.amp = torch.randn(n_out, _FEATURES, generator=g) * math.sqrt(2.0 / _FEATURES)
        self.inherit_sign = torch.where(torch.rand(N_PROPS, generator=g) < 0.75, 1.0, -1.0)
        self.edges = torch.tensor(BAND_EDGES)
        self._digest = self._compute_digest()
        self.to(self.device)

    # ----------------------------------------------------------------- utils
    def to(self, device) -> "Chemistry":
        self.device = torch.device(device)
        for k in _LAWS:
            setattr(self, k, getattr(self, k).to(self.device))
        return self

    def _compute_digest(self) -> str:
        h = hashlib.sha256(PHYSICS_VERSION.encode())
        for k in _LAWS:
            h.update(getattr(self, k).detach().cpu().contiguous().numpy().tobytes())
        h.update(repr((_FEATURES, _FREQ, _REACT_THRESHOLD, _INHERIT, IDENTITY_TOL)).encode())
        return h.hexdigest()[:16]

    def digest(self) -> str:
        """Identity of the law (stored with every registry; a mismatch is never silently accepted)."""
        return self._digest

    # ------------------------------------------------------------------- law
    def band_of(self, temp: torch.Tensor) -> torch.Tensor:
        """Temperature band (0..N_BANDS-1) of cells at ``temp`` °C."""
        return torch.bucketize(temp.contiguous(), self.edges.to(temp.device, temp.dtype))

    def react(self, pa: torch.Tensor, pb: torch.Tensor, band: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Process carried material ``pa`` against substrate ``pb`` at temperature ``band``.

        Args: pa, pb: (K, N_PROPS) property vectors. band: (K,) long.
        Returns: (product properties (K, N_PROPS), reacted (K,) bool). Deterministic.
        """
        if pa.shape[0] == 0:
            return pa.clone(), torch.zeros(0, dtype=torch.bool, device=pa.device)
        pa, pb = pa.float(), pb.float()
        b = band.long().clamp(0, N_BANDS - 1)
        x = torch.cat([pa, pb], 1)                                          # (K, 2P)
        arg = torch.einsum("kd,ofd->kof", x, self.freq) + self.phase + self.band_phase[b]
        z = (torch.cos(arg) * self.amp).sum(-1)                             # (K, P+1) ≈ N(0, 1)
        mix = (pa + pb - 1.0) * self.inherit_sign * 1.6                     # reagents' trend
        zp = _INHERIT * mix + math.sqrt(1 - _INHERIT ** 2) * z[:, :N_PROPS]
        prod = 0.02 + 0.96 * 0.5 * (1 + torch.erf(zp / math.sqrt(2.0)))
        affinity = z[:, N_PROPS] + 0.9 * (pa[:, 4] + pb[:, 4] - 1.0)        # reactive matter reacts more
        distinct = (pa - pb).norm(dim=1) > IDENTITY_TOL
        return prod, (affinity > _REACT_THRESHOLD) & distinct

    def describe(self) -> dict:
        return {"version": PHYSICS_VERSION, "digest": self.digest(), "prop_names": list(PROP_NAMES),
                "n_base": N_BASE, "band_edges_c": list(BAND_EDGES),
                "base_props": [[round(v, 4) for v in row] for row in self.base_props.cpu().tolist()]}


@dataclass
class MaterialRecord:
    id: int
    base: bool
    depth: int
    props: list
    recipe: tuple | None = None          # (carried material, substrate material, temperature band)
    first_tick: int = 0
    first_agent: int = -1
    first_cell: int = -1
    first_lineage: int = -1
    first_species: int = -1
    names: dict = field(default_factory=dict)  # reserved for observer annotations (never seen by organisms)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["recipe"] = list(self.recipe) if self.recipe is not None else None
        d["props"] = [round(float(v), 4) for v in self.props]
        d["prop_names"] = list(PROP_NAMES)
        return d

    @staticmethod
    def from_dict(d: dict) -> "MaterialRecord":
        d = {k: v for k, v in d.items() if k in MaterialRecord.__dataclass_fields__}
        if d.get("recipe") is not None:
            d["recipe"] = tuple(int(v) for v in d["recipe"])
        return MaterialRecord(**d)


class MaterialRegistry:
    """Per-world record of every material that exists (base deposits + products)."""

    def __init__(self, chem: Chemistry, capacity: int, device):
        self.chem = chem
        self.cap = int(capacity)
        self.device = torch.device(device)
        self.tol = IDENTITY_TOL
        d = self.device
        self.props = torch.zeros(self.cap, N_PROPS, device=d)
        self.produced = torch.zeros(self.cap, dtype=torch.long, device=d)    # production events
        self.producers = torch.zeros(self.cap, dtype=torch.long, device=d)   # distinct first-time producers
        self.built = torch.zeros(self.cap, device=d)                         # units placed into structures
        self.records: list[MaterialRecord] = []
        self.n = 0
        self.saturated = False
        for m in range(N_BASE):
            self.props[m] = chem.base_props[m].to(d)
            self.records.append(MaterialRecord(m, True, 0, chem.base_props[m].cpu().tolist()))
        self.n = N_BASE

    # ------------------------------------------------------------- queries
    @property
    def active_props(self) -> torch.Tensor:
        return self.props[: self.n]

    def identify(self, props: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Nearest registered material within tolerance: (ids or -1, distance)."""
        k = props.shape[0]
        if k == 0 or self.n == 0:
            return (torch.full((k,), -1, dtype=torch.long, device=props.device),
                    torch.full((k,), float("inf"), device=props.device))
        d = torch.cdist(props.float(), self.active_props.to(props.device))
        dmin, j = d.min(1)
        return torch.where(dmin <= self.tol, j, torch.full_like(j, -1)), dmin

    def register_products(self, prod: torch.Tensor, recipes: torch.Tensor, tick: int, agents: torch.Tensor,
                          cells: torch.Tensor, lineages: torch.Tensor,
                          species: torch.Tensor | None = None) -> tuple[torch.Tensor, list[int]]:
        """Map reaction products to material ids, registering materials never made before.

        Returns (ids (K,) long on the products' device, list of newly registered ids)."""
        ids, _ = self.identify(prod)
        new_ids: list[int] = []
        unknown = torch.nonzero(ids < 0).flatten().tolist()
        if not unknown:
            return ids, new_ids
        prod_c = prod.detach().float().cpu()
        rec_c = recipes.detach().long().cpu().tolist()
        ag = agents.detach().cpu().tolist()
        ce = cells.detach().cpu().tolist()
        li = lineages.detach().cpu().tolist()
        sp = species.detach().cpu().tolist() if species is not None else [-1] * len(ag)
        props_c = self.props[: self.n].detach().cpu()
        for row in unknown:
            p = prod_c[row]
            # a near-duplicate registered earlier in this same batch
            if new_ids:
                d = (props_c[new_ids] - p).norm(dim=1)
                j = int(d.argmin())
                if float(d[j]) <= self.tol:
                    ids[row] = new_ids[j]
                    continue
            if self.n >= self.cap:
                # capacity reached: the product is recorded as the closest known material (flagged)
                self.saturated = True
                ids[row] = int((props_c - p).norm(dim=1).argmin())
                continue
            mid = self.n
            a, b, band = rec_c[row]
            depth = 1 + max(self.records[a].depth, self.records[b].depth)
            self.props[mid] = p.to(self.device)
            self.records.append(MaterialRecord(mid, False, depth, p.tolist(), (int(a), int(b), int(band)), int(tick),
                                               int(ag[row]), int(ce[row]), int(li[row]), int(sp[row])))
            self.n += 1
            props_c = torch.cat([props_c, p.view(1, -1)])
            ids[row] = mid
            new_ids.append(mid)
        return ids, new_ids

    def summary(self) -> list[dict]:
        produced = self.produced[: self.n].tolist()
        producers = self.producers[: self.n].tolist()
        built = self.built[: self.n].tolist()
        out = []
        for m, r in enumerate(self.records[: self.n]):
            out.append({"id": m, "base": r.base, "depth": r.depth, "props": [round(float(v), 4) for v in r.props],
                        "recipe": list(r.recipe) if r.recipe is not None else None, "produced": int(produced[m]),
                        "producers": int(producers[m]), "built_units": round(float(built[m]), 3),
                        "first_tick": r.first_tick, "first_agent": r.first_agent, "first_cell": r.first_cell,
                        "first_lineage": r.first_lineage})
        return out

    def max_depth(self) -> int:
        return max((r.depth for r in self.records[: self.n]), default=0)

    # --------------------------------------------------------- persistence
    def state_dict(self) -> dict:
        return {"props": self.props, "produced": self.produced, "producers": self.producers, "built": self.built,
                "n": self.n, "saturated": self.saturated, "records": [r.to_dict() for r in self.records],
                "chem_digest": self.chem.digest()}

    def load_state_dict(self, d: dict) -> None:
        dig = d.get("chem_digest")
        if dig is not None and dig != self.chem.digest():
            raise RuntimeError("material chemistry digest mismatch: the reaction law changed; refusing silent change")
        props = d["props"].to(self.device)
        if props.shape[0] != self.cap:
            raise ValueError(f"material capacity {props.shape[0]} != {self.cap}")
        self.props = props.clone()
        self.produced = d["produced"].to(self.device).clone()
        self.producers = d["producers"].to(self.device).clone()
        self.built = d["built"].to(self.device).clone()
        self.n = int(d["n"])
        self.saturated = bool(d.get("saturated", False))
        self.records = [MaterialRecord.from_dict(r) for r in d["records"]]
