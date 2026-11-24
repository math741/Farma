"""EvolutionEngine + QualityDiversityEngine (spec §42-49, §277-278, §309, §641).

Selection in a world is natural: organisms that survive and reproduce leave
descendants. This module keeps the *observer-side* bookkeeping:

* :class:`SpeciesTracker` — genetic clustering into species with a phylogeny
  (species are discovered, never declared).
* :class:`QDArchive` — a MAP-Elites grid over lifetime behaviour descriptors
  plus a novelty ("weirdness") archive. Archives are diagnostics and sources
  for laboratory experiments; they never feed back into the prime timeline.
* :class:`LineageStore` — individual genealogy (parents, birth, death, cause).
* :class:`SeedBank` — dormant propagules left by adults. Germination when a
  population is depleted is a declared law of this universe (PROGRAMMED).
"""

from __future__ import annotations

import colorsys
import math

import numpy as np
import torch

from ..core.rng import derive_seed

DESCRIPTORS = ("aquatic", "speed", "emission", "build", "process", "strike", "give", "mass")
DOMINANT = ("none", "build", "process", "strike", "give", "emit")


# ------------------------------------------------------------- species
class SpeciesTracker:
    def __init__(self, genome_size: int, seed: int, threshold: float = 0.09, proj_dim: int = 32):
        g = torch.Generator().manual_seed(derive_seed(seed, "species-proj"))
        self.proj = torch.randn(genome_size, proj_dim, generator=g) / math.sqrt(proj_dim)
        self.scale = 1.0 / math.sqrt(genome_size)
        self.threshold = threshold
        self.species: dict[int, dict] = {}
        self.next_id = 1
        self._proj_dev = None

    def project(self, genomes: torch.Tensor) -> np.ndarray:
        if self._proj_dev is None or self._proj_dev.device != genomes.device:
            self._proj_dev = self.proj.to(genomes.device)
        return (genomes @ self._proj_dev * self.scale).cpu().numpy().astype(np.float32)

    def _new(self, centroid: np.ndarray, parent: int, tick: int) -> int:
        sid = self.next_id
        self.next_id += 1
        h = (derive_seed("species-color", sid) % 1000) / 1000.0
        r, gg, b = colorsys.hls_to_rgb(h, 0.6, 0.75)
        self.species[sid] = {"id": sid, "parent": int(parent), "birth": int(tick), "extinct": None,
                             "centroid": centroid.astype(np.float32), "pop": 0, "peak": 0,
                             "color": [round(r, 3), round(gg, 3), round(b, 3)], "traits": {}, "announced": False}
        return sid

    def assign_founders(self, proj: np.ndarray, tick: int) -> int:
        return self._new(proj.mean(0), 0, tick)

    def update(self, proj: np.ndarray, current: np.ndarray, tick: int, traits: dict | None = None):
        """Re-assign every living organism to a species. Returns (assignment, events)."""
        events = []
        living = [s for s in self.species.values() if s["extinct"] is None]
        n = proj.shape[0]
        out = current.copy()
        if n == 0:
            for s in living:
                if s["pop"] > 0:
                    s["pop"] = 0
                s["extinct"] = int(tick)
                events.append(("extinct", s["id"]))
            return out, events
        ids = np.array([s["id"] for s in living], dtype=np.int64)
        cents = np.stack([s["centroid"] for s in living]) if living else np.zeros((0, proj.shape[1]), np.float32)
        if len(ids):
            a2 = (proj ** 2).sum(1)[:, None]
            b2 = (cents ** 2).sum(1)[None, :]
            d = np.sqrt(np.maximum(a2 + b2 - 2.0 * proj @ cents.T, 0.0))
            j = d.argmin(1)
            dmin = d[np.arange(n), j]
            out = np.where(dmin <= self.threshold, ids[j], -1)
            # members keep their current species if still within threshold of it (hysteresis)
            cur_idx = {int(sid): k for k, sid in enumerate(ids)}
            kk = np.array([cur_idx.get(int(c), -1) for c in current], dtype=np.int64)
            has = kk >= 0
            keep = np.zeros(n, dtype=bool)
            keep[has] = d[np.where(has)[0], kk[has]] <= self.threshold * 1.15
            out = np.where(keep, current, out)
        else:
            out = np.full(n, -1, dtype=np.int64)
        orphans = np.where(out < 0)[0]
        # leader clustering of orphans (deterministic order)
        while orphans.size:
            lead = orphans[0]
            dd = np.sqrt(((proj[orphans] - proj[lead]) ** 2).sum(-1))
            grp = orphans[dd <= self.threshold]
            parent = int(current[lead]) if current[lead] > 0 else 0
            sid = self._new(proj[grp].mean(0), parent, tick)
            out[grp] = sid
            events.append(("new", sid))
            orphans = orphans[dd > self.threshold]
        # update centroids / populations
        uniq, counts = np.unique(out, return_counts=True)
        popmap = dict(zip(uniq.tolist(), counts.tolist()))
        for s in self.species.values():
            if s["extinct"] is not None:
                continue
            p = popmap.get(s["id"], 0)
            s["pop"] = int(p)
            s["peak"] = max(s["peak"], int(p))
            if p:
                mem = proj[out == s["id"]].mean(0)
                s["centroid"] = (0.7 * s["centroid"] + 0.3 * mem).astype(np.float32)
                if traits is not None and s["id"] in traits:
                    s["traits"] = traits[s["id"]]
            else:
                s["extinct"] = int(tick)
                events.append(("extinct", s["id"]))
        return out, events

    def table(self) -> list[dict]:
        rows = []
        for s in self.species.values():
            r = {k: v for k, v in s.items() if k != "centroid"}
            rows.append(r)
        return rows

    def state_dict(self) -> dict:
        return {"species": {k: dict(v, centroid=v["centroid"].tolist()) for k, v in self.species.items()},
                "next_id": self.next_id}

    def load_state_dict(self, d: dict) -> None:
        self.species = {int(k): dict(v, centroid=np.asarray(v["centroid"], np.float32))
                        for k, v in d["species"].items()}
        self.next_id = int(d["next_id"])


# ------------------------------------------------------------- QD
class QDArchive:
    """MAP-Elites over (aquatic bin × mass bin × dominant activity) + novelty archive (on device)."""

    def __init__(self, genome_size: int, device, novelty_size: int = 256, novelty_threshold: float = 0.35):
        self.bins = (4, 4, len(DOMINANT))
        n = int(np.prod(self.bins))
        self.n_cells = n
        self.device = torch.device(device)
        self.genomes = torch.zeros(n, genome_size, device=self.device)
        self.fitness = torch.full((n,), -1.0, device=self.device)
        self.desc = torch.zeros(n, len(DESCRIPTORS), device=self.device)
        self.meta = torch.zeros(n, 4, dtype=torch.long, device=self.device)  # id, species, tick, lineage
        self.nov_genomes = torch.zeros(novelty_size, genome_size, device=self.device)
        self.nov_desc = torch.zeros(novelty_size, len(DESCRIPTORS), device=self.device)
        self.nov_meta = torch.zeros(novelty_size, 4, dtype=torch.long, device=self.device)
        self.nov_score = torch.zeros(novelty_size, device=self.device)
        self.nov_count = 0
        self.novelty_threshold = novelty_threshold
        self.filled_history: list[tuple[int, int]] = []

    @staticmethod
    def descriptors(stats: torch.Tensor, caps_mass: torch.Tensor, st: dict, max_speed: float) -> torch.Tensor:
        ticks = stats[:, st["ticks"]].clamp_min(1.0)
        d = torch.stack([
            stats[:, st["water_ticks"]] / ticks,
            (stats[:, st["distance"]] / ticks / max_speed).clamp(0, 1),
            (stats[:, st["emitted"]] / ticks).clamp(0, 1),
            (stats[:, st["builds"]] / ticks * 10).clamp(0, 1),
            (stats[:, st["processes"]] / ticks * 10).clamp(0, 1),
            (stats[:, st["strikes"]] / ticks * 10).clamp(0, 1),
            (stats[:, st["gives"]] / ticks * 10).clamp(0, 1),
            (caps_mass / 15.0).clamp(0, 1),
        ], 1)
        return d

    def cell_index(self, d: torch.Tensor) -> torch.Tensor:
        aq = (d[:, 0] * 4).floor().clamp(0, 3).long()
        ms = (d[:, 7] * 4).floor().clamp(0, 3).long()
        acts = torch.stack([torch.full_like(d[:, 0], 0.05), d[:, 3], d[:, 4], d[:, 5], d[:, 6], d[:, 2] * 0.5], 1)
        dom = acts.argmax(1)
        return (aq * 4 + ms) * len(DOMINANT) + dom

    def ingest(self, genomes: torch.Tensor, desc: torch.Tensor, fitness: torch.Tensor, meta: torch.Tensor,
               tick: int) -> list[int]:
        """Offer dying organisms to the archive. Returns indices of newly filled cells."""
        if genomes.shape[0] == 0:
            return []
        cell = self.cell_index(desc)
        n = genomes.shape[0]
        key = fitness * 1e3 + torch.arange(n, device=fitness.device, dtype=fitness.dtype) * 1e-6
        best = torch.full((self.n_cells,), -1e18, device=self.device, dtype=key.dtype)
        best.scatter_reduce_(0, cell, key, reduce="amax")
        winner = (key == best[cell])
        improve = winner & (fitness > self.fitness[cell])
        newly = []
        if bool(improve.any()):
            rows = torch.nonzero(improve).flatten()
            cells = cell[rows]
            was_empty = self.fitness[cells] < 0
            self.genomes[cells] = genomes[rows]
            self.fitness[cells] = fitness[rows]
            self.desc[cells] = desc[rows]
            self.meta[cells] = meta[rows]
            newly = cells[was_empty].tolist()
            for c in newly:
                self.filled_history.append((int(tick), int(c)))
        # novelty: the most novel dying organism of this batch, if novel enough
        if self.nov_count > 0:
            dd = torch.cdist(desc, self.nov_desc[: self.nov_count])
            k = min(5, self.nov_count)
            nov = dd.topk(k, dim=1, largest=False).values.mean(1)
        else:
            nov = torch.full((n,), 1.0, device=self.device)
        j = int(torch.argmax(nov))
        if float(nov[j]) >= self.novelty_threshold or self.nov_count == 0:
            slot = self.nov_count % self.nov_genomes.shape[0]
            self.nov_genomes[slot] = genomes[j]
            self.nov_desc[slot] = desc[j]
            self.nov_meta[slot] = meta[j]
            self.nov_score[slot] = nov[j]
            self.nov_count += 1
        return newly

    def coverage(self) -> float:
        return float((self.fitness >= 0).float().mean())

    def qd_score(self) -> float:
        return float(self.fitness.clamp(min=0).sum())

    def elites(self) -> list[dict]:
        rows = []
        f = self.fitness.tolist()
        meta = self.meta.tolist()
        desc = self.desc.tolist()
        for c in range(self.n_cells):
            if f[c] < 0:
                continue
            aq, rest = divmod(c, 4 * len(DOMINANT))
            ms, dom = divmod(rest, len(DOMINANT))
            rows.append({"cell": c, "aquatic_bin": aq, "mass_bin": ms, "dominant": DOMINANT[dom], "fitness": f[c],
                         "agent_id": meta[c][0], "species": meta[c][1], "tick": meta[c][2],
                         "descriptors": dict(zip(DESCRIPTORS, [round(v, 3) for v in desc[c]]))})
        return rows

    def state_dict(self) -> dict:
        return {"genomes": self.genomes, "fitness": self.fitness, "desc": self.desc, "meta": self.meta,
                "nov_genomes": self.nov_genomes, "nov_desc": self.nov_desc, "nov_meta": self.nov_meta,
                "nov_score": self.nov_score, "nov_count": self.nov_count, "filled_history": self.filled_history}

    def load_state_dict(self, d: dict) -> None:
        for k in ("genomes", "fitness", "desc", "meta", "nov_genomes", "nov_desc", "nov_meta", "nov_score"):
            setattr(self, k, d[k].to(self.device))
        self.nov_count = int(d["nov_count"])
        self.filled_history = [tuple(x) for x in d["filled_history"]]


# ------------------------------------------------------------ genealogy
class LineageStore:
    """Ring table of individual life records (CPU)."""

    COLS = ("id", "parent", "lineage", "species", "generation", "birth", "death", "cause", "offspring", "slot")
    CAUSES = {0: "alive", 1: "starvation", 2: "injury", 3: "age", 4: "predation"}

    def __init__(self, capacity: int):
        self.cap = int(capacity)
        self.data = np.full((self.cap, len(self.COLS)), -1, dtype=np.int64)
        self.n = 0
        self.index: dict[int, int] = {}

    def add_births(self, ids, parents, lineages, species, generations, tick, slots) -> None:
        for k in range(len(ids)):
            row = self.n % self.cap
            old = int(self.data[row, 0])
            if old >= 0:
                self.index.pop(old, None)
            self.data[row] = (ids[k], parents[k], lineages[k], species[k], generations[k], tick, -1, 0, 0, slots[k])
            self.index[int(ids[k])] = row
            self.n += 1

    def add_deaths(self, ids, causes, offspring, tick) -> None:
        for k in range(len(ids)):
            row = self.index.get(int(ids[k]))
            if row is not None:
                self.data[row, 6] = tick
                self.data[row, 7] = causes[k]
                self.data[row, 8] = offspring[k]

    def set_species(self, ids, species) -> None:
        for i, s in zip(ids, species):
            row = self.index.get(int(i))
            if row is not None:
                self.data[row, 3] = s

    def record(self, agent_id: int) -> dict | None:
        row = self.index.get(int(agent_id))
        if row is None:
            return None
        r = dict(zip(self.COLS, (int(v) for v in self.data[row])))
        r["cause"] = self.CAUSES.get(r["cause"], "?")
        return r

    def ancestors(self, agent_id: int, depth: int = 12) -> list[dict]:
        out = []
        cur = self.record(agent_id)
        while cur and len(out) < depth and cur["parent"] >= 0:
            par = self.record(cur["parent"])
            if par is None:
                out.append({"id": cur["parent"], "known": False})
                break
            out.append(par)
            cur = par
        return out

    def children(self, agent_id: int, limit: int = 64) -> list[int]:
        rows = np.where(self.data[:, 1] == int(agent_id))[0][:limit]
        return [int(self.data[r, 0]) for r in rows]

    def state_dict(self) -> dict:
        return {"data": self.data.copy(), "n": self.n}

    def load_state_dict(self, d: dict) -> None:
        self.data = np.asarray(d["data"], dtype=np.int64).copy()
        self.n = int(d["n"])
        self.index = {int(self.data[r, 0]): r for r in range(min(self.n, self.cap)) if self.data[r, 0] >= 0}


# ------------------------------------------------------------ seed bank
class SeedBank:
    def __init__(self, capacity: int, genome_size: int, device):
        self.cap = int(capacity)
        d = torch.device(device)
        self.genomes = torch.zeros(self.cap, genome_size, device=d)
        self.pos = torch.zeros(self.cap, 3, device=d)
        self.meta = torch.zeros(self.cap, 4, dtype=torch.long, device=d)  # lineage, species, generation, tick
        self.valid = torch.zeros(self.cap, dtype=torch.bool, device=d)
        self.ptr = 0

    def deposit(self, genomes, pos, meta) -> None:
        n = genomes.shape[0]
        if n == 0:
            return
        idx = (torch.arange(n, device=genomes.device) + self.ptr) % self.cap
        self.genomes[idx] = genomes
        self.pos[idx] = pos
        self.meta[idx] = meta
        self.valid[idx] = True
        self.ptr = (self.ptr + n) % self.cap

    def count(self) -> int:
        return int(self.valid.sum())

    def state_dict(self) -> dict:
        return {"genomes": self.genomes, "pos": self.pos, "meta": self.meta, "valid": self.valid, "ptr": self.ptr}

    def load_state_dict(self, d: dict) -> None:
        for k in ("genomes", "pos", "meta", "valid"):
            setattr(self, k, d[k].to(self.genomes.device))
        self.ptr = int(d["ptr"])
