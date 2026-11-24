"""ConstructionEngine / SettlementEngine detectors (spec §64-69, §96-98, §307).

There is no BuildHouse(): organisms can only dig, carry and place matter.
These observer-side detectors look at the resulting fields and assign
*external labels* (INFERRED) when measured criteria hold:

* persistent structure — matter that stays above threshold for days;
* house-like — enclosed cells used preferentially during cold periods;
* storage-like — sheltered cells where resources are deposited *and* later
  withdrawn;
* village-like / city-like — clusters of persistent structures with
  occupancy (city-like requires density, persistence, infrastructure,
  activity, flow, specialisation and population together);
* trail / road-like — persistent compacted paths (desire paths);
* ruins — structures nobody has visited for a long time.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage


def label_periodic(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """8-connected components with longitude wrap-around."""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return lab, 0
    parent = np.arange(n + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    H = mask.shape[0]
    for i in range(H):
        for di in (-1, 0, 1):
            j = i + di
            if 0 <= j < H:
                a, b = lab[i, -1], lab[j, 0]
                if a and b:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(k) for k in range(n + 1)])
    uniq = np.unique(roots[1:])
    remap = np.zeros(n + 1, dtype=np.int64)
    remap[uniq] = np.arange(1, len(uniq) + 1)
    out = remap[roots[lab]]
    return out, len(uniq)


class ConstructionDetector:
    def __init__(self, world):
        self.w = world
        self.history: list[dict] = []
        self.clusters_prev: dict[int, dict] = {}
        self.cluster_first_seen: dict[tuple, int] = {}

    def analyze(self) -> dict:
        w = self.w
        p = w.phys
        day = w.xenos.day_ticks
        t = w.tick
        sh = w.struct_h.cpu().numpy()
        age = w.struct_age.cpu().numpy()
        occ = w.occupancy.cpu().numpy()
        cold = w.cold_occupancy.cpu().numpy()
        warm = w.warm_occupancy.cpu().numpy()
        comp = w.compaction.cpu().numpy()
        sin_, sout = w.store_in.cpu().numpy(), w.store_out.cpu().numpy()
        shelter = w.shelter.cpu().numpy()
        last = w.last_visit.cpu().numpy()
        builder = w.struct_builder.cpu().numpy()
        H, W = sh.shape
        persistent = (sh >= 0.3) & (age >= 3 * day)
        walls = sh >= p.wall_height
        out: dict = {"tick": t, "structure_cells": int((sh > 0.05).sum()), "persistent_cells": int(persistent.sum()),
                     "wall_cells": int(walls.sum())}
        # --- house-like: enclosed, persistent, preferentially occupied in the cold
        wall_f = walls.astype(np.float32)
        nb = sum(np.roll(np.roll(wall_f, di, 0), dj, 1) for di in (-1, 0, 1) for dj in (-1, 0, 1)
                 if (di, dj) != (0, 0))
        enclosed = (nb >= 5) & (~walls)
        wall_age = ndimage.maximum_filter(np.where(walls, age, 0), size=3, mode="wrap")
        houses = []
        if enclosed.any():
            lab, n = label_periodic(enclosed)
            for k in range(1, n + 1):
                cells = lab == k
                if cells.sum() > 12:
                    continue
                c_occ = float(cold[cells].sum())
                w_occ = float(warm[cells].sum())
                tot = float(occ[cells].sum())
                persist = float(wall_age[cells].min())
                ratio = (c_occ + 1e-6) / (w_occ + 1e-6)
                ok = persist >= 3 * day and tot > 0.05 and ratio > 1.5
                if ok:
                    ii, jj = np.nonzero(cells)
                    houses.append({"cells": int(cells.sum()), "lat": _lat(ii.mean(), H), "lon": _lon(jj.mean(), W),
                                   "cold_warm_ratio": round(ratio, 3), "occupancy": round(tot, 4),
                                   "wall_age_days": round(persist / day, 2)})
        out["house_like"] = houses
        # --- storage-like: sheltered cells with deposits and later withdrawals
        storage = (sin_ > 0.5) & (sout > 0.25) & (shelter > 0.3)
        out["storage_like_cells"] = int(storage.sum())
        # --- clusters of persistent structures (settlement candidates)
        clusters = []
        if persistent.any():
            dil = ndimage.binary_dilation(persistent, structure=np.ones((5, 5), bool))
            lab, n = label_periodic(dil)
            trail = comp >= 0.5
            for k in range(1, n + 1):
                region = lab == k
                cells = int((persistent & region).sum())
                if cells < 6:
                    continue
                ii, jj = np.nonzero(region)
                lat, lon = _lat(ii.mean(), H), _lon(jj.mean(), W)
                occ_sum = float(occ[region].sum())
                key = (round(lat), round(lon))
                first = self.cluster_first_seen.setdefault(key, t)
                trail_cells = int((trail & ndimage.binary_dilation(region, iterations=3)).sum())
                abandoned = (t - float(last[region & (sh > 0.05)].max())) > 10 * day
                lineages = np.unique(builder[region & (builder >= 0)])
                clusters.append({"cells": cells, "lat": lat, "lon": lon, "occupancy": round(occ_sum, 3),
                                 "age_days": round((t - first) / day, 2), "trail_cells": trail_cells,
                                 "abandoned": bool(abandoned), "builder_lineages": [int(v) for v in lineages[:8]]})
        out["clusters"] = clusters
        out["village_like"] = [c for c in clusters if c["occupancy"] >= 3.0 and not c["abandoned"]]
        out["ruins"] = [c for c in clusters if c["abandoned"]]
        out["city_criteria"] = [self._city_criteria(c) for c in out["village_like"]]
        out["city_like"] = [c for c, crit in zip(out["village_like"], out["city_criteria"]) if all(crit.values())]
        # --- trails / road-like infrastructure
        trail = comp >= 0.5
        roads = []
        if trail.any():
            lab, n = label_periodic(trail)
            sizes = ndimage.sum(trail, lab, index=np.arange(1, n + 1)) if n else []
            for k, size in enumerate(sizes, start=1):
                if size >= 8:
                    ii, jj = np.nonzero(lab == k)
                    roads.append({"cells": int(size), "lat": _lat(ii.mean(), H), "lon": _lon(jj.mean(), W)})
        out["trails"] = roads
        self.history.append({"tick": t, "persistent": out["persistent_cells"], "houses": len(houses),
                             "villages": len(out["village_like"]), "trails": len(roads)})
        self.history = self.history[-500:]
        return out

    def _city_criteria(self, c: dict) -> dict:
        year_days = self.w.xenos.year_days
        return {"density": c["cells"] >= 40, "population": c["occupancy"] >= 25.0,
                "persistence": c["age_days"] >= year_days, "infrastructure": c["trail_cells"] >= 20,
                "activity": c["occupancy"] >= 10.0, "flow": c["trail_cells"] >= 40,
                "specialization": False}  # requires behavioural specialisation evidence (Scientist)


def _lat(i: float, H: int) -> float:
    return round(90 - (i + 0.5) * 180 / H, 3)


def _lon(j: float, W: int) -> float:
    return round(-180 + (j + 0.5) * 360 / W, 3)


def shelter_effect(world) -> dict:
    """Physical check used by tests: shelter on structure-adjacent cells vs elsewhere."""
    s = world.shelter
    near = torch.nn.functional.max_pool2d((world.struct_h > 0).float()[None, None], 3, 1, 1)[0, 0] > 0
    return {"max": float(s.max()), "mean": float(s.mean()),
            "mean_near_structures": float(s[near].mean()) if bool(near.any()) else 0.0}
