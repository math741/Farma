"""TechnologyEngine + CultureEngine views (spec §61-63, §70-71, §129-131, §306, §310).

The technology graph is *historical*: nodes are materials organisms actually
produced, edges are the recipes they actually used, annotated with who first
made each one, how often, by how many distinct producers, and how much went
into structures. Cultural lineage asks, for every material, whether later
first-time producers had prior social contact with earlier producers
(candidate social transmission), shared a matriline (possible inheritance),
or neither (independent invention → convergence).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..world.materials import N_BASE, PROP_NAMES


def technology_graph(world) -> dict:
    reg = world.materials
    summary = reg.summary()
    nodes, edges = [], []
    for m in summary:
        nodes.append({"id": m["id"], "base": m["base"], "depth": m["depth"], "props": m["props"],
                      "produced": m["produced"], "producers": m["producers"], "built_units": m["built_units"],
                      "first_tick": m["first_tick"], "first_agent": m["first_agent"]})
        if m["recipe"]:
            a, b, band = m["recipe"]
            edges.append({"from": a, "to": m["id"], "role": "carried", "band": band})
            edges.append({"from": b, "to": m["id"], "role": "substrate", "band": band})
    candidates = [n for n in nodes if not n["base"] and n["producers"] >= 3 and n["produced"] >= 10]
    used = [n for n in candidates if n["built_units"] >= 1.0]
    return {"nodes": nodes, "edges": edges, "max_depth": max((n["depth"] for n in nodes), default=0),
            "n_products": sum(1 for n in nodes if not n["base"]), "candidates": [n["id"] for n in candidates],
            "used_candidates": [n["id"] for n in used], "saturated": reg.saturated,
            "prop_names": list(PROP_NAMES)}


def cultural_lineage(world) -> dict:
    """Classify each first-time production by prior contact with earlier producers."""
    log = list(world.production_log)
    if not log:
        return {"materials": {}, "totals": {"social": 0, "kin": 0, "independent": 0}}
    firsts = [r for r in log if r[6] == 1]  # (tick, agent, material, cell, lineage, species, first)
    by_mat: dict[int, list] = defaultdict(list)
    for r in firsts:
        by_mat[r[2]].append(r)
    edges = world.social.edges
    totals = {"social": 0, "kin": 0, "independent": 0}
    out = {}
    for mid, rows in by_mat.items():
        if mid < N_BASE:
            continue
        rows.sort(key=lambda r: r[0])
        earlier: list = []
        cls = []
        for r in rows:
            agent, lineage = r[1], r[4]
            kind = "first"
            if earlier:
                contact = any((min(agent, e[1]), max(agent, e[1])) in edges for e in earlier[-64:])
                same_line = any(e[4] == lineage for e in earlier)
                kind = "social" if contact else ("kin" if same_line else "independent")
                totals[kind] += 1
            cls.append(kind)
            earlier.append(r)
        out[int(mid)] = {"producers": len(rows), "classes": {k: cls.count(k) for k in set(cls)},
                         "first_tick": rows[0][0], "last_tick": rows[-1][0]}
    return {"materials": out, "totals": totals}


def convergence(worlds: list) -> list[dict]:
    """Materials independently discovered in more than one world (same universal law)."""
    found = []
    for i, a in enumerate(worlds):
        pa = a.materials.active_props[N_BASE:].cpu().numpy()
        for b in worlds[i + 1:]:
            pb = b.materials.active_props[N_BASE:].cpu().numpy()
            if len(pa) == 0 or len(pb) == 0:
                continue
            d = np.sqrt(((pa[:, None] - pb[None]) ** 2).sum(-1))
            for x, y in zip(*np.where(d < 0.15)):
                found.append({"world_a": a.index, "material_a": int(x + N_BASE), "world_b": b.index,
                              "material_b": int(y + N_BASE), "distance": round(float(d[x, y]), 4)})
    return found
