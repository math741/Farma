"""Level systems (spec §235-240). Levels are *assessments from evidence*, never goals.

Each level reports a status from the validation ladder and the evidence that
supports it. There is deliberately no consciousness level (spec §240).
"""

from __future__ import annotations

from ..core.status import Validation

ORGANISM = ("O0 organization", "O1 body", "O2 control", "O3 lifetime learning", "O4 memory", "O5 generalization",
            "O6 prediction", "O7 planning", "O8 social learning", "O9 communication", "O10 cumulative culture")
CIVILIZATION = ("C0 population", "C1 stable groups", "C2 communication network", "C3 culture", "C4 settlement",
                "C5 specialization", "C6 cumulative knowledge", "C7 infrastructure", "C8 planetary civilization",
                "C9 spacefaring", "C10 interstellar")
SCIENCE = ("S0 observation", "S1 repeated experiment", "S2 predictive model", "S3 transferable theory",
           "S4 engineering from model", "S5 advanced scientific tradition")
OBSERVER = ("X0 no anomaly recognition", "X1 intervention detection", "X2 prediction", "X3 external-cause model",
            "X4 systematic study of intervention boundary")

NOT_OBSERVED = "NOT OBSERVED"
NOT_IMPLEMENTED = "NOT IMPLEMENTED"


def _best(discoveries: list, kind: str, world: int) -> dict | None:
    best = None
    for d in discoveries:
        if d["kind"] == kind and d["world"] == world:
            if best is None or Validation(d["status"]).rank > Validation(best["status"]).rank:
                best = d
    return best


def _row(name, status, evidence=None):
    return {"level": name, "status": status, "evidence": evidence}


def assess(world, chronicle, discoveries: list, social_stats: dict, comm: dict | None) -> dict:
    w = world.index
    s = world.summary()
    c = chronicle.last_construction if chronicle else {}
    org = []
    alive = s["population"] > 0
    org.append(_row(ORGANISM[0], Validation.TESTED.value if alive else NOT_OBSERVED,
                    "bodies grow deterministically from genomes (test_life)"))
    moving = alive and s["counters"].get("births", 0) > 0
    org.append(_row(ORGANISM[1], Validation.TESTED.value if moving else NOT_OBSERVED,
                    "body-derived locomotion and feeding sustain reproduction"))
    org.append(_row(ORGANISM[2], Validation.TESTED.value if moving else NOT_OBSERVED,
                    "effectors are functions of sensed signals through the substrate"))
    for idx, kind in ((3, "learning"), (4, "memory")):
        d = _best(discoveries, kind, w)
        org.append(_row(ORGANISM[idx], d["status"] if d else NOT_OBSERVED, d["id"] if d else None))
    d = _best(discoveries, "self_world", w)
    org.append(_row(ORGANISM[5], NOT_IMPLEMENTED))
    b = _best(discoveries, "body_schema", w)
    org.append(_row(ORGANISM[6], (b or d or {}).get("status", NOT_OBSERVED), (b or d or {}).get("id")))
    org.append(_row(ORGANISM[7], NOT_IMPLEMENTED))
    org.append(_row(ORGANISM[8], NOT_IMPLEMENTED))
    cm = _best(discoveries, "communication", w)
    org.append(_row(ORGANISM[9], cm["status"] if cm else NOT_OBSERVED, cm["id"] if cm else None))
    org.append(_row(ORGANISM[10], NOT_OBSERVED))
    civ = []
    civ.append(_row(CIVILIZATION[0], Validation.TESTED.value if s["population"] >= 20 else NOT_OBSERVED,
                    {"population": s["population"]}))
    pers = social_stats.get("persistence", 0) if social_stats else 0
    groups = social_stats.get("n_groups", 0) if social_stats else 0
    civ.append(_row(CIVILIZATION[1], Validation.CANDIDATE.value if groups and pers >= 0.3 else NOT_OBSERVED,
                    {"groups": groups, "persistence": pers}))
    civ.append(_row(CIVILIZATION[2], cm["status"] if cm else NOT_OBSERVED, cm["id"] if cm else None))
    civ.append(_row(CIVILIZATION[3], NOT_OBSERVED))
    villages = len(c.get("village_like", []))
    civ.append(_row(CIVILIZATION[4], Validation.CANDIDATE.value if villages else NOT_OBSERVED,
                    {"village_like": villages}))
    civ.append(_row(CIVILIZATION[5], NOT_OBSERVED))
    tech = _best(discoveries, "technology", w)
    civ.append(_row(CIVILIZATION[6], tech["status"] if tech else NOT_OBSERVED, tech["id"] if tech else None))
    infra = bool(c.get("trails")) and c.get("storage_like_cells", 0) > 0
    civ.append(_row(CIVILIZATION[7], Validation.CANDIDATE.value if infra else NOT_OBSERVED,
                    {"trails": len(c.get("trails", [])), "storage_cells": c.get("storage_like_cells", 0)}))
    civ.append(_row(CIVILIZATION[8], NOT_OBSERVED))
    civ.append(_row(CIVILIZATION[9], NOT_IMPLEMENTED, "no spaceflight physics exists"))
    civ.append(_row(CIVILIZATION[10], NOT_IMPLEMENTED, "no spaceflight physics exists"))
    sci = [_row(SCIENCE[0], NOT_IMPLEMENTED)] + [_row(n, NOT_IMPLEMENTED) for n in SCIENCE[1:]]
    obs = [_row(OBSERVER[0], Validation.TESTED.value, "default: no detection probe has succeeded")] + \
        [_row(n, NOT_IMPLEMENTED) for n in OBSERVER[1:]]
    aff = []
    a = _best(discoveries, "affect", w)
    levels = (a or {}).get("metrics", {}).get("levels", {})
    for k in ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"):
        info = levels.get(k)
        if info is None:
            aff.append(_row(k, NOT_IMPLEMENTED if k >= "A5" else NOT_OBSERVED))
        elif info.get("status") == "NOT IMPLEMENTED":
            aff.append(_row(f"{k} {info['name']}", NOT_IMPLEMENTED))
        else:
            aff.append(_row(f"{k} {info['name']}", a["status"] if info.get("met") else NOT_OBSERVED, a["id"]))
    return {"world": w, "organism": org, "affect": aff, "civilization": civ, "science": sci, "observer": obs,
            "note": "Levels are evidence assessments. No level measures consciousness or feeling."}
