"""Scientist DSL — safe external experiments (spec §246-251).

Experiments are JSON documents built from a closed vocabulary. They are
validated against whitelists and numeric bounds and interpreted by plain
Python; nothing is ever evaluated, executed, imported or spawned. Every
experiment runs in a fork, never on the prime timeline.

    {"world": 0, "ticks": 400, "replicates": 3,
     "steps": [{"op": "ablate", "mechanism": "vibration"},
               {"op": "perturb", "param": "solar_flux", "value": 1.05},
               {"op": "observe", "metrics": ["population", "mean_energy"]}]}
"""

from __future__ import annotations

from dataclasses import dataclass, field

OPS = ("fork", "ablate", "perturb", "observe", "inject", "cull", "expedition")
MECHANISMS = ("vibration", "plasticity", "chemistry", "marks", "scent", "material")
PARAMS = {  # name: (min, max)
    "solar_flux": (0.8, 1.2),
    "growth_rate": (0.25, 4.0),
    "move_scale": (0.5, 2.0),
    "sound_decay": (0.1, 0.95),
}
METRICS = ("population", "mean_energy", "births", "deaths", "kills", "structure_cells", "materials", "eaten",
           "emission_rate", "species_alive", "reactions", "builds")
MAX_TICKS = 20_000
MAX_REPLICATES = 8


class ExperimentError(ValueError):
    pass


@dataclass
class Experiment:
    world: int
    ticks: int
    replicates: int
    steps: list = field(default_factory=list)
    label: str = ""

    def to_dict(self) -> dict:
        return {"world": self.world, "ticks": self.ticks, "replicates": self.replicates, "steps": self.steps,
                "label": self.label}


def _num(v, lo, hi, name) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ExperimentError(f"{name} must be a number")
    v = float(v)
    if not (lo <= v <= hi):
        raise ExperimentError(f"{name}={v} outside [{lo}, {hi}]")
    return v


def parse(doc: dict, n_worlds: int) -> Experiment:
    """Validate an experiment document. Raises ExperimentError on anything unknown."""
    if not isinstance(doc, dict):
        raise ExperimentError("experiment must be an object")
    allowed = {"world", "ticks", "replicates", "steps", "label"}
    extra = set(doc) - allowed
    if extra:
        raise ExperimentError(f"unknown fields {sorted(extra)}")
    world = int(_num(doc.get("world", 0), 0, max(0, n_worlds - 1), "world"))
    ticks = int(_num(doc.get("ticks", 400), 1, MAX_TICKS, "ticks"))
    reps = int(_num(doc.get("replicates", 1), 1, MAX_REPLICATES, "replicates"))
    label = str(doc.get("label", ""))[:80]
    steps = doc.get("steps", [])
    if not isinstance(steps, list) or len(steps) > 16:
        raise ExperimentError("steps must be a list of at most 16 operations")
    clean = []
    for st in steps:
        if not isinstance(st, dict) or st.get("op") not in OPS:
            raise ExperimentError(f"unknown operation {st!r}")
        op = st["op"]
        if op == "ablate":
            mech = st.get("mechanism")
            if mech not in MECHANISMS:
                raise ExperimentError(f"unknown mechanism {mech!r}")
            item = {"op": op, "mechanism": mech}
            if mech == "material":
                item["material"] = int(_num(st.get("material"), 6, 4096, "material"))
            clean.append(item)
        elif op == "perturb":
            name = st.get("param")
            if name not in PARAMS:
                raise ExperimentError(f"unknown parameter {name!r}")
            lo, hi = PARAMS[name]
            clean.append({"op": op, "param": name, "value": _num(st.get("value"), lo, hi, name)})
        elif op == "observe":
            mets = st.get("metrics", [])
            if not isinstance(mets, list) or any(m not in METRICS for m in mets):
                raise ExperimentError(f"unknown metrics {mets!r}")
            clean.append({"op": op, "metrics": list(mets)})
        elif op == "inject":
            field_ = st.get("field")
            if field_ not in ("light", "vibration"):
                raise ExperimentError("inject field must be 'light' or 'vibration'")
            clean.append({"op": op, "field": field_, "lat": _num(st.get("lat", 0), -90, 90, "lat"),
                          "lon": _num(st.get("lon", 0), -180, 180, "lon"),
                          "gain": _num(st.get("gain", 1.0), 0.0, 5.0, "gain"),
                          "band": int(_num(st.get("band", 0), 0, 3, "band")),
                          "period": int(_num(st.get("period", 0), 0, 10_000, "period"))})
        elif op == "cull":
            clean.append({"op": op, "fraction": _num(st.get("fraction", 0.5), 0.0, 0.95, "fraction")})
        elif op == "expedition":
            clean.append({"op": op, "source_world": int(_num(st.get("source_world", 0), 0, max(0, n_worlds - 1),
                                                             "source_world")),
                          "count": int(_num(st.get("count", 16), 1, 512, "count")),
                          "lat": _num(st.get("lat", 0), -90, 90, "lat"),
                          "lon": _num(st.get("lon", 0), -180, 180, "lon")})
        else:
            clean.append({"op": "fork"})
    return Experiment(world, ticks, reps, clean, label)
