"""HistoryEngine — THE SECOND HISTORY (spec §110-114, §149, §354-359, §579).

No narrative is generated. Events are *derived* from simulation state by
detectors that run on the slow clock, each with an epistemic tag, a validation
status, a location, actors and candidate causes (temporal/spatial association
with earlier events — labelled as candidate, never as proven). Nothing is
random: the same state always yields the same events.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field, asdict

import numpy as np

from ..core.ids import event_label
from ..core.status import Epistemic, Validation
from ..society.construction import ConstructionDetector

SIG_ROUTINE, SIG_NOTABLE, SIG_HISTORIC, SIG_EXTRAORDINARY = 0, 1, 2, 3

PANTHEON_KEYS = ("first_stable_organism", "first_speciation", "first_extinction", "first_predation",
                 "first_novel_material", "first_construction", "first_house_like", "first_storage_like",
                 "first_village_like", "first_city_like", "first_trail_network", "first_ruin", "first_ruin_reuse",
                 "first_technology_candidate", "first_learner", "first_memory", "first_communication",
                 "first_culture", "first_scientific_behavior", "first_spacecraft", "first_interstellar_craft",
                 "first_contact", "first_observer_model_candidate")


@dataclass
class Event:
    id: int
    type: str
    title: str
    utick: int
    world: int | None = None
    planet: str | None = None
    tick: int | None = None
    epistemic: str = Epistemic.SIMULATED.value
    validation: str = Validation.NOT_VALIDATED.value
    significance: int = SIG_ROUTINE
    detail: dict = field(default_factory=dict)
    lat: float | None = None
    lon: float | None = None
    actors: list = field(default_factory=list)
    causes: list = field(default_factory=list)
    fork: str | None = None
    snapshot: str | None = None

    @property
    def label(self) -> str:
        return event_label(self.id)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["label"] = self.label
        return d


class EventLog:
    """Append-only event record (JSONL on disk + bounded in-memory index)."""

    def __init__(self, path: str | None, keep: int = 50_000):
        self.path = path
        self.keep = keep
        self.events: list[Event] = []
        self._lock = threading.Lock()
        self._fh = None
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)

    def append(self, ev: Event) -> Event:
        with self._lock:
            self.events.append(ev)
            if len(self.events) > self.keep:
                self.events = self.events[-self.keep:]
            if self.path:
                if self._fh is None:
                    self._fh = open(self.path, "a", encoding="utf-8")
                self._fh.write(json.dumps(ev.to_dict(), default=_json_default, ensure_ascii=False) + "\n")
        return ev

    def flush(self) -> None:
        with self._lock:
            if self._fh:
                self._fh.flush()
                os.fsync(self._fh.fileno())

    def close(self) -> None:
        with self._lock:
            if self._fh:
                self._fh.flush()
                self._fh.close()
                self._fh = None

    def since(self, event_id: int = 0, min_significance: int = 0, limit: int = 500) -> list[dict]:
        with self._lock:
            out = [e.to_dict() for e in self.events if e.id > event_id and e.significance >= min_significance]
        return out[-limit:]

    def by_type(self, typ: str) -> list[Event]:
        with self._lock:
            return [e for e in self.events if e.type == typ]

    def get(self, event_id: int) -> Event | None:
        with self._lock:
            for e in reversed(self.events):
                if e.id == event_id:
                    return e
        return None

    def recent(self, world: int | None, since_utick: int, types: tuple = ()) -> list[Event]:
        with self._lock:
            return [e for e in self.events if e.utick >= since_utick and (world is None or e.world == world)
                    and (not types or e.type in types)]

    def load_jsonl(self, max_id: int | None = None) -> int:
        """Reload the in-memory index from disk after a restart.

        Events with ``id > max_id`` were written after the checkpoint being resumed: they
        belong to a continuation that no longer exists (a crash between checkpoints).
        They are moved — never deleted — to ``events.discarded-<time>.jsonl`` so the
        main log stays consistent with the timeline and event ids are never reused."""
        if not self.path or not os.path.exists(self.path):
            return 0
        self.close()
        evs, keep_lines, dropped = [], [], []
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if not s:
                    continue
                try:
                    d = json.loads(s)
                except json.JSONDecodeError:
                    dropped.append(s)  # a torn final line after a crash
                    continue
                if max_id is not None and int(d.get("id", 0)) > max_id:
                    dropped.append(s)
                    continue
                keep_lines.append(s)
                d.pop("label", None)
                evs.append(Event(**d))
        if dropped:
            import time as _time

            base = self.path[:-len(".jsonl")] if self.path.endswith(".jsonl") else self.path
            with open(f"{base}.discarded-{_time.strftime('%Y%m%d-%H%M%S')}-{_time.time_ns()}.jsonl", "w",
                      encoding="utf-8") as fh:
                fh.write("\n".join(dropped) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            tmp = f"{self.path}.tmp-{os.getpid()}"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write("".join(x + "\n" for x in keep_lines))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        self.events = evs[-self.keep:]
        self.discarded = len(dropped)
        return len(evs)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class Pantheon:
    """Firsts and milestones (spec §354-358). Never deleted."""

    def __init__(self):
        self.firsts: dict[str, dict] = {}

    def offer(self, key: str, ev: Event) -> bool:
        if key in self.firsts:
            return False
        self.firsts[key] = {"event": ev.id, "label": ev.label, "title": ev.title, "world": ev.world,
                            "utick": ev.utick, "tick": ev.tick, "epistemic": ev.epistemic,
                            "validation": ev.validation}
        return True

    def table(self) -> list[dict]:
        rows = []
        for k in PANTHEON_KEYS:
            r = self.firsts.get(k)
            rows.append({"key": k, **(r or {"event": None})})
        return rows

    def state_dict(self) -> dict:
        return {"firsts": dict(self.firsts)}

    def load_state_dict(self, d: dict) -> None:
        self.firsts = dict(d.get("firsts", {}))


class Chronicle:
    """Per-world event detectors (slow clock)."""

    def __init__(self, world):
        self.w = world
        self.pop_hist: list[tuple[int, int]] = []
        self.temp_hist: list[float] = []
        self.res_hist: list[float] = []
        self.species_seen_events: set[int] = set()
        self.reported: set[str] = set()
        self.construction = ConstructionDetector(world)
        self.last_construction: dict = {}
        self.species_pos: dict[int, tuple[float, float, int]] = {}
        self.ruins_seen: set[tuple] = set()
        self.village_seen: set[tuple] = set()

    def state_dict(self) -> dict:
        return {"pop_hist": self.pop_hist[-400:], "temp_hist": self.temp_hist[-400:],
                "res_hist": self.res_hist[-400:], "species_seen": sorted(self.species_seen_events),
                "reported": sorted(self.reported), "species_pos": {int(k): list(v) for k, v in self.species_pos.items()},
                "ruins_seen": [list(r) for r in self.ruins_seen], "village_seen": [list(v) for v in self.village_seen],
                "cluster_first_seen": [[k[0], k[1], v] for k, v in self.construction.cluster_first_seen.items()]}

    def load_state_dict(self, d: dict) -> None:
        self.pop_hist = [tuple(x) for x in d.get("pop_hist", [])]
        self.temp_hist = list(d.get("temp_hist", []))
        self.res_hist = list(d.get("res_hist", []))
        self.species_seen_events = set(d.get("species_seen", []))
        self.reported = set(d.get("reported", []))
        self.species_pos = {int(k): tuple(v) for k, v in d.get("species_pos", {}).items()}
        self.ruins_seen = {tuple(r) for r in d.get("ruins_seen", [])}
        self.village_seen = {tuple(v) for v in d.get("village_seen", [])}
        self.construction.cluster_first_seen = {(a, b): c for a, b, c in d.get("cluster_first_seen", [])}

    # ------------------------------------------------------------------
    def observe(self, make, species_events: list) -> None:
        """Run all detectors. ``make(type, title, **kw)`` creates and logs an event."""
        w = self.w
        s = w.summary()
        t = w.tick
        day = w.xenos.day_ticks
        pop = s["population"]
        self.pop_hist.append((t, pop))
        self.temp_hist.append(s["global_temp"])
        self.res_hist.append(s["resource_total"])
        for lst in (self.pop_hist, self.temp_hist, self.res_hist):
            del lst[:-400]
        # ---- population boom / crash (vs. ~5 days earlier)
        back = [p for (tt, p) in self.pop_hist if tt <= t - 5 * day]
        if back:
            ref = back[-1]
            if ref >= 30 and pop <= 0.5 * ref and self._once(f"crash@{t // (10 * day)}"):
                causes = self._recent(("CLIMATE_EXTREME", "ECOLOGICAL_CRISIS", "PREDATION_NICHE"), 20 * day)
                make("POPULATION_CRASH", f"Population crash on {w.planet['id']}: {ref} → {pop}",
                     significance=SIG_NOTABLE, detail={"before": ref, "after": pop}, causes=causes)
            if ref >= 10 and pop >= 2.0 * ref and self._once(f"boom@{t // (10 * day)}"):
                make("POPULATION_BOOM", f"Population boom on {w.planet['id']}: {ref} → {pop}",
                     significance=SIG_ROUTINE, detail={"before": ref, "after": pop})
        if pop == 0 and self.pop_hist and max(p for _, p in self.pop_hist) > 0 and w.bank.count() == 0 \
                and self._once("sterile"):
            make("EXTINCTION_OF_ALL_LIFE", f"All organisms on {w.planet['id']} are gone",
                 significance=SIG_EXTRAORDINARY, causes=self._recent(("POPULATION_CRASH", "CLIMATE_EXTREME"), 50 * day))
        # ---- climate & ecology
        if len(self.temp_hist) > 60:
            hist = np.array(self.temp_hist[:-1])
            mu, sd = hist.mean(), hist.std() + 1e-6
            z = (self.temp_hist[-1] - mu) / sd
            if abs(z) > 3.0 and self._once(f"climate@{t // (20 * day)}"):
                make("CLIMATE_EXTREME", f"Climate extreme on {w.planet['id']}: {self.temp_hist[-1]:.1f}°C "
                     f"(z={z:.1f})", significance=SIG_NOTABLE, detail={"z": round(float(z), 2)})
        if len(self.res_hist) > 30:
            ref = max(self.res_hist[-30:-1])
            if ref > 0 and self.res_hist[-1] < 0.5 * ref and self._once(f"eco@{t // (20 * day)}"):
                make("ECOLOGICAL_CRISIS", f"Primary production collapsed by {100 * (1 - self.res_hist[-1] / ref):.0f}%",
                     significance=SIG_NOTABLE, causes=self._recent(("CLIMATE_EXTREME", "POPULATION_BOOM"), 20 * day))
        # ---- species
        for kind, sid in species_events:
            rec = w.species.species.get(sid, {})
            if kind == "new":
                continue
            if kind == "extinct" and rec.get("peak", 0) >= 15:
                make("SPECIES_EXTINCT", f"Species SP-{w.index}-{sid} went extinct (peak {rec.get('peak')})",
                          significance=SIG_NOTABLE, actors=[sid], epistemic=Epistemic.INFERRED.value,
                          causes=self._recent(("POPULATION_CRASH", "ECOLOGICAL_CRISIS", "CLIMATE_EXTREME",
                                               "PREDATION_NICHE"), 30 * day), pantheon="first_extinction")
        for sid, rec in w.species.species.items():
            if rec["extinct"] is None and rec["pop"] >= 8 and sid not in self.species_seen_events and \
                    t - rec["birth"] >= 2 * w.ucfg.speciation_every:
                self.species_seen_events.add(sid)
                if rec["parent"] > 0:
                    make("SPECIES_EMERGED", f"New species SP-{w.index}-{sid} diverged from SP-{w.index}-{rec['parent']}",
                         significance=SIG_NOTABLE, actors=[sid, rec["parent"]], epistemic=Epistemic.INFERRED.value,
                         detail={"pop": rec["pop"], "traits": rec.get("traits", {})}, pantheon="first_speciation")
        # ---- stable organisms (deep lineages)
        if s.get("max_generation", 0) >= 10 and self._once("stable"):
            make("FIRST_STABLE_ORGANISM", "A lineage persisted for 10 generations", significance=SIG_HISTORIC,
                 detail={"generation": s["max_generation"]}, pantheon="first_stable_organism")
        # ---- mass migration (species centroid moved > 25° in ~a year)
        self._migration(make)
        # ---- predation
        if w.first_kill_tick is not None and self._once("first_kill"):
            make("FIRST_PREDATION", "An organism obtained energy by killing another", significance=SIG_HISTORIC,
                 tick=w.first_kill_tick, pantheon="first_predation")
        kills = w.counters.get("kills", 0)
        if kills >= 50 and self._once("predation_niche"):
            make("PREDATION_NICHE", f"Predation became a persistent way of life ({kills} kills)",
                 significance=SIG_NOTABLE, epistemic=Epistemic.INFERRED.value)
        # ---- body plans (MAP-Elites niches first filled)
        if w.qd_new_cells:
            for c in w.qd_new_cells[:5]:
                aq, rest = divmod(c, 24)
                ms, dom = divmod(rest, 6)
                from ..life.evolution import DOMINANT
                if dom != 0 and self._once(f"qd{c}"):
                    make("NEW_WAY_OF_LIFE", f"First organism living mainly by '{DOMINANT[dom]}' "
                         f"(aquatic bin {aq}, mass bin {ms})", significance=SIG_ROUTINE,
                         epistemic=Epistemic.INFERRED.value, detail={"cell": c})
            w.qd_new_cells.clear()
        # ---- germination waves
        if w.counters.get("germinations", 0) >= 10 and self._once(f"germ@{w.counters['germinations'] // 200}"):
            make("SEED_BANK_GERMINATION", "Dormant propagules re-established a depleted population",
                 significance=SIG_ROUTINE, epistemic=Epistemic.PROGRAMMED.value,
                 detail={"germinations": w.counters["germinations"]})
        # ---- materials / technology
        for nm in w.new_material_events:
            rec = nm["record"]
            ev = make("NOVEL_MATERIAL", f"New material M-{w.index}-{nm['material']} produced "
                      f"(depth {rec['depth']})", significance=SIG_HISTORIC if rec["depth"] >= 2 else SIG_NOTABLE,
                      tick=nm["tick"], detail=rec, actors=[rec["first_agent"]], cell=rec["first_cell"],
                      epistemic=Epistemic.EMERGENT.value, pantheon="first_novel_material")
            if rec["depth"] >= 3 and self._once(f"chain{nm['material']}"):
                make("CUMULATIVE_TECH_CHAIN", f"Material M-{w.index}-{nm['material']} required a chain of 3+ "
                     "earlier products", significance=SIG_HISTORIC, causes=[{"event": ev.id, "relation": "product"}],
                     epistemic=Epistemic.EMERGENT.value)
        w.new_material_events.clear()
        from ..society.technology import technology_graph
        tg = technology_graph(w)
        for mid in tg["used_candidates"]:
            if self._once(f"techcand{mid}"):
                make("TECHNOLOGY_CANDIDATE", f"M-{w.index}-{mid} is made by many and built with",
                     significance=SIG_HISTORIC, epistemic=Epistemic.HYPOTHESIS.value,
                     validation=Validation.CANDIDATE.value, detail={"material": mid},
                     pantheon="first_technology_candidate")
        # ---- construction & settlement
        if s["structure_cells"] > 0 and w.tick % (2 * w.ucfg.analysis_every) == 0 or not self.last_construction:
            self.last_construction = self.construction.analyze()
        c = self.last_construction
        if c.get("persistent_cells", 0) > 0 and self._once("first_persistent"):
            make("FIRST_PERSISTENT_STRUCTURE", "Matter placed by organisms persisted for days",
                 significance=SIG_HISTORIC, epistemic=Epistemic.EMERGENT.value, pantheon="first_construction")
        for h in c.get("house_like", [])[:3]:
            if self._once(f"house{round(h['lat'])},{round(h['lon'])}"):
                make("HOUSE_LIKE_STRUCTURE", "An enclosure used preferentially in the cold (house-like)",
                     significance=SIG_HISTORIC, epistemic=Epistemic.INFERRED.value, lat=h["lat"], lon=h["lon"],
                     detail=h, validation=Validation.CANDIDATE.value, pantheon="first_house_like")
        if c.get("storage_like_cells", 0) > 0 and self._once("storage"):
            make("STORAGE_LIKE_PLACE", "Resources deposited in shelter and later withdrawn (storage-like)",
                 significance=SIG_HISTORIC, epistemic=Epistemic.INFERRED.value, pantheon="first_storage_like")
        for v in c.get("village_like", []):
            key = (round(v["lat"] / 3), round(v["lon"] / 3))
            if key not in self.village_seen:
                self.village_seen.add(key)
                make("VILLAGE_LIKE_CLUSTER", f"A cluster of {v['cells']} persistent structures with residents",
                     significance=SIG_HISTORIC, epistemic=Epistemic.INFERRED.value, lat=v["lat"], lon=v["lon"],
                     detail=v, pantheon="first_village_like")
        for v in c.get("city_like", []):
            if self._once(f"city{round(v['lat'])},{round(v['lon'])}"):
                make("CITY_LIKE_CLUSTER", "A settlement met every city-like criterion", significance=SIG_EXTRAORDINARY,
                     epistemic=Epistemic.INFERRED.value, lat=v["lat"], lon=v["lon"], detail=v,
                     pantheon="first_city_like")
        if c.get("trails") and self._once("trails"):
            tr = c["trails"][0]
            make("TRAIL_NETWORK", f"Persistent compacted paths ({tr['cells']} cells) — road-like",
                 significance=SIG_HISTORIC, epistemic=Epistemic.EMERGENT.value, lat=tr["lat"], lon=tr["lon"],
                 pantheon="first_trail_network")
        for r in c.get("ruins", []):
            key = (round(r["lat"] / 3), round(r["lon"] / 3))
            if key not in self.ruins_seen:
                self.ruins_seen.add(key)
                make("RUIN", "Structures abandoned for >10 days persist as ruins", significance=SIG_NOTABLE,
                     epistemic=Epistemic.INFERRED.value, lat=r["lat"], lon=r["lon"], detail=r, pantheon="first_ruin")
        self._ruin_reuse(make)

    # ------------------------------------------------------------------
    def _once(self, key: str) -> bool:
        if key in self.reported:
            return False
        self.reported.add(key)
        return True

    def _recent(self, types: tuple, horizon: int) -> list[dict]:
        log = getattr(self, "log", None)
        if log is None:
            return []
        out = []
        for e in log.recent(self.w.index, 0, types):
            if e.tick is not None and self.w.tick - e.tick <= horizon:
                out.append({"event": e.id, "relation": "candidate cause (temporal association)", "type": e.type})
        return out[-3:]

    def _migration(self, make) -> None:
        w = self.w
        A = w.agents
        if A.count() == 0:
            return
        import torch
        act = A.active
        sp = A.species[act]
        pos = A.pos[act]
        year = max(w.xenos.year_ticks, 1)
        for sid in torch.unique(sp).tolist():
            m = sp == sid
            if int(m.sum()) < 10:
                continue
            c = pos[m].mean(0)
            c = c / c.norm().clamp_min(1e-9)
            lat = float(torch.asin(c[2].clamp(-1, 1))) * 57.2958
            lon = float(torch.atan2(c[1], c[0])) * 57.2958
            prev = self.species_pos.get(sid)
            if prev is None or w.tick - prev[2] > year:
                if prev is not None:
                    dlat = lat - prev[0]
                    dlon = (lon - prev[1] + 180) % 360 - 180
                    dist = (dlat ** 2 + (dlon * np.cos(np.radians(lat))) ** 2) ** 0.5
                    if dist > 25 and self._once(f"mig{sid}@{w.tick // year}"):
                        make("MASS_MIGRATION", f"SP-{w.index}-{sid} shifted its range by {dist:.0f}°",
                             significance=SIG_NOTABLE, lat=lat, lon=lon, epistemic=Epistemic.INFERRED.value,
                             actors=[sid], detail={"from": prev[:2], "to": [lat, lon]})
                self.species_pos[sid] = (lat, lon, w.tick)

    def _ruin_reuse(self, make) -> None:
        """Archaeology: organisms of another lineage active inside a known ruin."""
        w = self.w
        ruins = self.last_construction.get("ruins", [])
        if not ruins or not w.ring.frames:
            return
        f = w.ring.frames[-1]
        if f.get("n", 0) == 0:
            return
        H, W = w.grid.H, w.grid.W
        cells = f["cell"]
        lat = 90 - (cells // W + 0.5) * 180 / H
        lon = -180 + (cells % W + 0.5) * 360 / W
        for r in ruins:
            near = (np.abs(lat - r["lat"]) < 3) & (np.abs(((lon - r["lon"] + 180) % 360) - 180) < 3)
            if near.sum() >= 2:
                ids = f["id"][near]
                lineages = {int(w.lineage.record(int(i))["lineage"]) for i in ids[:16] if w.lineage.record(int(i))}
                builders = set(r.get("builder_lineages", []))
                if lineages and builders and not (lineages & builders) and \
                        self._once(f"reuse{round(r['lat'])},{round(r['lon'])}"):
                    make("RUIN_REUSE", "Organisms of an unrelated lineage are active inside ancient ruins",
                         significance=SIG_HISTORIC, epistemic=Epistemic.INFERRED.value, lat=r["lat"], lon=r["lon"],
                         detail={"visitors": sorted(lineages)[:8], "builders": sorted(builders)[:8]},
                         pantheon="first_ruin_reuse")
