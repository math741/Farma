"""GenesisCore — the Universe (spec §1, §5, §110-121, §139, §143-148, §330, §551-564).

One Universe = one timeline: the galaxy (F0 aggregate biospheres everywhere,
Keplerian orbits), every F4 world with individuals, the historical record,
the pantheon, the Scientist, keyframes for replay and the registry of forks.
The prime timeline is protected: loading, replaying or experimenting never
writes into it; experiments run in forks, and every external action is an
EXTERNAL event in the timeline where it happens.
"""

from __future__ import annotations

import copy
import math
import time
from collections import deque

import numpy as np
import torch

from ..core.config import UniverseConfig
from ..core.hashing import tensor_digest
from ..core.ids import IdAllocator, cosmos_label, fork_label
from ..core.logging import get_logger
from ..core.status import Epistemic, Validation, feature_table
from ..cosmos.galaxy import Galaxy
from ..history.events import (Event, EventLog, Pantheon, Chronicle, SIG_ROUTINE, SIG_NOTABLE, SIG_HISTORIC,
                              SIG_EXTRAORDINARY)
from ..life.interface import CAP
from ..science import levels as levels_mod
from ..science.experiments import parse as parse_experiment
from ..science.scientist import Scientist
from ..storage.checkpoint import StorageLayout
from ..version import version_info
from ..world.materials import Chemistry
from ..world.physics import WorldPhysics
from ..world.world import World

log = get_logger("universe")

class Universe:
    def __init__(self, cfg: UniverseConfig, storage: StorageLayout | None, timeline: str = "PRIME",
                 parent: str | None = None):
        self.cfg = cfg
        self.seed = int(cfg.seed)
        self.label = cosmos_label(self.seed)
        self.timeline = timeline
        self.parent = parent
        self.storage = storage
        self.device = torch.device(cfg.device)
        self.galaxy = Galaxy.generate(self.seed, cfg.n_stars)
        self.phys = WorldPhysics()
        self.chem = Chemistry(self.seed, self.device)
        self.ids = IdAllocator()
        self.utick = 0
        self.worlds: list[World] = []
        self.chronicles: list[Chronicle] = []
        path = storage.path("events", "events.jsonl") if storage else None
        self.events = EventLog(path)
        self.pantheon = Pantheon()
        self.scientist = Scientist(self)
        self.keyframes: deque = deque(maxlen=max(1, cfg.keyframes_keep))
        self.frozen: dict[int, dict] = {}
        self.latest_comm: dict[int, dict] = {}
        self.social_stats: dict[int, dict] = {}
        self.levels: dict[int, dict] = {}
        self.construction: dict[int, dict] = {}
        self.forks: list[dict] = []
        self.created_at = time.time()
        self.wall_seconds = 0.0
        self.intervention_lock = True        # experiment lock (spec §604)
        self._pending_post_freeze: list[tuple[int, int, int]] = []

    # =============================================================== create
    @classmethod
    def create(cls, cfg: UniverseConfig, storage: StorageLayout | None, run_controls: bool = True) -> "Universe":
        u = cls(cfg, storage)
        u.emit("COSMOS_BORN", f"Cosmos {u.label} formed: {cfg.n_stars} stars, {u.galaxy.n_planets} planets",
               epistemic=Epistemic.PROGRAMMED.value, significance=SIG_HISTORIC,
               detail={"seed": u.seed, "versions": version_info(),
                       "life_bearing_planets": int(u.galaxy.planets["life"].sum())})
        rows = u.galaxy.choose_focus_worlds(cfg.active_worlds)
        for k, row in enumerate(rows):
            u._add_world(row, k, reason="initial focus", external=False)
        if run_controls:
            u.scientist.run_controls()
        return u

    def _add_world(self, row: int, k: int, reason: str, external: bool) -> World:
        planet = self.galaxy.planet_dict(row)
        star = self.galaxy.star_dict(int(planet["star"]))
        wcfg = copy.deepcopy(self.cfg.world)
        if k > 0:
            wcfg.n_max = max(64, int(wcfg.n_max * self.cfg.secondary_n_max_scale))
        w = World(len(self.worlds), planet, star, self.cfg, wcfg, self.phys, self.chem, self.seed, self.device)
        self.worlds.append(w)
        ch = Chronicle(w)
        ch.log = self.events
        self.chronicles.append(ch)
        self.emit("WORLD_FORMED", f"{planet['id']} ({star['class']}-type star) simulated at F4",
                  world=w.index, epistemic=Epistemic.PROGRAMMED.value, significance=SIG_NOTABLE,
                  detail={"planet": planet, "xenos": w.xenos.to_dict(), "climate_calibration": w.calibration,
                          "reason": reason})
        sites = w.seed_life(3, float(planet.get("f0_biomass") or 0.5))
        self.emit("LIFE_SEEDED", f"Founder populations introduced on {planet['id']} (initial condition)",
                  world=w.index, epistemic=(Epistemic.EXTERNAL if external else Epistemic.PROGRAMMED).value,
                  significance=SIG_NOTABLE, detail={"sites": sites,
                                                    "note": "no abiogenesis chemistry is simulated"},
                  lat=sites[0]["lat"], lon=sites[0]["lon"])
        return w

    def promote(self, planet_row: int) -> World:
        """F0 → F4 promotion of a life-bearing planet (spec §13). Always an EXTERNAL event."""
        if any(w.planet["row"] == planet_row for w in self.worlds):
            raise ValueError("planet already simulated at F4")
        if np.isnan(self.galaxy.f0_biomass[planet_row]):
            raise ValueError("planet has no life to promote")
        w = self._add_world(planet_row, len(self.worlds), reason="observer-requested scientific inspection",
                            external=True)
        self.emit("FIDELITY_PROMOTION", f"{w.planet['id']} promoted from F0 (aggregate) to F4 (individuals)",
                  world=w.index, epistemic=Epistemic.EXTERNAL.value, significance=SIG_HISTORIC)
        return w

    # ================================================================ step
    @torch.no_grad()
    def step_round(self) -> None:
        cfg = self.cfg
        for w in self.worlds:
            w.step()
        self.utick += 1
        u = self.utick
        if u % cfg.speciation_every == 0:
            self._speciation()
        if u % cfg.analysis_every == 0:
            self._analysis()
        if u % cfg.galaxy_every == 0:
            self._galaxy_step()
        if u % cfg.keyframe_every == 0:
            self.keyframe()
        if self._pending_post_freeze:
            self._complete_freezes()

    def _speciation(self) -> None:
        for w in self.worlds:
            A = w.agents
            act = torch.nonzero(A.active).flatten()
            if act.numel() == 0:
                evs = w.species.update(np.zeros((0, 32), np.float32), np.zeros(0, np.int64), w.tick)[1]
                w._species_events = getattr(w, "_species_events", []) + evs
                continue
            proj = w.species.project(A.genome[act])
            cur = A.species[act].cpu().numpy()
            traits = self._species_traits(w, act)
            new, evs = w.species.update(proj, cur, w.tick, traits)
            A.species[act] = torch.tensor(new, device=w.device, dtype=torch.long)
            w.lineage.set_species(A.id[act].cpu().numpy(), new)
            w._species_events = getattr(w, "_species_events", []) + evs

    @staticmethod
    def _species_traits(w, act) -> dict:
        A = w.agents
        sp = A.species[act]
        out = {}
        for sid in torch.unique(sp).tolist():
            m = act[sp == sid]
            c = A.caps[m]
            out[int(sid)] = {k: round(float(c[:, CAP[k]].mean()), 3) for k in
                             ("mass", "land_speed", "water_speed", "aquatic", "emit_power", "grasp", "strike",
                              "armor", "units", "cells")}
        return out

    def _analysis(self) -> None:
        for w, ch in zip(self.worlds, self.chronicles):
            w.flush_social()
            sp_ev = getattr(w, "_species_events", [])
            w._species_events = []

            def make(typ, title, _w=w, **kw):
                return self._make_world_event(_w, typ, title, **kw)

            ch.observe(make, sp_ev)
            self.construction[w.index] = ch.last_construction
            alive = set(w.agents.id[w.agents.active].cpu().tolist())
            self.social_stats[w.index] = w.social.group_stats(alive)
            self.levels[w.index] = levels_mod.assess(w, ch, [d.to_dict() for d in self.scientist.discoveries],
                                                     self.social_stats[w.index], self.latest_comm.get(w.index))
            for ev in self.events.recent(w.index, self.utick - self.cfg.analysis_every, ("TECHNOLOGY_CANDIDATE",)):
                mid = ev.detail.get("material")
                if mid is not None and not any(t.kind == "technology" and t.params.get("material") == mid
                                               for t in self.scientist.queue):
                    self.scientist.enqueue("technology", w.index, 1.8, material=int(mid))
        self.scientist.schedule()
        self.events.flush()

    def _galaxy_step(self) -> None:
        w0 = self.worlds[0] if self.worlds else None
        if w0 is None:
            return
        # galaxy time: one focus-world year = that planet's orbital period
        years_per_tick = float(w0.planet["period"]) / max(w0.xenos.year_ticks, 1)
        exclude = {w.planet["row"] for w in self.worlds}
        evs = self.galaxy.step_f0(years_per_tick * self.cfg.galaxy_every, self.utick, exclude)
        for e in evs:
            p = self.galaxy.planet_dict(e["planet_row"])
            if e["type"] == "F0_IMPACT":
                self.emit("F0_IMPACT", f"Impact on {p['id']}: aggregate biomass {e['before']:.2f} → {e['after']:.2f}",
                          epistemic=Epistemic.SIMULATED.value, significance=SIG_NOTABLE,
                          detail={"planet": p["id"], **e, "model": "F0 aggregate"})
            else:
                self.emit("F0_STERILIZED", f"Aggregate biosphere of {p['id']} collapsed", significance=SIG_NOTABLE,
                          epistemic=Epistemic.SIMULATED.value, detail={"planet": p["id"], **e})

    # ============================================================== events
    def emit(self, typ: str, title: str, world: int | None = None, significance: int = SIG_ROUTINE,
             epistemic: str = Epistemic.SIMULATED.value, validation: str = Validation.NOT_VALIDATED.value,
             detail: dict | None = None, lat=None, lon=None, actors=None, causes=None, tick=None) -> Event:
        n = self.ids.next("event")
        w = self.worlds[world] if world is not None and world < len(self.worlds) else None
        ev = Event(id=n, type=typ, title=title, utick=self.utick, world=world,
                   planet=w.planet["id"] if w else None, tick=w.tick if (w and tick is None) else tick,
                   epistemic=epistemic, validation=validation, significance=significance, detail=detail or {},
                   lat=lat, lon=lon, actors=list(actors or []), causes=list(causes or []),
                   fork=None if self.timeline == "PRIME" else self.timeline)
        self.events.append(ev)
        return ev

    def _make_world_event(self, w: World, typ: str, title: str, significance: int = SIG_ROUTINE,
                          epistemic: str = Epistemic.SIMULATED.value, validation: str = Validation.NOT_VALIDATED.value,
                          detail=None, lat=None, lon=None, actors=None, causes=None, tick=None, cell=None,
                          pantheon: str | None = None) -> Event:
        if cell is not None and cell >= 0 and lat is None:
            i, j = divmod(int(cell), w.grid.W)
            lat = 90 - (i + 0.5) * 180 / w.grid.H
            lon = -180 + (j + 0.5) * 360 / w.grid.W
        if lat is None and w.agents.count():
            A = w.agents
            c = A.pos[A.active].mean(0)
            c = c / c.norm().clamp_min(1e-9)
            lat = math.degrees(math.asin(max(-1.0, min(1.0, float(c[2])))))
            lon = math.degrees(math.atan2(float(c[1]), float(c[0])))
        ev = self.emit(typ, title, world=w.index, significance=significance, epistemic=epistemic,
                       validation=validation, detail=detail, lat=lat, lon=lon, actors=actors, causes=causes,
                       tick=tick)
        if significance >= SIG_HISTORIC:
            ev.detail = dict(ev.detail, digest=self.world_digest(w))
            self.freeze(w, ev)
        if pantheon:
            self.pantheon.offer(pantheon, ev)
        return ev

    def emit_system(self, typ: str, title: str, **detail) -> Event:
        return self.emit(typ, title, epistemic=Epistemic.PROGRAMMED.value, significance=SIG_ROUTINE, detail=detail)

    def emit_discovery_event(self, d, old_status: str, why: str) -> None:
        sig = SIG_HISTORIC if d.status in (Validation.REPLICATED.value, Validation.CAUSALLY_SUPPORTED.value) \
            else SIG_NOTABLE
        if d.status == Validation.REJECTED.value:
            sig = SIG_ROUTINE
        w = self.worlds[d.world] if d.world < len(self.worlds) else None
        typ = "DISCOVERY_" + d.kind.upper()
        ev = self.emit(typ, f"{d.label} {d.title}: {old_status} → {d.status}", world=d.world, significance=sig,
                       epistemic=Epistemic.MEASURED.value, validation=d.status,
                       detail={"discovery": d.id, "why": why, "kind": d.kind})
        if w is not None and d.status in (Validation.TESTED.value, Validation.REPLICATED.value,
                                          Validation.CAUSALLY_SUPPORTED.value):
            key = {"learning": "first_learner", "memory": "first_memory",
                   "communication": "first_communication"}.get(d.kind)
            if key and d.status in (Validation.REPLICATED.value, Validation.CAUSALLY_SUPPORTED.value):
                self.pantheon.offer(key, ev)
        if self.storage:
            from ..storage.checkpoint import atomic_write_json
            atomic_write_json(self.storage.path("discoveries", f"{d.label}.json"), d.to_dict())

    # ===================================================== freeze / replay
    def world_digest(self, w: World) -> str:
        A = w.agents
        act = A.active
        return tensor_digest(A.id[act], A.pos[act], A.energy[act])[:16]

    def snapshot_ref(self, w: World) -> dict:
        kf = self.keyframes[-1] if self.keyframes else None
        return {"keyframe_utick": kf["utick"] if kf else None, "world": w.index, "tick": w.tick,
                "utick": self.utick, "digest": self.world_digest(w)}

    def freeze(self, w: World, ev: Event) -> None:
        """Freeze the ring-buffer window around an event (pre now, post later)."""
        pre = w.ring.window()[-48:]
        self.frozen[ev.id] = {"event": ev.id, "world": w.index, "pre": pre, "post": [], "utick": self.utick,
                              "keyframe_utick": self.keyframes[-1]["utick"] if self.keyframes else None}
        self._pending_post_freeze.append((ev.id, w.index, w.tick + 24 * w.cfg.ring_every))
        while len(self.frozen) > 64:
            self.frozen.pop(next(iter(self.frozen)))

    def _complete_freezes(self) -> None:
        keep = []
        for ev_id, wi, until in self._pending_post_freeze:
            w = self.worlds[wi]
            if w.tick >= until and ev_id in self.frozen:
                fr = self.frozen[ev_id]
                start = fr["pre"][-1]["tick"] if fr["pre"] else 0
                fr["post"] = [f for f in w.ring.window() if f["tick"] > start][:24]
                self._save_frozen(ev_id)
            elif ev_id in self.frozen:
                keep.append((ev_id, wi, until))
        self._pending_post_freeze = keep

    def _save_frozen(self, ev_id: int) -> None:
        if not self.storage:
            return
        fr = self.frozen[ev_id]
        frames = fr["pre"] + fr["post"]
        arrays = {}
        for k, f in enumerate(frames):
            if f.get("n", 0) == 0:
                continue
            for key in ("id", "lat", "lon", "species", "bits", "energy"):
                arrays[f"f{k}_{key}"] = f[key]
            arrays[f"f{k}_tick"] = np.array([f["tick"]])
        np.savez_compressed(self.storage.path("discoveries", f"frames_E{ev_id:07d}.npz"), **arrays)

    def keyframe(self) -> None:
        """In-memory full-state snapshot for deterministic replay (CPU copy)."""
        st = {"utick": self.utick, "worlds": [_to_cpu(w.state_dict()) for w in self.worlds],
              "digests": [self.world_digest(w) for w in self.worlds]}
        self.keyframes.append(st)

    def replay_copy(self, keyframe_utick: int | None = None) -> "Universe":
        """A detached copy of this timeline restored at a keyframe — never written back (spec §116, §384)."""
        kf = None
        for k in self.keyframes:
            if keyframe_utick is None or k["utick"] == keyframe_utick:
                kf = k
        if kf is None:
            raise ValueError("no such keyframe")
        r = Universe(self.cfg, None, timeline=f"REPLAY@{kf['utick']}", parent=self.timeline)
        r.utick = kf["utick"]
        r.scientist.control_results = dict(self.scientist.control_results)
        for wi, ws in enumerate(kf["worlds"]):
            src = self.worlds[wi]
            w = World(wi, src.planet, src.star, self.cfg, src.cfg, self.phys, self.chem, self.seed, self.device)
            w.load_state_dict(_to_device(copy.deepcopy(ws), self.device))
            r.worlds.append(w)
            ch = Chronicle(w)
            ch.log = r.events
            r.chronicles.append(ch)
        return r

    def verify_replay(self, keyframe_utick: int, target_utick: int, expected: list[str]) -> dict:
        """Re-simulate from a keyframe and compare world digests (spec §284, §435)."""
        r = self.replay_copy(keyframe_utick)
        while r.utick < target_utick:
            for w in r.worlds:
                w.step()
            r.utick += 1
        got = [r.world_digest(w) for w in r.worlds]
        return {"keyframe": keyframe_utick, "target": target_utick, "expected": expected, "got": got,
                "identical": got == expected}

    # ================================================================ forks
    def fork(self, label: str = "", reason: str = "", salt: int = 0) -> "Universe":
        """Create a scientific fork of this timeline (spec §117-121, §605-606)."""
        n = self.ids.next("fork")
        fid = fork_label(n)
        storage = self.storage.fork_layout(fid) if self.storage else None
        f = Universe.__new__(Universe)
        f.__dict__.update({k: v for k, v in self.__dict__.items()
                           if k not in ("worlds", "chronicles", "events", "scientist", "keyframes", "frozen",
                                        "storage", "forks", "pantheon", "ids", "_pending_post_freeze")})
        f.timeline = fid
        f.parent = self.timeline
        f.storage = storage
        if storage:
            storage.ensure()
        f.ids = copy.deepcopy(self.ids)
        f.events = EventLog(storage.path("events", "events.jsonl") if storage else None)
        f.pantheon = copy.deepcopy(self.pantheon)
        f.keyframes = deque(maxlen=self.keyframes.maxlen)
        f.frozen = {}
        f._pending_post_freeze = []
        f.forks = []
        f.worlds = [w.fork(salt) for w in self.worlds]
        f.chronicles = []
        for w, ch in zip(f.worlds, self.chronicles):
            nch = Chronicle(w)
            nch.load_state_dict(ch.state_dict())
            nch.log = f.events
            f.chronicles.append(nch)
        f.scientist = Scientist(f)
        f.scientist.control_results = dict(self.scientist.control_results)
        f.intervention_lock = False
        meta = {"id": fid, "label": label or fid, "reason": reason, "parent": self.timeline,
                "fork_utick": self.utick, "created_at": time.time(), "salt": salt, "interventions": []}
        f.fork_meta = meta
        self.forks.append(meta)
        self.emit("FORK_CREATED", f"Fork {fid} branched from {self.timeline} at cycle {self.utick}",
                  significance=SIG_NOTABLE, epistemic=Epistemic.EXTERNAL.value, detail=meta)
        f.emit("TIMELINE_BEGINS", f"Timeline {fid} begins (copy of {self.timeline} at cycle {self.utick})",
               significance=SIG_NOTABLE, epistemic=Epistemic.EXTERNAL.value, detail=meta)
        return f

    def intervene(self, doc: dict, source_worlds: list[World] | None = None) -> list[Event]:
        """Apply Scientist-DSL interventions to this timeline. Refused on the prime
        timeline while the experiment lock is on. Every step is an EXTERNAL event."""
        if self.timeline == "PRIME" and self.intervention_lock:
            raise PermissionError("prime timeline is protected: create a fork or unlock explicitly")
        exp = parse_experiment(doc, len(self.worlds))
        w = self.worlds[exp.world]
        out = []
        for st in exp.steps:
            op = st["op"]
            if op == "ablate":
                if st["mechanism"] == "material":
                    w.blocked_materials.add(int(st["material"]))
                else:
                    w.ablations.add(st["mechanism"])
            elif op == "perturb":
                if st["param"] == "solar_flux":
                    w.xenos.flux_rel *= st["value"]
                    w.climate.xenos = w.xenos
                elif st["param"] == "growth_rate":
                    w.param_scale["growth_rate"] = st["value"]
                elif st["param"] == "move_scale":
                    w.xenos.move_scale *= st["value"]
                elif st["param"] == "sound_decay":
                    w.xenos.sound_decay = st["value"]
            elif op == "inject":
                i = int(min(w.grid.H - 1, max(0, (90 - st["lat"]) / 180 * w.grid.H)))
                j = int(((st["lon"] + 180) / 360 * w.grid.W)) % w.grid.W
                w.external_sources.append({"field": st["field"], "cell": i * w.grid.W + j, "gain": st["gain"],
                                           "band": st["band"], "value": 1.0, "period": st["period"]})
            elif op == "cull":
                A = w.agents
                act = torch.nonzero(A.active).flatten()
                g = torch.Generator(device=w.device).manual_seed(self.utick)
                k = int(act.numel() * st["fraction"])
                victims = act[torch.randperm(act.numel(), generator=g, device=w.device)[:k]]
                A.energy[victims] = -1.0
            elif op == "expedition":
                self._expedition(w, (source_worlds or self.worlds)[st["source_world"]], st)
            rec = dict(st, utick=self.utick, world=w.index)
            w.interventions.append(rec)
            if hasattr(self, "fork_meta"):
                self.fork_meta["interventions"].append(rec)
            out.append(self.emit("EXTERNAL_INTERVENTION", f"External intervention on {w.planet['id']}: {op}",
                                 world=w.index, significance=SIG_NOTABLE, epistemic=Epistemic.EXTERNAL.value,
                                 detail=rec))
        return out

    def _expedition(self, w: World, src: World, st: dict) -> None:
        """Transplant organisms from another world (marked fork only)."""
        A, S = w.agents, src.agents
        act = torch.nonzero(S.active).flatten()[: st["count"]]
        free = torch.nonzero(~A.active).flatten()[: act.numel()]
        n = free.numel()
        if n == 0:
            return
        lat, lon = math.radians(st["lat"]), math.radians(st["lon"])
        c = torch.tensor([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)],
                         device=w.device)
        gen = torch.Generator(device=w.device).manual_seed(self.utick + 17)
        pos = w.grid.random_points_near(c, n, 3 * w.grid.dang, gen)
        head = w.grid.random_tangent(pos, gen)
        genomes = S.genome[act[:n]].to(w.device)
        sid = w.species._new(w.species.project(genomes).mean(0), 0, w.tick)
        w.agents.spawn(free, genomes, pos, head, torch.full((n,), 8.0, device=w.device),
                       torch.full((n,), -1, dtype=torch.long, device=w.device),
                       torch.full((n,), -1, dtype=torch.long, device=w.device),
                       torch.zeros(n, dtype=torch.long, device=w.device),
                       torch.full((n,), sid, dtype=torch.long, device=w.device), w.tick, w.next_agent_id)
        w.next_agent_id += n

    # ============================================================== queries
    def status(self) -> dict:
        return {
            "cosmos": self.label, "timeline": self.timeline, "parent": self.parent, "utick": self.utick,
            "galaxy_years": round(self.galaxy.time_years, 4), "worlds": [w.summary() for w in self.worlds],
            "life_bearing_f0": int(np.sum(~np.isnan(self.galaxy.f0_biomass))),
            "events": self.ids.peek("event"), "discoveries": len(self.scientist.discoveries),
            "forks": len(self.forks), "versions": version_info(), "lock": self.intervention_lock,
            "controls_valid": self.scientist.control_results.get("all_valid"),
            "profile": self.cfg.profile, "device": str(self.device),
        }

    def news(self, since_utick: int) -> dict:
        """What happened while the observer was away (spec §147-149, §497)."""
        evs = [e for e in self.events.events if e.utick > since_utick]
        important = [e.to_dict() for e in evs if e.significance >= SIG_NOTABLE]
        counts: dict[str, int] = {}
        for e in evs:
            counts[e.type] = counts.get(e.type, 0) + 1
        return {"since": since_utick, "now": self.utick, "elapsed": self.utick - since_utick,
                "counts": counts, "highlights": important[-40:],
                "extraordinary": [e.to_dict() for e in evs if e.significance >= SIG_EXTRAORDINARY]}

    def galaxy_payload(self) -> dict:
        g = self.galaxy
        life = np.zeros(g.n_stars, dtype=np.uint8)
        stars_of_life = g.planets["star"][~np.isnan(g.f0_biomass)]
        life[stars_of_life] = 1
        for w in self.worlds:
            life[int(w.planet["star"])] = 2
        return {"n": g.n_stars, "pos": g.pos, "teff": g.teff.astype(np.float32), "lum": g.lum.astype(np.float32),
                "stage": g.stage.astype(np.uint8), "life": life,
                "focus_stars": [int(w.planet["star"]) for w in self.worlds]}

    def system_payload(self, star: int) -> dict:
        d = self.galaxy.system_dict(star)
        for p in d["planets"]:
            p["position_au"] = self.galaxy.orbit_position(p["row"])
            p["f4_world"] = next((w.index for w in self.worlds if w.planet["row"] == p["row"]), None)
        return d

    def features(self) -> list[dict]:
        return feature_table()

    # ========================================================== persistence
    def state_dict(self) -> dict:
        return {
            "label": self.label, "seed": self.seed, "timeline": self.timeline, "parent": self.parent,
            "utick": self.utick, "ids": self.ids.state_dict(), "galaxy": self.galaxy.state_dict(),
            "worlds": [w.state_dict() for w in self.worlds],
            "world_rows": [w.planet["row"] for w in self.worlds],
            "world_cfgs": [w.cfg.to_dict() for w in self.worlds],
            "chronicles": [c.state_dict() for c in self.chronicles],
            "pantheon": self.pantheon.state_dict(), "scientist": self.scientist.state_dict(),
            "forks": list(self.forks), "created_at": self.created_at, "wall_seconds": self.wall_seconds,
            "lock": self.intervention_lock, "levels": self.levels, "latest_comm": self.latest_comm,
            "config": self.cfg.to_dict(),
        }

    @classmethod
    def from_state(cls, cfg: UniverseConfig, storage: StorageLayout | None, st: dict) -> "Universe":
        from ..core.config import WorldConfig

        if int(st["seed"]) != int(cfg.seed):
            cfg.seed = int(st["seed"])
        u = cls(cfg, storage, timeline=st.get("timeline", "PRIME"), parent=st.get("parent"))
        u.utick = int(st["utick"])
        u.ids.load_state_dict(st["ids"])
        u.galaxy.load_state_dict(st["galaxy"])
        for k, (row, ws) in enumerate(zip(st["world_rows"], st["worlds"])):
            planet = u.galaxy.planet_dict(int(row))
            star = u.galaxy.star_dict(int(planet["star"]))
            wc = WorldConfig(**st["world_cfgs"][k])
            w = World(k, planet, star, cfg, wc, u.phys, u.chem, u.seed, u.device)
            w.load_state_dict(_to_device(ws, u.device))
            u.worlds.append(w)
            ch = Chronicle(w)
            ch.load_state_dict(st["chronicles"][k])
            ch.log = u.events
            u.chronicles.append(ch)
        u.pantheon.load_state_dict(st["pantheon"])
        u.scientist.load_state_dict(st["scientist"])
        u.forks = list(st.get("forks", []))
        u.created_at = float(st.get("created_at", time.time()))
        u.wall_seconds = float(st.get("wall_seconds", 0.0))
        u.intervention_lock = bool(st.get("lock", True))
        u.levels = st.get("levels", {})
        u.latest_comm = st.get("latest_comm", {})
        u.events.load_jsonl()
        return u


def _to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu(v) for v in obj)
    return copy.deepcopy(obj)


def _to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_device(v, device) for v in obj]
    return obj


def cpu_state(u: Universe) -> dict:
    """State dict copied to host memory (safe to serialise from another thread)."""
    return _to_cpu(u.state_dict())
