"""ScientistEngine + ValidationEngine (spec §199, §223-234, §273-279, §399-405, §546-548, §641).

The Scientist is an observer outside the cosmos. It runs a research queue on
its own compute budget, and follows one rule: *if something looks
extraordinary, try to break it first*. A finding climbs the ladder

    CANDIDATE → TESTED → REPLICATED → CAUSALLY SUPPORTED

only by surviving controls, independent samples with new seeds, and
ablation. Fitness is never evidence (spec §233): what reproduces best is
irrelevant to what a probe measured. Negative results are archived, not
discarded. Pipelines whose positive/negative controls fail are never allowed
to promote anything.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field, asdict

import numpy as np
import torch

from ..core.ids import discovery_label
from ..core.logging import get_logger
from ..core.rng import derive_seed
from ..core.status import Validation, Epistemic
from ..life.interface import CAP
from ..society import communication
from ..version import VALIDATOR_VERSION
from . import probes, controls

log = get_logger("scientist")


@dataclass
class Discovery:
    id: int
    kind: str
    title: str
    world: int
    tick: int
    utick: int
    seed: int
    mechanism: str
    status: str = Validation.CANDIDATE.value
    epistemic: str = Epistemic.MEASURED.value
    metrics: dict = field(default_factory=dict)
    controls: dict = field(default_factory=dict)
    replication: list = field(default_factory=list)
    ablation: dict = field(default_factory=dict)
    generalization: dict = field(default_factory=dict)
    causal: dict = field(default_factory=dict)
    replay: dict = field(default_factory=dict)
    subject: dict = field(default_factory=dict)
    validator: str = VALIDATOR_VERSION

    @property
    def label(self) -> str:
        return discovery_label(self.id)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["label"] = self.label
        return d


@dataclass(order=True)
class Task:
    priority: float
    seq: int
    kind: str = field(compare=False)
    world: int = field(compare=False)
    params: dict = field(compare=False, default_factory=dict)


class Scientist:
    SURVEY_EVERY = 3000
    COMM_EVERY = 1200
    AFFECT_EVERY = 2400

    def __init__(self, universe):
        self.u = universe
        self.queue: list[Task] = []
        self.seq = 0
        self.discoveries: list[Discovery] = []
        self.failures: list[dict] = []
        self.results: list[dict] = []
        self.control_results: dict = {}
        self.last: dict[tuple, int] = {}
        self.lab_batch = universe.cfg.lab_batch
        self.boost_until = 0.0
        self.busy_s = 0.0
        self.worker = None
        self._future = None
        self._pending = None
        self._started = 0.0

    # ----------------------------------------------------------- controls
    def run_controls(self) -> dict:
        w = self.u.worlds[0]
        t0 = time.time()
        try:
            self.control_results = controls.run_all(w.layout, w.device)
        except Exception as e:  # a failing control run disables promotion; it is not hidden
            self.control_results = {"error": repr(e), "all_valid": False}
        self.control_results["seconds"] = round(time.time() - t0, 2)
        log.info("pipeline controls", extra={"fields": {k: v.get("valid") if isinstance(v, dict) else v
                                                         for k, v in self.control_results.items()}})
        return self.control_results

    def pipeline_valid(self, name: str) -> bool:
        r = self.control_results.get(name)
        return bool(isinstance(r, dict) and r.get("valid"))

    # ------------------------------------------------------------- queue
    def enqueue(self, kind: str, world: int, priority: float = 1.0, **params) -> None:
        self.seq += 1
        self.queue.append(Task(-priority, self.seq, kind, world, params))
        self.queue.sort()

    def schedule(self) -> None:
        """Periodic research (called on the slow clock)."""
        for w in self.u.worlds:
            t = w.tick
            for kind, every, prio in (("survey", self.SURVEY_EVERY, 1.0), ("communication", self.COMM_EVERY, 1.5),
                                      ("affect", self.AFFECT_EVERY, 1.2)):
                key = (kind, w.index)
                if t - self.last.get(key, -10 ** 9) >= every and w.agents.count() >= 8:
                    self.last[key] = t
                    if not any(q.kind == kind and q.world == w.index for q in self.queue):
                        self.enqueue(kind, w.index, prio)

    # ------------------------------------------------------------ worker
    def start_worker(self) -> None:
        """Run heavy research on a separate thread. Only cloned data ever reaches it,
        so research can never alter (or stall) the timeline it studies."""
        from concurrent.futures import ThreadPoolExecutor

        if self.worker is None:
            self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scientist")

    def stop_worker(self) -> None:
        if self.worker is not None:
            self.worker.shutdown(wait=False, cancel_futures=True)
            self.worker = None

    @property
    def busy(self) -> bool:
        return self._future is not None and not self._future.done()

    def step(self, budget_s: float) -> int:
        """Advance research. With a worker: commit finished work, launch the next task, return.
        Without one (tests / scripts): run tasks synchronously until the budget is used."""
        done = 0
        if self._future is not None and self._future.done():
            task, prep = self._pending
            fut, self._future, self._pending = self._future, None, None
            self.busy_s += time.time() - self._started
            try:
                self._commit(task, prep, fut.result())
            except Exception as e:
                self._fail(task, e)
            done += 1
        if self.worker is not None:
            while self._future is None and self.queue:
                task = self.queue.pop(0)
                prep = self._prepare(task)
                if prep is None:
                    continue
                self._pending = (task, prep)
                self._started = time.time()
                self._future = self.worker.submit(self._execute, task, prep)
            return done
        t0 = time.time()
        while self.queue and time.time() - t0 < budget_s:
            task = self.queue.pop(0)
            prep = self._prepare(task)
            if prep is not None:
                try:
                    self._commit(task, prep, self._execute(task, prep))
                except Exception as e:
                    self._fail(task, e)
            done += 1
        self.busy_s += time.time() - t0
        return done

    def _prepare(self, task: Task):
        try:
            return getattr(self, f"_prep_{task.kind}")(task)
        except Exception as e:
            self._fail(task, e)
            return None

    def _execute(self, task: Task, prep):
        try:
            return getattr(self, f"_exec_{task.kind}")(prep)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return {"__oom__": True}

    def _commit(self, task: Task, prep, res) -> None:
        if isinstance(res, dict) and res.get("__oom__"):
            self.lab_batch = max(8, self.lab_batch // 2)
            self.u.emit_system("OOM_SCIENCE_BATCH_REDUCED", f"Lab batch reduced to {self.lab_batch}")
            return
        getattr(self, f"_commit_{task.kind}")(prep, res)

    def _fail(self, task: Task, e: BaseException) -> None:
        self.failures.append({"task": task.kind, "world": task.world, "error": repr(e),
                              "trace": traceback.format_exc()[-2000:], "utick": self.u.utick})
        log.warning(f"research task {task.kind} failed: {e!r}")

    @property
    def wants_boost(self) -> bool:
        return time.time() < self.boost_until

    # ---------------------------------------------------------- sampling
    def _sample(self, w, n: int, seed: int, species: int | None = None):
        A = w.agents
        act = torch.nonzero(A.active).flatten()
        if species is not None:
            act = act[A.species[act] == species]
        if act.numel() == 0:
            return None
        if species is None:
            sp, cnt = torch.unique(A.species[act], return_counts=True)
            species = int(sp[torch.argmax(cnt)])
            act = act[A.species[act] == species]
        g = torch.Generator(device=w.device).manual_seed(seed)
        perm = torch.randperm(act.numel(), generator=g, device=w.device)
        slots = act[perm[:n]]
        return {"slots": slots, "species": species, "genomes": A.genome[slots].clone(),
                "units": A.caps[slots, CAP["units"]].clone(), "sens": A.sens[slots].clone(),
                "W": A.W[slots].clone(), "ids": A.id[slots].cpu().tolist()}

    # -------------------------------------------------------- discoveries
    def _discover(self, kind, title, w, seed, mechanism, metrics, subject) -> Discovery:
        n = self.u.ids.next("discovery")
        snap = self.u.snapshot_ref(w)
        d = Discovery(n, kind, title, w.index, w.tick, self.u.utick, seed, mechanism, metrics=metrics,
                      subject=subject, replay=snap)
        self.discoveries.append(d)
        self.boost_until = time.time() + 120
        return d

    def _existing(self, kind: str, w, species: int | None) -> Discovery | None:
        for d in reversed(self.discoveries):
            if d.kind == kind and d.world == w.index and (species is None or d.subject.get("species") == species) \
                    and d.status != Validation.REJECTED.value:
                return d
        return None

    def _promote(self, d: Discovery, status: Validation, why: str) -> None:
        if Validation(d.status).rank >= status.rank:
            return
        old = d.status
        d.status = status.value
        self.u.emit_discovery_event(d, old, why)

    def _archive_negative(self, kind, w, metrics) -> None:
        self.results.append({"kind": kind, "world": w.index, "tick": w.tick, "result": "negative",
                             "metrics": metrics})
        self.results = self.results[-400:]

    # ============================================================= survey
    def _prep_survey(self, task: Task):
        w = self.u.worlds[task.world]
        seed = derive_seed(self.u.seed, "survey", w.index, w.tick)
        smp = self._sample(w, min(self.lab_batch, 32), seed)
        if smp is None or len(smp["ids"]) < 6:
            return None
        return {"w": w, "seed": seed, "smp": smp, "layout": w.layout}

    def _exec_survey(self, prep):
        L, smp, seed = prep["layout"], prep["smp"], prep["seed"] % 100000
        G, U, S = smp["genomes"], smp["units"], smp["sens"]
        out = {}
        out["memory"] = probes.memory_probe(L, G, U, S, seed, trials=24)
        if out["memory"]["span"] >= 4:
            out["memory_ablation"] = probes.memory_probe(L, G, U, S, seed, trials=24, recurrent_scale=0.0)
        lp = probes.learning_probe(L, G, U, S, seed, plastic=True)
        lf = probes.learning_probe(L, G, U, S, seed, plastic=False)
        out["learning"] = probes.learning_evidence(lp, lf)
        out["self_world"] = probes.self_world_probe(L, G[:16], U[:16], S[:16], seed)
        bp = probes.body_schema_probe(L, G, U, S, seed, plastic=True)
        bf = probes.body_schema_probe(L, G, U, S, seed, plastic=False)
        out["body_schema"] = probes.body_schema_evidence(bp, bf)
        return out

    def _commit_survey(self, prep, res) -> None:
        w, seed, smp = prep["w"], prep["seed"], prep["smp"]
        sp = smp["species"]
        mem = res["memory"]
        abl = None
        if "memory_ablation" in res:
            abl = {"mechanism": "recurrence and leak removed", "span_after": res["memory_ablation"]["span"],
                   "supported": res["memory_ablation"]["span"] < mem["span"]}
        self._judge("memory", w, sp, seed, mem, smp, passed=mem["span"] >= 4, ablation=abl,
                    title=f"Substrate memory ≥{mem['span']} ticks in SP-{w.index}-{sp}",
                    mechanism="stimulus side decodable from substrate state after a delay")
        ev = res["learning"]
        self._judge("learning", w, sp, seed, ev, smp, passed=ev["supported"],
                    ablation={"mechanism": "plasticity frozen (built into every run)", "supported": True},
                    title=f"Lifetime learning (plasticity-dependent) in SP-{w.index}-{sp}",
                    mechanism="choice adapts to a randomised contingency only when plasticity is on")
        swp = res["self_world"]
        self._judge("self_world", w, sp, seed, swp, smp, passed=swp["supported"], ablation=None,
                    title=f"Self- vs world-caused change distinguishable in SP-{w.index}-{sp}",
                    mechanism="substrate state separates own-action consequences from yoked replay")
        be = res["body_schema"]
        self._judge("body_schema", w, sp, seed, be, smp, passed=be["supported"],
                    ablation={"mechanism": "plasticity frozen (built into every run)", "supported": True},
                    title=f"Motor adaptation to a reversed body in SP-{w.index}-{sp}",
                    mechanism="tracking recovers after turning is reversed only with plasticity")

    def _judge(self, kind, w, species, seed, metrics, smp, passed: bool, title: str, mechanism: str,
               ablation: dict | None) -> None:
        pipe = {"memory": "memory", "learning": "learning", "self_world": "memory", "body_schema": "learning",
                "affect": "affect", "communication": "communication"}.get(kind, kind)
        valid = self.pipeline_valid(pipe)
        existing = self._existing(kind, w, species)
        if not passed:
            self._archive_negative(kind, w, metrics)
            if existing is not None and existing.status in (Validation.CANDIDATE.value, Validation.TESTED.value):
                existing.replication.append({"tick": w.tick, "seed": seed, "passed": False, "metrics": metrics})
                fails = sum(1 for r in existing.replication if not r["passed"])
                if fails >= 2:
                    old = existing.status
                    existing.status = Validation.REJECTED.value
                    self.u.emit_discovery_event(existing, old, "failed independent replication")
            return
        if existing is None:
            d = self._discover(kind, title, w, seed, mechanism, metrics, {"species": species, "ids": smp["ids"][:12]})
            d.controls = {pipe: self.control_results.get(pipe)}
            if ablation is not None:
                d.ablation = ablation
            if valid:
                self._promote(d, Validation.TESTED, "passed probe with valid pipeline controls")
            else:
                d.controls["note"] = "pipeline controls not valid — cannot be promoted"
            return
        existing.replication.append({"tick": w.tick, "seed": seed, "passed": True, "metrics": metrics,
                                     "organisms": smp["ids"][:12]})
        if ablation is not None and not existing.ablation:
            existing.ablation = ablation
        n_ok = sum(1 for r in existing.replication if r["passed"])
        if valid and n_ok >= 2:
            self._promote(existing, Validation.REPLICATED, "replicated on independent samples with new seeds")
            if existing.ablation.get("supported"):
                self._promote(existing, Validation.CAUSALLY_SUPPORTED,
                              "effect removed by ablation of the mechanism" if kind not in ("learning", "body_schema")
                              else "effect requires plasticity (frozen-substrate control)")

    # ====================================================== communication
    def _prep_communication(self, task: Task):
        w = self.u.worlds[task.world]
        seed = derive_seed(self.u.seed, "comm", w.index, w.tick)
        return {"w": w, "seed": seed, "frames": list(w.ring.window())}

    def _exec_communication(self, prep):
        rng = np.random.default_rng(prep["seed"] % (2 ** 32))
        return communication.analyze(prep["frames"], rng)

    def _commit_communication(self, prep, res) -> None:
        w, seed = prep["w"], prep["seed"]
        self.u.latest_comm[w.index] = res
        if res.get("status") != "candidate":
            self._archive_negative("communication", w, res)
            return
        existing = self._existing("communication", w, None)
        if existing is None:
            d = self._discover("communication", f"Candidate signalling system on {w.planet['id']}", w, seed,
                               "vibration emissions carry information about the sender's situation beyond "
                               "time-of-day", res, {"world": w.index})
            d.controls = {"communication": self.control_results.get("communication")}
            if self.pipeline_valid("communication"):
                self._promote(d, Validation.TESTED, "information beyond confound with permutation null")
                self.enqueue("comm_causal", w.index, 2.0, discovery=d.id)
            return
        existing.replication.append({"tick": w.tick, "passed": True, "metrics": res})
        if sum(1 for r in existing.replication if r["passed"]) >= 2 and self.pipeline_valid("communication"):
            self._promote(existing, Validation.REPLICATED, "information criterion met in independent windows")

    def _prep_comm_causal(self, task: Task):
        w = self.u.worlds[task.world]
        d = next((x for x in self.discoveries if x.id == task.params.get("discovery")), None)
        if d is None:
            return None
        ticks = min(4 * w.xenos.day_ticks, 1200)
        return {"d": d, "ticks": ticks, "pairs": self._prep_forks(w, "vibration", 3, d.seed), "mechanism": "vibration"}

    def _exec_comm_causal(self, prep):
        return self._run_forks(prep["pairs"], prep["ticks"], prep["mechanism"])

    def _commit_comm_causal(self, prep, res) -> None:
        d = prep["d"]
        d.causal = res
        if res["supported"]:
            self._promote(d, Validation.CAUSALLY_SUPPORTED, "silencing vibration changes outcomes in forks")

    # ============================================================= affect
    def _prep_affect(self, task: Task):
        w = self.u.worlds[task.world]
        seed = derive_seed(self.u.seed, "affect", w.index, w.tick)
        cands = []
        A = w.agents
        for j in range(w.focus.k):
            slot = int(w.focus.slots[j])
            ser = w.focus.series(j)
            if slot < 0 or ser["x"].shape[0] < 400 or not bool(A.active[slot]):
                continue
            cands.append({"series": ser, "genome": A.genome[slot].clone(), "units": A.caps[slot, CAP["units"]].clone(),
                          "sens": A.sens[slot].clone(), "species": int(A.species[slot])})
        if not cands:
            return None
        return {"w": w, "seed": seed, "cands": cands, "layout": w.layout}

    def _exec_affect(self, prep):
        rng = np.random.default_rng(prep["seed"] % (2 ** 32))
        for c in prep["cands"]:
            res = probes.affect_analysis(c["series"], rng)
            if res["levels"].get("A1", {}).get("met"):
                causal = probes.affect_causal_test(prep["layout"], c["genome"], c["units"], c["sens"], c["series"],
                                                   np.array(res["direction"]), prep["seed"] % 100000)
                return {"found": True, "cand": c, "analysis": res, "causal": causal}
        return {"found": False}

    def _commit_affect(self, prep, res) -> None:
        w, seed = prep["w"], prep["seed"]
        if not res["found"]:
            self._archive_negative("affect", w, {"note": "no focus organism met A1"})
            return
        c, causal = res["cand"], res["causal"]
        levels = probes.affect_levels(res["analysis"], causal)
        metrics = {"levels": levels, "causal": causal, "focus_id": c["series"]["id"]}
        passed = levels["A2"]["met"]
        self._judge("affect", w, c["species"], seed, metrics, {"ids": [c["series"]["id"]]}, passed=passed,
                    ablation={"mechanism": "latent fluctuations clamped", "supported": causal["supported"]},
                    title=f"Functional global state (A1–A2) in SP-{w.index}-{c['species']}",
                    mechanism="a substrate direction tracking homeostatic need modulates ≥3 effectors; "
                              "blocking it changes behaviour")

    # ========================================================= technology
    def _prep_technology(self, task: Task):
        w = self.u.worlds[task.world]
        mid = int(task.params["material"])
        seed = derive_seed(self.u.seed, "tech", w.index, mid)
        return {"w": w, "mid": mid, "seed": seed, "ticks": min(6 * w.xenos.day_ticks, 1500),
                "pairs": self._prep_forks(w, f"material:{mid}", 3, seed), "mechanism": f"material:{mid}"}

    def _exec_technology(self, prep):
        return self._run_forks(prep["pairs"], prep["ticks"], prep["mechanism"])

    def _commit_technology(self, prep, res) -> None:
        w, mid, seed = prep["w"], prep["mid"], prep["seed"]
        d = self._discover("technology", f"Technology candidate M-{w.index}-{mid}", w, seed,
                           "blocking the reaction that yields this material in forks", res, {"material": mid})
        d.causal = res
        self._promote(d, Validation.TESTED, "fork ablation executed")
        if res["supported"]:
            self._promote(d, Validation.CAUSALLY_SUPPORTED, "blocking the material changes outcomes")

    # ------------------------------------------------------ fork machinery
    def _prep_forks(self, w, mechanism: str, reps: int, seed: int) -> list:
        """Paired forks (same noise salt) with and without a mechanism (created in the sim thread)."""
        pairs = []
        for r in range(reps):
            salt = derive_seed(seed, "fork", r) % (2 ** 31)
            a = w.fork(salt)
            b = w.fork(salt)
            if mechanism.startswith("material:"):
                b.blocked_materials.add(int(mechanism.split(":")[1]))
            else:
                b.ablations.add(mechanism)
            pairs.append((a, b))
        return pairs

    @staticmethod
    def _run_forks(pairs: list, ticks: int, mechanism: str) -> dict:
        diffs = {"population": [], "mean_energy": [], "eaten": [], "deaths": []}
        for a, b in pairs:
            ma = _run_measure(a, ticks)
            mb = _run_measure(b, ticks)
            for k in diffs:
                diffs[k].append(mb[k] - ma[k])
        out = {"mechanism": mechanism, "ticks": ticks, "replicates": len(pairs), "diffs": diffs}
        sup = False
        for v in diffs.values():
            v = np.array(v, dtype=np.float64)
            same_sign = (v > 0).all() or (v < 0).all()
            if same_sign and abs(v.mean()) > 2 * (v.std() + 1e-9) and abs(v.mean()) > 1e-6:
                sup = True
        out["supported"] = bool(sup)
        return out

    def fork_ablation(self, w, mechanism: str, ticks: int, reps: int, seed: int) -> dict:
        return self._run_forks(self._prep_forks(w, mechanism, reps, seed), ticks, mechanism)

    # ------------------------------------------------------------- state
    def state_dict(self) -> dict:
        return {"seq": self.seq, "discoveries": [d.to_dict() for d in self.discoveries],
                "failures": self.failures[-200:], "results": self.results[-400:], "controls": self.control_results,
                "last": [[k[0], k[1], v] for k, v in self.last.items()], "lab_batch": self.lab_batch,
                "queue": [[t.priority, t.seq, t.kind, t.world, t.params] for t in self.queue]}

    def load_state_dict(self, d: dict) -> None:
        self.seq = int(d.get("seq", 0))
        self.discoveries = []
        for x in d.get("discoveries", []):
            x = dict(x)
            x.pop("label", None)
            self.discoveries.append(Discovery(**x))
        self.failures = list(d.get("failures", []))
        self.results = list(d.get("results", []))
        self.control_results = dict(d.get("controls", {}))
        self.last = {(a, b): c for a, b, c in d.get("last", [])}
        self.lab_batch = int(d.get("lab_batch", self.lab_batch))
        self.queue = [Task(p, s, k, w, prm) for p, s, k, w, prm in d.get("queue", [])]


def _run_measure(w, ticks: int) -> dict:
    deaths0 = w.counters["deaths"]
    eaten0 = float(w.acc[0])
    for _ in range(ticks):
        w.step()
    A = w.agents
    n = A.count()
    return {"population": float(n), "mean_energy": float(A.energy[A.active].mean()) if n else 0.0,
            "eaten": float(w.acc[0]) - eaten0, "deaths": float(w.counters["deaths"] - deaths0)}
