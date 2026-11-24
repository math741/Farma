"""The engine room: background simulation thread (spec §145-148, §259-263, §270-276, §296, §553-556).

The cosmos keeps running while the backend runs — nobody needs to watch.
The loop interleaves: prime-timeline rounds, active forks (lower share),
Scientist research (its own budget, boosted after a discovery), frame
publication for viewers (fixed wall-clock rate, independent of clients),
periodic atomic checkpoints written asynchronously from host copies, and
recovery: OOM → shed science batch, then keyframe restore with a recorded,
reduced fidelity; crashes → recorded failure + restore; repeated failure →
halt with a visible reason. Never a silent reset.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import traceback
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor

import torch

from ..core.hardware import memory_stats
from ..core.logging import get_logger
from ..core.status import Epistemic
from ..history.events import SIG_NOTABLE
from ..storage.checkpoint import save_checkpoint, snapshot_history, atomic_write_json, load_latest
from .frames import FrameHub
from .universe import Universe, cpu_state, _to_device

log = get_logger("runner")


class Runner:
    def __init__(self, universe: Universe, hub: FrameHub | None = None):
        self.u = universe
        self.cfg = universe.cfg
        self.storage = universe.storage
        self.hub = hub or FrameHub()
        self.forks: dict[str, Universe] = {}
        self.cmd_q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ckpt")
        self._pending_write: Future | None = None
        self.last_ckpt = time.time()
        self.n_ckpt = 0
        self.rounds = 0
        self.tps = 0.0
        self.science_s = 0.0
        self.sim_s = 0.0
        self.errors: deque = deque(maxlen=20)
        self.ooms: deque = deque(maxlen=10)
        self.halted: str | None = None
        self.target_tps = float(self.cfg.target_tps)
        self.last_frame = 0.0
        self.last_fields = 0.0
        self.fault_injection: str | None = None   # tests only: "crash" | "oom"
        self.started_at = time.time()

    # --------------------------------------------------------- lifecycle
    def start(self) -> None:
        self.u.scientist.start_worker()
        self.thread = threading.Thread(target=self._loop, name="genesis-sim", daemon=True)
        self.thread.start()

    def stop(self, checkpoint: bool = True) -> None:
        self._stop.set()
        if self.thread:
            self.thread.join(timeout=120)
        self.u.scientist.stop_worker()
        if checkpoint and self.storage:
            self._checkpoint(sync=True)
        self.writer.shutdown(wait=True)
        self.u.events.flush()

    def submit(self, fn) -> Future:
        """Run ``fn(runner)`` inside the simulation thread between rounds."""
        fut: Future = Future()
        self.cmd_q.put((fn, fut))
        return fut

    def call(self, fn, timeout: float = 120.0):
        if self.thread is None or not self.thread.is_alive():
            return fn(self)
        return self.submit(fn).result(timeout=timeout)

    # ------------------------------------------------------------- loop
    def _loop(self) -> None:
        tps_t0, tps_n = time.time(), 0
        while not self._stop.is_set():
            self._drain()
            if self.halted:
                self._publish(force=True)
                time.sleep(0.5)
                continue
            t0 = time.time()
            try:
                n = 0
                while time.time() - t0 < 0.05:
                    if self.fault_injection:
                        kind, self.fault_injection = self.fault_injection, None
                        if kind == "oom":
                            raise torch.cuda.OutOfMemoryError("injected OOM (test)")
                        raise RuntimeError("injected crash (test)")
                    self.u.step_round()
                    n += 1
                    if self.target_tps > 0 and n >= self.target_tps * 0.05:
                        break
                self.rounds += n
                tps_n += n
            except torch.cuda.OutOfMemoryError as e:
                self._on_oom(e)
            except Exception as e:  # recorded, recovered, never silent
                self._on_crash(e)
            sim_dt = time.time() - t0
            self.sim_s += sim_dt
            self.u.wall_seconds += sim_dt
            # forks share compute
            if self.forks:
                f_budget = sim_dt * self.cfg.fork_fraction / max(1 - self.cfg.fork_fraction, 1e-3)
                tf = time.time()
                for f in list(self.forks.values()):
                    try:
                        while time.time() - tf < f_budget:
                            f.step_round()
                    except Exception as e:
                        self.errors.append({"fork": f.timeline, "error": repr(e), "t": time.time()})
                        self.forks.pop(f.timeline, None)
            # science: commit finished research, launch the next task on the Scientist's worker thread
            ts = time.time()
            try:
                self.u.scientist.step(0.0)
            except Exception as e:
                self.errors.append({"science": repr(e), "t": time.time()})
            self.science_s += time.time() - ts
            self._publish()
            if self.storage and time.time() - self.last_ckpt >= self.cfg.checkpoint_interval_s:
                self._checkpoint()
            now = time.time()
            if now - tps_t0 >= 2.0:
                self.tps = tps_n / (now - tps_t0)
                tps_t0, tps_n = now, 0
                self.hub.set_telemetry(self.telemetry())
            if self.target_tps > 0:
                spare = 0.05 - (time.time() - t0)
                if spare > 0:
                    time.sleep(spare)

    def _drain(self) -> None:
        while True:
            try:
                fn, fut = self.cmd_q.get_nowait()
            except queue.Empty:
                return
            try:
                fut.set_result(fn(self))
            except Exception as e:
                fut.set_exception(e)

    def _publish(self, force: bool = False) -> None:
        now = time.time()
        if force or now - self.last_frame >= 1.0 / max(self.cfg.frame_hz, 0.1):
            with_fields = now - self.last_fields >= 2.0
            try:
                self.hub.publish(self.u, with_fields)
            except Exception as e:
                self.errors.append({"publish": repr(e), "t": now})
            self.last_frame = now
            if with_fields:
                self.last_fields = now

    # ------------------------------------------------------- persistence
    def _checkpoint(self, sync: bool = False) -> None:
        if not self.storage:
            return
        if self._pending_write is not None and not self._pending_write.done():
            if not sync:
                return
            self._pending_write.result()
        t0 = time.time()
        state = cpu_state(self.u)
        meta = {"utick": self.u.utick, "label": self.u.label, "timeline": self.u.timeline, "seed": self.u.seed}
        self.n_ckpt += 1
        hist = self.n_ckpt % max(1, self.cfg.history_every_checkpoints) == 0
        copy_s = time.time() - t0

        def write():
            side = save_checkpoint(self.storage, state, meta)
            if hist:
                snapshot_history(self.storage, self.cfg.history_keep)
            self.u.events.flush()
            return side

        self._pending_write = self.writer.submit(write)
        if sync:
            self._pending_write.result()
        self.last_ckpt = time.time()
        log.info("checkpoint queued", extra={"fields": {"utick": self.u.utick, "host_copy_s": round(copy_s, 2)}})

    def checkpoint_now(self) -> None:
        self.call(lambda r: r._checkpoint(sync=True), timeout=600)

    # ----------------------------------------------------------- recovery
    def _record_failure(self, kind: str, e: BaseException) -> dict:
        rec = {"kind": kind, "error": repr(e), "trace": traceback.format_exc()[-4000:], "utick": self.u.utick,
               "t": time.time(), "memory": memory_stats(str(self.u.device))}
        self.errors.append(rec)
        if self.storage:
            try:
                atomic_write_json(self.storage.path("failures", f"{kind}_{int(time.time() * 1000)}.json"), rec)
            except OSError:
                pass
        return rec

    def _restore_latest(self, reason: str) -> bool:
        """Roll back to the newest keyframe (else the last checkpoint); the rollback is recorded."""
        crashed_at = self.u.utick
        if self.u.keyframes:
            kf = self.u.keyframes[-1]
            for w, ws in zip(self.u.worlds, kf["worlds"]):
                w.load_state_dict(_to_device(ws, self.u.device))
            self.u.utick = kf["utick"]
            src = f"keyframe@{kf['utick']}"
        elif self.storage:
            try:
                state, meta, path = load_latest(self.storage)
            except Exception as e:
                self.halted = f"recovery impossible: {e!r}"
                return False
            self.u.events.close()
            fresh = Universe.from_state(self.cfg, self.storage, state)
            fresh.scientist.control_results = self.u.scientist.control_results
            had_worker = self.u.scientist.worker is not None
            self.u.scientist.stop_worker()           # research in flight studied the discarded continuation
            self.u.__dict__.update(fresh.__dict__)
            # the restored scientist must point at this (same) Universe object, not the temporary one
            self.u.scientist.u = self.u
            if had_worker:
                self.u.scientist.start_worker()
            src = os.path.basename(path)
        else:
            self.halted = "recovery impossible: no keyframe and no storage"
            return False
        self.u.emit("TIMELINE_ROLLBACK", f"Recovered from {reason}: rolled back from cycle {crashed_at} to "
                    f"{self.u.utick} ({src})", significance=SIG_NOTABLE, epistemic=Epistemic.PROGRAMMED.value,
                    detail={"reason": reason, "from_utick": crashed_at, "to_utick": self.u.utick, "source": src,
                            "note": "events logged after the restore point belong to a discarded continuation"})
        return True

    def _on_oom(self, e: BaseException) -> None:
        self._record_failure("oom", e)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        now = time.time()
        self.ooms.append(now)
        recent = [t for t in self.ooms if now - t < 300]
        if len(recent) > 2:
            self.halted = "halted after repeated out-of-memory; checkpoint preserved"
            self.u.emit("SIMULATION_HALTED", self.halted, epistemic=Epistemic.PROGRAMMED.value,
                        significance=SIG_NOTABLE)
            return
        batch = max(8, self.u.scientist.lab_batch // 2)
        births = [max(2, w.max_births // 2) for w in self.u.worlds]
        self._restore_latest("out-of-memory" if len(recent) == 1 else "repeated out-of-memory")
        # reductions are applied *after* the rollback, otherwise restoring the state would undo them
        self.u.scientist.lab_batch = batch
        if len(recent) == 1:
            self.u.emit("RESOURCE_LIMIT", f"Out of memory: science batch reduced to {batch}",
                        epistemic=Epistemic.PROGRAMMED.value, significance=SIG_NOTABLE)
        else:
            for w, b in zip(self.u.worlds, births):
                w.max_births = b
            self.u.emit("RESOURCE_LIMIT", "Repeated out of memory: active fidelity reduced (fewer simultaneous "
                        "births per tick)", epistemic=Epistemic.PROGRAMMED.value, significance=SIG_NOTABLE,
                        detail={"max_births": [w.max_births for w in self.u.worlds], "lab_batch": batch})

    def _on_crash(self, e: BaseException) -> None:
        self._record_failure("crash", e)
        log.error(f"simulation error: {e!r}")
        recent = [r for r in self.errors if r.get("kind") == "crash" and time.time() - r["t"] < 600]
        if len(recent) > 3:
            self.halted = f"halted after repeated errors: {e!r}"
            self.u.emit("SIMULATION_HALTED", self.halted, epistemic=Epistemic.PROGRAMMED.value,
                        significance=SIG_NOTABLE)
            return
        self._restore_latest(f"error {type(e).__name__}")

    # ------------------------------------------------------------- forks
    def create_fork(self, label: str, reason: str, steps: list | None = None, salt: int = 0) -> str:
        def make(r):
            f = r.u.fork(label, reason, salt)
            if steps:
                f.intervene({"world": steps[0].get("world", 0) if isinstance(steps[0], dict) else 0,
                             "steps": [{k: v for k, v in s.items() if k != "world"} for s in steps]},
                            source_worlds=r.u.worlds)
            r.forks[f.timeline] = f
            return f.timeline
        return self.call(make)

    def fork_universe(self, fid: str) -> Universe | None:
        return self.forks.get(fid)

    # ---------------------------------------------------------- telemetry
    def telemetry(self) -> dict:
        up = max(time.time() - self.started_at, 1e-6)
        return {"tps": round(self.tps, 2), "rounds": self.rounds, "utick": self.u.utick,
                "sim_share": round(self.sim_s / up, 3),
                "science_share": round(self.u.scientist.busy_s / up, 3), "science_busy": self.u.scientist.busy,
                "science_queue": len(self.u.scientist.queue), "lab_batch": self.u.scientist.lab_batch,
                "forks": list(self.forks.keys()), "memory": memory_stats(str(self.u.device)),
                "checkpoints": self.n_ckpt, "last_checkpoint_age_s": round(time.time() - self.last_ckpt, 1),
                "halted": self.halted, "errors": list(self.errors)[-5:], "target_tps": self.target_tps,
                "uptime_s": round(up, 1)}


def write_report(runner: Runner) -> str | None:
    """Science report snapshot (spec §545)."""
    u = runner.u
    if not u.storage:
        return None
    rep = {"utick": u.utick, "status": u.status(), "discoveries": [d.to_dict() for d in u.scientist.discoveries],
           "negative_results": u.scientist.results[-100:], "controls": u.scientist.control_results,
           "levels": u.levels, "pantheon": u.pantheon.table()}
    path = u.storage.path("reports", f"report_u{u.utick:012d}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, default=str, indent=1)
    return path
