"""Viewer frame extraction (spec §293-298, §539). Telemetry-side only.

The simulation thread publishes immutable snapshots into the hub at a fixed
wall-clock rate; the server reads them without ever touching simulation
tensors. A slow browser therefore never slows the cosmos (spec §296).
"""

from __future__ import annotations

import threading
import time

import numpy as np


class FrameHub:
    def __init__(self):
        self._lock = threading.Lock()
        self.seq = 0
        self.agents: dict[int, dict] = {}
        self.fields: dict[int, dict] = {}
        self.fields_seq: dict[int, int] = {}
        self.structures: dict[int, dict] = {}
        self.struct_deltas: dict[int, list] = {}
        self.status: dict = {}
        self.telemetry: dict = {}
        self.terrain: dict[int, dict] = {}
        self.last_publish = 0.0
        self._prev_struct: dict[int, np.ndarray] = {}

    def publish(self, universe, with_fields: bool) -> None:
        agents, fields, structs, deltas = {}, {}, {}, {}
        for w in universe.worlds:
            agents[w.index] = w.agents_frame()
            agents[w.index]["tick"] = w.tick
            if w.index not in self.terrain:
                self.terrain[w.index] = w.terrain_export()
            if with_fields or w.index not in self.fields:
                fields[w.index] = w.fields_export()
            sc = w.structure_cells()
            structs[w.index] = sc
            h = np.zeros(w.grid.H * w.grid.W, np.float32)
            h[sc["cell"]] = sc["h"]
            prev = self._prev_struct.get(w.index)
            if prev is None or prev.shape != h.shape:
                changed = sc["cell"]
            else:
                changed = np.nonzero(np.abs(h - prev) > 0.02)[0].astype(np.int32)
            self._prev_struct[w.index] = h
            mat = np.full(h.shape, -1, np.int32)
            mat[sc["cell"]] = sc["mat"]
            deltas[w.index] = {"cell": changed, "h": h[changed], "mat": mat[changed]}
        status = universe.status()
        with self._lock:
            self.seq += 1
            self.agents = agents
            for k, v in fields.items():
                self.fields[k] = v
                self.fields_seq[k] = self.seq
            self.structures = structs
            for k, d in deltas.items():
                self.struct_deltas.setdefault(k, []).append((self.seq, d))
                self.struct_deltas[k] = self.struct_deltas[k][-50:]
            self.status = status
            self.last_publish = time.time()

    def snapshot(self) -> dict:
        with self._lock:
            return {"seq": self.seq, "agents": self.agents, "fields": dict(self.fields),
                    "fields_seq": dict(self.fields_seq), "structures": self.structures,
                    "struct_deltas": {k: list(v) for k, v in self.struct_deltas.items()},
                    "status": self.status, "telemetry": dict(self.telemetry), "published": self.last_publish}

    def set_telemetry(self, t: dict) -> None:
        with self._lock:
            self.telemetry = t
