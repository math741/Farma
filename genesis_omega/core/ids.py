"""Persistent identifiers (spec §557-564).

Counters live inside checkpoints, so IDs are never reused across restarts.
Display formats are stable and human-readable.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IdAllocator:
    counters: dict[str, int] = field(default_factory=dict)

    def next(self, kind: str) -> int:
        n = self.counters.get(kind, 0) + 1
        self.counters[kind] = n
        return n

    def peek(self, kind: str) -> int:
        return self.counters.get(kind, 0)

    def state_dict(self) -> dict:
        return {"counters": dict(self.counters)}

    def load_state_dict(self, d: dict) -> None:
        self.counters = {k: int(v) for k, v in d.get("counters", {}).items()}


def cosmos_label(seed: int) -> str:
    return f"CΩ-{seed & 0xFFFFFFFF:08X}"


def star_label(index: int) -> str:
    return f"S-{index:05d}"


def planet_label(star_index: int, planet_index: int) -> str:
    return f"P-{star_index:05d}-{planet_index}"


def agent_label(agent_id: int) -> str:
    return f"A-{agent_id}"


def species_label(world_index: int, species_id: int) -> str:
    return f"SP-{world_index}-{species_id}"


def event_label(n: int) -> str:
    return f"E-{n:07d}"


def discovery_label(n: int) -> str:
    return f"D-{n:05d}"


def fork_label(n: int) -> str:
    return f"F-{n:04d}"


def material_label(world_index: int, material_id: int) -> str:
    return f"M-{world_index}-{material_id}"
