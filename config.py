"""Configuration dataclasses and hardware profiles (spec §419-421, §634).

Strong defaults are chosen per profile; the autotuner may lower ``n_max`` from
measurements but never changes the laws of the universe (those live in
:mod:`genesis_omega.world.physics` and are versioned).
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field, asdict, fields


@dataclass
class WorldConfig:
    """Layout of an F4 world. All F4 worlds of one cosmos share the genome layout
    (hidden size, development lattice) so lineages can be transplanted between
    marked forks (expeditions)."""

    grid_w: int = 160
    grid_h: int = 80
    climate_div: int = 4          # coarse climate grid = grid / climate_div
    n_max: int = 512              # agent slots (static shapes on GPU)
    hidden: int = 24              # signal-substrate units per organism
    dev_size: int = 5             # development lattice edge (D×D×D)
    dev_channels: int = 16
    dev_hidden: int = 24
    dev_steps: int = 8
    emit_bands: int = 4
    focus_agents: int = 32        # F5 scientific-focus individuals (full internal recording)
    focus_ticks: int = 1024
    ring_frames: int = 256        # recent-history ring buffer (frames)
    ring_every: int = 4           # ticks between ring frames
    seed_bank: int = 512
    day_ticks: int = 240          # ticks per planetary day (base; planets scale it)
    founders_fraction: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class UniverseConfig:
    profile: str = "cpu"
    device: str = "cpu"
    deterministic: bool = True
    seed: int = 20261124
    n_stars: int = 3000
    active_worlds: int = 1        # how many life-bearing worlds run at F4
    world: WorldConfig = field(default_factory=WorldConfig)
    secondary_n_max_scale: float = 0.5
    # cadences (ticks)
    climate_every: int = 4
    analysis_every: int = 200
    speciation_every: int = 600
    galaxy_every: int = 1000
    keyframe_every: int = 2000
    keyframes_keep: int = 3
    # wall-clock cadences (seconds)
    checkpoint_interval_s: float = 300.0
    history_every_checkpoints: int = 6
    history_keep: int = 6
    frame_hz: float = 10.0
    # compute shares
    science_fraction: float = 0.12
    science_boost_fraction: float = 0.35
    fork_fraction: float = 0.25
    # server
    host: str = "0.0.0.0"
    port: int = 8765
    compile: str = "off"          # off | auto | on
    target_tps: float = 0.0       # 0 = as fast as possible
    lab_batch: int = 128

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: dict) -> "UniverseConfig":
        d = copy.deepcopy(d)
        wc = WorldConfig(**{k: v for k, v in d.pop("world", {}).items() if k in {f.name for f in fields(WorldConfig)}})
        known = {f.name for f in fields(UniverseConfig)}
        return UniverseConfig(world=wc, **{k: v for k, v in d.items() if k in known})


def profile_config(name: str, device: str | None = None) -> UniverseConfig:
    """Return the default configuration for a named hardware profile."""
    name = name.lower()
    if name == "cpu-test":
        cfg = UniverseConfig(
            profile=name, device="cpu", n_stars=400, active_worlds=1,
            world=WorldConfig(grid_w=64, grid_h=32, n_max=128, hidden=12, dev_size=4, dev_steps=5,
                              focus_agents=8, focus_ticks=256, ring_frames=64, ring_every=2, seed_bank=64,
                              day_ticks=120),
            climate_every=2, analysis_every=50, speciation_every=100, galaxy_every=200, keyframe_every=200,
            checkpoint_interval_s=3600, frame_hz=5, lab_batch=32,
        )
    elif name == "cpu":
        cfg = UniverseConfig(profile=name, device="cpu")
    elif name == "gpu":
        cfg = UniverseConfig(
            profile=name, device="cuda", n_stars=12000, active_worlds=2,
            world=WorldConfig(grid_w=384, grid_h=192, n_max=4096, hidden=40, focus_agents=48, focus_ticks=2048,
                              ring_frames=384, seed_bank=1024),
            lab_batch=256,
        )
    elif name == "a100":
        cfg = UniverseConfig(
            profile=name, device="cuda", n_stars=20000, active_worlds=2,
            world=WorldConfig(grid_w=512, grid_h=256, n_max=12288, hidden=48, focus_agents=64, focus_ticks=2048,
                              ring_frames=512, seed_bank=2048),
            secondary_n_max_scale=0.5, lab_batch=512, compile="auto",
        )
    else:
        raise ValueError(f"unknown profile {name!r}")
    if device is not None:
        cfg.device = device
    return cfg


def auto_profile(hw) -> str:
    """Pick a profile from detected hardware (never assumes an A100)."""
    forced = os.environ.get("GENESIS_PROFILE")
    if forced:
        return forced
    if hw.device == "cuda":
        if hw.is_a100 or hw.gpu_total_gb >= 70:
            return "a100"
        return "gpu"
    return "cpu"
