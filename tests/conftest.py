"""Shared fixtures. Everything runs on CPU with the small ``cpu-test`` profile."""

from __future__ import annotations

import os

import pytest
import torch

from genesis_omega.core.config import profile_config
from genesis_omega.core.hardware import configure_determinism
from genesis_omega.storage.checkpoint import StorageLayout

SEED = 424242


def small_cfg(seed: int = SEED, **over):
    cfg = profile_config("cpu-test")
    cfg.seed = seed
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


@pytest.fixture(scope="session", autouse=True)
def _determinism():
    configure_determinism("cpu", True)
    yield


@pytest.fixture
def cfg():
    return small_cfg()


@pytest.fixture
def storage(tmp_path) -> StorageLayout:
    lay = StorageLayout(str(tmp_path / "data"), "test-cosmos", str(tmp_path / "scratch"))
    lay.ensure()
    return lay


@pytest.fixture(scope="session")
def universe():
    """A small cosmos stepped for a while (shared, read-mostly: tests must not mutate it)."""
    from genesis_omega.runtime.universe import Universe

    configure_determinism("cpu", True)
    u = Universe.create(small_cfg(), None, run_controls=False)
    for _ in range(250):
        u.step_round()
    return u


@pytest.fixture
def fresh_world():
    """A brand-new F4 world (mutable)."""
    from genesis_omega.runtime.universe import Universe

    u = Universe.create(small_cfg(seed=SEED + 1), None, run_controls=False)
    return u.worlds[0]


def force_outputs(world, names, value: float = 3.0) -> None:
    """Set the output bias of the given effectors for every organism (test helper).

    This edits genomes directly — something only a test (never the simulation) does —
    to exercise the physics of actions that evolution has not discovered yet."""
    from genesis_omega.life.interface import O

    a, _, _ = world.layout.slices["br_bo"]
    for n in names:
        world.agents.genome[:, a + O[n]] = value
    world.agents.invalidate()


def has_display_free_chromium() -> bool:
    try:
        import playwright  # noqa: F401
    except ImportError:
        return False
    return os.environ.get("GENESIS_SKIP_BROWSER", "") != "1"


__all__ = ["small_cfg", "force_outputs", "SEED", "torch"]
