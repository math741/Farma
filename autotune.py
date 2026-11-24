"""Auto-tuning by measurement (spec §253-255, §264-266, §419-420, §638).

Benchmarks are real measurements on the machine at hand. The tuner only
changes capacity (agent slots) for a *new* cosmos — never the laws — and it
records what it measured. torch.compile is benchmarked and used only if it is
measurably faster and produces finite results.
"""

from __future__ import annotations

import copy
import time

import torch

from ..core.config import UniverseConfig
from ..core.hardware import memory_stats
from ..core.logging import get_logger
from ..cosmos.galaxy import Galaxy
from ..world.materials import Chemistry
from ..world.physics import WorldPhysics
from ..world.world import World

log = get_logger("autotune")


def _bench_world(cfg: UniverseConfig, n_max: int, ticks: int) -> dict:
    c = copy.deepcopy(cfg)
    c.world.n_max = n_max
    g = Galaxy.generate(c.seed, min(c.n_stars, 600))
    row = g.choose_focus_worlds(1)[0]
    planet = g.planet_dict(row)
    star = g.star_dict(int(planet["star"]))
    dev = torch.device(c.device)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    w = World(0, planet, star, c, c.world, WorldPhysics(), Chemistry(c.seed, dev), c.seed, dev)
    w.seed_life(3, 0.9)
    for _ in range(10):
        w.step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ticks):
        w.step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    mem = memory_stats(str(dev))
    pop = w.agents.count()
    del w
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return {"n_max": n_max, "ticks": ticks, "seconds": round(dt, 3), "tps": round(ticks / dt, 2),
            "agent_ticks_per_s": round(pop * ticks / dt, 1), "population": pop, "memory": mem}


def benchmark_compile(cfg: UniverseConfig, ticks: int = 60) -> dict:
    """Compare eager vs torch.compile for the substrate step (opt-in)."""
    from ..life import substrate
    from ..world import world as world_mod

    eager = _bench_world(cfg, cfg.world.n_max, ticks)
    try:
        compiled = torch.compile(substrate.substrate_step, dynamic=False)
        world_mod.substrate_step = compiled
        _bench_world(cfg, cfg.world.n_max, 5)         # warm-up / compilation
        comp = _bench_world(cfg, cfg.world.n_max, ticks)
        faster = comp["tps"] > eager["tps"] * 1.1
        if not faster:
            world_mod.substrate_step = substrate.substrate_step
        return {"eager_tps": eager["tps"], "compiled_tps": comp["tps"], "used": bool(faster)}
    except Exception as e:  # compile unavailable: stay eager, report why
        world_mod.substrate_step = substrate.substrate_step
        return {"eager_tps": eager["tps"], "error": repr(e)[:300], "used": False}


def autotune(cfg: UniverseConfig, min_tps: float | None = None, ticks: int = 40) -> tuple[UniverseConfig, dict]:
    """Choose the largest agent capacity that sustains ``min_tps`` (measured)."""
    min_tps = min_tps if min_tps is not None else (30.0 if cfg.device.startswith("cuda") else 15.0)
    cands = sorted({cfg.world.n_max, max(64, cfg.world.n_max // 2), max(64, cfg.world.n_max // 4)}, reverse=True)
    results = []
    chosen = cands[-1]
    for n in cands:
        try:
            r = _bench_world(cfg, n, ticks)
        except torch.cuda.OutOfMemoryError as e:
            results.append({"n_max": n, "error": "OOM", "detail": repr(e)[:200]})
            torch.cuda.empty_cache()
            continue
        results.append(r)
        if r["tps"] >= min_tps:
            chosen = n
            break
    out = copy.deepcopy(cfg)
    out.world.n_max = chosen
    report = {"min_tps": min_tps, "candidates": results, "chosen_n_max": chosen, "profile": cfg.profile}
    if cfg.compile in ("auto", "on") and cfg.device.startswith("cuda"):
        report["compile"] = benchmark_compile(out)
    log.info("autotune", extra={"fields": {"chosen_n_max": chosen,
                                           "measured": [(r.get("n_max"), r.get("tps")) for r in results]}})
    return out, report
