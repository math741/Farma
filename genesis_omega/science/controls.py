"""Positive / negative controls for every measurement pipeline (spec §437-440).

Hand-built synthetic organisms with known mechanisms (and matched ones
without) are pushed through the *same* probes used on evolved organisms. A
pipeline whose positive control fails or whose negative control "succeeds"
is reported as unvalidated, and its results on real organisms are not
promoted. Synthetic organisms exist only in the laboratory.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ..life.genome import GenomeLayout
from ..life.interface import I, O, N_INPUTS
from ..society import communication
from . import probes


def _blank(layout: GenomeLayout, device) -> torch.Tensor:
    g = torch.zeros(1, layout.size, device=device)
    a, b, _ = layout.slices["br_tau"]
    g[:, a:b] = 6.0                     # alpha ≈ 1 (memoryless) unless set otherwise
    a, b, _ = layout.slices["pl_eta"]
    g[:, a:b] = -12.0                   # negligible plasticity unless set otherwise
    a, _, _ = layout.slices["mod"]
    g[:, a + 2] = -6.0                  # tiny intrinsic noise
    return g


def _w(layout, g, row, col, value):
    a, _, shape = layout.slices["br_w"]
    g[0, a + row * shape[1] + col] = value


def _wo(layout, g, out, unit, value):
    a, _, shape = layout.slices["br_wo"]
    g[0, a + out * shape[1] + unit] = value


def _set(layout, g, seg, idx, value):
    a, _, _ = layout.slices[seg]
    g[0, a + idx] = value


def _alpha_gene(alpha: float) -> float:
    s = (alpha - 0.05) / 0.95
    s = min(max(s, 1e-4), 1 - 1e-4)
    return math.log(s / (1 - s))


def photo_sens(n: int, device) -> torch.Tensor:
    s = torch.zeros(n, 4, 4, device=device)
    s[:, :, 0] = 1.0
    return s


# ---------------------------------------------------------------- memory
def memory_organisms(layout, device, positive: bool, n: int = 8) -> tuple:
    g = _blank(layout, device)
    H = layout.hidden
    left = I["left.light"]
    right = I["right.light"]
    _w(layout, g, 2, left, 2.0)
    _w(layout, g, 2, right, -2.0)
    _set(layout, g, "mod", 2, 0.0)            # realistic intrinsic noise for both controls
    if positive:
        _w(layout, g, 2, N_INPUTS + 2, 1.6)   # self-excitation → bistable latch
        _set(layout, g, "br_tau", 2, _alpha_gene(0.5))
    else:
        a, b, _ = layout.slices["br_tau"]
        g[:, a:b] = 30.0                      # exactly memoryless
    genomes = g.repeat(n, 1)
    return genomes, torch.full((n,), float(H), device=device), photo_sens(n, device)


# -------------------------------------------------------------- learning
def learning_organisms(layout, device, positive: bool, n: int = 48) -> tuple:
    g = _blank(layout, device)
    H = layout.hidden
    e = I["energy"]
    bias = I["bias"]
    # modulators: fast (0) and slow (1) followers of the energy signal → m ≈ d(energy)/dt
    for unit, alpha in ((0, 1.0), (1, 0.08)):
        _w(layout, g, unit, e, 6.0)
        _w(layout, g, unit, bias, -3.0)
        _set(layout, g, "br_tau", unit, _alpha_gene(alpha))
    # choice unit (2): latched by self-excitation, biased by a plastic weight from the bias input
    _w(layout, g, 2, N_INPUTS + 2, 1.2)
    _set(layout, g, "br_tau", 2, _alpha_gene(0.6))
    _wo(layout, g, O["turn"], 2, 3.0)
    _set(layout, g, "mod", 2, 2.0)            # exploration noise so choices vary
    if positive:
        _set(layout, g, "pl_A", 2, 4.0)
        _set(layout, g, "pl_eta", 2, 6.0)
    genomes = g.repeat(n, 1)
    return genomes, torch.full((n,), float(H), device=device), photo_sens(n, device)


# ---------------------------------------------------------------- affect
def affect_organism(layout, device, positive: bool, seed: int = 0) -> tuple:
    g = _blank(layout, device)
    H = layout.hidden
    rng = np.random.default_rng(seed)
    e, bias = I["energy"], I["bias"]
    if positive:
        # unit 2: slow integrator of energy deficit that modulates several effectors
        _w(layout, g, 2, e, -3.0)
        _w(layout, g, 2, bias, 1.8)
        _w(layout, g, 2, N_INPUTS + 2, 0.6)
        _set(layout, g, "br_tau", 2, _alpha_gene(0.06))
        _wo(layout, g, O["eat"], 2, 3.0)
        _wo(layout, g, O["thrust"], 2, 2.0)
        _wo(layout, g, O["emit0"], 2, 2.0)
        _wo(layout, g, O["dig"], 2, -2.0)
        _w(layout, g, 0, N_INPUTS + 2, 2.0)  # drives the modulatory broadcast
    else:
        a, b, _ = layout.slices["br_wo"]
        g[0, a:b] = torch.tensor(rng.normal(0, 0.3, b - a), dtype=torch.float32, device=device)
    return g, torch.tensor(float(H), device=device), photo_sens(1, device)[0]


def simulate_series(layout, genome, units, sens, T: int, seed: int) -> dict:
    """A minimal foraging environment for affect-pipeline controls."""
    dev = genome.device
    rng = np.random.default_rng(seed)
    lab = probes.Lab(layout, genome.view(1, -1), units.view(1), sens.view(1, 4, 4))
    gen = torch.Generator(device=dev).manual_seed(seed)
    base_d, base_l = lab.baseline()
    base = base_d.shape[1] * base_d.shape[2]
    energy = 0.6
    patch = 0.5
    us, xs, ys, hs = [], [], [], []
    for t in range(T):
        patch = float(np.clip(patch + rng.normal(0, 0.08), 0, 1))
        loc = base_l.clone()
        loc[:, I["energy"] - base] = energy
        loc[:, I["resource_here"] - base] = patch
        u = lab.compose(base_d, loc)
        y, dW = lab.step(u, gen)
        eat = float(torch.sigmoid(y[0, O["eat"]]))
        gain = 0.04 * eat * patch
        energy = float(np.clip(energy + gain - 0.012, 0.0, 1.0))
        us.append(u[0].cpu().numpy())
        xs.append(lab.x[0].cpu().numpy())
        ys.append(y[0].cpu().numpy())
        hs.append([float(lab.m[0]), float(dW[0]), energy, 1.0, 0.0, 0.0, 15.0, 0.0])
    return {"u": np.array(us, np.float32), "x": np.array(xs, np.float32), "y": np.array(ys, np.float32),
            "h": np.array(hs, np.float32), "ctx": np.zeros((T, 3), np.float32), "ticks": np.arange(T), "id": -1}


# --------------------------------------------------------- communication
def synthetic_frames(positive: bool, n_agents: int = 120, n_frames: int = 40, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    frames = []
    ids = np.arange(1, n_agents + 1)
    for f in range(n_frames):
        r_here = rng.random(n_agents).astype(np.float16)
        hurt = rng.random(n_agents) < 0.15
        crowd = rng.integers(0, 4, n_agents).astype(np.float16)
        light = np.full(n_agents, 0.8 if (f // 10) % 2 == 0 else 0.02, np.float16)
        emit = np.zeros((n_agents, 4), np.uint8)
        if positive:
            band = np.where(hurt, 2, np.where(r_here > 0.6, 1, -1))
            speak = (band >= 0) & (rng.random(n_agents) < 0.8)
            emit[np.where(speak)[0], band[speak]] = 200
        else:
            speak = rng.random(n_agents) < 0.25
            emit[np.where(speak)[0], rng.integers(0, 4, speak.sum())] = 200
        frames.append({"tick": f * 4, "n": n_agents, "id": ids, "emit": emit, "r_here": r_here, "hurt": hurt,
                       "crowd": crowd, "light": light, "recv": rng.random((n_agents, 4)).astype(np.float16) * 0.2,
                       "turn": rng.normal(0, 0.3, n_agents).astype(np.float16),
                       "thrust": rng.random(n_agents).astype(np.float16),
                       "lon": rng.uniform(-3, 3, n_agents).astype(np.float32)})
    return frames


# ------------------------------------------------------------- runner
def run_all(layout, device, seed: int = 7) -> dict:
    """Run every pipeline's controls. Returns {pipeline: {positive, negative, valid}}."""
    out = {}
    gp, up, sp = memory_organisms(layout, device, True)
    gn, un, sn = memory_organisms(layout, device, False)
    mp = probes.memory_probe(layout, gp, up, sp, seed, delays=(1, 16), trials=32)
    mn = probes.memory_probe(layout, gn, un, sn, seed, delays=(1, 16), trials=32)
    ms = probes.memory_probe(layout, gp, up, sp, seed, delays=(16,), trials=32, shuffle=True)
    out["memory"] = {"positive_span": mp["span"], "negative_span": mn["span"],
                     "shuffled_acc": ms["per_delay"][0]["mean_acc"],
                     "valid": bool(mp["span"] >= 16 and mn["span"] == 0 and abs(ms["per_delay"][0]["mean_acc"] - 0.5)
                                   < 0.1)}
    gl, ul, sl = learning_organisms(layout, device, True)
    lp = probes.learning_probe(layout, gl, ul, sl, seed, plastic=True)
    lf = probes.learning_probe(layout, gl, ul, sl, seed, plastic=False)
    gneg, uneg, sneg = learning_organisms(layout, device, False)
    lnp = probes.learning_probe(layout, gneg, uneg, sneg, seed, plastic=True)
    lnf = probes.learning_probe(layout, gneg, uneg, sneg, seed, plastic=False)
    ev_pos = probes.learning_evidence(lp, lf)
    ev_neg = probes.learning_evidence(lnp, lnf)
    out["learning"] = {"positive": ev_pos, "negative": ev_neg,
                       "valid": bool(ev_pos["supported"] and not ev_neg["supported"])}
    rng = np.random.default_rng(seed)
    ga, ua, sa = affect_organism(layout, device, True)
    gb, ub, sb = affect_organism(layout, device, False, seed)
    sa_ser = simulate_series(layout, ga[0], ua, sa, 900, seed)
    sb_ser = simulate_series(layout, gb[0], ub, sb, 900, seed)
    aa = probes.affect_analysis(sa_ser, rng)
    ab = probes.affect_analysis(sb_ser, rng)
    ca = probes.affect_causal_test(layout, ga[0], ua, sa, sa_ser, np.array(aa["direction"]), seed)
    cb = probes.affect_causal_test(layout, gb[0], ub, sb, sb_ser, np.array(ab["direction"]), seed)
    la = probes.affect_levels(aa, ca)
    lb = probes.affect_levels(ab, cb)
    pos_levels = [k for k, v in la.items() if v["met"]]
    neg_levels = [k for k, v in lb.items() if v["met"]]
    out["affect"] = {"positive_levels": pos_levels, "negative_levels": neg_levels,
                     "positive_causal": ca["supported"], "negative_causal": cb["supported"],
                     "valid": bool({"A1", "A2"} <= set(pos_levels) and ca["supported"] and "A1" not in neg_levels
                                   and not cb["supported"])}
    cp = communication.analyze(synthetic_frames(True, seed=seed), np.random.default_rng(seed))
    cn = communication.analyze(synthetic_frames(False, seed=seed), np.random.default_rng(seed))
    out["communication"] = {"positive_p": cp.get("info_p"), "negative_p": cn.get("info_p"),
                            "valid": bool(cp.get("criteria", {}).get("information") and
                                          not cn.get("criteria", {}).get("information"))}
    out["all_valid"] = all(v["valid"] for k, v in out.items() if isinstance(v, dict))
    return out
