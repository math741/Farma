"""Laboratory probes on organisms' real substrates (spec §31-41, §232-233, §437, §439).

Probes clone organisms' substrates (genome-derived parameters, optionally
their current plastic weights) into an isolated laboratory and drive them
through controlled input sequences built from the *same* world interface the
organism lives with (raw signals gated by its own body's sensor gains).
Nothing in the lab touches the prime timeline.

Every probe reports its controls. A probe result is evidence about a
measured function, never a claim about experience.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ..life.interface import I, O, N_INPUTS, DIR_SIGNALS, DIR_MODALITY, N_DIR_SIGNALS
from ..life.substrate import SubstrateParams, substrate_step
from .stats import cv_decode_accuracy, binom_sf, ridge_cv_r2, welch_t

DIR_MOD = torch.tensor([DIR_MODALITY[s] for s in DIR_SIGNALS])
SIG = {s: i for i, s in enumerate(DIR_SIGNALS)}
FRONT, RIGHT, BACK, LEFT = 0, 1, 2, 3


class Lab:
    """A batch of cloned substrates (B = organisms × replicates)."""

    def __init__(self, layout, genomes: torch.Tensor, units: torch.Tensor, sens: torch.Tensor,
                 W: torch.Tensor | None = None):
        self.layout = layout
        self.genomes = genomes
        self.p = SubstrateParams(layout, genomes, units)
        self.W = (W if W is not None else self.p.W0).clone()
        self.sens = sens                     # (B,4,4)
        self.B = genomes.shape[0]
        self.H = layout.hidden
        self.device = genomes.device
        self.x = torch.zeros(self.B, self.H, device=self.device)
        self.m = torch.zeros(self.B, device=self.device)

    def baseline(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Neutral situation: daylight, some food, comfortable, alone."""
        d = torch.zeros(self.B, 4, N_DIR_SIGNALS, device=self.device)
        d[:, :, SIG["light"]] = 0.5
        d[:, :, SIG["resource"]] = 0.3
        loc = torch.zeros(self.B, N_INPUTS - 4 * N_DIR_SIGNALS, device=self.device)
        base = 4 * N_DIR_SIGNALS
        loc[:, I["resource_here"] - base] = 0.3
        loc[:, I["energy"] - base] = 0.6
        loc[:, I["integrity"] - base] = 1.0
        loc[:, I["speed"] - base] = 0.3
        loc[:, I["bias"] - base] = 1.0
        return d, loc

    def compose(self, d: torch.Tensor, loc: torch.Tensor) -> torch.Tensor:
        gains = self.sens[:, :, DIR_MOD.to(self.device)]
        return torch.cat([(d * gains).reshape(self.B, -1), loc], 1)

    def step(self, u, gen, plastic=True, clamp_dir=None, recurrent_scale=1.0, noise=True):
        nz = torch.randn(self.B, self.H, generator=gen, device=self.device) if noise else None
        self.W, self.x, y, self.m, dW = substrate_step(self.p, self.W, self.x, self.m, u, nz, plastic=plastic,
                                                       clamp_dir=clamp_dir, recurrent_scale=recurrent_scale)
        return y, dW


def _best_modality(sens: torch.Tensor) -> torch.Tensor:
    """Per organism, the modality (photo/chemo/mechano) its left+right sensors weight most."""
    lr = sens[:, LEFT, :3] + sens[:, RIGHT, :3]
    return lr.argmax(1)


def _lateral_stimulus(d: torch.Tensor, side: torch.Tensor, modality: torch.Tensor, amp: float) -> torch.Tensor:
    """Add a left (side=0) or right (side=1) pulse in each organism's best modality."""
    sig_for_mod = {0: SIG["light"], 1: SIG["resource"], 2: SIG["vib0"]}
    out = d.clone()
    for mod, sig in sig_for_mod.items():
        m = modality == mod
        if not bool(m.any()):
            continue
        left = m & (side == 0)
        right = m & (side == 1)
        out[left, LEFT, sig] += amp
        out[right, RIGHT, sig] += amp
    return out


# ------------------------------------------------------------------ O4
def memory_probe(layout, genomes, units, sens, seed: int, delays=(1, 4, 16, 48), trials: int = 32,
                 recurrent_scale: float = 1.0, shuffle: bool = False) -> dict:
    """Can a linear decoder read which side a brief stimulus came from, after a delay,
    from the substrate state alone? Per organism, cross-validated, with binomial tests."""
    n = genomes.shape[0]
    dev = genomes.device
    rng = np.random.default_rng(seed)
    res = {"delays": list(delays), "n_organisms": int(n), "trials": trials, "per_delay": []}
    rep = torch.arange(n, device=dev).repeat_interleave(trials)
    side = torch.tensor(np.tile(np.r_[np.zeros(trials // 2), np.ones(trials - trials // 2)], n), device=dev).long()
    for d_ in delays:
        lab = Lab(layout, genomes[rep], units[rep], sens[rep])
        gen = torch.Generator(device=dev).manual_seed(seed * 1000 + d_)
        base_d, base_l = lab.baseline()
        mod = _best_modality(lab.sens)
        stim = _lateral_stimulus(base_d, side, mod, 0.8)
        u0, us = lab.compose(base_d, base_l), lab.compose(stim, base_l)
        for _ in range(5):
            lab.step(u0, gen, recurrent_scale=recurrent_scale)
        for _ in range(3):
            lab.step(us, gen, recurrent_scale=recurrent_scale)
        for _ in range(d_):
            lab.step(u0, gen, recurrent_scale=recurrent_scale)
        X = lab.x.cpu().numpy().reshape(n, trials, -1)
        y = side.cpu().numpy().reshape(n, trials)
        accs, sig = [], 0
        for k in range(n):
            yy = rng.permutation(y[k]) if shuffle else y[k]
            acc = cv_decode_accuracy(X[k], yy, 4, rng)
            accs.append(acc)
            if binom_sf(int(round(acc * trials)), trials, 0.5) < 0.01:
                sig += 1
        res["per_delay"].append({"delay": d_, "mean_acc": round(float(np.mean(accs)), 4),
                                 "frac_significant": round(sig / n, 4)})
    # memory span: longest delay at which ≥ 25% of organisms decode significantly
    span = 0
    for r in res["per_delay"]:
        if r["frac_significant"] >= 0.25:
            span = r["delay"]
    res["span"] = span
    return res


# ------------------------------------------------------------------ O3
def learning_probe(layout, genomes, units, sens, seed: int, trials: int = 30, plastic: bool = True) -> dict:
    """Two-choice contingency task with symmetric cues; the rewarded side is fixed per
    organism and random across organisms, so only within-lifetime adaptation can
    raise success above an innate bias. Reward is delivered through the same
    channels feeding produces in the world (energy and local resource rise)."""
    n = genomes.shape[0]
    dev = genomes.device
    lab = Lab(layout, genomes, units, sens)
    gen = torch.Generator(device=dev).manual_seed(seed)
    rng = np.random.default_rng(seed)
    good = torch.tensor(rng.integers(0, 2, n), device=dev)              # 0 = left, 1 = right
    base_d, base_l = lab.baseline()
    mod = _best_modality(lab.sens)
    cue_d = _lateral_stimulus(_lateral_stimulus(base_d, torch.zeros(n, dtype=torch.long, device=dev), mod, 0.6),
                              torch.ones(n, dtype=torch.long, device=dev), mod, 0.6)
    base = 4 * N_DIR_SIGNALS
    energy = torch.full((n,), 0.5, device=dev)
    correct = np.zeros((n, trials), dtype=bool)

    def tick(win_mask=None, food=False):
        nonlocal energy
        energy = (energy - 0.025).clamp(min=0.1)
        if win_mask is not None:
            energy = torch.where(win_mask, (energy + 0.09).clamp(max=1.0), energy)
        loc = base_l.clone()
        loc[:, I["energy"] - base] = energy
        if food:
            loc[:, I["resource_here"] - base] = torch.where(win_mask, torch.full_like(energy, 0.9),
                                                            torch.full_like(energy, 0.05))
        return loc

    for k in range(trials):
        turn_sum = torch.zeros(n, device=dev)
        for _ in range(6):
            loc = tick()
            y, _ = lab.step(lab.compose(cue_d, loc), gen, plastic=plastic)
            turn_sum += torch.tanh(y[:, O["turn"]])
        choice = (turn_sum < 0).long()                                    # positive turn = left
        win = choice == good
        correct[:, k] = win.cpu().numpy()
        for _ in range(4):
            loc = tick(win, food=True)
            lab.step(lab.compose(base_d, loc), gen, plastic=plastic)
        for _ in range(4):
            loc = tick()
            lab.step(lab.compose(base_d, loc), gen, plastic=plastic)
    early = correct[:, : trials // 3].mean(1)
    late = correct[:, -trials // 3:].mean(1)
    return {"plastic": plastic, "n_organisms": int(n), "early": round(float(early.mean()), 4),
            "late": round(float(late.mean()), 4), "improvement": (late - early).tolist(),
            "mean_improvement": round(float((late - early).mean()), 4)}


def learning_evidence(plastic: dict, frozen: dict) -> dict:
    a = np.array(plastic["improvement"])
    b = np.array(frozen["improvement"])
    t, p = welch_t(a, b)
    return {"plastic_gain": plastic["mean_improvement"], "frozen_gain": frozen["mean_improvement"],
            "t": round(t, 3), "p": round(p, 5),
            "supported": bool(p < 0.01 and plastic["mean_improvement"] > frozen["mean_improvement"] + 0.05
                              and plastic["mean_improvement"] > 0.05)}


# --------------------------------------------------- tracking world (1D)
def _bearing_inputs(lab: Lab, theta: torch.Tensor, base_d, base_l, amp=0.8):
    d = base_d.clone()
    angles = torch.tensor([0.0, -math.pi / 2, math.pi, math.pi / 2], device=theta.device)
    inten = torch.cos(theta.unsqueeze(1) - angles.unsqueeze(0)).clamp(min=0) * amp
    mod = _best_modality(lab.sens)
    sig_for_mod = {0: SIG["light"], 1: SIG["resource"], 2: SIG["vib0"]}
    for mm, sig in sig_for_mod.items():
        m = mod == mm
        if bool(m.any()):
            d[m, :, sig] = d[m, :, sig] + inten[m]
    return lab.compose(d, base_l)


def self_world_probe(layout, genomes, units, sens, seed: int, trials: int = 16, T: int = 40) -> dict:
    """Self vs yoked (externally caused) sensory change with matched statistics:
    the yoked trajectory is the organism's *own* earlier closed-loop trajectory.
    Decoders: substrate state (test) vs raw inputs (confound control)."""
    n = genomes.shape[0]
    dev = genomes.device
    rng = np.random.default_rng(seed)
    rep = torch.arange(n, device=dev).repeat_interleave(trials)
    B = rep.numel()

    def closed_loop(gen, yoke=None):
        lab = Lab(layout, genomes[rep], units[rep], sens[rep])
        base_d, base_l = lab.baseline()
        theta = torch.tensor(rng.uniform(-math.pi, math.pi, B), device=dev, dtype=torch.float32)
        thetas, xs, us = [], [], []
        for t in range(T):
            th = yoke[t] if yoke is not None else theta
            u = _bearing_inputs(lab, th, base_d, base_l)
            y, _ = lab.step(u, gen)
            turn = torch.tanh(y[:, O["turn"]]) * 0.5
            theta = theta - turn + 0.05 * torch.randn(B, generator=gen, device=dev)
            theta = (theta + math.pi) % (2 * math.pi) - math.pi
            thetas.append(th.clone())
            xs.append(lab.x.clone())
            us.append(u.clone())
        return thetas, torch.stack(xs[T // 2:]), torch.stack(us[T // 2:])

    gen = torch.Generator(device=dev).manual_seed(seed)
    traj, xs_self, us_self = closed_loop(gen)
    gen2 = torch.Generator(device=dev).manual_seed(seed + 7)
    _, xs_self2, us_self2 = closed_loop(gen2)
    _, xs_yoke, us_yoke = closed_loop(gen2, yoke=traj)
    # per organism: classify (self2 vs yoke) from time-sampled snapshots
    acc_x, acc_u, acc_shuf = [], [], []
    for k in range(n):
        sl = slice(k * trials, (k + 1) * trials)
        Xa = xs_self2[:, sl].reshape(-1, xs_self2.shape[-1]).cpu().numpy()
        Xb = xs_yoke[:, sl].reshape(-1, xs_yoke.shape[-1]).cpu().numpy()
        Ua = us_self2[:, sl].reshape(-1, us_self2.shape[-1]).cpu().numpy()
        Ub = us_yoke[:, sl].reshape(-1, us_yoke.shape[-1]).cpu().numpy()
        sub = rng.choice(Xa.shape[0], size=min(160, Xa.shape[0]), replace=False)
        X = np.vstack([Xa[sub], Xb[sub]])
        U = np.vstack([Ua[sub], Ub[sub]])
        y = np.r_[np.zeros(len(sub)), np.ones(len(sub))].astype(np.int64)
        acc_x.append(cv_decode_accuracy(X, y, 4, rng))
        acc_u.append(cv_decode_accuracy(U, y, 4, rng))
        acc_shuf.append(cv_decode_accuracy(X, rng.permutation(y), 4, rng))
    ax, au, ash = float(np.mean(acc_x)), float(np.mean(acc_u)), float(np.mean(acc_shuf))
    return {"n_organisms": int(n), "acc_state": round(ax, 4), "acc_inputs_control": round(au, 4),
            "acc_shuffled_control": round(ash, 4),
            "supported": bool(ax > 0.6 and ax > au + 0.05 and abs(ash - 0.5) < 0.06)}


def body_schema_probe(layout, genomes, units, sens, seed: int, T: int = 120, flip_at: int = 40,
                      plastic: bool = True) -> dict:
    """Motor adaptation after a body perturbation (reversed turning, like prism
    adaptation): tracking error before, right after and late after the flip."""
    n = genomes.shape[0]
    dev = genomes.device
    rng = np.random.default_rng(seed)
    lab = Lab(layout, genomes, units, sens)
    base_d, base_l = lab.baseline()
    gen = torch.Generator(device=dev).manual_seed(seed)
    theta = torch.tensor(rng.uniform(-1.0, 1.0, n), device=dev, dtype=torch.float32)
    err = []
    for t in range(T):
        u = _bearing_inputs(lab, theta, base_d, base_l)
        y, _ = lab.step(u, gen, plastic=plastic)
        turn = torch.tanh(y[:, O["turn"]]) * 0.5
        sign = -1.0 if t < flip_at else 1.0
        theta = theta + sign * turn + 0.04 * torch.randn(n, generator=gen, device=dev)
        theta = (theta + math.pi) % (2 * math.pi) - math.pi
        if t % 15 == 0:
            theta = theta + torch.tensor(rng.uniform(-1.0, 1.0, n), device=dev, dtype=torch.float32)
        err.append(theta.abs().cpu().numpy())
    e = np.stack(err)
    pre = e[flip_at - 20:flip_at].mean(0)
    early = e[flip_at:flip_at + 20].mean(0)
    late = e[-20:].mean(0)
    return {"plastic": plastic, "pre": round(float(pre.mean()), 4), "after": round(float(early.mean()), 4),
            "late": round(float(late.mean()), 4), "recovery": (early - late).tolist(),
            "tracking_skill": round(float((math.pi / 2) - pre.mean()), 4)}


def body_schema_evidence(plastic: dict, frozen: dict) -> dict:
    a, b = np.array(plastic["recovery"]), np.array(frozen["recovery"])
    t, p = welch_t(a, b)
    return {"plastic_recovery": round(float(a.mean()), 4), "frozen_recovery": round(float(b.mean()), 4),
            "t": round(t, 3), "p": round(p, 5),
            "supported": bool(p < 0.01 and a.mean() > b.mean() + 0.05 and plastic["tracking_skill"] > 0.2)}


# ------------------------------------------------------------- affect
def _autocorr_time(z: np.ndarray, max_lag: int = 200) -> float:
    z = z - z.mean()
    v = (z * z).mean()
    if v <= 1e-12:
        return 0.0
    for lag in range(1, min(max_lag, len(z) - 1)):
        if (z[:-lag] * z[lag:]).mean() / v < math.exp(-1):
            return float(lag)
    return float(max_lag)


def _circular_null(a: np.ndarray, b: np.ndarray, rng, n: int = 50) -> float:
    obs = abs(np.corrcoef(a, b)[0, 1])
    null = []
    L = len(a)
    for _ in range(n):
        s = rng.integers(L // 5, L - L // 5)
        null.append(abs(np.corrcoef(np.roll(a, s), b)[0, 1]))
    return float((1 + (np.array(null) >= obs).sum()) / (n + 1))


def affect_analysis(series: dict, rng: np.random.Generator) -> dict:
    """Functional-affect pipeline A0–A4 on one F5 focus recording.

    h columns: m, dW, energy, integrity, thermal_dev, shelter, t_env, damage.
    """
    x, u, y, h = series["x"], series["u"], series["y"], series["h"]
    T = x.shape[0]
    out = {"ticks": int(T), "levels": {}}
    if T < 200:
        out["status"] = "insufficient recording"
        return out
    energy, integ = h[:, 2], h[:, 3]
    eat = 1 / (1 + np.exp(-y[:, O["eat"]]))
    # A0 regulation: feeding drive increases with energy deficit (controlling for food present)
    res_here = u[:, I["resource_here"]]
    Xc = np.stack([res_here, np.ones(T)], 1)
    beta = np.linalg.lstsq(Xc, eat, rcond=None)[0]
    eat_res = eat - Xc @ beta
    deficit = 1 - energy
    p0 = _circular_null(deficit, eat_res, rng)
    r0 = float(np.corrcoef(deficit, eat_res)[0, 1]) if eat_res.std() > 1e-9 and deficit.std() > 1e-9 else 0.0
    out["levels"]["A0"] = {"name": "regulation", "r_deficit_feeding": round(r0, 4), "p": round(p0, 4),
                           "met": bool(r0 > 0.1 and p0 < 0.02)}
    # Candidate latent: the substrate direction with the broadest influence on effectors
    # (rank-1 reduced-rank regression state → outputs). Then ask whether it is valence-like.
    Xc = x - x.mean(0)                       # raw (unstandardised): near-silent units get ~0 weight
    keep = y.std(0) > 1e-6
    Ys = (y[:, keep] - y[:, keep].mean(0)) / (y[:, keep].std(0) + 1e-6)
    lam = 1e-2 * float((Xc ** 2).sum(0).max() + 1e-9)
    B = np.linalg.solve(Xc.T @ Xc + lam * np.eye(x.shape[1]), Xc.T @ Ys)
    _, sv, vt = np.linalg.svd(Xc @ B, full_matrices=False)
    v = B @ vt[0]
    v = v / (np.linalg.norm(v) + 1e-9)
    z = Xc @ v
    need = (1 - energy) + 2 * (1 - integ) + np.abs(h[:, 4]) / 3.0
    if z.std() > 1e-9 and need.std() > 1e-9:
        r1 = float(np.corrcoef(z, need)[0, 1])
        p1 = _circular_null(z, need, rng)
    else:
        r1, p1 = 0.0, 1.0
    V = np.r_[0, np.diff(energy)] + 2 * np.r_[0, np.diff(integ)]
    r1b = float(np.corrcoef(z, V)[0, 1]) if z.std() > 1e-9 and V.std() > 1e-9 else 0.0
    a1 = bool(abs(r1) > 0.3 and p1 < 0.02)
    out["levels"]["A1"] = {"name": "valence-like", "r_need": round(r1, 4), "p": round(p1, 4),
                           "r_change": round(r1b, 4), "met": a1}
    sd = x.std(0)
    out["direction"] = v.tolist()
    out["active_units"] = int((sd > 0.01 * (sd.max() + 1e-12)).sum())
    # A2 candidates (correlational only; the level itself is decided causally)
    chans = {}
    for name, k in O.items():
        yk = y[1:, k]
        if yk.std() > 1e-6 and z[:-1].std() > 1e-9:
            chans[name] = round(float(np.corrcoef(z[:-1], yk)[0, 1]), 3)
    out["levels"]["A2"] = {"name": "global modulation", "correlations": chans, "met": False,
                           "decided_by": "causal ablation"}
    out["levels"]["A3"] = {"name": "persistent state", "met": False, "decided_by": "pulse test"}
    # A4 memory/plasticity modulation
    dW = h[:, 1]
    p4 = _circular_null(z, dW, rng) if dW.std() > 1e-12 else 1.0
    r4 = float(np.corrcoef(z, dW)[0, 1]) if dW.std() > 1e-12 else 0.0
    out["levels"]["A4"] = {"name": "memory modulation", "r_plasticity": round(r4, 4), "p": round(p4, 4),
                           "met": bool(a1 and abs(r4) > 0.15 and p4 < 0.02)}
    return out


def affect_causal_test(layout, genome, units, sens, series: dict, direction: np.ndarray, seed: int,
                       reps: int = 4) -> dict:
    """Causal tests of a candidate latent (spec §40: find state, block, repeat, compare).

    1. Block: replay the recorded inputs (open loop) with the latent's fluctuations
       removed at every step (projection held at its intact mean). Compare with the
       intact replay under the same noise, against the run-to-run variability of two
       intact replays with different noise. Global modulation (A2) = ≥3 effector
       channels changed beyond 3× that variability.
    2. Pulse (A3): a brief drop of the energy signal, then baseline — does the
       projection on the latent stay displaced after the input returns?
    No effect beyond noise ⇒ no functional evidence."""
    dev = genome.device
    u = torch.tensor(series["u"], device=dev, dtype=torch.float32)
    T = u.shape[0]
    H = layout.hidden
    v = np.asarray(direction, np.float32)
    v = v / (np.linalg.norm(v) + 1e-9)
    vt = torch.tensor(v, device=dev)

    def run(n_rows, noise_rows, clamp=None, mu=None):
        lab = Lab(layout, genome.view(1, -1).repeat(n_rows, 1), units.view(1).repeat(n_rows),
                  sens.view(1, 4, 4).repeat(n_rows, 1, 1))
        ys, xs = [], []
        for t in range(T):
            nz = torch.stack([g_noise[t] for g_noise in noise_rows])
            lab.W, lab.x, y, lab.m, _ = substrate_step(lab.p, lab.W, lab.x, lab.m, u[t].view(1, -1).repeat(n_rows, 1),
                                                       nz, plastic=True, clamp_dir=clamp, clamp_mu=mu)
            ys.append(y)
            xs.append(lab.x)
        return torch.stack(ys), torch.stack(xs)

    effects, noises, ratios = [], [], []
    for r in range(reps):
        g1 = torch.Generator(device=dev).manual_seed(seed * 100 + 2 * r)
        g2 = torch.Generator(device=dev).manual_seed(seed * 100 + 2 * r + 1)
        n1 = [torch.randn(H, generator=g1, device=dev) for _ in range(T)]
        n2 = [torch.randn(H, generator=g2, device=dev) for _ in range(T)]
        y_a, x_a = run(1, [n1])
        mu = (x_a[:, 0] * vt).sum(1).mean().view(1)
        y_b, _ = run(1, [n2])
        y_c, _ = run(1, [n1], clamp=vt.view(1, H), mu=mu)
        eff = (y_c[:, 0] - y_a[:, 0]).abs().mean(0).cpu().numpy()
        noi = (y_b[:, 0] - y_a[:, 0]).abs().mean(0).cpu().numpy()
        effects.append(eff)
        noises.append(noi)
        ratios.append(float(eff.mean() / (noi.mean() + 1e-6)))
    eff = np.mean(effects, 0)
    noi = np.mean(noises, 0)
    changed = [name for name, j in O.items() if eff[j] > 3 * noi[j] + 1e-4]
    supported = bool(min(ratios) > 2.0 and eff.mean() > 1e-3)
    # pulse test for persistence
    lab2 = Lab(layout, genome.view(1, -1), units.view(1), sens.view(1, 4, 4))
    bd, bl = lab2.baseline()
    gen2 = torch.Generator(device=dev).manual_seed(seed + 1)
    proj = []
    for t in range(60):
        loc = bl.clone()
        if 20 <= t < 23:
            loc[:, I["energy"] - 4 * N_DIR_SIGNALS] = 0.1
        lab2.step(lab2.compose(bd, loc), gen2, noise=False)
        proj.append(float((lab2.x[0] * vt).sum()))
    proj = np.array(proj)
    pre = proj[10:20].mean()
    peak = np.abs(proj[20:26] - pre).max()
    after = np.abs(proj[30:40] - pre).mean()
    persistent = bool(peak > 0.05 and after > 0.3 * peak)
    return {"effect": round(float(eff.mean()), 5), "noise": round(float(noi.mean()), 5),
            "ratios": [round(x, 3) for x in ratios], "changed_channels": changed, "supported": supported,
            "pulse": {"peak": round(float(peak), 4), "after": round(float(after), 4), "persistent": persistent}}


def affect_levels(analysis: dict, causal: dict | None) -> dict:
    """Combine correlational (A0, A1, A4) and causal (A2, A3) evidence."""
    lv = {k: dict(v) for k, v in analysis.get("levels", {}).items()}
    a1 = lv.get("A1", {}).get("met", False)
    if causal is not None:
        lv["A2"]["met"] = bool(a1 and causal["supported"] and len(causal["changed_channels"]) >= 3)
        lv["A2"]["changed_channels"] = causal["changed_channels"]
        lv["A3"]["met"] = bool(a1 and causal["pulse"]["persistent"])
        lv["A3"]["pulse"] = causal["pulse"]
    for k in ("A5", "A6", "A7", "A8", "A9"):
        lv[k] = {"name": {"A5": "contextual response", "A6": "anticipation", "A7": "learned affect",
                          "A8": "social affect", "A9": "emergent affective architecture"}[k],
                 "met": False, "status": "NOT IMPLEMENTED"}
    return lv
