"""CommunicationEngine — measurement of candidate signalling (spec §57-60, §211, §215, §438).

Organisms can emit vibration in a few frequency bands (what their bodies can
produce) and sense the vibration field. This module never gives signals
meaning; it measures, from recorded frames, whether emissions carry
information about the sender's situation beyond shared confounds (time of
day), whether receivers behave differently when signals arrive, whether the
mapping is stable over time and across regions, and whether signalling
depends on an audience. Causal influence is established separately by the
Scientist with ablation forks.
"""

from __future__ import annotations

import numpy as np

from ..science.stats import conditional_mi, permutation_test, js_similarity, welch_t

EMIT_THRESHOLD = 26  # uint8 ≈ 0.1


def _context(r_here, hurt, crowd):
    rb = np.digitize(r_here.astype(np.float32), [0.15, 0.6])            # 3 levels of local resource
    cb = np.digitize(crowd.astype(np.float32), [0.5, 2.5])              # alone / few / many
    return rb * 6 + hurt.astype(np.int64) * 3 + cb                       # 18 context states


def _light_bin(light):
    return np.digitize(light.astype(np.float32), [0.05, 0.4])


def frames_to_table(frames: list[dict]) -> dict | None:
    """Flatten ring-buffer frames into per-agent-frame arrays (+ next-frame responses)."""
    rows = [f for f in frames if f.get("n", 0) > 0]
    if len(rows) < 3:
        return None
    cols = {k: [] for k in ("id", "symbol", "ctx", "light", "recv", "dturn", "dthrust", "lon", "crowd", "tick")}
    for a, b in zip(rows[:-1], rows[1:]):
        ids_a, ids_b = a["id"], b["id"]
        order = np.argsort(ids_b)
        pos = np.searchsorted(ids_b[order], ids_a)
        pos = np.clip(pos, 0, len(ids_b) - 1)
        match = ids_b[order][pos] == ids_a
        nb = order[pos]
        emit = a["emit"].astype(np.int64)
        strongest = emit.argmax(1)
        symbol = np.where(emit.max(1) >= EMIT_THRESHOLD, strongest + 1, 0)
        cols["id"].append(ids_a[match])
        cols["symbol"].append(symbol[match])
        cols["ctx"].append(_context(a["r_here"], a["hurt"], a["crowd"])[match])
        cols["light"].append(_light_bin(a["light"])[match])
        cols["recv"].append(a["recv"].astype(np.float32).sum(1)[match])
        cols["dturn"].append(np.abs(b["turn"][nb].astype(np.float32) - a["turn"].astype(np.float32))[match])
        cols["dthrust"].append((b["thrust"][nb].astype(np.float32) - a["thrust"].astype(np.float32))[match])
        cols["lon"].append(a["lon"][match])
        cols["crowd"].append(a["crowd"].astype(np.float32)[match])
        cols["tick"].append(np.full(int(match.sum()), a["tick"]))
    return {k: np.concatenate(v) if v else np.zeros(0) for k, v in cols.items()}


def analyze(frames: list[dict], rng: np.random.Generator, n_perm: int = 200) -> dict:
    """Return communication-candidate metrics with honest null comparisons."""
    tab = frames_to_table(frames)
    if tab is None or tab["symbol"].size < 50:
        return {"status": "insufficient data", "samples": 0 if tab is None else int(tab["symbol"].size)}
    sym, ctx, light = tab["symbol"], tab["ctx"], tab["light"]
    n = sym.size
    emitting = sym > 0
    out: dict = {"samples": int(n), "emission_rate": round(float(emitting.mean()), 4),
                 "symbols_used": int(len(np.unique(sym[emitting]))) if emitting.any() else 0}
    if emitting.sum() < 30:
        out["status"] = "too few emissions"
        return out
    # 1. information about sender context beyond time-of-day
    stat = lambda x, y, z: conditional_mi(x, y, z, 5, 18)  # noqa: E731
    mi, p, null = permutation_test(stat, sym, ctx, light, n_perm, rng)
    out["info_bits"] = round(mi, 5)
    out["info_null_mean"] = round(float(null.mean()), 5)
    out["info_p"] = round(p, 4)
    # 2. receiver response (observational): action change when signal arrives
    recv = tab["recv"]
    hi = recv > 0.1
    if hi.sum() >= 20 and (~hi).sum() >= 20:
        t, pr = welch_t(tab["dturn"][hi], tab["dturn"][~hi])
        out["receiver_dturn_t"] = round(t, 3)
        out["receiver_p"] = round(pr, 4)
    # 3. stability: P(context | symbol) in the two halves of the window
    half = tab["tick"] < np.median(tab["tick"])
    sims = []
    for s in np.unique(sym[emitting]):
        a = np.bincount(ctx[(sym == s) & half], minlength=18).astype(float)
        b = np.bincount(ctx[(sym == s) & ~half], minlength=18).astype(float)
        if a.sum() >= 10 and b.sum() >= 10:
            sims.append(js_similarity(a, b))
    out["stability"] = round(float(np.mean(sims)), 4) if sims else None
    # 4. generalisation: information present in both longitudinal halves (spatial holdout)
    east = tab["lon"] >= 0
    gen = []
    for m in (east, ~east):
        if (sym[m] > 0).sum() >= 20:
            g_mi, g_p, _ = permutation_test(stat, sym[m], ctx[m], light[m], max(20, n_perm // 3), rng)
            gen.append(g_p)
    out["generalization_p"] = [round(v, 4) for v in gen]
    # 5. social use (audience effect), stratified by light and resource
    rate_with, rate_alone = [], []
    crowd = tab["crowd"]
    for lb in np.unique(light):
        for rb in range(3):
            m = (light == lb) & (ctx // 6 == rb)
            a = m & (crowd > 0.5)
            b = m & (crowd <= 0.5)
            if a.sum() >= 15 and b.sum() >= 15:
                rate_with.append(emitting[a].mean())
                rate_alone.append(emitting[b].mean())
    if rate_with:
        out["audience_effect"] = round(float(np.mean(np.array(rate_with) - np.array(rate_alone))), 4)
    crit = {
        "information": out["info_p"] < 0.01 and mi > 2 * max(out["info_null_mean"], 1e-4),
        "stability": (out["stability"] or 0) > 0.8,
        "generalization": len(gen) == 2 and all(v < 0.05 for v in gen),
        "social_use": abs(out.get("audience_effect", 0.0)) > 0.02,
        "causality": None,  # only the Scientist's ablation forks can establish this
    }
    out["criteria"] = crit
    out["status"] = "candidate" if crit["information"] else "no evidence"
    return out


def marks_analysis(world, rng: np.random.Generator) -> dict:
    """Writing-like substrate check: do persistent marks co-vary with places' properties?
    (Only an information test; transmission to readers would require a fork test.)"""
    import torch

    m = world.marks.sum(0)
    if float(m.max()) < 0.05:
        return {"status": "no marks"}
    has = (m > 0.05).flatten()
    feats = torch.stack([world.R.flatten() > 0.3 * world.K.flatten(), world.struct_h.flatten() > 0.1,
                         world.store.flatten() > 0.05], 1).long()
    code = (feats[:, 0] * 4 + feats[:, 1] * 2 + feats[:, 2]).cpu().numpy()
    x = has.long().cpu().numpy()
    z = np.zeros_like(x)
    stat = lambda a, b, c: conditional_mi(a, b, c, 2, 8)  # noqa: E731
    mi, p, _ = permutation_test(stat, x, code, z, 30, rng)
    return {"status": "measured", "marked_cells": int(x.sum()), "info_bits": round(mi, 5), "p": round(p, 4)}
