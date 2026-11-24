"""Ω2-Ω5 science: pipeline controls, statistics, communication metrics, the Scientist's validation ladder."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import small_cfg
from genesis_omega.core.status import Validation
from genesis_omega.life.genome import GenomeLayout
from genesis_omega.science import controls, probes, stats
from genesis_omega.society import communication

LAYOUT = GenomeLayout(12, 16, 24)


@pytest.fixture(scope="module")
def control_results():
    return controls.run_all(LAYOUT, torch.device("cpu"))


# ------------------------------------------------------------------ controls
def test_every_pipeline_passes_its_controls(control_results):
    r = control_results
    assert r["all_valid"], r


def test_memory_pipeline_controls(control_results):
    m = control_results["memory"]
    assert m["positive_span"] >= 16 and m["negative_span"] == 0
    assert abs(m["shuffled_acc"] - 0.5) < 0.1          # shuffled labels decode at chance


def test_learning_pipeline_controls(control_results):
    pos, neg = control_results["learning"]["positive"], control_results["learning"]["negative"]
    assert pos["supported"] and pos["plastic_gain"] > pos["frozen_gain"]
    assert not neg["supported"]


def test_affect_pipeline_controls(control_results):
    a = control_results["affect"]
    assert {"A1", "A2"} <= set(a["positive_levels"]) and a["positive_causal"]
    assert "A1" not in a["negative_levels"] and not a["negative_causal"]


def test_communication_pipeline_controls(control_results):
    c = control_results["communication"]
    assert c["positive_p"] < 0.01 and c["negative_p"] > 0.05


# ------------------------------------------------------------------ statistics
def test_mutual_information_estimators():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 4, 5000)
    assert stats.mutual_information(x, rng.integers(0, 4, 5000)) < 0.01
    assert stats.mutual_information(x, x) == pytest.approx(2.0, abs=0.02)
    z = rng.integers(0, 2, 5000)
    y = (x + z) % 4
    assert stats.conditional_mi(x, y, z) > 1.5
    obs, p, null = stats.permutation_test(lambda a, b, c: stats.conditional_mi(a, b, c), x, y, z, 50, rng)
    assert p < 0.05 and obs > null.max()


def test_tests_and_decoders():
    assert stats.binom_sf(0, 10, 0.5) == 1.0 and stats.binom_sf(11, 10, 0.5) == 0.0
    assert stats.binom_sf(10, 10, 0.5) == pytest.approx(0.5 ** 10)
    rng = np.random.default_rng(1)
    t, p = stats.welch_t(rng.normal(1, 1, 200), rng.normal(0, 1, 200))
    assert t > 5 and p < 1e-4
    X = rng.normal(0, 1, (400, 3))
    y = (X[:, 0] > 0).astype(int)
    assert stats.cv_decode_accuracy(X, y, 4, rng) > 0.9
    assert abs(stats.cv_decode_accuracy(X, rng.permutation(y), 4, rng) - 0.5) < 0.1
    yr = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(0, 0.1, 400)
    assert stats.ridge_cv_r2(X, yr, 4, rng) > 0.95
    assert stats.js_similarity(np.array([1.0, 1, 0]), np.array([1.0, 1, 0])) == pytest.approx(1.0)


# ------------------------------------------------------------------ communication
def test_communication_metrics_on_synthetic_frames():
    pos = communication.analyze(controls.synthetic_frames(True, seed=3), np.random.default_rng(3))
    neg = communication.analyze(controls.synthetic_frames(False, seed=3), np.random.default_rng(3))
    assert pos["status"] == "candidate" and pos["criteria"]["information"]
    assert neg["status"] == "no evidence"
    assert pos["criteria"]["causality"] is None          # only fork ablation can establish causality


def test_communication_needs_enough_data():
    assert communication.analyze([], np.random.default_rng(0))["status"] == "insufficient data"


# ------------------------------------------------------------------ probes on evolved organisms
def test_probes_run_on_real_organisms(universe):
    w = universe.worlds[0]
    A = w.agents
    act = torch.nonzero(A.active).flatten()[:6]
    from genesis_omega.life.interface import CAP

    G, U, S = A.genome[act], A.caps[act, CAP["units"]], A.sens[act]
    mem = probes.memory_probe(w.layout, G, U, S, 5, delays=(1, 4), trials=16)
    assert mem["n_organisms"] == 6 and len(mem["per_delay"]) == 2
    lp = probes.learning_probe(w.layout, G, U, S, 5, trials=9)
    assert 0 <= lp["early"] <= 1 and 0 <= lp["late"] <= 1
    sw = probes.self_world_probe(w.layout, G[:2], U[:2], S[:2], 5, trials=4, T=12)
    assert 0 <= sw["acc_state"] <= 1
    bs = probes.body_schema_probe(w.layout, G, U, S, 5, T=60, flip_at=30)
    assert len(bs["recovery"]) == 6


# ------------------------------------------------------------------ the Scientist's ladder
def _scientist():
    from genesis_omega.runtime.universe import Universe

    u = Universe.create(small_cfg(seed=99), None, run_controls=False)
    for _ in range(30):
        u.step_round()
    return u


def test_discoveries_climb_the_ladder_only_with_valid_controls():
    u = _scientist()
    sc = u.scientist
    w = u.worlds[0]
    smp = {"ids": [1, 2, 3]}
    # pipeline controls invalid → a passing probe is recorded but cannot be promoted
    sc.control_results = {"memory": {"valid": False}}
    sc._judge("memory", w, 1, 11, {"span": 16}, smp, passed=True, title="t", mechanism="m", ablation=None)
    d = sc.discoveries[-1]
    assert d.status == Validation.CANDIDATE.value and "cannot be promoted" in d.controls["note"]
    # valid controls: a new claim is TESTED at once; the earlier unpromotable one stays CANDIDATE
    sc.control_results = {"memory": {"valid": True}}
    sc._judge("memory", w, 2, 12, {"span": 16}, smp, passed=True, title="t2", mechanism="m", ablation=None)
    d2 = sc.discoveries[-1]
    assert d2 is not d and d2.status == Validation.TESTED.value and d.status == Validation.CANDIDATE.value
    # TESTED → REPLICATED (new seeds) → CAUSALLY SUPPORTED (ablation removes the effect)
    sc.discoveries.clear()
    sc._judge("memory", w, 3, 21, {"span": 16}, smp, passed=True, title="t3", mechanism="m",
              ablation={"mechanism": "recurrence removed", "supported": True})
    d3 = sc.discoveries[-1]
    assert d3.status == Validation.TESTED.value
    sc._judge("memory", w, 3, 22, {"span": 16}, smp, passed=True, title="t3", mechanism="m", ablation=None)
    assert d3.status == Validation.TESTED.value          # one passing replication is not enough
    sc._judge("memory", w, 3, 23, {"span": 16}, smp, passed=True, title="t3", mechanism="m", ablation=None)
    assert d3.status == Validation.CAUSALLY_SUPPORTED.value
    kinds = [e.type for e in u.events.events]
    assert "DISCOVERY_MEMORY" in kinds


def test_failed_replications_reject_a_claim():
    u = _scientist()
    sc = u.scientist
    w = u.worlds[0]
    sc.control_results = {"learning": {"valid": True}}
    sc._judge("learning", w, 4, 1, {}, {"ids": [1]}, passed=True, title="L", mechanism="m", ablation=None)
    d = sc.discoveries[-1]
    assert d.status == Validation.TESTED.value
    for seed in (2, 3):
        sc._judge("learning", w, 4, seed, {}, {"ids": [1]}, passed=False, title="L", mechanism="m", ablation=None)
    assert d.status == Validation.REJECTED.value
    assert len([r for r in sc.results if r["kind"] == "learning"]) == 2   # negative results are archived


def test_research_queue_runs_synchronously_without_worker():
    u = _scientist()
    sc = u.scientist
    sc.control_results = {"communication": {"valid": True}}
    sc.enqueue("communication", 0, 1.0)
    assert sc.step(120.0) == 1
    assert 0 in u.latest_comm
    st = sc.state_dict()
    from genesis_omega.science.scientist import Scientist
    s2 = Scientist(u)
    s2.load_state_dict(st)
    assert s2.state_dict()["discoveries"] == st["discoveries"]


def test_fork_ablation_runs_in_copies_only():
    u = _scientist()
    w = u.worlds[0]
    before = u.world_digest(w)
    res = u.scientist.fork_ablation(w, "vibration", ticks=20, reps=2, seed=5)
    assert res["replicates"] == 2 and set(res["diffs"]) == {"population", "mean_energy", "eaten", "deaths"}
    assert u.world_digest(w) == before                   # the studied timeline is untouched
