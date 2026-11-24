"""Ω0 security: organisms only touch the world interface; the Scientist DSL is data, never code."""

from __future__ import annotations

import ast
import inspect
import os

import pytest

from genesis_omega.life import interface
from genesis_omega.science.experiments import (MAX_REPLICATES, MAX_TICKS, ExperimentError, METRICS, MECHANISMS,
                                               PARAMS, parse)

PKG = os.path.dirname(os.path.dirname(os.path.abspath(interface.__file__)))


# ------------------------------------------------------------------ DSL
def test_valid_experiment_is_accepted():
    exp = parse({"world": 0, "ticks": 400, "replicates": 3, "label": "x" * 200,
                 "steps": [{"op": "ablate", "mechanism": "vibration"},
                           {"op": "perturb", "param": "solar_flux", "value": 1.05},
                           {"op": "observe", "metrics": ["population", "mean_energy"]},
                           {"op": "inject", "field": "vibration", "lat": 10, "lon": -20, "band": 2},
                           {"op": "cull", "fraction": 0.3},
                           {"op": "ablate", "mechanism": "material", "material": 9},
                           {"op": "fork"}]}, n_worlds=2)
    assert exp.ticks == 400 and len(exp.steps) == 7 and len(exp.label) == 80
    assert exp.steps[3]["gain"] == 1.0 and exp.steps[3]["period"] == 0


@pytest.mark.parametrize("doc", [
    "import os",
    {"world": 0, "code": "print(1)"},                                         # unknown field
    {"steps": [{"op": "exec", "source": "os.system('rm -rf /')"}]},          # unknown op
    {"steps": [{"op": "ablate", "mechanism": "__import__('os')"}]},          # unknown mechanism
    {"steps": [{"op": "perturb", "param": "gravity", "value": 2}]},          # unknown parameter
    {"steps": [{"op": "perturb", "param": "solar_flux", "value": 9.0}]},     # out of bounds
    {"steps": [{"op": "perturb", "param": "solar_flux", "value": "1.0"}]},   # strings are not numbers
    {"steps": [{"op": "perturb", "param": "solar_flux", "value": True}]},    # bools are not numbers
    {"steps": [{"op": "observe", "metrics": ["population", "secrets"]}]},
    {"steps": [{"op": "inject", "field": "gamma_rays"}]},
    {"steps": [{"op": "ablate", "mechanism": "material", "material": 2}]},   # base materials can't be blocked
    {"world": 5},                                                             # no such world
    {"ticks": MAX_TICKS + 1},
    {"replicates": MAX_REPLICATES + 1},
    {"steps": [{"op": "fork"}] * 17},                                         # too many steps
    {"steps": "fork"},
])
def test_invalid_experiments_are_rejected(doc):
    with pytest.raises(ExperimentError):
        parse(doc, n_worlds=1)


def test_vocabularies_are_closed():
    assert set(MECHANISMS) == {"vibration", "plasticity", "chemistry", "marks", "scent", "material"}
    assert all(lo < hi for lo, hi in PARAMS.values())
    assert len(METRICS) == len(set(METRICS))


def test_dsl_module_never_evaluates_anything():
    src = inspect.getsource(__import__("genesis_omega.science.experiments", fromlist=["x"]))
    tree = ast.parse(src)
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not called & {"eval", "exec", "compile", "__import__", "open", "getattr"}


# ------------------------------------------------------------------ the world interface
def test_world_interface_carries_no_semantic_labels():
    """Only raw physical signals cross into organisms — never categories like 'food' or 'enemy'."""
    forbidden = {"food", "enemy", "predator", "prey", "friend", "human", "house", "tool", "danger", "mate",
                 "reward", "good", "bad", "pain", "pleasure", "fear", "happy"}
    names = " ".join(interface.INPUT_NAMES + interface.OUTPUT_NAMES).lower()
    for word in forbidden:
        assert word not in names.replace(".", " ").split(), word
    assert interface.N_INPUTS == len(interface.INPUT_NAMES)


def test_simulation_core_has_no_io_or_process_access():
    """Laws, organisms and their substrates cannot reach files, network, processes or code."""
    banned_mods = {"os", "subprocess", "socket", "urllib", "requests", "shutil", "importlib", "ctypes", "pickle"}
    banned_calls = {"eval", "exec", "compile", "__import__", "open"}
    for sub in ("life", "world", "society"):
        for fn in os.listdir(os.path.join(PKG, sub)):
            if not fn.endswith(".py"):
                continue
            tree = ast.parse(open(os.path.join(PKG, sub, fn), encoding="utf-8").read())
            for n in ast.walk(tree):
                if isinstance(n, ast.Import):
                    assert not {a.name.split(".")[0] for a in n.names} & banned_mods, (sub, fn)
                elif isinstance(n, ast.ImportFrom) and n.level == 0:
                    assert (n.module or "").split(".")[0] not in banned_mods, (sub, fn)
                elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                    assert n.func.id not in banned_calls, (sub, fn, n.func.id)


def test_prime_timeline_refuses_interventions_while_locked(universe):
    with pytest.raises(PermissionError):
        universe.intervene({"world": 0, "steps": [{"op": "cull", "fraction": 0.9}]})


def test_server_has_no_code_execution_routes():
    from genesis_omega.server import app as app_mod

    src = inspect.getsource(app_mod)
    tree = ast.parse(src)
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not called & {"eval", "exec", "compile", "__import__"}
    assert "subprocess" not in src


def test_static_route_does_not_escape_its_directory(tmp_path):
    import socket
    import time
    import urllib.error
    import urllib.request

    from conftest import small_cfg
    from genesis_omega.runtime.runner import Runner
    from genesis_omega.runtime.universe import Universe
    from genesis_omega.server.app import ServerThread

    u = Universe.create(small_cfg(seed=91), None, run_controls=False)
    r = Runner(u)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = ServerThread(r, {"viewer": "v", "admin": "a"}, "127.0.0.1", port)
    srv.start()
    try:
        for path in ("/static/../server/app.py", "/static/%2e%2e/server/app.py", "/static/../../version.py",
                     "/static/..%2fserver%2fapp.py"):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as resp:
                    body = resp.read()
                    assert b"def make_app" not in body and b"ENGINE_VERSION" not in body, path
            except urllib.error.HTTPError as e:
                assert e.code in (400, 403, 404), path
    finally:
        srv.stop()
        time.sleep(0.1)
