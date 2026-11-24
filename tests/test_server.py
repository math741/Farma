"""Ω0 observatory backend: binary protocol, token auth, read-only API, admin API, WebSocket stream."""

from __future__ import annotations

import asyncio
import json
import socket
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from conftest import small_cfg
from genesis_omega.core.status import FEATURES
from genesis_omega.server.protocol import MAGIC, decode, encode, json_safe

VIEWER, ADMIN = "viewer-token-x", "admin-token-y"


# ------------------------------------------------------------------ protocol
def test_protocol_roundtrip_all_dtypes():
    arrays = {"f": np.linspace(0, 1, 5, dtype=np.float32), "i": np.arange(-3, 4, dtype=np.int32),
              "u32": np.arange(7, dtype=np.uint32), "i16": np.array([-32000, 0, 32000], np.int16),
              "u16": np.arange(3, dtype=np.uint16), "u8": np.arange(5, dtype=np.uint8),
              "i8": np.array([-5, 5], np.int8), "grid": np.ones((3, 4), np.uint8),
              "f64": np.array([1.5, 2.5]), "i64": np.array([1, 2], np.int64), "b": np.array([True, False]),
              "f16": np.array([0.5], np.float16)}
    buf = encode("test", arrays, {"hello": "Ω", "n": np.int64(3)})
    assert buf[:4] == MAGIC
    kind, meta, out = decode(buf)
    assert kind == "test" and meta == {"hello": "Ω", "n": 3}
    for k, v in arrays.items():
        assert np.array_equal(out[k], v.astype(out[k].dtype)), k
    hl = int.from_bytes(buf[4:8], "little")
    assert (8 + hl) % 4 == 0
    header = json.loads(buf[8:8 + hl])
    assert all(s["offset"] % 4 == 0 for s in header["arrays"])        # typed-array views need alignment
    assert out["f64"].dtype == np.float32 and out["i64"].dtype == np.int32 and out["b"].dtype == np.uint8


def test_json_safe_handles_numpy_and_nan():
    import torch

    out = json_safe({"a": np.float32(1.5), "b": float("nan"), "c": np.arange(3), 4: torch.tensor([1, 2])})
    assert out == {"a": 1.5, "b": None, "c": [0, 1, 2], "4": [1, 2]}


# ------------------------------------------------------------------ live server
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    from genesis_omega.runtime.runner import Runner
    from genesis_omega.runtime.universe import Universe
    from genesis_omega.server.app import ServerThread
    from genesis_omega.storage.checkpoint import StorageLayout

    tmp = tmp_path_factory.mktemp("srv")
    lay = StorageLayout(str(tmp / "data"), "srv", str(tmp / "scratch"))
    lay.ensure()
    u = Universe.create(small_cfg(seed=81, keyframe_every=50), lay, run_controls=False)
    r = Runner(u)
    r.start()
    port = _free_port()
    s = ServerThread(r, {"viewer": VIEWER, "admin": ADMIN}, "127.0.0.1", port)
    s.start()
    t0 = time.time()
    while r.hub.snapshot()["seq"] == 0 and time.time() - t0 < 30:
        time.sleep(0.05)
    yield {"base": f"http://127.0.0.1:{port}", "port": port, "runner": r, "universe": u}
    s.stop()
    r.stop(checkpoint=False)


def _req(live, path, token=VIEWER, data=None, raw=False):
    sep = "&" if "?" in path else "?"
    url = live["base"] + path + (f"{sep}token={token}" if token else "")
    req = urllib.request.Request(url, data=None if data is None else json.dumps(data).encode(),
                                 method="POST" if data is not None else "GET",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read()
            return resp.status, (body if raw else json.loads(body)), dict(resp.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def test_static_viewer_is_served_without_token(live):
    with urllib.request.urlopen(live["base"] + "/", timeout=30) as resp:
        html = resp.read().decode()
    assert "Genesis Ω Observatory" in html and "/static/js/main.js" in html
    for path in ("/static/js/main.js", "/static/css/app.css", "/static/vendor/three.module.min.js"):
        with urllib.request.urlopen(live["base"] + path, timeout=30) as resp:
            assert resp.status == 200 and len(resp.read()) > 100, path


def test_api_requires_tokens(live):
    assert _req(live, "/api/hello", token=None)[0] == 401
    assert _req(live, "/api/hello", token="wrong")[0] == 401
    st, body, headers = _req(live, "/api/hello")
    assert st == 200 and body["role"] == "viewer" and body["cosmos"] == live["universe"].label
    assert headers.get("X-Content-Type-Options") == "nosniff"
    assert VIEWER not in json.dumps(body) and ADMIN not in json.dumps(body)       # tokens are never echoed
    assert _req(live, "/api/admin/speed", data={"tps": 1})[0] == 403             # viewer cannot administrate
    assert _req(live, "/api/hello", token=ADMIN)[1]["role"] == "admin"


def test_binary_endpoints(live):
    u = live["universe"]
    st, b, _ = _req(live, "/api/galaxy", raw=True)
    kind, meta, arr = decode(b)
    assert kind == "galaxy" and arr["pos"].shape == (u.galaxy.n_stars, 3) and meta["focus_stars"]
    st, b, _ = _req(live, "/api/world/0/terrain", raw=True)
    kind, meta, arr = decode(b)
    w = u.worlds[0]
    assert arr["elev"].shape == (w.grid.H, w.grid.W) and meta["xenos"]["planet_id"] == w.planet["id"]
    st, b, _ = _req(live, "/api/world/0/fields", raw=True)
    kind, meta, arr = decode(b)
    assert {"temp", "resource", "cloud", "structure", "occupancy"} <= set(arr)
    st, b, _ = _req(live, "/api/world/0/structures", raw=True)
    assert decode(b)[0] == "structures"
    assert _req(live, "/api/world/9/terrain")[0] == 404


@pytest.mark.parametrize("path", ["/api/status", "/api/telemetry", "/api/world/0/species", "/api/world/0/technology",
                                  "/api/world/0/levels", "/api/world/0/construction", "/api/world/0/qd",
                                  "/api/world/0/social", "/api/world/0/communication", "/api/events?since=0",
                                  "/api/news?since=0", "/api/discoveries", "/api/pantheon", "/api/forks",
                                  "/api/search?q=E-1"])
def test_json_endpoints(live, path):
    st, body, _ = _req(live, path)
    assert st == 200, (path, body)


def test_features_endpoint_is_the_honest_registry(live):
    st, body, _ = _req(live, "/api/features")
    assert st == 200 and [f["key"] for f in body] == [f.key for f in FEATURES]
    assert any(f["status"] == "NOT IMPLEMENTED" for f in body)


def test_system_and_agent_inspection(live):
    u = live["universe"]
    star = int(u.worlds[0].planet["star"])
    st, body, _ = _req(live, f"/api/system/{star}")
    assert st == 200 and any(p["f4_world"] == 0 for p in body["planets"])
    aid = live["runner"].call(lambda rr: int(rr.u.worlds[0].agents.id[rr.u.worlds[0].agents.active][0]))
    st, body, _ = _req(live, f"/api/world/0/agent/{aid}")
    assert st == 200 and body["alive"] and len(body["substrate"]["inputs"]) > 0
    st, body, _ = _req(live, f"/api/world/0/agent/{aid}/body")
    assert st == 200 and body["voxels"]
    assert _req(live, "/api/world/0/agent/987654321")[0] == 404
    assert _req(live, f"/api/search?q=A-{aid}")[1][0]["kind"] == "agent"


def test_admin_flow_forks_and_protection(live):
    st, body, _ = _req(live, "/api/admin/intervene", token=ADMIN,
                       data={"timeline": "PRIME", "experiment": {"world": 0, "steps": [{"op": "cull"}]}})
    assert st == 403                                            # prime timeline protected by the lock
    st, body, _ = _req(live, "/api/admin/fork", token=ADMIN,
                       data={"label": "x", "steps": [{"op": "exec", "code": "__import__('os')"}]})
    assert st == 400
    st, body, _ = _req(live, "/api/admin/fork", token=ADMIN,
                       data={"label": "brighter", "reason": "test",
                             "steps": [{"op": "perturb", "param": "solar_flux", "value": 1.1}]})
    assert st == 200
    fid = body["fork"]
    st, body, _ = _req(live, f"/api/status?timeline={fid}")
    assert st == 200 and body["status"]["timeline"] == fid
    st, body, _ = _req(live, "/api/admin/intervene", token=ADMIN,
                       data={"timeline": fid, "experiment": {"world": 0, "steps": [
                           {"op": "observe", "metrics": ["population", "births"]}]}})
    assert st == 200 and body["events"][0]["epistemic"] == "EXTERNAL"
    st, body, _ = _req(live, f"/api/status?timeline={fid}")
    assert set(body["status"]["worlds"][0]["observed"]) == {"population", "births"}
    st, body, _ = _req(live, "/api/admin/bridge", token=ADMIN, data={"timeline": fid, "field": "light"})
    assert st == 200 and body["source"] == 0
    assert _req(live, "/api/admin/bridge", token=ADMIN, data={"timeline": "PRIME"})[0] == 400
    st, body, _ = _req(live, "/api/forks")
    assert fid in body["active"]
    assert _req(live, "/api/status?timeline=F-9999")[0] == 404
    st, body, _ = _req(live, "/api/admin/checkpoint", token=ADMIN, data={})
    assert st == 200 and body["ok"]


def test_websocket_streams_agents_fields_and_status(live):
    import aiohttp

    async def run():
        kinds, sizes = {}, []
        async with aiohttp.ClientSession() as sess:
            async with sess.ws_connect(f"ws://127.0.0.1:{live['port']}/ws?token={VIEWER}") as ws:
                hello = await ws.receive_json(timeout=10)
                assert hello["type"] == "hello" and hello["role"] == "viewer"
                await ws.send_json({"type": "subscribe", "world": 0, "max_agents": 100, "rate": 20,
                                    "roi": {"lat": 0, "lon": 0, "radius": 180}})
                t0 = time.time()
                while time.time() - t0 < 6 and not ({"agents", "fields", "status"} <= set(kinds)
                                                   and len(sizes) >= 3):
                    msg = await ws.receive(timeout=10)
                    if msg.type == aiohttp.WSMsgType.BINARY:
                        kind, meta, arr = decode(msg.data)
                        kinds[kind] = kinds.get(kind, 0) + 1
                        if kind == "agents":
                            sizes.append(arr["id"].size)
                            assert meta["world"] == 0 and meta["total"] >= arr["id"].size
                            assert arr["lat"].dtype == np.int16 and arr["species"].dtype == np.uint32
                    elif msg.type == aiohttp.WSMsgType.TEXT:
                        d = json.loads(msg.data)
                        kinds[d["type"]] = kinds.get(d["type"], 0) + 1
                        if d["type"] == "status":
                            assert d["timeline"] == "PRIME" and "telemetry" in d
        return kinds, sizes

    kinds, sizes = asyncio.run(run())
    assert {"agents", "fields", "status"} <= set(kinds)
    assert sizes[-1] <= 100            # the subscription's cap applies once the server has read it


def test_websocket_rejects_missing_token(live):
    import aiohttp

    async def run():
        async with aiohttp.ClientSession() as sess:
            with pytest.raises(aiohttp.WSServerHandshakeError):
                async with sess.ws_connect(f"ws://127.0.0.1:{live['port']}/ws"):
                    pass

    asyncio.run(run())
