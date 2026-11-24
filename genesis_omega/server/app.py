"""RealtimeServer (spec §287-300, §443-444, §601-608).

HTTP + WebSocket server (aiohttp) running in its own thread with its own
event loop. It reads published frames from the FrameHub and routes every
query that touches live simulation state through ``runner.call`` so it runs
inside the simulation thread between rounds (no torn reads).

Auth: a read-only *viewer* token (shareable with friends) and a separate
*admin* token. Every ``/api`` and ``/ws`` request needs one; ``/api/admin/*``
needs the admin token. There is no endpoint that executes code, touches the
filesystem outside the static directory, or spawns processes.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import threading
import time

import numpy as np
from aiohttp import web, WSMsgType

from ..core.logging import get_logger
from ..science.experiments import ExperimentError, parse as parse_experiment
from ..version import PROTOCOL_VERSION, version_info
from .protocol import encode, json_safe

log = get_logger("server")
RUNNER = web.AppKey("runner", object)
AUTH = web.AppKey("auth", dict)
CLIENTS = web.AppKey("clients", set)
STATIC = os.path.join(os.path.dirname(os.path.dirname(__file__)), "viewer", "static")
THREE_CDN = "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.min.js"


# ------------------------------------------------------------------ auth
def _token(request: web.Request) -> str:
    t = request.query.get("token", "")
    if not t:
        h = request.headers.get("Authorization", "")
        if h.startswith("Bearer "):
            t = h[7:]
    return t


def role_of(request: web.Request) -> str | None:
    auth = request.app[AUTH]
    t = _token(request)
    if not t:
        return None
    if t == auth["admin"]:
        return "admin"
    if t == auth["viewer"]:
        return "viewer"
    return None


@web.middleware
async def auth_middleware(request: web.Request, handler):
    path = request.path
    if path.startswith("/api") or path.startswith("/ws"):
        role = role_of(request)
        if role is None:
            return web.json_response({"error": "unauthorized"}, status=401)
        if path.startswith("/api/admin") and role != "admin":
            return web.json_response({"error": "admin token required"}, status=403)
    resp = await handler(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "no-referrer"
    return resp


# --------------------------------------------------------------- helpers
def _json(obj, status=200):
    return web.json_response(json_safe(obj), status=status, dumps=lambda o: json.dumps(o, default=str))


def _bin(kind, arrays, meta=None):
    return web.Response(body=encode(kind, arrays, meta), content_type="application/octet-stream")


async def _in_sim(request, fn, timeout=60.0):
    runner = request.app[RUNNER]
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: runner.call(fn, timeout=timeout))


def _timeline(runner, request):
    tl = request.query.get("timeline", "PRIME")
    if tl in ("", "PRIME"):
        return runner.u
    f = runner.forks.get(tl)
    if f is None:
        raise web.HTTPNotFound(text=json.dumps({"error": f"unknown timeline {tl}"}), content_type="application/json")
    return f


def _world(u, request):
    wi = int(request.match_info.get("w", 0))
    if wi < 0 or wi >= len(u.worlds):
        raise web.HTTPNotFound(text=json.dumps({"error": "no such world"}), content_type="application/json")
    return u.worlds[wi]


def bearing(lat, lon, head) -> np.ndarray:
    """Heading of tangent vectors as a compass bearing (radians)."""
    east = np.stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)], 1)
    north = np.stack([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)], 1)
    return np.arctan2((head * east).sum(1), (head * north).sum(1))


def encode_agents(fr: dict, roi: dict | None, cap: int) -> tuple[dict, int]:
    lat, lon = fr["lat"], fr["lon"]
    n = lat.shape[0]
    idx = np.arange(n)
    if roi and n:
        la, lo = math.radians(roi.get("lat", 0)), math.radians(roi.get("lon", 0))
        rad = math.radians(max(0.5, min(180, roi.get("radius", 180))))
        c = np.sin(lat) * math.sin(la) + np.cos(lat) * math.cos(la) * np.cos(lon - lo)
        d = np.arccos(np.clip(c, -1, 1))
        idx = idx[d <= rad]
        if idx.size > cap:
            idx = idx[np.argsort(d[idx], kind="stable")[:cap]]
    elif idx.size > cap:
        idx = idx[:: int(math.ceil(idx.size / cap))][:cap]
    b = bearing(lat[idx], lon[idx], fr["head"][idx])
    arrays = {
        "id": fr["id"][idx].astype(np.uint32),
        "lat": np.round(lat[idx] / (math.pi / 2) * 32767).astype(np.int16),
        "lon": np.round(lon[idx] / math.pi * 32767).astype(np.int16),
        "bearing": np.round(b / math.pi * 32767).astype(np.int16),
        "species": fr["species"][idx].astype(np.uint32),
        "bits": fr["bits"][idx].astype(np.uint16),
        "energy": np.round(np.clip(fr["energy"][idx], 0, 1) * 255).astype(np.uint8),
        "mass": np.round(np.clip(fr["mass"][idx] / 20.0, 0, 1) * 255).astype(np.uint8),
        "emit": np.round(np.clip(fr["emit"][idx], 0, 1) * 255).astype(np.uint8),
        "aquatic": np.round(np.clip(fr["aquatic"][idx], 0, 1) * 255).astype(np.uint8),
    }
    return arrays, n


# ------------------------------------------------------------------ app
def make_app(runner, auth: dict) -> web.Application:
    app = web.Application(middlewares=[auth_middleware], client_max_size=2 * 1024 * 1024)
    app[RUNNER] = runner
    app[AUTH] = auth
    app[CLIENTS] = set()
    r = app.router

    async def index(request):
        return web.FileResponse(os.path.join(STATIC, "index.html"))

    async def three(request):
        local = os.path.join(STATIC, "vendor", "three.module.min.js")
        if os.path.exists(local):
            return web.FileResponse(local, headers={"Content-Type": "application/javascript"})
        raise web.HTTPFound(THREE_CDN)

    r.add_get("/", index)
    r.add_get("/static/vendor/three.module.min.js", three)
    r.add_static("/static", STATIC, show_index=False, follow_symlinks=False)

    # ------------------------------------------------------- read-only API
    async def hello(request):
        u = runner.u
        return _json({"role": role_of(request), "protocol": PROTOCOL_VERSION, "versions": version_info(),
                      "cosmos": u.label, "utick": u.utick, "timelines": ["PRIME"] + list(runner.forks.keys()),
                      "worlds": len(u.worlds), "server_time": time.time()})

    async def status(request):
        snap = runner.hub.snapshot()
        u = _timeline(runner, request)
        st = snap["status"] if u is runner.u else await _in_sim(request, lambda rr: u.status())
        return _json({"status": st, "telemetry": snap["telemetry"], "published": snap["published"]})

    async def telemetry(request):
        return _json(runner.hub.snapshot()["telemetry"])

    async def galaxy(request):
        u = runner.u
        g = u.galaxy_payload()      # static arrays + slowly changing life flags: no sim-thread round trip
        return _bin("galaxy", {"pos": g["pos"], "teff": g["teff"], "lum": g["lum"], "stage": g["stage"],
                               "life": g["life"]}, {"n": g["n"], "focus_stars": g["focus_stars"],
                                                    "smbh_mass": u.galaxy.smbh_mass})

    async def system(request):
        u = _timeline(runner, request)
        s = int(request.match_info["star"])
        if s < 0 or s >= u.galaxy.n_stars:
            return _json({"error": "no such star"}, 404)
        return _json(await _in_sim(request, lambda rr: u.system_payload(s)))

    async def terrain(request):
        u = _timeline(runner, request)
        w = _world(u, request)
        t = runner.hub.terrain.get(w.index) if u is runner.u else None
        if t is None:
            t = await _in_sim(request, lambda rr: w.terrain_export())
        elev_m = np.clip(t["elev"].astype(np.float32) * 1000.0, -32000, 32000).astype(np.int16)
        meta = {"w": t["w"], "h": t["h"], "planet": w.planet, "xenos": w.xenos.to_dict(),
                "star": w.star, "world": w.index}
        return _bin("terrain", {"elev": elev_m, "ocean": t["ocean"]}, meta)

    async def fields(request):
        u = _timeline(runner, request)
        w = _world(u, request)
        f = runner.hub.snapshot()["fields"].get(w.index) if u is runner.u else None
        if f is None:
            f = await _in_sim(request, lambda rr: w.fields_export())
        arrays = {k: v for k, v in f.items() if isinstance(v, np.ndarray)}
        return _bin("fields", arrays, {"w": f["w"], "h": f["h"], "world": w.index})

    async def structures(request):
        u = _timeline(runner, request)
        w = _world(u, request)
        s = await _in_sim(request, lambda rr: w.structure_cells())
        return _bin("structures", {"cell": s["cell"].astype(np.uint32), "h": s["h"], "mat": s["mat"],
                                   "age": s["age"]}, {"world": w.index, "w": w.grid.W, "h": w.grid.H})

    async def agent(request):
        u = _timeline(runner, request)
        w = _world(u, request)
        aid = int(request.match_info["id"])
        d = await _in_sim(request, lambda rr: w.agent_detail(aid))
        return _json(d if d else {"error": "unknown agent"}, 200 if d else 404)

    async def body(request):
        u = _timeline(runner, request)
        w = _world(u, request)
        aid = int(request.match_info["id"])
        d = await _in_sim(request, lambda rr: w.agent_body(aid))
        return _json(d if d else {"error": "not alive"}, 200 if d else 404)

    async def species(request):
        u = _timeline(runner, request)
        w = _world(u, request)
        return _json(await _in_sim(request, lambda rr: w.species.table()))

    async def technology(request):
        from ..society.technology import technology_graph, cultural_lineage
        u = _timeline(runner, request)
        w = _world(u, request)
        return _json(await _in_sim(request, lambda rr: {"graph": technology_graph(w), "culture": cultural_lineage(w)}))

    async def levels(request):
        u = _timeline(runner, request)
        w = _world(u, request)
        return _json(await _in_sim(request, lambda rr: u.levels.get(w.index, {})))

    async def construction(request):
        u = _timeline(runner, request)
        w = _world(u, request)
        return _json(await _in_sim(request, lambda rr: u.construction.get(w.index, {})))

    async def qd(request):
        u = _timeline(runner, request)
        w = _world(u, request)
        return _json(await _in_sim(request, lambda rr: {"elites": w.qd.elites(), "coverage": w.qd.coverage(),
                                                        "qd_score": w.qd.qd_score(),
                                                        "novelty_archive": w.qd.nov_count}))

    async def social(request):
        u = _timeline(runner, request)
        w = _world(u, request)
        return _json(await _in_sim(request, lambda rr: u.social_stats.get(w.index, {})))

    async def comm(request):
        u = _timeline(runner, request)
        w = _world(u, request)
        return _json(await _in_sim(request, lambda rr: u.latest_comm.get(w.index, {"status": "not yet analysed"})))

    async def events(request):
        u = _timeline(runner, request)
        since = int(request.query.get("since", 0))
        sig = int(request.query.get("min_sig", 0))
        lim = min(int(request.query.get("limit", 300)), 2000)
        return _json(u.events.since(since, sig, lim))

    async def news(request):
        u = _timeline(runner, request)
        since = int(request.query.get("since", 0))
        return _json(await _in_sim(request, lambda rr: u.news(since)))

    async def discoveries(request):
        u = _timeline(runner, request)
        return _json(await _in_sim(request, lambda rr: {"discoveries": [d.to_dict() for d in u.scientist.discoveries],
                                                        "controls": u.scientist.control_results,
                                                        "negative": u.scientist.results[-60:],
                                                        "failures": u.scientist.failures[-20:],
                                                        "queue": len(u.scientist.queue)}))

    async def frames(request):
        u = _timeline(runner, request)
        eid = int(request.match_info["event"])
        fr = await _in_sim(request, lambda rr: u.frozen.get(eid))
        if not fr:
            return _json({"error": "no frozen window for this event"}, 404)
        seq = [f for f in fr["pre"] + fr["post"] if f.get("n", 0) > 0]
        arrays, counts, ticks = {}, [], []
        for k, f in enumerate(seq):
            a, _ = encode_agents({"lat": f["lat"].astype(np.float64), "lon": f["lon"].astype(np.float64),
                                  "head": np.zeros((f["n"], 3)), "id": f["id"], "species": f["species"],
                                  "bits": f["bits"], "energy": np.clip(f["energy"].astype(np.float32) / 30, 0, 1),
                                  "mass": np.ones(f["n"]), "emit": f["emit"].astype(np.float32).sum(1) / 255,
                                  "aquatic": np.zeros(f["n"])}, None, 100000)
            for name in ("id", "lat", "lon", "species", "bits", "emit"):
                arrays[f"{name}_{k}"] = a[name]
            counts.append(f["n"])
            ticks.append(f["tick"])
        return _bin("frames", arrays, {"event": eid, "world": fr["world"], "counts": counts, "ticks": ticks})

    async def pantheon(request):
        u = _timeline(runner, request)
        return _json(u.pantheon.table())

    async def features(request):
        return _json(runner.u.features())

    async def forks(request):
        return _json({"forks": runner.u.forks, "active": list(runner.forks.keys())})

    async def search(request):
        q = request.query.get("q", "").strip().upper()
        u = _timeline(runner, request)

        def find(rr):
            out = []
            digits = "".join(ch for ch in q if ch.isdigit())
            if q.startswith("A-") or q.isdigit():
                for w in u.worlds:
                    if digits and w.lineage.record(int(digits)):
                        out.append({"kind": "agent", "world": w.index, "id": int(digits)})
            if q.startswith("E-") and digits:
                e = u.events.get(int(digits))
                if e:
                    out.append({"kind": "event", "event": e.to_dict()})
            if q.startswith("D-") and digits:
                for d in u.scientist.discoveries:
                    if d.id == int(digits):
                        out.append({"kind": "discovery", "discovery": d.to_dict()})
            if q.startswith("SP-"):
                parts = q.split("-")
                if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                    wi, sid = int(parts[1]), int(parts[2])
                    if wi < len(u.worlds) and sid in u.worlds[wi].species.species:
                        rec = {k: v for k, v in u.worlds[wi].species.species[sid].items() if k != "centroid"}
                        out.append({"kind": "species", "world": wi, "species": rec})
            return out

        return _json(await _in_sim(request, find))

    # ------------------------------------------------------------ admin API
    async def admin_fork(request):
        body_ = await request.json()
        steps = body_.get("steps") or []
        try:
            if steps:
                parse_experiment({"world": int(body_.get("world", 0)), "steps": steps}, len(runner.u.worlds))
            fid = await asyncio.get_running_loop().run_in_executor(
                None, lambda: runner.create_fork(str(body_.get("label", ""))[:80], str(body_.get("reason", ""))[:200],
                                                 [dict(s, world=int(body_.get("world", 0))) for s in steps],
                                                 int(body_.get("salt", 0))))
        except ExperimentError as e:
            return _json({"error": str(e)}, 400)
        return _json({"fork": fid})

    async def admin_intervene(request):
        body_ = await request.json()
        tl = body_.get("timeline", "PRIME")
        doc = body_.get("experiment", {})

        def go(rr):
            u = rr.u if tl == "PRIME" else rr.forks[tl]
            return [e.to_dict() for e in u.intervene(doc, source_worlds=rr.u.worlds)]
        try:
            return _json({"events": await _in_sim(request, go)})
        except PermissionError as e:
            return _json({"error": str(e)}, 403)
        except (ExperimentError, KeyError) as e:
            return _json({"error": str(e)}, 400)

    async def admin_lock(request):
        body_ = await request.json()
        locked = bool(body_.get("locked", True))

        def go(rr):
            rr.u.intervention_lock = locked
            rr.u.emit("EXPERIMENT_LOCK", f"Prime-timeline experiment lock {'ON' if locked else 'OFF'}",
                      epistemic="EXTERNAL", significance=1)
            return locked
        return _json({"locked": await _in_sim(request, go)})

    async def admin_speed(request):
        body_ = await request.json()
        runner.target_tps = max(0.0, float(body_.get("tps", 0)))
        return _json({"target_tps": runner.target_tps})

    async def admin_checkpoint(request):
        await asyncio.get_running_loop().run_in_executor(None, runner.checkpoint_now)
        return _json({"ok": True, "utick": runner.u.utick})

    async def admin_promote(request):
        body_ = await request.json()
        row = int(body_.get("planet_row", -1))
        try:
            w = await _in_sim(request, lambda rr: rr.u.promote(row).summary(), timeout=300)
        except ValueError as e:
            return _json({"error": str(e)}, 400)
        return _json({"world": w})

    async def admin_replay(request):
        body_ = await request.json()

        def go(rr):
            u = rr.u
            if not u.keyframes:
                return {"error": "no keyframe yet"}
            kfs = list(u.keyframes)
            kf = kfs[0] if len(kfs) > 1 else kfs[-1]
            target = int(body_.get("target", kfs[-1]["utick"] if len(kfs) > 1 else kf["utick"]))
            expected = next((k["digests"] for k in kfs if k["utick"] == target), None)
            if expected is None:
                return {"error": "target must be a later keyframe"}
            return u.verify_replay(kf["utick"], target, expected)
        return _json(await _in_sim(request, go, timeout=900))

    async def admin_bridge(request):
        body_ = await request.json()
        tl = body_.get("timeline")

        def go(rr):
            if tl not in rr.forks:
                raise KeyError("reality bridge only in a fork")
            f = rr.forks[tl]
            f.intervene({"world": int(body_.get("world", 0)), "steps": [{
                "op": "inject", "field": body_.get("field", "light"), "lat": float(body_.get("lat", 0)),
                "lon": float(body_.get("lon", 0)), "gain": float(body_.get("gain", 1.0))}]})
            src = f.worlds[int(body_.get("world", 0))].external_sources[-1]
            src["value"] = 0.0
            src["bridge"] = True
            return len(f.worlds[int(body_.get("world", 0))].external_sources) - 1
        try:
            sid = await _in_sim(request, go)
        except (KeyError, ExperimentError) as e:
            return _json({"error": str(e)}, 400)
        return _json({"source": sid})

    for path, h in (("/api/hello", hello), ("/api/status", status), ("/api/telemetry", telemetry),
                    ("/api/galaxy", galaxy), ("/api/system/{star}", system),
                    ("/api/world/{w}/terrain", terrain), ("/api/world/{w}/fields", fields),
                    ("/api/world/{w}/structures", structures), ("/api/world/{w}/agent/{id}", agent),
                    ("/api/world/{w}/agent/{id}/body", body), ("/api/world/{w}/species", species),
                    ("/api/world/{w}/technology", technology), ("/api/world/{w}/levels", levels),
                    ("/api/world/{w}/construction", construction), ("/api/world/{w}/qd", qd),
                    ("/api/world/{w}/social", social), ("/api/world/{w}/communication", comm),
                    ("/api/events", events), ("/api/news", news), ("/api/discoveries", discoveries),
                    ("/api/frames/{event}", frames), ("/api/pantheon", pantheon), ("/api/features", features),
                    ("/api/forks", forks), ("/api/search", search)):
        r.add_get(path, h)
    for path, h in (("/api/admin/fork", admin_fork), ("/api/admin/intervene", admin_intervene),
                    ("/api/admin/lock", admin_lock), ("/api/admin/speed", admin_speed),
                    ("/api/admin/checkpoint", admin_checkpoint), ("/api/admin/promote", admin_promote),
                    ("/api/admin/replay_verify", admin_replay), ("/api/admin/bridge", admin_bridge)):
        r.add_post(path, h)

    # ------------------------------------------------------------ websocket
    async def ws_handler(request):
        ws = web.WebSocketResponse(heartbeat=20, max_msg_size=1 << 20)
        await ws.prepare(request)
        app[CLIENTS].add(ws)
        role = role_of(request)
        sub = {"world": 0, "timeline": "PRIME", "roi": None, "max_agents": 3000, "rate": 10.0}
        sent_seq, field_seq, struct_seq, ev_id, last_status = -1, -1, -1, 0, 0.0
        await ws.send_str(json.dumps({"type": "hello", "role": role, "protocol": PROTOCOL_VERSION,
                                      "cosmos": runner.u.label}))

        async def reader():
            nonlocal sub
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    m = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                if m.get("type") == "subscribe":
                    sub = {"world": int(m.get("world", 0)), "timeline": str(m.get("timeline", "PRIME")),
                           "roi": m.get("roi"), "max_agents": int(min(max(m.get("max_agents", 3000), 50), 20000)),
                           "rate": float(min(max(m.get("rate", 10), 0.5), 30))}
                elif m.get("type") == "bridge" and role == "admin":
                    f = runner.forks.get(m.get("timeline"))
                    if f is not None:
                        try:
                            src = f.worlds[int(m.get("world", 0))].external_sources[int(m.get("source", -1))]
                            if src.get("bridge"):
                                src["value"] = float(max(0.0, min(5.0, m.get("value", 0.0))))
                        except (IndexError, ValueError, TypeError):
                            pass

        rtask = asyncio.ensure_future(reader())
        loop = asyncio.get_running_loop()
        ev_tl, last_fork_frame, last_fork_fields = "PRIME", 0.0, 0.0

        def in_sim(fn):
            return loop.run_in_executor(None, lambda: runner.call(fn, timeout=30))

        try:
            while not ws.closed:
                t0 = time.time()
                snap = runner.hub.snapshot()
                wi = sub["world"]
                tl = sub["timeline"]
                fork = runner.forks.get(tl) if tl != "PRIME" else None
                if fork is not None and 0 <= wi < len(fork.worlds) and t0 - last_fork_frame >= 0.5:
                    # forks are not published to the frame hub: sample them from the sim thread at ≤ 2 Hz
                    fw = fork.worlds[wi]
                    fr = await in_sim(lambda rr: dict(fw.agents_frame(), tick=fw.tick))
                    arrays, total = encode_agents(fr, sub["roi"], sub["max_agents"])
                    await ws.send_bytes(encode("agents", arrays, {"world": wi, "tick": fr["tick"], "total": total,
                                                                   "timeline": tl, "t": time.time()}))
                    last_fork_frame = time.time()
                    if last_fork_frame - last_fork_fields >= 2.0:
                        f = await in_sim(lambda rr: fw.fields_export())
                        await ws.send_bytes(encode("fields", {k: v for k, v in f.items() if isinstance(v, np.ndarray)},
                                                   {"w": f["w"], "h": f["h"], "world": wi, "timeline": tl}))
                        s = await in_sim(lambda rr: fw.structure_cells())
                        await ws.send_bytes(encode("structures", {"cell": s["cell"].astype(np.uint32), "h": s["h"],
                                                                  "mat": s["mat"], "age": s["age"]},
                                                   {"world": wi, "timeline": tl, "w": fw.grid.W, "h": fw.grid.H}))
                        last_fork_fields = time.time()
                if tl == "PRIME":
                    if snap["seq"] != sent_seq and wi in snap["agents"]:
                        tr = request.transport
                        buffered = tr.get_write_buffer_size() if tr is not None else 0
                        cap = sub["max_agents"] if buffered < 2_000_000 else max(200, sub["max_agents"] // 4)
                        arrays, total = encode_agents(snap["agents"][wi], sub["roi"], cap)
                        await ws.send_bytes(encode("agents", arrays, {"world": wi, "tick": snap["agents"][wi]["tick"],
                                                                       "total": total, "seq": snap["seq"],
                                                                       "t": snap["published"]}))
                        sent_seq = snap["seq"]
                    if snap["fields_seq"].get(wi, -1) != field_seq and wi in snap["fields"]:
                        f = snap["fields"][wi]
                        await ws.send_bytes(encode("fields", {k: v for k, v in f.items() if isinstance(v, np.ndarray)},
                                                   {"w": f["w"], "h": f["h"], "world": wi}))
                        field_seq = snap["fields_seq"][wi]
                    deltas = [d for s, d in snap["struct_deltas"].get(wi, []) if s > struct_seq]
                    if deltas:
                        cells = np.concatenate([d["cell"] for d in deltas]).astype(np.uint32)
                        hs = np.concatenate([d["h"] for d in deltas]).astype(np.float32)
                        mats = np.concatenate([d["mat"] for d in deltas]).astype(np.int32)
                        if cells.size:
                            await ws.send_bytes(encode("structdelta", {"cell": cells, "h": hs, "mat": mats},
                                                       {"world": wi}))
                        struct_seq = snap["struct_deltas"][wi][-1][0]
                if time.time() - last_status >= 1.0:
                    if tl != ev_tl:                      # another timeline: its own event numbering
                        ev_id, ev_tl = 0, tl
                    if tl == "PRIME":
                        u, st = runner.u, snap["status"]
                    else:
                        u = fork
                        st = await in_sim(lambda rr: fork.status()) if fork is not None else None
                    new = u.events.since(ev_id, 0, 200) if u is not None else []
                    if new:
                        ev_id = new[-1]["id"]
                    await ws.send_str(json.dumps(json_safe({"type": "status", "timeline": tl, "missing": u is None,
                                                            "status": st if st is not None else snap["status"],
                                                            "telemetry": snap["telemetry"], "events": new}),
                                                 default=str))
                    last_status = time.time()
                await asyncio.sleep(max(0.01, 1.0 / sub["rate"] - (time.time() - t0)))
        except (ConnectionResetError, RuntimeError):
            pass
        except Exception as e:  # never kill the server for one client; the viewer reconnects
            log.warning(f"websocket stream ended: {e!r}")
        finally:
            rtask.cancel()
            app[CLIENTS].discard(ws)
        return ws

    r.add_get("/ws", ws_handler)
    return app


class ServerThread:
    """Runs the aiohttp app in a dedicated thread."""

    def __init__(self, runner, auth: dict, host: str, port: int):
        self.runner, self.auth, self.host, self.port = runner, auth, host, port
        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: threading.Thread | None = None
        self._runner = None
        self.ready = threading.Event()
        self.error: BaseException | None = None

    def start(self) -> None:
        def run():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            app = make_app(self.runner, self.auth)
            self._runner = web.AppRunner(app, access_log=None)
            try:
                self.loop.run_until_complete(self._runner.setup())
                site = web.TCPSite(self._runner, self.host, self.port)
                self.loop.run_until_complete(site.start())
            except BaseException as e:
                self.error = e
                self.ready.set()
                return
            self.ready.set()
            self.loop.run_forever()

        self.thread = threading.Thread(target=run, name="genesis-server", daemon=True)
        self.thread.start()
        self.ready.wait(30)
        if self.error:
            raise self.error

    def stop(self) -> None:
        if self.loop and self._runner:
            fut = asyncio.run_coroutine_threadsafe(self._runner.cleanup(), self.loop)
            try:
                fut.result(10)
            except Exception:
                pass
            self.loop.call_soon_threadsafe(self.loop.stop)
