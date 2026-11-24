"""One-button launcher (spec §408-421, §553-556, §591).

``launch()`` does everything the notebook's "Run All" needs: detect hardware,
pick a profile, open (or create) the persistent cosmos on Drive, auto-tune a
*new* cosmos by measurement, start the simulation thread, the server and the
tunnel, and return the links. If a cosmos already exists it is resumed —
never reset unless ``reset=True`` is passed explicitly (and even then it is
archived, not deleted).
"""

from __future__ import annotations

import json
import os
import secrets
import time
from dataclasses import dataclass, field

from ..core.config import profile_config, auto_profile, UniverseConfig
from ..core.hardware import detect, configure_determinism, in_colab
from ..core.ids import cosmos_label
from ..core.logging import get_logger, attach_file_log
from ..core.rng import derive_seed
from ..storage.checkpoint import (StorageLayout, load_latest, save_checkpoint, write_identity, read_identity,
                                  archive_cosmos, NoValidCheckpoint, atomic_write_json)
from ..version import version_info
from .autotune import autotune
from .runner import Runner
from .universe import Universe, cpu_state

log = get_logger("launcher")


@dataclass
class GenesisHandle:
    universe: Universe
    runner: Runner
    server: object
    tunnel: object | None
    storage: StorageLayout
    auth: dict
    urls: dict = field(default_factory=dict)
    hardware: dict = field(default_factory=dict)
    autotune: dict | None = None

    def status(self) -> dict:
        return {"status": self.universe.status(), "telemetry": self.runner.telemetry()}

    def stop(self) -> None:
        self.runner.stop(checkpoint=True)
        if self.server:
            self.server.stop()
        if self.tunnel:
            self.tunnel.stop()


def default_roots() -> tuple[str, str]:
    if in_colab():
        drive = "/content/drive/MyDrive"
        root = os.path.join(drive, "GenesisOmega") if os.path.isdir(drive) else "/content/GenesisOmega"
        return root, "/content/genesis_scratch"
    here = os.getcwd()
    return os.path.join(here, "genesis_data"), os.path.join(here, "genesis_scratch")


def _existing_cosmos(root: str) -> list[str]:
    if not os.path.isdir(root):
        return []
    return sorted(d for d in os.listdir(root) if os.path.exists(os.path.join(root, d, "cosmos.json")))


def _auth(layout: StorageLayout) -> dict:
    p = layout.path("auth.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    auth = {"viewer": secrets.token_urlsafe(18), "admin": secrets.token_urlsafe(24)}
    atomic_write_json(p, auth)
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass
    return auth


def open_or_create(storage_root: str, scratch_root: str, profile: str, seed: int | None, reset: bool,
                   device: str | None, cosmos: str | None, tune: bool) -> tuple[Universe, StorageLayout, dict]:
    existing = _existing_cosmos(storage_root)
    target = cosmos or (existing[0] if existing else None)
    report = None
    if target and reset:
        lay = StorageLayout(storage_root, target, os.path.join(scratch_root, target))
        dst = archive_cosmos(lay)
        log.warning(f"explicit reset: previous cosmos archived at {dst}")
        target = None
    if target:
        lay = StorageLayout(storage_root, target, os.path.join(scratch_root, target))
        ident = read_identity(lay)
        cfg = UniverseConfig.from_dict(ident["config"])
        if device:
            cfg.device = device
        try:
            state, meta, path = load_latest(lay, map_location="cpu")
        except NoValidCheckpoint as e:
            raise NoValidCheckpoint(
                f"Cosmos {target} exists at {lay.base} but no valid checkpoint could be loaded ({e}). "
                "Refusing to start a new history silently. Pass reset=True to archive it and begin again.") from e
        configure_determinism(cfg.device, cfg.deterministic)
        attach_file_log(lay.path("telemetry", "log.jsonl"))
        u = Universe.from_state(cfg, lay, state)
        u.scientist.run_controls()
        u.emit("SESSION_RESUMED", f"Observatory reconnected; cosmos resumed at cycle {u.utick} from "
               f"{os.path.basename(path)}", epistemic="PROGRAMMED", significance=1,
               detail={"checkpoint": path, "versions": version_info()})
        return u, lay, {"resumed": True, "checkpoint": path}
    prof = profile
    cfg = profile_config(prof, device)
    cfg.seed = int(seed) if seed is not None else derive_seed(time.time_ns(), os.getpid()) % (2 ** 31)
    configure_determinism(cfg.device, cfg.deterministic)
    if tune:
        cfg, report = autotune(cfg)
    label = cosmos_label(cfg.seed)
    lay = StorageLayout(storage_root, label, os.path.join(scratch_root, label))
    lay.ensure()
    attach_file_log(lay.path("telemetry", "log.jsonl"))
    write_identity(lay, {"label": label, "seed": cfg.seed, "config": cfg.to_dict(), "versions": version_info(),
                         "created_at": time.time()})
    if report:
        atomic_write_json(lay.path("telemetry", "autotune.json"), report)
    u = Universe.create(cfg, lay)
    if report:
        u.emit("AUTOTUNE", f"Capacity chosen by measurement: {cfg.world.n_max} organism slots",
               epistemic="MEASURED", significance=0, detail=report)
    save_checkpoint(lay, cpu_state(u), {"utick": u.utick, "label": u.label, "timeline": u.timeline, "seed": u.seed})
    return u, lay, {"resumed": False, "autotune": report}


def launch(storage_root: str | None = None, scratch_root: str | None = None, profile: str = "auto",
           seed: int | None = None, reset: bool = False, port: int = 8765, host: str = "0.0.0.0",
           tunnel: str = "auto", device: str | None = None, cosmos: str | None = None, tune: bool = True,
           checkpoint_interval_s: float | None = None) -> GenesisHandle:
    from ..server.app import ServerThread
    from ..server.tunnel import Tunnel, colab_proxy_url, wait_port

    hw = detect()
    prof = auto_profile(hw) if profile == "auto" else profile
    sr, sc = default_roots()
    storage_root = storage_root or sr
    scratch_root = scratch_root or sc
    os.makedirs(storage_root, exist_ok=True)
    os.makedirs(scratch_root, exist_ok=True)
    u, lay, info = open_or_create(storage_root, scratch_root, prof, seed, reset, device or hw.device, cosmos, tune)
    if checkpoint_interval_s is not None:
        u.cfg.checkpoint_interval_s = checkpoint_interval_s
    runner = Runner(u)
    runner.start()
    auth = _auth(lay)
    server = ServerThread(runner, auth, host, port)
    server.start()
    wait_port(port)
    urls = {"local_viewer": f"http://127.0.0.1:{port}/?token={auth['viewer']}",
            "local_admin": f"http://127.0.0.1:{port}/?token={auth['admin']}"}
    tun = None
    if tunnel in ("auto", "cloudflared"):
        tun = Tunnel(port, os.path.join(scratch_root, "bin"))
        url = tun.start()
        if url:
            urls["public_viewer"] = f"{url}/?token={auth['viewer']}"
            urls["public_admin"] = f"{url}/?token={auth['admin']}"
        else:
            urls["tunnel_error"] = tun.error
    proxy = colab_proxy_url(port)
    if proxy:
        urls["colab_viewer"] = f"{proxy.rstrip('/')}/?token={auth['viewer']}"
        urls["colab_admin"] = f"{proxy.rstrip('/')}/?token={auth['admin']}"
    return GenesisHandle(u, runner, server, tun, lay, auth, urls, hw.to_dict(), info.get("autotune"))
