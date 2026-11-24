"""HTTPS exposure for remote observers (spec §287-288).

Host-side tooling only: organisms and the Scientist DSL can never reach this
module. Uses a Cloudflare quick tunnel (no account) when available, and the
Colab authenticated port proxy when running inside Colab.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import threading
import time
import urllib.request

from ..core.logging import get_logger

log = get_logger("tunnel")
CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


class Tunnel:
    def __init__(self, port: int, bin_dir: str):
        self.port = port
        self.bin_dir = bin_dir
        self.proc: subprocess.Popen | None = None
        self.url: str | None = None
        self.error: str | None = None

    def _binary(self) -> str | None:
        found = shutil.which("cloudflared")
        if found:
            return found
        path = os.path.join(self.bin_dir, "cloudflared")
        if os.path.exists(path):
            return path
        try:
            os.makedirs(self.bin_dir, exist_ok=True)
            urllib.request.urlretrieve(CLOUDFLARED_URL, path)
            os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
            return path
        except Exception as e:
            self.error = f"cloudflared unavailable: {e!r}"
            return None

    def start(self, timeout: float = 45.0) -> str | None:
        exe = self._binary()
        if exe is None:
            return None
        self.proc = subprocess.Popen([exe, "tunnel", "--url", f"http://127.0.0.1:{self.port}", "--no-autoupdate"],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        found = threading.Event()

        def pump():
            for line in self.proc.stdout:
                m = URL_RE.search(line)
                if m and not self.url:
                    self.url = m.group(0)
                    found.set()

        threading.Thread(target=pump, daemon=True, name="tunnel-log").start()
        found.wait(timeout)
        if not self.url:
            self.error = "tunnel did not report a URL in time"
        return self.url

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def colab_proxy_url(port: int) -> str | None:
    try:
        from google.colab.output import eval_js  # type: ignore

        return eval_js(f"google.colab.kernel.proxyPort({port})")
    except Exception:
        return None


def wait_port(port: int, timeout: float = 10.0) -> bool:
    import socket

    t0 = time.time()
    while time.time() - t0 < timeout:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.1)
    return False
