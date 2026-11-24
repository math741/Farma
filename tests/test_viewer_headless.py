"""Ω0 observatory, end to end in a real (headless) browser: boot, enter, every mode, every panel,
organism inspection, overlays, replay of a frozen window. Fails on any JavaScript error.

Visual quality is not machine-verified (see the feature registry); this checks that the viewer
boots, talks the protocol and renders real state without errors. Skipped when Playwright with a
Chromium build is not installed (``pip install playwright && playwright install chromium``).
"""

from __future__ import annotations

import socket
import time

import pytest

from conftest import small_cfg

playwright = pytest.importorskip("playwright.sync_api")


@pytest.fixture(scope="module")
def cosmos():
    from genesis_omega.history.events import SIG_HISTORIC
    from genesis_omega.runtime.runner import Runner
    from genesis_omega.runtime.universe import Universe
    from genesis_omega.server.app import ServerThread

    u = Universe.create(small_cfg(seed=777), None, run_controls=False)
    for _ in range(150):
        u.step_round()
    # a historic event freezes the ring-buffer window around it (replayable in the viewer)
    ev = u._make_world_event(u.worlds[0], "TEST_WINDOW", "a recorded window for the viewer test",
                             significance=SIG_HISTORIC)
    for _ in range(120):
        u.step_round()
    assert ev.id in u.frozen and u.frozen[ev.id]["post"]
    r = Runner(u)
    r.target_tps = 15
    r.start()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = ServerThread(r, {"viewer": "viewer-x", "admin": "admin-y"}, "127.0.0.1", port)
    srv.start()
    yield {"port": port, "event": ev.id, "runner": r}
    srv.stop()
    r.stop(checkpoint=False)


@pytest.fixture(scope="module")
def page(cosmos):
    errors = []
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        except Exception as e:  # browser binary missing
            pytest.skip(f"no Chromium for Playwright: {e}")
        pg = browser.new_page(viewport={"width": 1280, "height": 800})
        pg.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        pg.on("console", lambda m: errors.append(f"console.error: {m.text}") if m.type == "error" else None)
        pg.goto(f"http://127.0.0.1:{cosmos['port']}/?token=admin-y")
        pg.wait_for_selector("body[data-ready='1']", timeout=90_000)
        pg.errors = errors
        yield pg
        browser.close()


def _no_errors(pg):
    assert not pg.errors, pg.errors


def test_boot_and_enter(page):
    assert "token=" not in page.url                                    # the token leaves the address bar
    assert page.is_enabled("#enter")
    assert "admin access" in page.inner_text("#start-status")
    assert page.evaluate("() => !!document.createElement('canvas').getContext('webgl2')")
    page.click("#enter")
    page.wait_for_selector("body[data-entered='1']", timeout=30_000)
    page.wait_for_function("() => window.genesis.status !== null", timeout=30_000)  # live stream arrived
    assert page.is_visible("#hud") and page.is_visible("#admin-btn")
    assert page.evaluate("() => genesis.mode") == "galaxy"
    _no_errors(page)


def test_modes_render_real_state(page):
    page.click("#modes [data-mode='system']")
    page.wait_for_function("() => genesis.mode === 'system' && !genesis.switching", timeout=30_000)
    assert page.evaluate("() => genesis.view.planets.length") > 0
    page.click("#modes [data-mode='planet']")
    page.wait_for_function("() => genesis.mode === 'planet' && !genesis.switching", timeout=60_000)
    page.wait_for_function("() => genesis.view.n > 0", timeout=30_000)  # organisms streamed over the socket
    n = page.evaluate("() => genesis.view.n")
    assert n > 0
    for k in range(1, 9):
        page.select_option("#overlay", str(k))
    page.select_option("#overlay", "0")
    page.click("#modes [data-mode='galaxy']")
    page.wait_for_function("() => genesis.mode === 'galaxy' && !genesis.switching", timeout=30_000)
    _no_errors(page)


@pytest.mark.parametrize("name", ["history", "science", "life", "tech", "status", "admin"])
def test_panels(page, name):
    page.click(f"#modes [data-panel='{name}']")
    page.wait_for_function("() => !document.querySelector('#panel-body').innerText.includes('loading…')",
                           timeout=30_000)
    text = page.inner_text("#panel-body")
    assert "could not load" not in text, text[:300]
    page.click(f"#modes [data-panel='{name}']")
    _no_errors(page)


def test_inspect_an_organism_and_replay_a_window(page, cosmos):
    page.evaluate("() => genesisDebug.setMode('planet', {world: 0})")
    page.wait_for_function("() => genesis.mode === 'planet' && !genesis.switching && genesis.view.n > 0",
                           timeout=60_000)
    aid = page.evaluate("() => Math.max(...genesis.view.ids)")           # the youngest: alive for a while
    page.evaluate(f"() => genesisDebug.selectAgent({aid})")
    page.wait_for_function(f"() => document.querySelector('#insp-body').innerText.includes('A-{aid}')",
                           timeout=30_000)
    assert page.is_visible("#inspector")
    page.evaluate(f"() => genesisDebug.replayEvent({cosmos['event']})")
    page.wait_for_selector("#replay-badge:not(.hidden)", timeout=30_000)
    page.wait_for_function("() => document.querySelector('#replay-badge span').innerText.includes('tick')",
                           timeout=30_000)
    page.click("#replay-stop")
    assert page.is_hidden("#replay-badge")
    _no_errors(page)
