# -*- coding: utf-8 -*-
"""
tests/test_e2e_a11y.py, Browser end-to-end audits: axe-core accessibility scan
and the Efficiency-tab hover regression.

Runs a REAL Streamlit server (corpus-less demo mode) and drives it with
Playwright/Chromium, so it exercises what a visitor actually gets, injected
CSS, both themes, Vega tooltips, none of which AppTest can see.

    Run:      pytest -m e2e
    Skipped:  automatically, when Playwright (or its Chromium) is not installed.

The axe-core engine is vendored at tests/assets/axe.min.js (no network needed
at test time). The scan gates on SERIOUS + CRITICAL violations, scoped to the
app's own UI: Streamlit's built-in chrome (toolbar, status widget) is excluded
because we cannot fix upstream markup from this repo; anything excluded is
listed here, on purpose, as the audit's documented blind spot.
"""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pw = pytest.importorskip("playwright.sync_api")

pytestmark = pytest.mark.e2e

ROOT = Path(__file__).parent.parent
AXE_JS = Path(__file__).parent / "assets" / "axe.min.js"
PORT = 8610
URL = f"http://localhost:{PORT}"

_AXE_EXCLUDE = [
    '[data-testid="stToolbar"]',
    '[data-testid="stStatusWidget"]',
    '[data-testid="stDecoration"]',
    "iframe",
]

_ALLOWED: set = {("aria-allowed-attr", ".stSidebar")}


@pytest.fixture(scope="module")
def server():
    env = dict(os.environ, DEEPFAKE_FORCE_DEMO="1")
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py",
         "--server.headless", "true", "--server.port", str(PORT),
         "--browser.gatherUsageStats", "false"],
        cwd=ROOT, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(60):
            try:
                socket.create_connection(("localhost", PORT), timeout=1).close()
                break
            except OSError:
                time.sleep(1)
        else:
            pytest.skip("Streamlit server did not start")
        yield URL
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="module")
def browser():
    try:
        p = pw.sync_playwright().start()
        b = p.chromium.launch()
    except Exception as exc:                    # noqa: BLE001
        pytest.skip(f"Chromium unavailable: {exc}")
    yield b
    b.close()
    p.stop()


def _open(browser, url: str):
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto(url, wait_until="networkidle")
    page.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=30_000)
    page.wait_for_timeout(2_500)
    return page


def _axe_scan(page) -> list:
    """Inject vendored axe-core and return serious/critical violations."""
    page.add_script_tag(content=AXE_JS.read_text(encoding="utf-8"))
    result = page.evaluate(
        """async (exclude) => {
            const ctx = { exclude: exclude.map(sel => [sel]) };
            const res = await axe.run(ctx, {
                runOnly: { type: 'tag', values: ['wcag2a', 'wcag2aa'] },
            });
            return res.violations.map(v => ({
                id: v.id, impact: v.impact, help: v.help,
                nodes: v.nodes.slice(0, 5).map(n => n.target.join(' ')),
            }));
        }""",
        _AXE_EXCLUDE,
    )
    def _allowed(v) -> bool:
        return all((v["id"], node) in _ALLOWED for node in v["nodes"])

    return [v for v in result
            if v["impact"] in ("serious", "critical") and not _allowed(v)]


def _assert_clean(violations, label: str) -> None:
    lines = [f"- [{v['impact']}] {v['id']}: {v['help']} → {v['nodes']}"
             for v in violations]
    assert not violations, f"axe violations on {label}:\n" + "\n".join(lines)


def test_axe_home_dark(server, browser):
    page = _open(browser, server)
    try:
        _assert_clean(_axe_scan(page), "Home (Dark Side)")
    finally:
        page.close()


def test_axe_settings_both_sides(server, browser):
    page = _open(browser, server + "/settings")
    try:
        _assert_clean(_axe_scan(page), "Settings (Dark Side)")
        page.get_by_text("Light Side", exact=True).first.click()
        page.wait_for_timeout(2_500)
        _assert_clean(_axe_scan(page), "Settings (Light Side)")
    finally:
        page.close()


def test_efficiency_hover_survives_interaction(server, browser):
    """Regression test for the Full-comparison Efficiency bug: Vega tooltips
    used to die after the first click (a keyed selection chart re-rendered on
    rerun and lost its hover handler). Hover must work on entry AND after
    switching tabs / clicking around."""
    page = _open(browser, server + "/benchmark")
    page.get_by_role("button", name="Open Full comparison").click(timeout=45_000)
    page.get_by_role("tab", name="Efficiency").wait_for(timeout=240_000)
    page.get_by_role("tab", name="Efficiency").click()
    page.wait_for_timeout(2_500)

    def hover_shows_tooltip() -> bool:
        charts = page.locator('[data-testid*="VegaLiteChart"]:visible')
        if not charts.count():
            return False
        charts.first.scroll_into_view_if_needed()
        page.wait_for_timeout(400)
        marks = charts.first.locator("svg .mark-symbol path")
        n = marks.count()
        if not n:
            return False
        for i in range(min(3, n)):
            page.mouse.move(10, 10)
            page.wait_for_timeout(300)
            marks.nth(i).hover(force=True)
            for _ in range(16):
                tip = page.locator("#vg-tooltip-element")
                if tip.count() and "visible" in (tip.first.get_attribute("class")
                                                 or ""):
                    return True
                page.wait_for_timeout(250)
        return False

    try:
        assert hover_shows_tooltip(), "tooltip never appeared on first entry"

        chart = page.locator('[data-testid*="VegaLiteChart"]:visible').first
        chart.scroll_into_view_if_needed()
        box = chart.bounding_box()
        page.mouse.click(box["x"] + box["width"] / 2,
                         box["y"] + box["height"] / 2)
        page.wait_for_timeout(1_000)
        page.get_by_role("tab", name="Ranking").click()
        page.wait_for_timeout(1_200)
        page.get_by_role("tab", name="Efficiency").click()
        page.wait_for_timeout(1_800)
        ok = hover_shows_tooltip()
        if not ok:
            page.screenshot(path="/tmp/e2e_hover_fail.png", full_page=True)
            with open("/tmp/e2e_hover_fail.txt", "w") as fh:
                fh.write("visible charts: %d\n" % page.locator(
                    '[data-testid*="VegaLiteChart"]:visible').count())
                tip = page.locator("#vg-tooltip-element")
                fh.write("tooltip count: %d\n" % tip.count())
                if tip.count():
                    fh.write("tooltip class: %s\n"
                             % tip.first.get_attribute("class"))
                fh.write("tabs: %s\n" % page.get_by_role("tab").all_inner_texts())
        assert ok, "tooltip died after chart click + tab bounce"
    finally:
        page.close()
