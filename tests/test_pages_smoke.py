# -*- coding: utf-8 -*-
"""
tests/test_pages_smoke.py, Headless smoke test for every Streamlit page.

Uses Streamlit's AppTest to actually execute each page script in a simulated
runtime and asserts it renders without raising. Runs in the corpus-less
web-demo mode (DEEPFAKE_FORCE_DEMO=1) so it is deterministic and needs no
dataset, GPU or network. This is the only automated coverage of the page layer
(the UI refactors are otherwise untested), so it guards against import/render
regressions across the ui_helpers split and the page de-duplications.
"""

import os
from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

ROOT = Path(__file__).parent.parent

PAGES = [
    "app_pages/0_Home.py",
    "app_pages/1_Signal_Explorer.py",
    "app_pages/2_Benchmark.py",
    "app_pages/3_Detection_Analysis.py",
    "app_pages/4_Methodology.py",
    "app_pages/5_Settings.py",
]

MODES = [
    "app_pages/modes/_mode_classic.py",
    "app_pages/modes/_mode_cnn.py",
    "app_pages/modes/_mode_full.py",
]


@pytest.fixture(autouse=True)
def _force_demo(monkeypatch):
    monkeypatch.setenv("DEEPFAKE_FORCE_DEMO", "1")


@pytest.mark.parametrize("page", PAGES + MODES,
                         ids=[p.split("/")[-1] for p in PAGES + MODES])
def test_page_renders_without_exception(page):
    at = AppTest.from_file(str(ROOT / page), default_timeout=90).run()
    assert not at.exception, f"{page} raised: {at.exception}"


def test_settings_survives_app_seeded_state():
    """Settings must render with the plain keys app.py seeds on every rerun,
    including a stale legacy 'Auto' saber value, which its colour selectbox
    does not offer and must therefore be coerced, not passed through."""
    at = AppTest.from_file(str(ROOT / "app_pages/5_Settings.py"),
                           default_timeout=90)
    at.session_state["sw_theme"] = "dark"
    at.session_state["sw_saber"] = "Auto"
    at.run()
    assert not at.exception, f"Settings raised: {at.exception}"
    assert at.session_state["sw_color_ctl"] in (
        "Red", "Blue", "Green", "Purple", "Amber")
