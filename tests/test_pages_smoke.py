# -*- coding: utf-8 -*-
"""
tests/test_pages_smoke.py — Headless smoke test for every Streamlit page.

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

# The three Benchmark modes (dispatched via runpy from 2_Benchmark.py). Smoke-run
# directly so the suite also covers the mode files — including the Full-comparison
# leaderboard table/chart, which renders from the committed leaderboard.json.
MODES = [
    "app_pages/modes/_mode_classic.py",
    "app_pages/modes/_mode_cnn.py",
    "app_pages/modes/_mode_full.py",
]


@pytest.fixture(autouse=True)
def _force_demo(monkeypatch):
    # Corpus-less web-demo path: deterministic, no dataset/GPU/network needed.
    monkeypatch.setenv("DEEPFAKE_FORCE_DEMO", "1")


@pytest.mark.parametrize("page", PAGES + MODES,
                         ids=[p.split("/")[-1] for p in PAGES + MODES])
def test_page_renders_without_exception(page):
    at = AppTest.from_file(str(ROOT / page), default_timeout=90).run()
    assert not at.exception, f"{page} raised: {at.exception}"
