# -*- coding: utf-8 -*-
"""
app.py, main entry point.

Configures the page, global CSS and multi-page navigation BEFORE delegating to
each page, so the first render is already stable (no flash of auto-discovered
navigation or unstyled content).

NOTE: the pages directory is named `app_pages/` (not `pages/`) on purpose: a
`pages/` directory triggers Streamlit's v1 auto-navigation, which appears for a
moment ("app", "0 Home", …) before st.navigation replaces it. Renaming it
removes that intermediate state.

Launch with:
    streamlit run app.py
"""

import logging as _logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st  # noqa: E402

# Streamlit's module watcher calls hasattr(m, "__path__") on every entry in
# sys.modules each rerun. Some transformers subpackages are lazy proxies whose
# __getattr__ triggers importing the real file, which needs torchvision (not
# installed). Suppress the resulting WARNING from the watcher logger, the app
# works correctly, these are pure noise.
class _NoTransformerVisionWarning(_logging.Filter):
    def filter(self, record):
        return "transformers.models." not in record.getMessage()

_wl = _logging.getLogger("streamlit.watcher.local_sources_watcher")
if not any(isinstance(f, _NoTransformerVisionWarning) for f in _wl.filters):
    _wl.addFilter(_NoTransformerVisionWarning())

# The only set_page_config call in the whole app: individual pages must NOT
# call it (each st.Page supplies its own tab title).
st.set_page_config(
    page_title="Deepfake Audio Detection",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.ui_helpers import (  # noqa: E402
    apply_mpl_theme, build_page_css, preload_wav2vec_background, theme_mode,
)

# Start loading wav2vec 2.0 (raw-waveform SSL detector) in the background the
# moment the app boots, instead of the first visitor paying for it mid-navigation
# on Benchmark → Full comparison or Detection Analysis. Non-blocking (returns
# immediately, the model finishes loading on a background thread) and runs
# exactly once per server process no matter how many sessions rerun this file.
preload_wav2vec_background()

# ── Light Side / Dark Side ───────────────────────────────────────────────────
# Read the chosen side (default Dark Side) and apply the matching palette to both
# the matplotlib figures and the global stylesheet, BEFORE the page draws. The
# sidebar toggle (below) writes st.session_state["sw_theme"] in an on_change
# callback that runs at the start of the rerun, so this reads the up-to-date
# choice with no one-rerun lag.
# Persistent (non-widget) preference keys so the choices survive page changes.
# Streamlit garbage-collects a WIDGET's state when its page stops rendering, so the
# Settings widgets write to these plain keys via callbacks (see 5_Settings.py) and
# everything here reads the plain keys.
# Defaults MUST match _DEFAULTS in 5_Settings.py: the Settings widgets are seeded
# from these plain keys, so a value outside a widget's options (e.g. the old
# "Auto" saber) would silently reset the control and desync the two pages.
for _k, _v in {
    "sw_theme": "dark", "sw_saber": "Red", "sw_bg": "Star Wars",
    "sw_bg_intensity": "Normal", "sw_reduced_motion": False,
    "sw_contrast": False, "sw_text_scale": "Normal",
    "sw_underline": False, "sw_text_spacing": False,
    "sw_show_ships": True, "sw_show_deathstar": True,
    "sw_audio_color": "Red",
}.items():
    st.session_state.setdefault(_k, _v)
_theme = theme_mode()
apply_mpl_theme(_theme)

# ── Native Streamlit theme follows the chosen side ───────────────────────────
# The injected CSS cannot reach canvas-based widgets (st.dataframe is a
# glide-data-grid canvas), so on the Light Side they stayed dark. Flipping the
# CONFIG theme makes native chrome and the dataframe follow. The new values are
# only picked up by the NewSession message of the NEXT run, so when something
# actually changed we rerun once immediately (the comparison guard makes this a
# strict one-shot, no loop).
# KNOWN LIMITATION: config options are process-wide, not per-session. On a
# shared deployment, another visitor's native chrome adopts this theme until
# their own rerun re-asserts their choice (this block runs before every page,
# so each session self-corrects on its next interaction; the injected CSS keeps
# their page consistent meanwhile). Acceptable for a low-traffic demo.
_NATIVE_THEME = {
    "dark":  {"base": "dark", "primaryColor": "#4F8BF9",
              "backgroundColor": "#0E1117",
              "secondaryBackgroundColor": "#161C2D", "textColor": "#E8EDF8"},
    "light": {"base": "light", "primaryColor": "#1E5FCF",
              "backgroundColor": "#D5DEEE",
              "secondaryBackgroundColor": "#FFFFFF", "textColor": "#1B2438"},
}
_theme_changed = False
try:
    from streamlit import config as _st_config

    for _opt, _val in _NATIVE_THEME[_theme].items():
        if _st_config.get_option(f"theme.{_opt}") != _val:
            _st_config.set_option(f"theme.{_opt}", _val)
            _theme_changed = True
except Exception:  # noqa: BLE001, never let theming break the app
    _theme_changed = False
# Outside the try: st.rerun() raises a control-flow exception that the guard
# above must not swallow. NEVER rerun on a session's FIRST script run: at that
# point st.navigation has not routed the URL yet, and the rerun would land the
# visitor on the default page instead of the deep link they opened (only the
# multi-user stale-config case even reaches here on a first run; it
# self-corrects on the next interaction and is documented in the README).
if _theme_changed and st.session_state.get("_nav_routed"):
    st.rerun()
st.session_state["_nav_routed"] = True

# Global CSS injected here (not per page): it always occupies the same position
# in the element tree, so Streamlit reconciles it across pages and there is no
# flash of default styles when navigating.
st.markdown(build_page_css(_theme), unsafe_allow_html=True)

# May the 4th, a one-line banner that only shows on Star Wars Day (4 May).
import datetime as _dt  # noqa: E402

if (_dt.date.today().month, _dt.date.today().day) == (5, 4):
    st.markdown(
        '<div style="text-align:center;font-size:0.8rem;font-weight:700;'
        'letter-spacing:0.18em;text-transform:uppercase;color:var(--saber);'
        'padding:0.4rem 0 0.2rem;text-shadow:0 0 10px var(--saber-glow);">'
        'May the 4th be with you</div>',
        unsafe_allow_html=True,
    )

# ── Script host (0-height, same-origin iframe) ───────────────────────────────
# Hosts, in a single IIFE: the Star Wars background canvas (ships, Death Star,
# hyperspace), the Konami-code listener and the sidebar resize-handle hider. The
# script lives in static/canvas.js (externalised for JS linting and browser
# caching); it is wrapped in <script> tags via CONCATENATION, not an f-string,
# the JS is full of braces, and injected into a 0-height same-origin iframe.
@st.cache_data(show_spinner=False)
def _canvas_js() -> str:
    # app.py re-executes on every rerun; cache the 24 KB read instead of
    # hitting the disk each interaction.
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "static", "canvas.js")
    with open(path, encoding="utf-8") as f:
        return f.read()


_CANVAS_JS = _canvas_js()
with st.container(key="ddac_host"):
    st.iframe(
        "<script>\n" + _CANVAS_JS + "\n</script>",
        # st.iframe rejects height=0 (unlike the old components.html); use the
        # minimum positive height and let the ddac_host CSS collapse it to 0.
        height=1,
    )

# ── Live settings → background canvas ────────────────────────────────────────
# The canvas script (above) runs once and reads these window flags every frame.
#
# Helper-iframe design note (evaluated for merging, kept separate on purpose):
#   • ddac_host (canvas.js) must stay STATIC, if its srcdoc carried the live
#     flags it would change every rerun, remounting the iframe and resetting the
#     whole background animation on every interaction.
#   • eqramp_host lives on Home only (page-scoped script over Home's DOM).
#   • This flags iframe is the only script channel Streamlit offers for pushing
#     values into the parent window (st.markdown strips <script>). What we CAN
#     save is re-rendering it when nothing changed: the flags persist on
#     window.parent once set, so the iframe is emitted only when their value
#     ACTUALLY differs from what this session last pushed, i.e. one component
#     render on the first run and on each Settings change, zero on every other
#     rerun (the common case).
_BG_MODES = {"Star Wars": "starwars", "Particle network": "network", "Off": "off"}
_bg_mode = _BG_MODES.get(st.session_state.get("sw_bg", "Star Wars"), "starwars")
_intensity = st.session_state.get("sw_bg_intensity", "Normal")
_reduce = "1" if st.session_state.get("sw_reduced_motion") else "0"
_ships = "1" if st.session_state.get("sw_show_ships", True) else "0"
_deathstar = "1" if st.session_state.get("sw_show_deathstar", True) else "0"
_sw_flags = (_bg_mode, _theme, _intensity, _reduce, _ships, _deathstar)
with st.container(key="swflags_host"):
    if st.session_state.get("_swflags_sent") != _sw_flags:
        st.iframe(
            f"""
<script>
(function(){{
  var w; try {{ w = window.parent; }} catch (e) {{ return; }}
  w.__swBg = '{_bg_mode}';
  w.__swTheme = '{_theme}';
  w.__swIntensity = '{_intensity}';
  w.__reduceMotion = {_reduce};
  w.__swShips = {_ships};
  w.__swDeathStar = {_deathstar};
  try {{ w.document.documentElement.setAttribute('data-reduce-motion', '{_reduce}'); }} catch (e) {{}}
}})();
</script>
""",
            height=1,
        )
        st.session_state["_swflags_sent"] = _sw_flags

# Collect a finished background benchmark (runs on every page) and clear the
# "operation in progress" flag so run/train buttons re-enable everywhere.
import src.jobs as _jobs  # noqa: E402

_fut = st.session_state.get("bench_future")
if _fut is not None and _fut.done():
    try:
        _classic_rows, _cnn_rows = _fut.result()
        if _jobs.cancel_requested():            # cancelled → discard partial work
            _classic_rows, _cnn_rows = [], []
            st.session_state["bench_cancelled"] = True
        # "Eval" = eval-only: keep just the eval rows (Split starts with "eval").
        elif st.session_state.get("bench_score") == "Eval":
            _classic_rows = [r for r in _classic_rows
                             if str(r.get("Split", "")).startswith("eval")]
            _cnn_rows = [r for r in _cnn_rows
                         if str(r.get("Split", "")).startswith("eval")]
        st.session_state.setdefault("experiment_rows", []).extend(_classic_rows)
        st.session_state.setdefault("cnn_runs", []).extend(_cnn_rows)
        if not _jobs.cancel_requested():
            st.session_state["bench_done"] = True   # enables "See full comparison"
    except Exception as _exc:  # noqa: BLE001, surface compute errors
        st.session_state["bench_error"] = str(_exc)
    st.session_state["bench_future"] = None
    st.session_state["op_running"] = False

# Collect a finished background deep network training the same way: store the model and
# its results so the Deep Networks Learning page shows them, exactly as if trained inline.
_cnn_fut = st.session_state.get("cnn_future")
if _cnn_fut is not None and _cnn_fut.done():
    try:
        _model, _hist, _results = _cnn_fut.result()
        if _jobs.cancel_requested():            # cancelled → keep nothing
            st.session_state["cnn_cancelled"] = True
        else:
            _pend = st.session_state.get("cnn_pending", {})
            _train_only = st.session_state.pop("cnn_train_only", False)
            st.session_state["cnn_model"]        = _model
            st.session_state["cnn_history"]      = _hist
            st.session_state["cnn_dev"]          = _pend.get("dev", [])
            st.session_state["cnn_train_corpus"] = _pend.get("corpus", ", ")
            st.session_state["cnn_arch_trained"] = _pend.get("arch", ", ")
            if not _train_only:
                # Produce result rows only when NOT in train-only mode; the
                # Evaluate button in Deep Networks mode adds them on-demand instead.
                st.session_state["cnn_results"] = _results
                _board = _results
                if _pend.get("score") == "Eval":
                    _ev = [r for r in _results if "[EVAL]" in str(r.get("Model", ""))]
                    _board = _ev or _results
                st.session_state.setdefault("cnn_runs", []).extend(_board)
    except Exception as _exc:  # noqa: BLE001, surface compute errors
        st.session_state["cnn_error"] = str(_exc)
    st.session_state["cnn_future"] = None
    st.session_state.pop("cnn_pending", None)

# Global background-job banner, pinned to the bottom of the sidebar. Rendered
# BEFORE the page runs (so it appears on every page, even ones that call
# st.stop()) and as a self-refreshing fragment (so it updates its progress every
# 2 s on its own and triggers a single full rerun when the job finishes, no
# disruptive whole-app polling).
from src.ui_helpers import (  # noqa: E402
    op_banner_fragment, op_in_progress,
)

# Rendered on every page inside a stable-key container. The banner itself only
# attaches a run_every=2 s timer while a job is actually running (see
# op_banner_fragment): an idle app keeps no timer, so the rapid reruns of startup
# and page navigation no longer leave a dangling timer that logs
# "The fragment ... does not exist anymore".
with st.sidebar:
    with st.container(key="sidebar_banner"):
        op_banner_fragment()

pg = st.navigation([
    st.Page("app_pages/0_Home.py",               title="Home",               icon=":material/home:", default=True),
    st.Page("app_pages/1_Signal_Explorer.py",    title="Signal Explorer",    icon=":material/graphic_eq:", url_path="signal_explorer"),
    st.Page("app_pages/2_Benchmark.py",          title="Benchmark",          icon=":material/science:", url_path="benchmark"),
    st.Page("app_pages/3_Detection_Analysis.py", title="Detection Analysis", icon=":material/insights:", url_path="detection_analysis"),
    st.Page("app_pages/6_Voice_Cloner.py",       title="Voice Cloner",       icon=":material/record_voice_over:", url_path="voice_cloner"),
    st.Page("app_pages/4_Methodology.py",        title="Methodology",        icon=":material/menu_book:", url_path="methodology"),
    st.Page("app_pages/5_Settings.py",           title="Settings",           icon=":material/settings:", url_path="settings"),
])
pg.run()
