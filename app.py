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

class _NoTransformerVisionWarning(_logging.Filter):
    def filter(self, record):
        return "transformers.models." not in record.getMessage()

_wl = _logging.getLogger("streamlit.watcher.local_sources_watcher")
if not any(isinstance(f, _NoTransformerVisionWarning) for f in _wl.filters):
    _wl.addFilter(_NoTransformerVisionWarning())

st.set_page_config(
    page_title="Deepfake Audio Detection",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.ui_helpers import (  # noqa: E402
    apply_mpl_theme, build_page_css, preload_figure_backend,
    preload_wav2vec_background, theme_mode,
)

preload_wav2vec_background()

preload_figure_backend()

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

_NATIVE_THEME = {
    "dark": {"base": "dark", "primaryColor": "#4F8BF9",
              "backgroundColor": "#0E1117",
              "secondaryBackgroundColor": "#161C2D", "textColor": "#E8EDF8"},
    "light": {"base": "light", "primaryColor": "#1E5FCF",
              "backgroundColor": "#D5DEEE",
              "secondaryBackgroundColor": "#FFFFFF", "textColor": "#1B2438"},
}
try:
    from streamlit import config as _st_config

    for _opt, _val in _NATIVE_THEME[_theme].items():
        if _st_config.get_option(f"theme.{_opt}") != _val:
            _st_config.set_option(f"theme.{_opt}", _val)
except Exception:  # noqa: BLE001, never let theming break the app
    pass

st.markdown(build_page_css(_theme), unsafe_allow_html=True)

import datetime as _dt  # noqa: E402

if (_dt.date.today().month, _dt.date.today().day) == (5, 4):
    st.markdown(
        '<div style="text-align:center;font-size:0.8rem;font-weight:700;'
        'letter-spacing:0.18em;text-transform:uppercase;color:var(--saber);'
        'padding:0.4rem 0 0.2rem;text-shadow:0 0 10px var(--saber-glow);">'
        'May the 4th be with you</div>',
        unsafe_allow_html=True,
    )

@st.cache_data(show_spinner=False)
def _canvas_js() -> str:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "static", "canvas.js")
    with open(path, encoding="utf-8") as f:
        return f.read()


_CANVAS_JS = _canvas_js()
with st.container(key="ddac_host"):
    st.iframe(
        "<script>\n" + _CANVAS_JS + "\n</script>",
        height=1,
    )

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

import src.jobs as _jobs  # noqa: E402

_fut = st.session_state.get("bench_future")
if _fut is not None and _fut.done():
    try:
        _classic_rows, _cnn_rows = _fut.result()
        if _jobs.cancel_requested():
            _classic_rows, _cnn_rows = [], []
            st.session_state["bench_cancelled"] = True
        elif st.session_state.get("bench_score") == "Eval":
            _classic_rows = [r for r in _classic_rows
                             if str(r.get("Split", "")).startswith("eval")]
            _cnn_rows = [r for r in _cnn_rows
                         if str(r.get("Split", "")).startswith("eval")]
        st.session_state.setdefault("experiment_rows", []).extend(_classic_rows)
        st.session_state.setdefault("cnn_runs", []).extend(_cnn_rows)
        if not _jobs.cancel_requested():
            st.session_state["bench_done"] = True
    except Exception as _exc:  # noqa: BLE001, surface compute errors
        st.session_state["bench_error"] = str(_exc)
    st.session_state["bench_future"] = None
    st.session_state["op_running"] = False

_cnn_fut = st.session_state.get("cnn_future")
if _cnn_fut is not None and _cnn_fut.done():
    try:
        _model, _hist, _results = _cnn_fut.result()
        if _jobs.cancel_requested():
            st.session_state["cnn_cancelled"] = True
        else:
            _pend = st.session_state.get("cnn_pending", {})
            _train_only = st.session_state.pop("cnn_train_only", False)
            st.session_state["cnn_model"] = _model
            st.session_state["cnn_history"] = _hist
            st.session_state["cnn_dev"] = _pend.get("dev", [])
            st.session_state["cnn_train_corpus"] = _pend.get("corpus", ", ")
            st.session_state["cnn_arch_trained"] = _pend.get("arch", ", ")
            if not _train_only:
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

from src.ui_helpers import (  # noqa: E402
    op_banner_fragment, op_in_progress,
)

with st.sidebar:
    with st.container(key="sidebar_banner"):
        op_banner_fragment()

pg = st.navigation([
    st.Page("app_pages/0_Home.py", title="Home", icon=":material/home:", default=True),
    st.Page("app_pages/1_Signal_Explorer.py", title="Signal Explorer", icon=":material/graphic_eq:", url_path="signal_explorer"),
    st.Page("app_pages/2_Benchmark.py", title="Benchmark", icon=":material/science:", url_path="benchmark"),
    st.Page("app_pages/3_Detection_Analysis.py", title="Detection Analysis", icon=":material/insights:", url_path="detection_analysis"),
    st.Page("app_pages/6_Voice_Cloner.py", title="Voice Cloner", icon=":material/record_voice_over:", url_path="voice_cloner"),
    st.Page("app_pages/4_Methodology.py", title="Methodology", icon=":material/menu_book:", url_path="methodology"),
    st.Page("app_pages/5_Settings.py", title="Settings", icon=":material/settings:", url_path="settings"),
])
pg.run()
