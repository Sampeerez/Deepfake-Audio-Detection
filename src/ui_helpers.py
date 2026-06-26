# -*- coding: utf-8 -*-
"""
src/ui_helpers.py — Shared helpers for the Streamlit GUI.

Cached resource loaders, plotting helpers, and reusable UI components.
Keeps page files thin and avoids recomputation across reruns.
"""

import os
from typing import Dict, List, Optional, Tuple

import librosa
import librosa.display
import matplotlib
import numpy as np
import streamlit as st
import yaml
from scipy.fft import dct

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data_loader import (  # noqa: E402
    LABEL_BONAFIDE, LABEL_SPOOF, parse_protocol, parse_protocol_2021,
)
from src.features import FeatureExtractor  # noqa: E402

CONFIG_PATH = os.path.join("config", "config.yaml")
_EPS = 1e-10


def _forced_demo() -> bool:
    """Env switch to simulate the corpus-less cloud deployment on a local machine
    (set DEEPFAKE_FORCE_DEMO=1): the dataset loaders report empty so every page
    renders exactly as it will in the public web demo."""
    return os.environ.get("DEEPFAKE_FORCE_DEMO") in ("1", "true", "True")

# ── Colour palette ────────────────────────────────────────────────────────── #
BONAFIDE_COLOR = "#42A5F5"   # blue  — real voice  (lighter for dark bg)
SPOOF_COLOR    = "#EF5350"   # red   — deepfake
NEUTRAL_COLOR  = "#78909C"   # slate — unknown / upload

# ── matplotlib figure palettes ────────────────────────────────────────────── #
# Two themes: Dark Side (default, matches the dark chrome) and Light Side. The
# data-semantic colours (BONAFIDE/SPOOF above) stay constant across both so the
# charts read the same; only the canvas/text/grid flip.
_DARK_FIG  = {"bg": "#161C2D", "axes": "#1E2640", "grid": "#263050",
              "text": "#C5CDE8", "edge": "#2E3A58"}
_LIGHT_FIG = {"bg": "#FFFFFF", "axes": "#F4F7FD", "grid": "#D5DEEF",
              "text": "#27324C", "edge": "#B7C3DC"}

# Active palette: module globals so the fig_* helpers below pick up the current
# theme at call time (they read these names when a figure is built). app.py calls
# apply_mpl_theme() once per rerun before any page draws; default to dark so any
# figure built at import time still renders correctly.
_FIG_BG   = _DARK_FIG["bg"]
_FIG_AXES = _DARK_FIG["axes"]
_FIG_GRID = _DARK_FIG["grid"]
_FIG_TEXT = _DARK_FIG["text"]
_FIG_EDGE = _DARK_FIG["edge"]


def apply_mpl_theme(theme: str = "dark") -> None:
    """Restyle every matplotlib figure for the active side (Light/Dark). Called
    from app.py on each rerun before the page runs, so all fig_* helpers match
    the chosen theme with no per-figure changes."""
    global _FIG_BG, _FIG_AXES, _FIG_GRID, _FIG_TEXT, _FIG_EDGE
    pal = _LIGHT_FIG if theme == "light" else _DARK_FIG
    _FIG_BG, _FIG_AXES = pal["bg"], pal["axes"]
    _FIG_GRID, _FIG_TEXT, _FIG_EDGE = pal["grid"], pal["text"], pal["edge"]
    plt.rcParams.update({
        "figure.facecolor":  _FIG_BG,
        "axes.facecolor":    _FIG_AXES,
        "axes.edgecolor":    _FIG_EDGE,
        "axes.labelcolor":   _FIG_TEXT,
        "xtick.color":       _FIG_TEXT,
        "ytick.color":       _FIG_TEXT,
        "text.color":        _FIG_TEXT,
        "grid.color":        _FIG_GRID,
        "grid.alpha":        0.55,
        "font.size":         9,
        "legend.facecolor":  _FIG_AXES,
        "legend.edgecolor":  _FIG_EDGE,
        "figure.dpi":        110,
    })


apply_mpl_theme("dark")

# ── Shared CSS injected at the top of every page ──────────────────────────── #
PAGE_CSS = """
<style>
/* ═══════════════════════════════════════════════════════════════════════════
   KEYFRAMES — only continuous ambient motion; never entry animations
   (those replay on every rerun and read as flicker).
   ═══════════════════════════════════════════════════════════════════════════ */
@keyframes gradientShift {
    0%,100% { background-position:0% 50%; }
    50%      { background-position:100% 50%; }
}
@keyframes float {
    0%,100% { transform:translateY(0); }
    50%      { transform:translateY(-10px); }
}

/* ═══════════════════════════════════════════════════════════════════════════
   SCROLLBAR
   ═══════════════════════════════════════════════════════════════════════════ */
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:rgba(9,28,78,0.25); border-radius:3px; }
::-webkit-scrollbar-thumb { background:rgba(79,139,249,0.38); border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:rgba(79,139,249,0.65); }

/* ═══════════════════════════════════════════════════════════════════════════
   STREAMLIT HEADER — visually hidden so the sidebar reaches the very top, but
   NOT display:none: the collapsed-sidebar "expand" control lives inside the
   header, and removing the header removed the only way to re-open a closed
   sidebar. visibility:hidden + height:0 hides it and frees the space, while the
   expand control below is flipped back to visible (visibility is overridable on
   children, unlike display).
   ═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stHeader"] {
    visibility: hidden !important;
    height: 0 !important;
    min-height: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}
/* The thin top "decoration" gradient strip is purely cosmetic — hide it. (Do NOT
   hide the toolbar: in this Streamlit build the collapsed-sidebar EXPAND control
   lives in that region, and display:none-ing it left no way to re-open the
   sidebar. The black strip that used to flash on navigation is handled instead by
   the transparent-header rules injected from app.py.) */
[data-testid="stDecoration"] {
    display: none !important;
}

/* Invisible helper host for the dropdown auto-close script: the iframe must
   stay in the DOM (so its script runs) but must NOT occupy a slot in the main
   flex flow — otherwise it adds one vertical-block gap at the top of every page
   and makes the content overflow the viewport by a hair (a faint scroll on
   otherwise-short pages). position:absolute removes it from the flow entirely. */
[class*="st-key-ddac_host"] {
    position: absolute !important;
    width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
    pointer-events: none !important;
}
[class*="st-key-ddac_host"] iframe { height: 0 !important; }

/* Same treatment for Home's equaliser-ramp helper iframe (height=1 otherwise
   shows as a thin white line above the stat strip). */
[class*="st-key-eqramp_host"] {
    position: absolute !important;
    width: 0 !important; height: 0 !important;
    margin: 0 !important; padding: 0 !important;
    overflow: hidden !important; pointer-events: none !important;
}
[class*="st-key-eqramp_host"] iframe { height: 0 !important; }

/* Signal Explorer panel-height equalizer iframe (seqh = Signal Explorer
   Qt Height). Script runs on every render via setTimeout chains. */
[class*="st-key-seqh_host"] {
    position: absolute !important;
    width: 0 !important; height: 0 !important;
    margin: 0 !important; padding: 0 !important;
    overflow: hidden !important; pointer-events: none !important;
}
[class*="st-key-seqh_host"] iframe { height: 0 !important; }

/* Live-settings flags iframe (background mode / reduced motion) — keep it in the
   DOM so its script runs every rerun, but out of the layout flow. */
[class*="st-key-swflags_host"] {
    position: absolute !important;
    width: 0 !important; height: 0 !important;
    margin: 0 !important; padding: 0 !important;
    overflow: hidden !important; pointer-events: none !important;
}
[class*="st-key-swflags_host"] iframe { height: 0 !important; }

/* ═══════════════════════════════════════════════════════════════════════════
   MAIN CONTENT
   — sin animación de entrada: se re-dispararía en cada rerun y provoca el
     "flash" al navegar entre páginas o tocar cualquier widget.
   ═══════════════════════════════════════════════════════════════════════════ */
.main .block-container,
[data-testid="stMainBlockContainer"] {
    padding-top: 0.6rem !important;      /* titles/hero sit higher on the page */
    padding-bottom: 1.4rem !important;   /* no dead scroll past the footer */
}

/* A page-local style block injected via st.markdown still occupies an element
   slot (plus a vertical-block gap), which pushes that page's title lower than
   pages without one — so titles ended up at different heights. Collapse any
   element container whose only child is a style element; its rules still apply
   (a style element works regardless of an ancestor's display:none).
   NOTE: never write the literal closing style tag inside this block — the HTML
   parser would end the whole stylesheet there and dump the rest as text. */
[data-testid="stElementContainer"]:has(> div[data-testid="stMarkdown"] style),
[data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] style) {
    display: none !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   SIDEBAR — fixed, always visible, navigation-first.
   (The previous hover auto-hide collapsed while interacting with dropdown
   popovers — selectbox menus render outside the sidebar, hover was lost and
   the panel slid shut mid-selection. Root cause removed, not patched.)
   ═══════════════════════════════════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,
        #06101F 0%, #09193A 35%, #0B1E48 65%, #07131E 100%) !important;
    border-right: 1px solid rgba(79,139,249,0.12) !important;
    width: 17.5rem !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
    margin-top: 0 !important;
}
[data-testid="stSidebarContent"] {
    padding-top: 0 !important;
}
/* The sidebar opens by default (initial_sidebar_state="expanded"), but stays
   fully collapsible AND re-openable: the in-sidebar collapse button and the
   collapsed-state expand control are always visible and clickable. The expand
   control lives inside the (visibility:hidden) header, so it must be flipped back
   to visible + pointer-events:auto + a high z-index to sit above the canvas. */
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stExpandSidebarButton"] {
    display: inline-flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    z-index: 1000 !important;
}
/* The header is collapsed to height:0, so the "expand" control (shown when the
   sidebar is closed) would otherwise sit jammed against the very top edge. Push
   it well down so it lines up with the page content/title. */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stExpandSidebarButton"] {
    top: 5.5rem !important;
    margin-top: 0.5rem !important;
    /* transform is honoured regardless of how the control is positioned, so it
       reliably pushes the arrow down even if `top` has no effect in this build. */
    transform: translateY(3rem) !important;
}

/* Remove the sidebar resize/drag handle — the 8px-wide `cursor: col-resize`
   strip on the sidebar's inner edge that lets you drag its width. It is a
   styled div with the emotion target class below (build-specific to this
   Streamlit version); hiding it also drops the draggable border affordance. */
section[data-testid="stSidebar"] [class*="eelgd2m3"],
[class*="st-key-"] [class*="eelgd2m3"],
[class*="eelgd2m3"] { display: none !important; }

/* ═══════════════════════════════════════════════════════════════════════════
   SIDEBAR NAV LINKS
   ═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stSidebarNav"]::before {
    content: "DEEPFAKE AUDIO DETECTION";
    display: block;
    font-size: 0.72rem;
    font-weight: 800;
    color: #8AABEF;
    background: linear-gradient(90deg,
        rgba(79,139,249,0.12) 0%, rgba(8,145,178,0.08) 100%);
    padding: 1.25rem 1.1rem 1.05rem;
    margin-bottom: 0.55rem;       /* breathing room before the first nav link */
    letter-spacing: 0.14em;
    text-transform: uppercase;
    border-bottom: 1px solid rgba(79,139,249,0.18);
    white-space: nowrap;
}
/* Extra gap between the section title and the nav list */
[data-testid="stSidebarNav"] ul { padding-top: 0.35rem !important; }
[data-testid="stSidebarNavLink"] {
    border-radius: 0.55rem !important;
    margin: 0.2rem 0.6rem !important;
    padding: 0.6rem 0.95rem !important;
    font-size: 1.02rem !important;       /* larger, more readable nav labels */
    transition: background 0.22s ease, transform 0.22s ease !important;
    position: relative;
    overflow: hidden;
}
/* The label text inside each nav link */
[data-testid="stSidebarNavLink"] span,
[data-testid="stSidebarNavLink"] p { font-size: 1.02rem !important; }
[data-testid="stSidebarNavLink"]::after {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg,
        transparent 0%, rgba(79,139,249,0.07) 50%, transparent 100%);
    transform: translateX(-100%);
    transition: transform 0.5s ease;
}
[data-testid="stSidebarNavLink"]:hover {
    background: rgba(79,139,249,0.12) !important;
    transform: translateX(5px) !important;
}
[data-testid="stSidebarNavLink"]:hover::after {
    transform: translateX(100%);
}
[data-testid="stSidebarNavLink"][aria-current="page"] {
    background: linear-gradient(90deg,
        rgba(79,139,249,0.22) 0%, rgba(79,139,249,0.06) 100%) !important;
    border-left: 3px solid #4F8BF9 !important;
    font-weight: 700 !important;
    box-shadow: 0 0 14px rgba(79,139,249,0.3) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   METRIC CARDS
   ═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stMetric"] {
    background: linear-gradient(135deg,
        rgba(12,25,70,0.65) 0%, rgba(18,38,95,0.45) 100%) !important;
    backdrop-filter: blur(14px) !important;
    border: 1px solid rgba(79,139,249,0.18) !important;
    border-radius: 0.8rem !important;
    padding: 0.9rem 1.1rem !important;
    transition: transform 0.24s ease, box-shadow 0.24s ease !important;
    position: relative;
    overflow: hidden;
}
[data-testid="stMetric"]::after {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg,
        transparent, rgba(79,139,249,0.5), transparent);
}
[data-testid="stMetric"]:hover {
    transform: translateY(-4px) !important;
    box-shadow: 0 10px 32px rgba(79,139,249,0.22),
                0 0 0 1px rgba(79,139,249,0.28) !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    opacity: 0.5 !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #82B1FF, #4FC3F7) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   BUTTONS
   ═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #2952C4 0%, #0891B2 100%) !important;
    border: none !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease !important;
}
[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 22px rgba(79,139,249,0.5) !important;
    filter: brightness(1.1) !important;
}
[data-testid="stBaseButton-secondary"] {
    border-color: rgba(79,139,249,0.35) !important;
    transition: transform 0.2s ease, border-color 0.2s ease,
                box-shadow 0.2s ease !important;
}
[data-testid="stBaseButton-secondary"]:hover {
    transform: translateY(-2px) !important;
    border-color: rgba(79,139,249,0.7) !important;
    box-shadow: 0 4px 16px rgba(79,139,249,0.18) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   TABS
   ═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stTabs"] { margin-top: 0.2rem; }
button[data-baseweb="tab"] {
    transition: color 0.18s ease, background 0.18s ease !important;
    border-radius: 0.45rem 0.45rem 0 0 !important;
    padding: 0.5rem 1.1rem !important;
}
button[data-baseweb="tab"]:hover {
    color: #82B1FF !important;
    background: rgba(79,139,249,0.07) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    font-weight: 700 !important;
    color: #82B1FF !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   EXPANDERS
   ═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stExpander"] {
    border: 1px solid rgba(79,139,249,0.16) !important;
    border-radius: 0.7rem !important;
    transition: box-shadow 0.22s ease, border-color 0.22s ease !important;
    overflow: hidden !important;
}
[data-testid="stExpander"]:hover {
    border-color: rgba(79,139,249,0.32) !important;
    box-shadow: 0 4px 18px rgba(79,139,249,0.1) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   PROGRESS BARS
   ═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #4F8BF9, #00BCD4) !important;
    border-radius: 4px !important;
    transition: width 0.3s ease !important;
}
/* The progress label (e.g. "5/5 · CQCC … done") sat flush against the bar's left
   edge — give it a little breathing room so it never touches the border. */
[data-testid="stProgress"] [data-testid="stMarkdownContainer"] p {
    padding-left: 0.65rem !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   DIVIDER
   ═══════════════════════════════════════════════════════════════════════════ */
hr { border-color: rgba(79,139,249,0.1) !important; }

/* ═══════════════════════════════════════════════════════════════════════════
   GRADIENT ACCENT BAR
   ═══════════════════════════════════════════════════════════════════════════ */
.gradient-bar {
    height: 3px;
    background: linear-gradient(90deg, #4F8BF9, #00BCD4, #9C27B0, #4F8BF9);
    background-size: 300% 300%;
    animation: gradientShift 5s ease infinite;
    border-radius: 2px;
    margin: 0.6rem 0 1.4rem;
}

/* ═══════════════════════════════════════════════════════════════════════════
   HERO BANNER
   ═══════════════════════════════════════════════════════════════════════════ */
.hero-banner {
    background: linear-gradient(135deg,
        #070F2B 0%, #0C2272 28%, #1548C0 55%, #0B7AB0 80%, #065A8A 100%);
    background-size: 300% 300%;
    animation: gradientShift 12s ease infinite;
    padding: 1.9rem 3rem 1.8rem;          /* shorter hero — less wasted vertical space */
    border-radius: 1.1rem;
    color: #E8EDF8;
    box-shadow: 0 16px 60px rgba(7,15,43,0.7),
                0 0 0 1px rgba(79,139,249,0.18),
                inset 0 1px 0 rgba(255,255,255,0.06);
    position: relative;
    overflow: hidden;
}
/* Floating orbs */
.hero-banner::before {
    content: "";
    position: absolute;
    top: -50%; right: -8%;
    width: 520px; height: 520px;
    background: radial-gradient(circle,
        rgba(79,139,249,0.14) 0%, transparent 65%);
    pointer-events: none;
    animation: float 9s ease-in-out infinite;
}
.hero-banner::after {
    content: "";
    position: absolute;
    bottom: -45%; left: 25%;
    width: 380px; height: 380px;
    background: radial-gradient(circle,
        rgba(8,145,178,0.1) 0%, transparent 60%);
    pointer-events: none;
    animation: float 14s ease-in-out infinite reverse;
}
.hero-banner h1 {
    font-size: 3.1rem;
    font-weight: 850;
    margin: 0 0 0.5rem;
    line-height: 1.08;
    letter-spacing: -0.028em;
    text-shadow: 0 3px 30px rgba(79,139,249,0.45);
}
.hero-banner p {
    font-size: 1rem;
    margin: 0;
    opacity: 0.82;
    line-height: 1.68;
    max-width: none;          /* span the full hero width, no early wrap */
    padding-right: 1rem;
}
.hero-author {
    margin-top: 1.15rem;
    padding-top: 0.85rem;
    border-top: 1px solid rgba(255,255,255,0.1);
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.55rem 0.9rem;
    font-size: 0.82rem;
    letter-spacing: 0.03em;
}
.hero-author .ha-name { font-weight: 700; color: #B9CCF2; }
.hero-author a {
    color: #9BB8F4 !important;
    text-decoration: none;
    font-weight: 600;
    border: 1px solid rgba(155,184,244,0.35);
    border-radius: 1.4rem;
    padding: 0.12rem 0.7rem;
    transition: background 0.2s ease, border-color 0.2s ease,
                transform 0.2s ease, color 0.2s ease;
}
.hero-author a:hover {
    background: rgba(155,184,244,0.16);
    border-color: rgba(155,184,244,0.7);
    color: #E8EDF8 !important;
    transform: translateY(-1px);
}

/* ═══════════════════════════════════════════════════════════════════════════
   HERO WAVE — two audio sine-waves travelling across the WHOLE header, behind
   the text (continuous scroll, no entry flicker).
   ═══════════════════════════════════════════════════════════════════════════ */
@keyframes heroWaveScroll { from { transform: translateX(0); } to { transform: translateX(-50%); } }
.hero-wave {
    position: absolute;
    inset: 0;
    overflow: hidden;
    z-index: 0;
    pointer-events: none;
}
.hero-wave .hw {
    position: absolute;
    top: 0; left: 0;
    width: 200%; height: 100%;
    will-change: transform;
}
.hero-wave .hw path { fill: none; vector-effect: non-scaling-stroke; stroke-width: 2; }
.hero-wave .hw1 { animation: heroWaveScroll 9s linear infinite; }
.hero-wave .hw1 path { stroke: rgba(130,177,255,0.42); }
.hero-wave .hw2 { animation: heroWaveScroll 15s linear infinite; opacity: 0.85; }
.hero-wave .hw2 path { stroke: rgba(79,195,247,0.26); }
/* Keep all hero text/badges/author above the wave. */
.hero-banner > div:not(.hero-wave),
.hero-banner > h1,
.hero-banner > p { position: relative; z-index: 1; }
.hero-meta {
    position: relative; z-index: 1;
    margin-top: 1rem;
    font-size: 0.8rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    color: #9BB8F4;
    opacity: 0.92;
}
.hero-badge {
    display: inline-block;
    background: rgba(79,139,249,0.16);
    border: 1px solid rgba(79,139,249,0.36);
    border-radius: 2rem;
    padding: 0.2rem 0.75rem;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    margin: 0 0.28rem 0.7rem 0;
    color: #A8C4FF;
    backdrop-filter: blur(4px);
    transition: background 0.22s ease, transform 0.22s ease,
                box-shadow 0.22s ease;
}
.hero-badge:hover {
    background: rgba(79,139,249,0.28);
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(79,139,249,0.25);
}

/* ═══════════════════════════════════════════════════════════════════════════
   PANEL CARDS
   — equal height via min-height (NOT a global column-flex hack: stretching
     every column's first child distorted unrelated layouts and caused
     overlapping controls).
   ═══════════════════════════════════════════════════════════════════════════ */
.panel-card {
    min-height: 16.5rem;
    margin-bottom: 0.65rem;
    background: linear-gradient(145deg,
        rgba(10,22,62,0.62) 0%, rgba(14,28,75,0.38) 100%);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(79,139,249,0.18);
    border-radius: 1rem;
    padding: 1.6rem 1.7rem 1.8rem;
    flex: 1;
    display: flex;
    flex-direction: column;
    position: relative;
    overflow: hidden;
    transition: transform 0.3s cubic-bezier(0.34,1.56,0.64,1),
                box-shadow 0.3s ease,
                border-color 0.3s ease;
}
/* Top gradient line */
.panel-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg,
        transparent 0%, #4F8BF9 40%, #00BCD4 60%, transparent 100%);
    opacity: 0;
    transition: opacity 0.3s ease;
}
/* Shimmer sweep on hover */
.panel-card::after {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(105deg,
        transparent 35%, rgba(79,139,249,0.06) 50%, transparent 65%);
    transform: translateX(-100%) skewX(-10deg);
    transition: none;
}
.panel-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 18px 44px rgba(7,15,43,0.6),
                0 0 0 1px rgba(79,139,249,0.35),
                0 0 32px rgba(79,139,249,0.08);
    border-color: rgba(79,139,249,0.35);
}
.panel-card:hover::before { opacity: 1; }
.panel-card:hover::after {
    transform: translateX(200%) skewX(-10deg);
    transition: transform 0.65s ease;
}
.panel-card h4 {
    margin: 0 0 0.9rem;
    font-size: 1.1rem;
    font-weight: 700;
    background: linear-gradient(135deg, #82B1FF 0%, #4FC3F7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.panel-card ul { padding-left: 1.1rem; margin: 0; flex: 1; }
.panel-card li {
    margin-bottom: 0.4rem;
    font-size: 0.87rem;
    opacity: 0.72;
    line-height: 1.45;
    transition: opacity 0.2s ease, transform 0.2s ease;
}
.panel-card:hover li { opacity: 0.88; }

/* ═══════════════════════════════════════════════════════════════════════════
   EMPTY STATE — sober glyph in a dashed ring instead of an emoji
   ═══════════════════════════════════════════════════════════════════════════ */
.empty-state {
    text-align: center;
    padding: 2.4rem 2rem 2.6rem;
    background: rgba(79,139,249,0.04);
    border: 1px dashed rgba(79,139,249,0.22);
    border-radius: 1rem;
    margin: 0.8rem 0;
}
.empty-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 64px; height: 64px;
    border: 1px dashed rgba(79,139,249,0.45);
    border-radius: 50%;
    font-size: 1.6rem;
    font-weight: 300;
    color: #5E7FD4;
    margin-bottom: 1rem;
    animation: float 5s ease-in-out infinite;
}

/* ═══════════════════════════════════════════════════════════════════════════
   PILLS & SEGMENTED CONTROL  — modern chip-style selectors.
   Streamlit renders segmented-control buttons FLUSH against each other;
   chip styling (rounded + bordered segments) makes them overlap. Forcing a
   real flex gap on the group turns every option into a separate chip.
   ═══════════════════════════════════════════════════════════════════════════ */
/* The real flex row of options is baseweb's button-group INSIDE stButtonGroup
   (stButtonGroup itself also wraps the widget label, so spacing it does
   nothing to the buttons). Space the options on the button-group and reset the
   negative border-collapse margins baseweb puts between adjacent segments. */
[data-testid="stButtonGroup"] [data-baseweb="button-group"] {
    display: inline-flex !important;
    flex-wrap: wrap !important;
    gap: 0.7rem !important;
}
[data-testid="stButtonGroup"] [data-baseweb="button-group"] > * {
    margin: 0 !important;
}
[data-testid="stBaseButton-pills"],
[data-testid="stBaseButton-segmented_control"] {
    background: rgba(79,139,249,0.06) !important;
    border: 1px solid rgba(79,139,249,0.22) !important;
    color: #AFC3E8 !important;
    border-radius: 2rem !important;
    transition: background 0.18s ease, border-color 0.18s ease,
                transform 0.18s ease, box-shadow 0.18s ease !important;
}
[data-testid="stBaseButton-pills"]:hover,
[data-testid="stBaseButton-segmented_control"]:hover {
    border-color: rgba(79,139,249,0.55) !important;
    background: rgba(79,139,249,0.14) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stBaseButton-pillsActive"],
[data-testid="stBaseButton-segmented_controlActive"] {
    background: linear-gradient(135deg, #2952C4 0%, #0891B2 100%) !important;
    border: 1px solid rgba(130,177,255,0.55) !important;
    color: #FFFFFF !important;
    border-radius: 2rem !important;
    font-weight: 600 !important;
    box-shadow: 0 3px 14px rgba(41,82,196,0.45) !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   BORDERED CONTAINERS  (st.container(border=True)) — glass panels
   ═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"] {
    gap: 0.65rem;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(150deg,
        rgba(10,22,62,0.45) 0%, rgba(14,28,75,0.22) 100%);
    border: 1px solid rgba(79,139,249,0.16) !important;
    border-radius: 0.9rem !important;
    transition: border-color 0.25s ease, box-shadow 0.25s ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(79,139,249,0.3) !important;
    box-shadow: 0 6px 24px rgba(7,15,43,0.35);
}

/* ═══════════════════════════════════════════════════════════════════════════
   INFO CARDS  — generic editorial card (CNN Learning, Run Experiment, …)
   ═══════════════════════════════════════════════════════════════════════════ */
.info-card {
    background: linear-gradient(145deg,
        rgba(8,18,52,0.55) 0%, rgba(12,24,65,0.32) 100%);
    border: 1px solid rgba(79,139,249,0.14);
    border-left: 3px solid rgba(79,139,249,0.45);
    border-radius: 0.75rem;
    padding: 0.85rem 1.05rem 0.9rem;
    margin-bottom: 0.7rem;
    transition: transform 0.22s ease, border-color 0.22s ease,
                box-shadow 0.22s ease;
}
.info-card:hover {
    transform: translateX(3px);
    border-left-color: #4F8BF9;
    box-shadow: 0 6px 20px rgba(9,28,78,0.4);
}
.info-card .ic-title {
    font-size: 0.84rem;
    font-weight: 700;
    color: #82B1FF;
    margin-bottom: 0.3rem;
    letter-spacing: 0.01em;
}
.info-card .ic-tag {
    display: inline-block;
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #B39DDB;
    background: rgba(156,39,176,0.14);
    border: 1px solid rgba(156,39,176,0.3);
    border-radius: 1rem;
    padding: 0.06rem 0.55rem;
    margin-left: 0.4rem;
    vertical-align: 1px;
}
.info-card .ic-body {
    font-size: 0.8rem;
    line-height: 1.6;
    opacity: 0.74;
    margin: 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
   INFORMATIONAL CARDS — normal arrow cursor (not the text I-beam) over any
   non-interactive info surface. cursor is inherited, so children get it too;
   real links/buttons inside keep their own pointer cursor.
   ═══════════════════════════════════════════════════════════════════════════ */
.info-card, .panel-card, .corpus-card, .method-card, .pipe-step, .pipe-arrow,
.stat-strip, .stat-cell, .cfg-chip, .chip-row, .side-status, .dist-cards,
.dist-row, .mini-note, .sec-head, .sec-sub, .best-banner, .empty-state,
.hero-badge, .rep-card, .rep-grid,
.hero-banner h1, .hero-banner p, .hero-overline, .hero-author .ha-name {
    cursor: default;
}
.hero-author a { cursor: pointer; }

/* ═══════════════════════════════════════════════════════════════════════════
   REPRESENTATION CARDS — pretty grid for the Signal Explorer "what does each
   representation show?" expander.
   ═══════════════════════════════════════════════════════════════════════════ */
.rep-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.55rem;
    margin: 0.2rem 0 0.55rem;   /* bottom gap so the last cards don't touch the expander edge */
}
.rep-card {
    background: linear-gradient(145deg,
        rgba(8,18,52,0.5) 0%, rgba(12,24,65,0.3) 100%);
    border: 1px solid rgba(79,139,249,0.14);
    border-left: 3px solid #4F8BF9;
    border-radius: 0.6rem;
    padding: 0.6rem 0.85rem 0.65rem;
    transition: transform 0.2s ease, border-color 0.2s ease,
                box-shadow 0.2s ease;
}
.rep-card:hover {
    transform: translateX(3px);
    border-left-color: #00BCD4;
    box-shadow: 0 5px 16px rgba(9,28,78,0.35);
}
.rep-card .rep-name {
    font-weight: 700;
    color: #82B1FF;
    font-size: 0.84rem;
    margin-bottom: 0.2rem;
    letter-spacing: 0.01em;
}
.rep-card .rep-desc {
    font-size: 0.76rem;
    opacity: 0.74;
    line-height: 1.45;
}

/* ═══════════════════════════════════════════════════════════════════════════
   PIPELINE STEPS  — Run Experiment header strip
   ═══════════════════════════════════════════════════════════════════════════ */
.pipe-row {
    display: grid;
    grid-template-columns: 1fr auto 1fr auto 1fr;
    gap: 0.6rem;
    align-items: stretch;
    margin: 0.6rem 0 0.9rem;
}
.pipe-step {
    background: linear-gradient(145deg,
        rgba(10,22,62,0.6) 0%, rgba(14,28,75,0.35) 100%);
    border: 1px solid rgba(79,139,249,0.18);
    border-radius: 0.85rem;
    padding: 0.85rem 1.05rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.24s ease, border-color 0.24s ease,
                box-shadow 0.24s ease;
}
.pipe-step::after {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #4F8BF9, #00BCD4);
    opacity: 0.55;
}
.pipe-step:hover {
    transform: translateY(-3px);
    border-color: rgba(79,139,249,0.4);
    box-shadow: 0 10px 28px rgba(9,28,78,0.45);
}
.pipe-step .ps-step {
    font-size: 0.62rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #6E87C9;
    margin-bottom: 0.18rem;
}
.pipe-step .ps-value {
    font-size: 0.98rem;
    font-weight: 700;
    background: linear-gradient(135deg, #82B1FF, #4FC3F7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.3;
}
.pipe-step .ps-sub {
    font-size: 0.72rem;
    opacity: 0.58;
    margin-top: 0.15rem;
    line-height: 1.45;
}
.pipe-arrow {
    align-self: center;
    color: rgba(79,139,249,0.55);
    font-size: 1.25rem;
    font-weight: 700;
    padding: 0 0.1rem;
}

/* ═══════════════════════════════════════════════════════════════════════════
   CONFIG CHIPS  — small key:value summary badges
   ═══════════════════════════════════════════════════════════════════════════ */
.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin: 0.15rem 0 0.6rem;
}
.cfg-chip {
    display: inline-flex;
    align-items: baseline;
    gap: 0.35rem;
    background: rgba(79,139,249,0.07);
    border: 1px solid rgba(79,139,249,0.2);
    border-radius: 2rem;
    padding: 0.18rem 0.75rem;
    font-size: 0.73rem;
    color: #AFC3E8;
}
.cfg-chip b { color: #82B1FF; font-weight: 700; }

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION LABEL  — small uppercase heading used above control groups
   ═══════════════════════════════════════════════════════════════════════════ */
.section-label {
    font-size: 0.68rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #6E87C9;
    margin: 0.1rem 0 0.35rem;
}

/* ═══════════════════════════════════════════════════════════════════════════
   EDITORIAL SECTION HEADER — index number + title + fading rule
   ═══════════════════════════════════════════════════════════════════════════ */
.sec-head {
    display: flex;
    align-items: baseline;
    gap: 0.85rem;
    margin: 0.6rem 0 0.35rem;
}
.sec-head .sh-num {
    font-size: 0.78rem;
    font-weight: 800;
    color: #4F8BF9;
    letter-spacing: 0.1em;
    font-variant-numeric: tabular-nums;
}
.sec-head .sh-title {
    margin: 0;
    font-size: 1.32rem;
    font-weight: 750;
    letter-spacing: -0.015em;
    color: #E8EDF8;
    line-height: 1.2;
}
.sec-head .sh-rule {
    /* Fill ALL the remaining width up to the content edge (long accent that
       adapts to the page). The sweep uses a percentage background-position, so
       the highlight sits at the same FRACTION of every rule at any instant —
       they stay coordinated even though each fills a slightly different width. */
    flex: 1 1 auto;
    min-width: 0;
    height: 1px;
    align-self: center;
    background: linear-gradient(90deg, rgba(79,139,249,0.45), transparent);
}
.sec-sub {
    font-size: 0.82rem;
    opacity: 0.55;
    margin: 0 0 0.9rem;
}

/* ═══════════════════════════════════════════════════════════════════════════
   PAGE LINK — ghost-button navigation cards (Home panel links)
   ═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stPageLink"] a {
    border: 1px solid rgba(79,139,249,0.3) !important;
    border-radius: 0.6rem !important;
    padding: 0.42rem 0.9rem !important;
    transition: background 0.2s ease, border-color 0.2s ease,
                transform 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="stPageLink"] a:hover {
    background: rgba(79,139,249,0.14) !important;
    border-color: rgba(79,139,249,0.6) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(79,139,249,0.2) !important;
}
[data-testid="stPageLink"] a p { font-weight: 600 !important; }

/* ═══════════════════════════════════════════════════════════════════════════
   ALIGN-RIGHT UTILITY — wrap a widget in st.container(key="alignr_…").
   Covers button groups, buttons and download buttons so right-edge
   alignment reads as intentional, not floating.
   ═══════════════════════════════════════════════════════════════════════════ */
[class*="st-key-alignr"] [data-testid="stButtonGroup"],
[class*="st-key-alignr"] [data-testid="stButton"],
[class*="st-key-alignr"] [data-testid="stDownloadButton"] {
    display: flex !important;
    justify-content: flex-end !important;
    width: 100% !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   DROPDOWN-ONLY SELECTBOXES — wrap in st.container(key="nosearch_…") to block
   type-to-filter so the control behaves as a pure dropdown (still clickable).
   ═══════════════════════════════════════════════════════════════════════════ */
[class*="st-key-nosearch"] [data-baseweb="select"] input {
    pointer-events: none !important;
    caret-color: transparent !important;
}
[class*="st-key-nosearch"] [data-baseweb="select"] > div {
    cursor: pointer !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   LABEL GAP — extra space between a segmented control's own label and its
   buttons. Wrap the control in st.container(key="lblgap_…"). Only affects
   visible labels; collapsed-label controls render no stWidgetLabel.
   ═══════════════════════════════════════════════════════════════════════════ */
[class*="st-key-lblgap"] [data-testid="stWidgetLabel"] {
    margin-bottom: 0.6rem !important;
}

/* COMPACT "Evaluation" group — "Train on" (fixed) + "Evaluate on" + "Score on"
   laid out side by side in a fit-content blue frame (short, not full width). */
[class*="st-key-evalgrp_"] {
    display: flex !important; flex-direction: row; flex-wrap: wrap;
    column-gap: 1.2rem; row-gap: 0.2rem; align-items: flex-start;
    width: fit-content; max-width: 100%;
    border: 1px solid rgba(79,139,249,0.2);
    border-left: 3px solid rgba(79,139,249,0.45);
    border-radius: 0.55rem;
    padding: 0.35rem 0.7rem 0.45rem;
    background: rgba(79,139,249,0.05);
}
[class*="st-key-evalgrp_"] > [data-testid="stElementContainer"] { width: auto !important; }
/* Vertical divider between every pair of controls (Train on | Evaluate on | Score on). */
[class*="st-key-evalgrp_"] > [data-testid="stElementContainer"]:not(:first-child) {
    border-left: 1px solid rgba(79,139,249,0.28);
    padding-left: 1.1rem; margin-left: 0.1rem;
}
/* The fixed "Score on · Eval" shown for the eval-only 2021 corpora. */
.eval-fixed { display: flex; flex-direction: column; gap: 0.32rem; padding-top: 0.05rem; }
.eval-fixed .ef-lbl { font-size: 0.7rem; font-weight: 700; opacity: 0.75;
    text-transform: uppercase; letter-spacing: 0.05em; }
.eval-fixed .ef-val { font-size: 0.82rem; font-weight: 700; color: #C9D7F5;
    background: rgba(79,139,249,0.12); border: 1px solid rgba(79,139,249,0.25);
    border-radius: 0.45rem; padding: 0.18rem 0.7rem; width: fit-content; }
[class*="st-key-evalgrp_"] [data-testid="stWidgetLabel"] {
    margin-bottom: 0.2rem !important;
}
[class*="st-key-evalgrp_"] [data-testid="stWidgetLabel"] p {
    font-size: 0.7rem !important; font-weight: 700; opacity: 0.75;
    text-transform: uppercase; letter-spacing: 0.05em;
}
[class*="st-key-evalgrp_"] [data-baseweb="button-group"] button {
    padding-top: 0.18rem !important; padding-bottom: 0.18rem !important;
    font-size: 0.78rem !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   HERO REFINEMENTS — overline kicker, flush-top placement
   ═══════════════════════════════════════════════════════════════════════════ */
.hero-overline {
    font-size: 0.7rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.22em;
    color: #9BB8F4;
    margin-bottom: 0.55rem;
}
.hero-banner { margin-top: 0; }

/* ═══════════════════════════════════════════════════════════════════════════
   STAT STRIP — large editorial numerals under the hero
   ═══════════════════════════════════════════════════════════════════════════ */
.stat-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: rgba(79,139,249,0.16);
    border: 1px solid rgba(79,139,249,0.16);
    border-radius: 0.9rem;
    overflow: hidden;
    margin: 1rem 0 0.4rem;
}
.stat-cell {
    background: #0D1426;
    padding: 1rem 1.3rem 0.95rem;
    transition: background 0.22s ease;
}
.stat-cell:hover { background: #111A33; }
.stat-cell .st-num {
    font-size: 1.55rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #82B1FF, #4FC3F7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    font-variant-numeric: tabular-nums;
}
.stat-cell .st-lbl {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    opacity: 0.5;
    margin-top: 0.18rem;
    font-weight: 600;
}

/* ═══════════════════════════════════════════════════════════════════════════
   STATUS DOTS — sober availability indicators (sidebar + cards)
   ═══════════════════════════════════════════════════════════════════════════ */
.dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    margin-right: 0.45rem;
    vertical-align: 1px;
}
.dot-ok  { background: #66BB6A; box-shadow: 0 0 6px rgba(102,187,106,0.6); }
.dot-err { background: #EF5350; box-shadow: 0 0 6px rgba(239,83,80,0.5); }
.dot-na  { background: #5C6B8A; }
.side-status {
    border: 1px solid rgba(79,139,249,0.14);
    border-radius: 0.7rem;
    background: rgba(79,139,249,0.05);
    padding: 0.7rem 0.9rem 0.65rem;
    margin-bottom: 0.85rem;        /* air between stacked sidebar panels */
    font-size: 0.86rem;            /* larger, more readable sidebar panels */
    line-height: 1.9;
    color: #AFC3E8;
}
.side-status .ss-title {
    font-size: 0.72rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #6E87C9;
    margin-bottom: 0.25rem;
}
.side-status .ss-row {
    display: flex;
    justify-content: space-between;
    gap: 0.6rem;
    line-height: 1.75;
}
.side-status .ss-row .ss-k { opacity: 0.6; white-space: nowrap; }
.side-status .ss-row .ss-v {
    color: #C9D7F5;
    font-weight: 600;
    text-align: right;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 11rem;
}

/* ═══════════════════════════════════════════════════════════════════════════
   MINI NOTE — fixed-height inline notice. Always rendered (info or warn
   variant) so toggling options never resizes the surrounding panel.
   ═══════════════════════════════════════════════════════════════════════════ */
.mini-note {
    display: flex;
    align-items: center;          /* text vertically centred against the bar */
    gap: 0.55rem;
    min-height: 2.2em;
    font-size: 0.74rem;
    line-height: 1.45;
    color: #8FA3CE;
    margin: 0.1rem 0 0.55rem;     /* hugs the control above, breathes below */
}
.mini-note::before {
    content: "";
    flex: 0 0 3px;
    align-self: stretch;
    border-radius: 2px;
    background: rgba(79,139,249,0.45);
}
.mini-note.warn { color: #D9BC8A; }
.mini-note.warn::before { background: rgba(255,179,0,0.55); }

/* ═══════════════════════════════════════════════════════════════════════════
   APP FOOTER — closed page ending: top rule, colophon, no trailing scroll
   ═══════════════════════════════════════════════════════════════════════════ */
.app-footer {
    margin-top: 2.2rem;
    border-top: 1px solid rgba(79,139,249,0.16);
    padding: 1.1rem 0 0;
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: baseline;
    gap: 0.4rem 1.5rem;
    font-size: 0.74rem;
    color: #7487B0;
}
.app-footer .af-left  { font-weight: 700; letter-spacing: 0.04em; color: #8FA3CE; }
.app-footer .af-right { opacity: 0.8; }

/* Pin the footer to the very bottom of the viewport on pages that have one
   (only when content is shorter than the screen). Scoped via :has(.app-footer)
   so it never affects the tool pages. NOTE: the page's elements live inside a
   stVerticalBlock *inside* the block-container, so the flex column + auto
   margin must be applied there (the footer is the last element of THAT block). */
[data-testid="stMainBlockContainer"]:has(.app-footer) {
    min-height: calc(100vh - 2rem);
    display: flex;
    flex-direction: column;
}
[data-testid="stMainBlockContainer"]:has(.app-footer) > [data-testid="stVerticalBlock"] {
    flex: 1 1 auto;
    display: flex;
    flex-direction: column;
}
[data-testid="stMainBlockContainer"]:has(.app-footer) > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:last-child {
    margin-top: auto;
}

/* ═══════════════════════════════════════════════════════════════════════════
   MOTION & DYNAMISM — cohesive, slow, CONTINUOUS ambient motion only (never
   entry animations: those replay on every rerun and read as flicker). Overrides
   a few earlier static rules to make the signal/audio theme feel alive.
   ═══════════════════════════════════════════════════════════════════════════ */

/* Animated particle-network ("spider-web") background, drawn on a <canvas>
   that the helper iframe injects behind everything. The dark base colour moves
   to the page root so the canvas (z-index:-1) shows through the now-transparent
   app containers; the sidebar keeps its own opaque background. */
html, body { background-color: #0E1117 !important; }
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"] { background: transparent !important; }
#bgWeb {
    position: fixed;
    inset: 0;
    width: 100vw;
    height: 100vh;
    z-index: -1;
    pointer-events: none;
}

/* A bright highlight that TRAVELS along thin accent rules — clearly moving,
   unlike a subtle hue shift on a 1-3px bar. */
@keyframes sweepX { 0% { background-position: -45% 0; } 100% { background-position: 145% 0; } }
.gradient-bar {
    height: 3px; border-radius: 2px; margin: 0.6rem 0 1.4rem;
    background-color: rgba(79,139,249,0.18) !important;
    background-image: linear-gradient(90deg,
        rgba(0,229,255,0) 0%, #00E5FF 45%, #4FC3F7 55%, rgba(0,229,255,0) 100%) !important;
    background-size: 36% 100% !important;
    background-repeat: no-repeat !important;
    animation: sweepX 2.8s linear infinite !important;
}
.sec-head .sh-rule {
    background-color: rgba(79,139,249,0.14) !important;
    background-image: linear-gradient(90deg,
        rgba(79,195,247,0) 0%, #4FC3F7 50%, rgba(79,195,247,0) 100%) !important;
    background-size: 45% 100% !important;
    background-repeat: no-repeat !important;
    animation: sweepX 5s linear infinite !important;
}

/* Headline numerals breathe with a slow flowing gradient. */
[data-testid="stMetricValue"],
.stat-cell .st-num,
.pipe-step .ps-value {
    background-size: 220% 220% !important;
    animation: gradientShift 7s ease infinite;
}

/* Metric cards: a permanent (not hover-only) faint flowing top accent. */
[data-testid="stMetric"]::after {
    background: linear-gradient(90deg,
        transparent, rgba(79,139,249,0.55), rgba(0,188,212,0.55), transparent) !important;
    background-size: 220% 100% !important;
    animation: gradientShift 6s ease infinite;
}

/* Panel cards: reveal a softly flowing top line even at rest (subtle). */
.panel-card::before {
    opacity: 0.45;
    background: linear-gradient(90deg,
        transparent 0%, #4F8BF9 40%, #00BCD4 60%, transparent 100%);
    background-size: 220% 100%;
    animation: gradientShift 6s ease infinite;
}
.panel-card:hover::before { opacity: 1; }

/* ── Global background-job banner — PINNED to the bottom of the sidebar ────── */
@keyframes obGlow { 0%,100% { box-shadow: 0 0 0 1px rgba(179,136,255,0.30),
                                          0 0 16px rgba(156,39,176,0.20); }
                    50%      { box-shadow: 0 0 0 1px rgba(179,136,255,0.6),
                                          0 0 30px rgba(156,39,176,0.42); } }
@keyframes obDot  { 0%,100% { opacity: 0.35; transform: scale(0.8); }
                    50%      { opacity: 1;    transform: scale(1.15); } }
/* The sidebar has a fixed 17.5rem width, so pin the banner there near the
   bottom. It is position:fixed (does NOT add to the content height), so it never
   forces the sidebar to scroll; the page's own panels are short enough to sit
   above it. */
[class*="st-key-opbanner"], [class*="st-key-opcta"] {
    position: fixed; bottom: 3rem; left: 0; width: 17.5rem; z-index: 60;
    padding: 0 0.9rem;
}
.op-cta-head {
    font-size: 0.72rem; line-height: 1.4; color: #9FB6E0;
    margin-bottom: 0.45rem; opacity: 0.85;
}
/* The whole running banner is a click target: an invisible button overlays it
   and jumps to the page where the job is running. */
[class*="st-key-opbanner"] .op-banner { cursor: pointer; }
[class*="st-key-opbanner"] [data-testid="stElementContainer"]:last-child {
    position: absolute; inset: 0; margin: 0 !important; z-index: 5;
}
[class*="st-key-opbanner"] [data-testid="stButton"],
[class*="st-key-opbanner"] [data-testid="stButton"] button {
    width: 100% !important; height: 100% !important; margin: 0 !important;
}
[class*="st-key-opbanner"] [data-testid="stButton"] button {
    opacity: 0 !important; min-height: 0 !important; border: none !important;
    padding: 0 !important;
}
.op-banner .ob-go { margin-left: auto; font-size: 0.66rem; color: #C9A6FF;
    font-weight: 700; letter-spacing: 0.04em; white-space: nowrap; flex: 0 0 auto; }
.op-banner .ob-head { flex-wrap: nowrap; }
.op-banner .ob-head > span:nth-child(3) {
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.op-banner {
    background: linear-gradient(150deg, rgba(74,20,110,0.62), rgba(34,16,74,0.6));
    border: 1px solid rgba(179,136,255,0.42); border-radius: 0.8rem;
    padding: 0.85rem 0.95rem; animation: obGlow 2.6s ease-in-out infinite;
    backdrop-filter: blur(3px);
}
.op-banner .ob-head {
    display: flex; align-items: center; gap: 0.45rem;
    font-size: 0.86rem; font-weight: 750; color: #E6D6FF; margin-bottom: 0.5rem;
}
.op-banner .ob-ic { display: inline-flex; color: #C9A6FF; }
.op-banner .ob-dot {
    width: 9px; height: 9px; border-radius: 50%; background: #C77DFF;
    box-shadow: 0 0 8px #C77DFF; animation: obDot 1.1s ease-in-out infinite; flex: 0 0 auto;
}
.op-banner .ob-sub { font-size: 0.74rem; color: #C7B6E6; margin-bottom: 0.45rem; }
.op-banner .ob-track {
    height: 6px; border-radius: 4px; background: rgba(255,255,255,0.08); overflow: hidden;
}
.op-banner .ob-track span {
    display: block; height: 100%; border-radius: 4px;
    background: linear-gradient(90deg, #B388FF, #7C4DFF);
    transition: width 0.4s ease;
}
.op-banner .ob-pct { font-size: 0.7rem; color: #A98FD0; margin-top: 0.35rem; }
.op-banner .ob-note { font-size: 0.68rem; color: #8A78AE; margin-top: 0.5rem; line-height: 1.4; }

/* ═══════════════════════════════════════════════════════════════════════════
   RESPONSIVE — tablet & phone. The hero title scales fluidly; dense custom grids
   collapse; and the sidebar collapse control (hidden on desktop) comes back so
   the sidebar can be opened/closed on touch devices.
   ═══════════════════════════════════════════════════════════════════════════ */
.hero-banner h1 { font-size: clamp(1.85rem, 5.2vw, 3.1rem); }

@media (max-width: 1024px) {
    .stat-strip { grid-template-columns: repeat(2, 1fr); }
    .pipe-row   { grid-template-columns: 1fr; gap: 0.5rem; }
    .pipe-arrow { display: none; }
    /* Tablets auto-collapse the sidebar — bring the open/close control back so it
       is recoverable (it is hidden on the desktop always-open layout). */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"] { display: inline-flex !important; }
    .set-grid   { grid-template-columns: 1fr !important; }
}
@media (max-width: 680px) {
    [data-testid="stMainBlockContainer"] {
        padding-left: 0.7rem !important; padding-right: 0.7rem !important;
    }
    .hero-banner   { padding: 1.25rem 1.25rem 1.35rem; }
    .hero-overline { font-size: 0.6rem; letter-spacing: 0.15em; }
    .hero-banner p { font-size: 0.9rem; line-height: 1.55; }
    .hero-author   { gap: 0.4rem 0.6rem; font-size: 0.76rem; }
    .stat-strip    { grid-template-columns: 1fr 1fr; }
    .stat-cell     { padding: 0.8rem 0.9rem 0.75rem; }
    .stat-cell .st-num { font-size: 1.3rem; }
    .rep-grid      { grid-template-columns: 1fr; }
    .panel-card    { min-height: auto; }
    .mode-card     { height: auto !important; min-height: 11rem; }
    .corpus-card, .method-card { height: auto; }
    .hcw-step      { grid-template-columns: 44px 1fr; gap: 0.7rem; }
    section[data-testid="stSidebar"] { width: 15rem !important; }
    /* Re-enable the sidebar open/close affordance on small screens. */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"] { display: inline-flex !important; }
}
</style>
"""


# ===========================================================================
# Theme — Light Side / Dark Side
# ===========================================================================
# The app ships a fully-styled Dark Side (the default — examiners see exactly the
# original look). The Light Side is an accessibility light theme generated from
# the SAME stylesheet at build time: the dark colour literals are swapped for
# light ones via _LIGHT_MAP, so the dark path stays byte-identical and there is a
# single source of truth for the layout. theme_mode() reads the user's choice
# from session_state; app.py injects the right palette on every rerun.

def theme_mode() -> str:
    """Active UI side: 'light' (Light Side) or 'dark' (Dark Side, default).
    Driven by the sidebar toggle (session key 'sw_theme')."""
    return "light" if st.session_state.get("sw_theme") == "light" else "dark"


# Dark colour literal -> Light colour literal. Only chrome flips; the data-
# semantic colours (bonafide blue, spoof red, status green/gold) are preserved so
# charts and badges stay honest. The hero banner keeps its deep-blue identity in
# both sides and is restored by _LIGHT_PATCH. Order is irrelevant (no key is a
# substring of another, and no replacement re-introduces a key).
_LIGHT_MAP = {
    # ── solid page / cell surfaces ──────────────────────────────────────────
    # A deeper blue-grey canvas (the previous near-white made cards and accents
    # vanish — too low-contrast). Cards stay white, so they now read as elevated.
    "#0E1117": "#D5DEEE",          # page background (html, body)
    "#0D1426": "#FFFFFF",          # stat-cell
    "#111A33": "#E4EBF7",          # stat-cell hover
    # ── sidebar gradient ────────────────────────────────────────────────────
    "#06101F": "#E8EEF8", "#09193A": "#DCE7F6",
    "#0B1E48": "#CFDDF1", "#07131E": "#E4EBF7",
    # ── card surfaces (translucent navy -> translucent white) ───────────────
    "rgba(12,25,70,0.65)": "rgba(255,255,255,0.9)",
    "rgba(10,22,62,0.65)": "rgba(255,255,255,0.9)",
    "rgba(10,22,62,0.62)": "rgba(255,255,255,0.88)",
    "rgba(10,22,62,0.6)":  "rgba(255,255,255,0.88)",
    "rgba(10,22,62,0.55)": "rgba(255,255,255,0.86)",
    "rgba(10,22,62,0.5)":  "rgba(255,255,255,0.84)",
    "rgba(10,22,62,0.45)": "rgba(255,255,255,0.8)",
    "rgba(8,18,52,0.55)":  "rgba(255,255,255,0.88)",
    "rgba(8,18,52,0.5)":   "rgba(255,255,255,0.85)",
    "rgba(18,38,95,0.45)": "rgba(228,237,250,0.8)",
    "rgba(14,28,78,0.4)":  "rgba(228,237,250,0.7)",
    "rgba(14,28,75,0.38)": "rgba(228,237,250,0.68)",
    "rgba(14,28,75,0.35)": "rgba(228,237,250,0.62)",
    "rgba(14,28,75,0.3)":  "rgba(228,237,250,0.55)",
    "rgba(14,28,75,0.28)": "rgba(228,237,250,0.52)",
    "rgba(14,28,75,0.22)": "rgba(228,237,250,0.45)",
    "rgba(14,28,75,0.2)":  "rgba(228,237,250,0.42)",
    "rgba(12,24,65,0.35)": "rgba(228,237,250,0.62)",
    "rgba(12,24,65,0.32)": "rgba(228,237,250,0.58)",
    "rgba(12,24,65,0.3)":  "rgba(228,237,250,0.55)",
    # ── drop shadows (navy -> soft slate) ───────────────────────────────────
    "rgba(7,15,43,0.7)":  "rgba(70,90,140,0.18)",
    "rgba(7,15,43,0.6)":  "rgba(70,90,140,0.16)",
    "rgba(7,15,43,0.35)": "rgba(70,90,140,0.12)",
    "rgba(9,28,78,0.55)": "rgba(70,90,140,0.16)",
    "rgba(9,28,78,0.45)": "rgba(70,90,140,0.14)",
    "rgba(9,28,78,0.4)":  "rgba(70,90,140,0.13)",
    "rgba(9,28,78,0.35)": "rgba(70,90,140,0.12)",
    "rgba(9,28,78,0.25)": "rgba(70,90,140,0.1)",
    # ── light-blue text -> darker blue (readable on a light background) ──────
    "#E8EDF8": "#1B2438",          # main text + section titles + card bodies
    "#82B1FF": "#1E5FCF", "#90CAF9": "#1F6FC0", "#64B5F6": "#1565C0",
    "#BFE0FF": "#2C6FCF", "#C2CFEC": "#3A4E78", "#C9D7F5": "#3A4E78",
    "#AFC3E8": "#42567F", "#8FA3CE": "#46588A", "#6E87C9": "#46588A",
    "#7487B0": "#586A92", "#5E7FD4": "#3458AE", "#9FB6E0": "#46588A",
    "#8AABEF": "#2E5AA8", "#4FC3F7": "#1597C7",
    # ── purple operation banner ─────────────────────────────────────────────
    "rgba(74,20,110,0.62)": "rgba(124,63,176,0.14)",
    "rgba(34,16,74,0.6)":   "rgba(124,63,176,0.08)",
    "#E6D6FF": "#5B2A86", "#C7B6E6": "#6A4A92", "#C9A6FF": "#7C3FB8",
    "#A98FD0": "#6A4A92", "#8A78AE": "#7A6A98", "#C77DFF": "#9B30D8",
    "#C9A6E8": "#7C3FB0",
    # ── gold / amber text (champion + ratio) ────────────────────────────────
    "#FFD454": "#B8860B", "#FFE08A": "#A8780A", "#FFB74D": "#C77B00",
}


def _light_swap(css: str) -> str:
    """Return css with every dark literal replaced by its Light Side equivalent."""
    for dark, light in _LIGHT_MAP.items():
        css = css.replace(dark, light)
    return css


# Lightsaber accent: the one decorative colour, exposed as a CSS variable so the
# blade styling (below) and the verdict glow can read it. The user picks a blade
# colour in Settings (session key 'sw_saber'); "Auto" follows the side — a red
# Sith blade on the Dark Side, a blue Jedi blade on the Light Side.
_SABER_NAMED = {
    "Blue":   ("#2C82FF", "44,130,255"),
    "Green":  ("#46E36B", "70,227,107"),
    "Red":    ("#FF3B3B", "255,59,59"),
    "Purple": ("#B36BFF", "179,107,255"),
    "Amber":  ("#FFB23B", "255,178,59"),
}
_SABER_AUTO = {"dark": "Red", "light": "Blue"}


def saber_choice(theme: str) -> Tuple[str, str]:
    """(blade hex, glow-rgb) for the active side, honouring the Settings choice."""
    pick = st.session_state.get("sw_saber", "Auto")
    if pick not in _SABER_NAMED:
        pick = _SABER_AUTO.get(theme, "Red")
    return _SABER_NAMED[pick]


# Turns the existing flat accent rules (.gradient-bar and the section header rule)
# into a GLOWING, animated lightsaber blade with an ignition flicker + travelling
# shimmer and a small hilt cap. Appended LAST so it wins over the earlier cyan
# definitions. Blade colour comes from --saber. Motion stops under reduce-motion.
_SABER_CSS = """
@keyframes saberPulse {
    0%,100% { box-shadow: 0 0 8px var(--saber-glow), 0 0 20px var(--saber-glow),
                          0 0 34px var(--saber-glow); filter: brightness(1); }
    50%     { box-shadow: 0 0 14px var(--saber-glow), 0 0 30px var(--saber-glow),
                          0 0 52px var(--saber-glow); filter: brightness(1.18); }
}
@keyframes saberShimmer { 0% { background-position: -40% 0; } 100% { background-position: 140% 0; } }
.gradient-bar {
    position: relative; height: 5px; border-radius: 4px; margin: 0.6rem 0 1.4rem;
    background-color: var(--saber) !important;
    background-image: linear-gradient(90deg,
        transparent 0%, rgba(255,255,255,0.85) 50%, transparent 100%) !important;
    background-size: 28% 100% !important; background-repeat: no-repeat !important;
    box-shadow: 0 0 10px var(--saber-glow), 0 0 22px var(--saber-glow),
                0 0 36px var(--saber-glow) !important;
    animation: saberPulse 2.4s ease-in-out infinite,
               saberShimmer 3.6s linear infinite !important;
}
.sec-head .sh-rule {
    position: relative;
    background-color: var(--saber) !important;
    background-image: linear-gradient(90deg,
        transparent 0%, rgba(255,255,255,0.8) 50%, transparent 100%) !important;
    background-size: 30% 100% !important; background-repeat: no-repeat !important;
    height: 4px !important; border-radius: 2px;
    box-shadow: 0 0 8px var(--saber-glow), 0 0 18px var(--saber-glow) !important;
    animation: saberPulse 3s ease-in-out infinite,
               saberShimmer 5s linear infinite !important;
    overflow: visible !important;
}
/* ── Lightsaber HILT — a properly modelled metal handle on the title blades ──
   A short cylinder (top-lit shine), a few machined grip rings, a glowing emitter
   ring where the blade ignites, and a rounded pommel. Shared by the section-title
   rule and the free-standing accent bar so every title blade has a real hilt. */
.gradient-bar::before, .sec-head .sh-rule::before {
    content: ""; position: absolute; top: 50%; transform: translateY(-50%);
    width: 30px; height: 14px; border-radius: 4px 2px 2px 4px;
    background:
        /* emitter: bright saber-coloured ring at the blade (right) end */
        linear-gradient(90deg, transparent 0 79%, rgba(255,255,255,0.55) 79% 81%,
                         var(--saber) 81% 100%),
        /* machined grip ring shadows */
        repeating-linear-gradient(90deg, rgba(255,255,255,0.12) 0 1px,
                         rgba(0,0,0,0.22) 3px 4px, transparent 4px 7px),
        /* the metal body, top-lit */
        linear-gradient(180deg, #f1f3f7 0%, #c4cad4 20%, #889 50%,
                         #4d5362 80%, #2a2f3a 100%);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.5),
                inset 0 -1px 0 rgba(0,0,0,0.55),
                0 1px 3px rgba(0,0,0,0.55);
    z-index: 2;
}
/* The free-standing bar sits clear of any text, so its hilt extends fully left. */
.gradient-bar { overflow: visible; }
.gradient-bar::before { left: -28px; }
/* The title rule has a heading to its left; nudge the hilt only slightly so the
   emitter sits at the rule's start and the blade glows away from the title. */
.sec-head .sh-rule::before { left: -8px; }
/* Every other luminous accent becomes a lightsaber blade too (colour + glow). */
[data-testid="stProgress"] > div > div {
    background: var(--saber) !important;
    background-image: none !important;
    box-shadow: 0 0 8px var(--saber-glow), 0 0 16px var(--saber-glow) !important;
    animation: saberPulse 2.6s ease-in-out infinite !important;
}
/* Card accent lines: a soft resting glow so they are never dull, igniting to a
   full blade only on hover (keeps dense pages like Methodology from looking
   overloaded with always-on bars). */
.panel-card::before, [data-testid="stMetric"]::after, .pipe-step::after,
.nav-tile::before, .method-card::before, .corpus-card::after {
    background: var(--saber) !important;
    background-image: none !important;
    box-shadow: 0 0 5px var(--saber-glow) !important;
    opacity: 0.33 !important;
    transition: opacity 0.3s ease, box-shadow 0.3s ease !important;
}
.panel-card:hover::before, [data-testid="stMetric"]:hover::after,
.pipe-step:hover::after, .nav-tile:hover::before,
.method-card:hover::before, .corpus-card:hover::after {
    opacity: 1 !important;
    box-shadow: 0 0 9px var(--saber-glow), 0 0 18px var(--saber-glow) !important;
}
.info-card, .rep-card {
    border-left-color: var(--saber) !important;
    box-shadow: -2px 0 7px -3px var(--saber-glow) !important;
    transition: box-shadow 0.25s ease, transform 0.22s ease,
                border-color 0.22s ease !important;
}
.info-card:hover, .rep-card:hover {
    box-shadow: -3px 0 13px -2px var(--saber-glow) !important;
}
.hcw-body, [class*="st-key-evalgrp_"] { border-left-color: var(--saber) !important; }
"""


def _root_block(theme: str) -> str:
    """:root variables consumed by the lightsaber/verdict styling."""
    blade, glow = saber_choice(theme)
    return (f":root{{--saber:{blade};"
            f"--saber-glow:rgba({glow},0.65);}}")


def _accessibility_css() -> str:
    """Extra CSS from the Settings page: reduced motion, high contrast, text size.
    Appended LAST so it overrides the base + saber rules."""
    out = []
    # Reduced motion (also gated on the <html data-reduce-motion> attribute set by
    # the live-settings iframe; the attribute selector beats !important animations).
    out.append(
        'html[data-reduce-motion="1"] *,'
        'html[data-reduce-motion="1"] *::before,'
        'html[data-reduce-motion="1"] *::after'
        '{animation:none !important;transition:none !important;}'
    )
    # Bump only the *content* text, not the whole rem system (scaling the root
    # font grew the sidebar/hero and deformed the layout). Targeting text-bearing
    # elements keeps the structure intact.
    scale = st.session_state.get("sw_text_scale", "Normal")
    if scale in ("Large", "Larger"):
        f = "1.12em" if scale == "Large" else "1.26em"
        out.append(
            "[data-testid=\"stMarkdownContainer\"] p,"
            "[data-testid=\"stMarkdownContainer\"] li,"
            "[data-testid=\"stText\"], .info-card .ic-body, .panel-card li,"
            ".method-card li, .rep-card .rep-desc, .sec-sub, .stCaption,"
            "[data-testid=\"stCaptionContainer\"] p,"
            "[data-baseweb=\"select\"] div, .stMarkdown li"
            "{font-size:" + f + " !important;line-height:1.6 !important;}"
            "[data-testid=\"stWidgetLabel\"] p{font-size:" + f + " !important;}"
        )
    if st.session_state.get("sw_contrast"):
        # Boost text and borders for legibility (works on both sides).
        out.append(
            '[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] *,'
            '[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *'
            '{opacity:1 !important;}'
            '[data-testid="stMetric"], [data-testid="stVerticalBlockBorderWrapper"],'
            '.info-card, .panel-card, .rep-card, .corpus-card, .method-card,'
            '.pipe-step, .stat-strip{border-width:2px !important;}'
            'a, .hero-author a{text-decoration:underline !important;}'
        )
    return "".join(out)


# Re-applied AFTER the light swap. Two jobs: (1) keep the hero banner's deep-blue
# identity (its text must stay light even on the Light Side); (2) flip Streamlit's
# own native chrome (text, dropdown popovers) which is driven by config.toml's
# dark base and so isn't reached by our class-level CSS.
_LIGHT_PATCH = """
/* Native Streamlit text -> dark on the light canvas */
[data-testid="stApp"] {
    --text-color: #1B2438; --default-textColor: #1B2438;
    --background-color: #D5DEEE; --secondary-background-color: #FFFFFF;
}
body, [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li, [data-testid="stText"],
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
[data-testid="stMetricLabel"], [data-testid="stCaptionContainer"],
[data-testid="stHeadingWithActionElements"], h1, h2, h3, h4, h5, h6,
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span,
[data-testid="stSidebarNavLink"] span, [data-testid="stSidebarNavLink"] p {
    color: #1B2438 !important;
}

/* ── baseweb popovers / dropdown menus (portaled to <body>, so config.toml's
   dark base leaks through unless we override them explicitly) ─────────────── */
[data-baseweb="popover"], [data-baseweb="popover"] > div,
[data-baseweb="popover"] [data-baseweb="menu"],
[data-baseweb="popover"] ul[role="listbox"], [data-baseweb="menu"], [role="listbox"] {
    background: #FFFFFF !important;
    border-color: rgba(40,90,180,0.22) !important;
}
[data-baseweb="popover"] [data-baseweb="menu"], [data-baseweb="popover"] ul[role="listbox"] {
    box-shadow: 0 10px 30px rgba(40,70,130,0.20) !important;
    border: 1px solid rgba(40,90,180,0.20) !important;
}
[role="option"], [data-baseweb="menu"] li, [role="listbox"] li,
[data-baseweb="popover"] * { color: #1B2438 !important; }
[role="option"]:hover, [data-baseweb="menu"] li:hover,
li[role="option"][aria-selected="true"], [role="option"][aria-selected="true"] {
    background: rgba(40,90,180,0.12) !important;
}

/* ── closed controls: select, text/number inputs, textareas, file uploader ── */
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"],
[data-baseweb="base-input"] input, textarea, input[type="text"], input[type="number"],
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input, [data-testid="stFileUploaderDropzone"] {
    background: #FFFFFF !important; color: #1B2438 !important;
    border-color: rgba(40,90,180,0.25) !important;
}
[data-baseweb="select"] *, [data-baseweb="select"] input { color: #1B2438 !important; }
[data-baseweb="select"] svg, [data-baseweb="input"] svg,
[data-testid="stWidgetLabel"] svg { fill: #46588A !important; color: #46588A !important; }
/* multiselect chips / tags */
[data-baseweb="tag"] { background: rgba(40,90,180,0.14) !important; color: #1B2438 !important; }
[data-baseweb="tag"] svg { fill: #1B2438 !important; }
/* tooltips stay dark-on-light for legibility */
[data-baseweb="tooltip"], [role="tooltip"] { background: #1B2438 !important; color: #FFFFFF !important; }
[data-baseweb="tooltip"] * { color: #FFFFFF !important; }
/* sliders */
[data-testid="stSlider"] [role="slider"] { background: #1E5FCF !important; }

/* ── more separation on the light canvas: stronger borders + soft elevation so
   the white cards lift off the blue-grey background and accents are visible ── */
.panel-card, .corpus-card, .method-card, .pipe-step, .dist-row, .side-status,
[data-testid="stMetric"], [data-testid="stVerticalBlockBorderWrapper"],
.stat-strip, .empty-state, [data-testid="stExpander"] {
    border-color: rgba(40,90,180,0.30) !important;
}
.panel-card, .corpus-card, .method-card, .pipe-step, .info-card, .rep-card,
[data-testid="stMetric"], [data-testid="stVerticalBlockBorderWrapper"] {
    box-shadow: 0 3px 14px rgba(40,70,130,0.11) !important;
}
.panel-card:hover, .corpus-card:hover, .method-card:hover, .pipe-step:hover,
.info-card:hover, .rep-card:hover, [data-testid="stMetric"]:hover {
    box-shadow: 0 10px 28px rgba(40,70,130,0.20) !important;
}
/* scrollbar contrast on the light side */
::-webkit-scrollbar-track { background: rgba(40,90,180,0.10) !important; }
::-webkit-scrollbar-thumb  { background: rgba(40,90,180,0.40) !important; }

/* selected pill / segmented option keeps white text on its blue gradient */
[data-testid="stBaseButton-pillsActive"],
[data-testid="stBaseButton-segmented_controlActive"],
[data-testid="stBaseButton-pillsActive"] *,
[data-testid="stBaseButton-segmented_controlActive"] * {
    color: #FFFFFF !important;
}
/* Hero banner stays a deep-blue colored banner on BOTH sides — restore its light
   text (placed last so it beats the generic dark-text rule above). */
.hero-banner, .hero-banner h1, .hero-banner p, .hero-overline, .hero-meta,
.hero-author .ha-name, .hero-author a { color: #EAF1FF !important; }
.hero-banner h1 { -webkit-text-fill-color: #EAF1FF; }
"""


def build_page_css(theme: str) -> str:
    """The global stylesheet for the active side. Dark returns PAGE_CSS unchanged
    (plus the saber variables); light applies the colour swap and native-chrome
    patch. Injected once per rerun from app.py."""
    inject = _root_block(theme) + _SABER_CSS
    css = PAGE_CSS
    if theme == "light":
        css = _light_swap(css)
        inject += _LIGHT_PATCH
    inject += _accessibility_css()          # appended LAST so it always wins
    return css.replace("</style>", inject + "\n</style>", 1)


def themed(html: str) -> str:
    """Light-swap a CSS/HTML string when the Light Side is active (no-op on dark).
    Lets page-local <style> blocks reuse the single _LIGHT_MAP."""
    return _light_swap(html) if theme_mode() == "light" else html


def inject_css(html: str) -> None:
    """st.markdown for a page-local style block, theming it for the active side."""
    st.markdown(themed(html), unsafe_allow_html=True)


# ===========================================================================
# Cached resources
# ===========================================================================

@st.cache_resource
def load_config() -> Dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_resource
def get_extractor() -> FeatureExtractor:
    return FeatureExtractor(CONFIG_PATH)


@st.cache_data(show_spinner=False)
def get_samples(subset: str) -> List[Tuple[str, int]]:
    if _forced_demo():
        return []
    config    = load_config()
    root_dir  = config["dataset"]["path_la2019"]
    proto_dir = os.path.join(root_dir, config["dataset"]["protocols_dir"])
    proto     = config["dataset"]["protocols"].get(subset)
    if not proto:
        return []
    try:
        return parse_protocol(os.path.join(proto_dir, proto), root_dir, subset)
    except (FileNotFoundError, ValueError):
        return []


@st.cache_data(show_spinner=False)
def get_samples_2021_la() -> List[Tuple[str, int]]:
    """Load and cache the full ASVspoof 2021 LA eval split."""
    if _forced_demo():
        return []
    config = load_config()
    cfg    = config.get("dataset_2021", {}).get("la", {})
    eval_dir = cfg.get("eval_dir", "")
    keys     = cfg.get("keys", "")
    try:
        return parse_protocol_2021(keys, [eval_dir])
    except (FileNotFoundError, ValueError) as exc:
        print(f"[WARNING] 2021 LA unavailable: {exc}")
        return []


@st.cache_data(show_spinner=False)
def get_samples_2021_df() -> List[Tuple[str, int]]:
    """Load and cache the full ASVspoof 2021 DF eval split (all 3 partitions)."""
    if _forced_demo():
        return []
    config    = load_config()
    cfg       = config.get("dataset_2021", {}).get("df", {})
    eval_dirs = cfg.get("eval_dirs", [])
    keys      = cfg.get("keys", "")
    try:
        return parse_protocol_2021(keys, eval_dirs)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[WARNING] 2021 DF unavailable: {exc}")
        return []


def corpus_available_2021_la() -> bool:
    return len(get_samples_2021_la()) > 0


def corpus_available_2021_df() -> bool:
    return len(get_samples_2021_df()) > 0


def corpus_configured_2021_la() -> bool:
    """Lightweight check (single stat call) — does NOT load audio index."""
    cfg = load_config().get("dataset_2021", {}).get("la", {})
    return os.path.isfile(cfg.get("keys", ""))


def corpus_configured_2021_df() -> bool:
    """Lightweight check (single stat call) — does NOT load audio index."""
    cfg = load_config().get("dataset_2021", {}).get("df", {})
    return os.path.isfile(cfg.get("keys", ""))


def split_by_label(
    samples: List[Tuple[str, int]],
) -> Tuple[List[str], List[str]]:
    bonafide = [p for p, e in samples if e == LABEL_BONAFIDE]
    spoof    = [p for p, e in samples if e == LABEL_SPOOF]
    return bonafide, spoof


def corpus_available() -> bool:
    return len(get_samples("train")) > 0


# ===========================================================================
# Public / CPU demo mode + pretrained model registry (Streamlit Cloud)
# ===========================================================================
# On the free public cloud there is no GPU and the multi-GB ASVspoof corpus is
# not on disk, so training / full-benchmark features cannot run. Instead the whole
# pretrained zoo — the two CNNs and the classic LR / SVM / XGBoost over every DSP
# front-end — is COMMITTED to the repo under models/ (the weights are small, a few
# MB total) and loaded directly from disk on CPU. Only the multi-GB DATASETS still
# stream from Hugging Face (HF_EVAL_DATASETS); the model weights do NOT.

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Pretrained weights live HERE, committed to the repo (the trainer also writes a
# legacy single checkpoint at the repo root, used by src/pipeline.py).
MODELS_DIR      = os.path.join(_REPO_ROOT, "models")
CHECKPOINT_PATH = os.path.join(_REPO_ROOT, "asvspoof_model_checkpoint.pth")

# ── Weight source ───────────────────────────────────────────────────────────
# Models are loaded straight from the committed models/ folder — no runtime
# download. Leave HF_BASE_URL EMPTY to keep it that way. Only set it to a Hugging
# Face "resolve/main" folder if you ever prefer to stream the weights instead of
# committing them (then each model's URL is derived from this one base). The
# DATASETS are unrelated and always come from HF (see HF_EVAL_DATASETS).
HF_BASE_URL = ""
MODEL_URL   = HF_BASE_URL


def _hf_url(file: str) -> str:
    """Derive a model's download URL from HF_BASE_URL (empty until it is set)."""
    base = HF_BASE_URL.rstrip("/")
    if "TU_ENLACE" in base or not base.startswith(("http://", "https://")):
        return ""
    return f"{base}/{file}"


# Build the registry of every servable detector. Fields:
#   kind  : "cnn" (torch .pth) or "classic" (joblib-dumped sklearn/xgb estimator)
#   clf   : classifier name (classic only) for get_classic_model / row matching
#   feat  : FeatureExtractor option key for classic models (CNNs read the STFT
#           spectrogram directly, so feat is None for them)
#   front : human-readable front-end, for the comparison table
_CLF_DEFS = [
    ("lr",  "logistic_regression", "Logistic Regression"),
    ("svm", "svm_lineal",          "SVM (RBF)"),
    ("xgb", "xgboost",             "XGBoost"),
]
# (key suffix, FeatureExtractor option, label) — every DSP front-end.
_FEAT_DEFS = [
    ("rms",  "1", "RMS"),
    ("mfcc", "2", "MFCC"),
    ("lfcc", "3", "LFCC"),
    ("dwt",  "4", "DWT"),
    ("cqcc", "6", "CQCC"),
]

PRETRAINED_REGISTRY: List[Dict] = [
    {"key": "resnet", "name": "ResNet + SE", "kind": "cnn", "clf": None,
     "feat": None, "front": "STFT-dB spectrogram",
     "file": "resnet.pth", "url": _hf_url("resnet.pth")},
    {"key": "cnn3x3", "name": "3-Block CNN (3×3)", "kind": "cnn", "clf": None,
     "feat": None, "front": "STFT-dB spectrogram",
     "file": "cnn3x3.pth", "url": _hf_url("cnn3x3.pth")},
    # Self-supervised raw-waveform detector (fine-tuned wav2vec 2.0 base + linear
    # head). "raw" kind: no DSP front-end, no spectrogram — it eats the 16 kHz
    # waveform directly. Inference-only (it is evaluated, never trained, by the
    # Full-comparison sweep). On the web demo it is too large to commit to GitHub
    # (≈469 MB), so it is fetched from a PUBLIC Hugging Face model repo on demand
    # (hf_repo / hf_file) — no local heavy file, no HF_BASE_URL needed.
    {"key": "wav2vec2", "name": "wav2vec 2.0 (SSL)", "kind": "raw", "clf": None,
     "feat": None, "front": "Self-supervised raw-waveform",
     "file": "wav2vec2.pth", "url": _hf_url("wav2vec2.pth"),
     "hf_repo": "Sara1708/deepfake-audio-wav2vec2", "hf_file": "stage2_best.pt"},
]
for _ck, _cname, _clabel in _CLF_DEFS:
    for _fk, _fopt, _flabel in _FEAT_DEFS:
        _file = f"{_ck}_{_fk}.joblib"
        PRETRAINED_REGISTRY.append({
            "key":   f"{_ck}_{_fk}",
            "name":  f"{_clabel} · {_flabel}",
            "kind":  "classic",
            "clf":   _cname,
            "feat":  _fopt,
            "front": _flabel,
            "file":  _file,
            "url":   _hf_url(_file),
        })


def running_on_gpu() -> bool:
    """True when a CUDA GPU is available (local workstation), False on the
    CPU-only public cloud."""
    import torch
    return torch.cuda.is_available()


def demo_mode() -> bool:
    """Public-demo mode: the heavy ASVspoof corpus is NOT on disk (the case on
    Streamlit Community Cloud). Corpus-dependent sections degrade to notices;
    the pretrained multi-model file analysis remains fully usable.

    Honours DEEPFAKE_FORCE_DEMO=1 transparently (the loaders report empty, so
    corpus_available() is False) — handy to preview the cloud UI locally."""
    return not corpus_available()


def _url_set(url: Optional[str]) -> bool:
    """Whether a download URL has been filled in (not a placeholder)."""
    return (isinstance(url, str) and "TU_ENLACE" not in url
            and url.startswith(("http://", "https://")))


def _model_path(entry: Dict) -> str:
    return os.path.join(MODELS_DIR, entry["file"])


def _hf_cached(repo: str, fname: str) -> bool:
    """True if the file from a Hugging Face repo is already in the local HF cache
    (a cheap, offline-only probe — never hits the network)."""
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id=repo, filename=fname, local_files_only=True)
        return True
    except Exception:                       # not cached / hub unavailable
        return False


def model_available(entry: Dict) -> bool:
    """Servable if the weights are on disk, a direct URL is set, or the model has a
    public Hugging Face source (hf_repo) it can stream on demand."""
    return (os.path.isfile(_model_path(entry)) or _url_set(entry.get("url"))
            or bool(entry.get("hf_repo")))


def available_pretrained_models() -> List[Dict]:
    """Every registry entry whose weights are present or downloadable."""
    return [e for e in PRETRAINED_REGISTRY if model_available(e)]


def pretrained_available() -> bool:
    """True when at least one pretrained model can be served."""
    return len(available_pretrained_models()) > 0


def model_downloaded(entry: Dict) -> bool:
    """True when a model's weights are already cached locally — on disk, or (for an
    HF-sourced model like wav2vec2) in the Hugging Face cache."""
    if os.path.isfile(_model_path(entry)):
        return True
    if entry.get("hf_repo"):
        return _hf_cached(entry["hf_repo"], entry["hf_file"])
    return False


def models_trained() -> bool:
    """True once EVERY registry model has been trained and saved to disk — used
    to switch the Benchmark from 'train everything' to 'evaluate the saved zoo'."""
    return bool(PRETRAINED_REGISTRY) and all(
        model_downloaded(e) for e in PRETRAINED_REGISTRY)


# Pre-computed metrics for the pretrained models, written automatically by a
# local Benchmark → Full comparison and committed, so the web demo can show the
# real EER/minDCF leaderboard without the corpus. Maps model key ->
# {"eer_dev", "mindcf_dev", "eer_eval", "mindcf_eval"}.
LEADERBOARD_PATH = os.path.join(_REPO_ROOT, "leaderboard.json")
# Previous filename — still read as a fallback so an un-regenerated repo keeps
# showing its committed metrics after the rename.
_LEGACY_LEADERBOARD_PATH = os.path.join(_REPO_ROOT, "demo_leaderboard.json")

# Back-compat alias: some imports still reference the old constant name.
DEMO_LEADERBOARD_PATH = LEADERBOARD_PATH


def load_leaderboard() -> Dict:
    """Read the committed leaderboard file as the full structured object
    ``{"models": {key: {eer_dev, mindcf_dev, eer_eval, mindcf_eval}}, "rows": [...]}``.

    Migrates the legacy FLAT format (a bare ``{key: metrics}`` dict, no per-corpus
    rows) into ``{"models": <flat>, "rows": []}`` so old files keep working."""
    import json
    path = (LEADERBOARD_PATH if os.path.isfile(LEADERBOARD_PATH)
            else _LEGACY_LEADERBOARD_PATH)
    if not os.path.isfile(path):
        return {"models": {}, "rows": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (ValueError, OSError):
        return {"models": {}, "rows": []}
    if isinstance(data, dict) and ("models" in data or "rows" in data):
        data.setdefault("models", {})
        data.setdefault("rows", [])
        return data
    # Legacy flat dict {key: metrics}.
    return {"models": data if isinstance(data, dict) else {}, "rows": []}


def load_leaderboard_models() -> Dict[str, Dict]:
    """Per-model dev/eval metrics map (keyed by registry key). The headline-verdict
    fusion and the web model-hub use this for ranking; empty until generated."""
    return load_leaderboard().get("models", {})


def load_leaderboard_rows() -> List[Dict]:
    """Every leaderboard ROW (one per model × split/corpus) exactly as produced by a
    local Full comparison — so the web demo can render the IDENTICAL filterable
    table/chart, not just a condensed summary."""
    return load_leaderboard().get("rows", [])


# Back-compat alias for the old name (now returns the per-model metrics map).
def load_demo_leaderboard() -> Dict[str, Dict]:
    return load_leaderboard_models()


# ── Bundled sample clips (so Signal Explorer always has audio to show) ─────── #
# A handful of real clips per corpus/subset are committed under
# samples/<key>/<subset>/ so the explorer works even without the multi-GB
# datasets (e.g. on the cloud). For whatever folder is still empty they are
# auto-populated from the live corpus the first time it is browsed locally.
# The label is encoded in the filename prefix (spoof__* / bonafide__*).
SAMPLES_DIR  = os.path.join(_REPO_ROOT, "samples")
_SAMPLE_KEYS = {"2019 LA": "2019_la", "2021 LA": "2021_la", "2021 DF": "2021_df"}


def _sample_dir(corpus: str, subset: Optional[str] = None) -> str:
    base = os.path.join(SAMPLES_DIR, _SAMPLE_KEYS.get(corpus, corpus))
    return os.path.join(base, subset) if subset else base


def _label_for(fn: str) -> int:
    return LABEL_SPOOF if fn.startswith("spoof") else LABEL_BONAFIDE


def _scan_clips(d: str) -> List[Tuple[str, int]]:
    if not os.path.isdir(d):
        return []
    return [(os.path.join(d, fn), _label_for(fn))
            for fn in sorted(os.listdir(d))
            if fn.lower().endswith((".flac", ".wav"))]


def bundled_samples(corpus: str, subset: Optional[str] = None
                    ) -> List[Tuple[str, int]]:
    """(path, label) list of committed clips for a corpus/subset.

    Reads samples/<corpus>/<subset>/ when a subset is given; falls back to the
    flat samples/<corpus>/ folder (legacy layout) so older bundles keep working.
    Empty when nothing has been bundled.
    """
    if subset:
        clips = _scan_clips(_sample_dir(corpus, subset))
        if clips:
            return clips
    return _scan_clips(_sample_dir(corpus))


def bundle_samples(corpus: str, samples: List[Tuple[str, int]],
                   subset: Optional[str] = None, n_per_class: int = 10) -> None:
    """Copy a few bonafide + spoof clips from the live corpus into
    samples/<corpus>/<subset>/ (once). Idempotent: does nothing if that folder
    already holds clips."""
    import shutil
    d = _sample_dir(corpus, subset)
    if _scan_clips(d):
        return
    bona  = [p for p, e in samples if e == LABEL_BONAFIDE][:n_per_class]
    spoof = [p for p, e in samples if e == LABEL_SPOOF][:n_per_class]
    if not bona and not spoof:
        return
    os.makedirs(d, exist_ok=True)
    for tag, paths in (("bonafide", bona), ("spoof", spoof)):
        for i, p in enumerate(paths):
            try:
                shutil.copy(p, os.path.join(d, f"{tag}__{i}_{os.path.basename(p)}"))
            except OSError:
                pass


# ── Eval clips streamed from public Hugging Face datasets ──────────────────── #
# On the corpus-less web demo the EVAL splits are far too large to commit, so we
# pull a small, balanced sample on demand from public HF datasets (the dev/train
# splits keep using the committed samples/ tree above). We use the lightweight
# datasets-server /rows API — no `datasets` dependency and no multi-GB parquet
# download — read each row's label + presigned audio URL, and cache a capped
# number of clips locally so browsing and scoring reuse them.
HF_EVAL_DATASETS: Dict[str, Dict[str, str]] = {
    "2019 LA": {"id": "Bisher/ASVspoof_2019_LA",
                "config": "default", "split": "test", "label_col": "key"},
    "2021 LA": {"id": "SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA",
                "config": "default", "split": "test", "label_col": "label"},
    "2021 DF": {"id": "SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF",
                "config": "default", "split": "test", "label_col": "label"},
}
HF_EVAL_PER_CLASS = 50          # clips PER CLASS to cache (web-demo friendly cap)
HF_EVAL_MAX_PAGES = 10          # /rows pages (×100 rows) scanned to find bonafide
_HF_CACHE_DIR     = os.path.join(SAMPLES_DIR, "_hf_cache")
_DS_ROWS_URL      = "https://datasets-server.huggingface.co/rows"


def _hf_get_json(url: str) -> Dict:
    import json
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "tfg-deepfake-demo"})
    with urllib.request.urlopen(req, timeout=30) as r:   # noqa: S310 — fixed host
        return json.load(r)


def _hf_download(src_dst):
    import urllib.request
    src, dst, label = src_dst
    if not os.path.isfile(dst):
        try:
            req = urllib.request.Request(src, headers={"User-Agent": "tfg-deepfake-demo"})
            with urllib.request.urlopen(req, timeout=60) as r:   # noqa: S310
                data = r.read()
            with open(dst, "wb") as fh:
                fh.write(data)
        except Exception:                                # noqa: BLE001 — skip bad clip
            return None
    return (dst, label)


def _hf_eval_impl(corpus: str, n_per_class: int) -> List[Tuple[str, int]]:
    import concurrent.futures as cf
    import random
    from urllib.parse import quote

    spec = HF_EVAL_DATASETS.get(corpus)
    if not spec:
        return []

    cache_dir = os.path.join(_HF_CACHE_DIR, _SAMPLE_KEYS.get(corpus, corpus), "eval")
    cached = _scan_clips(cache_dir)
    if len(cached) >= 2 * n_per_class:                   # already populated → reuse
        return cached

    base = (f"{_DS_ROWS_URL}?dataset={quote(spec['id'], safe='')}"
            f"&config={spec['config']}&split={spec['split']}")

    # Read rows across random windows: ASVspoof is ~90% spoof, so sequential
    # pages can be all-spoof. Random offsets give both classes quickly.
    collected: Dict[int, List[str]] = {LABEL_BONAFIDE: [], LABEL_SPOOF: []}
    try:
        first = _hf_get_json(base + "&offset=0&length=100")
    except Exception:                                    # noqa: BLE001 — HF unreachable
        return cached
    total = int(first.get("num_rows_total", 100))
    rng = random.Random(42)
    offsets = [0] + [rng.randint(0, max(0, total - 100))
                     for _ in range(HF_EVAL_MAX_PAGES - 1)]

    def _ingest(payload):
        for row in payload.get("rows", []):
            r   = row.get("row", {})
            lab = r.get(spec["label_col"])
            if lab not in (LABEL_BONAFIDE, LABEL_SPOOF):
                continue
            if len(collected[lab]) >= n_per_class:
                continue
            audio = r.get("audio")
            src   = (audio[0].get("src") if isinstance(audio, list) and audio
                     else None)
            if src:
                collected[lab].append(src)

    _ingest(first)
    for off in offsets[1:]:
        if all(len(collected[c]) >= n_per_class for c in collected):
            break
        try:
            _ingest(_hf_get_json(base + f"&offset={off}&length=100"))
        except Exception:                                # noqa: BLE001 — skip bad page
            continue

    os.makedirs(cache_dir, exist_ok=True)
    ckey = _SAMPLE_KEYS.get(corpus, corpus)
    tasks = []
    for lab, srcs in collected.items():
        tag = "bonafide" if lab == LABEL_BONAFIDE else "spoof"
        for i, src in enumerate(srcs):
            # Embed the corpus in the filename: the DSP/spectrogram caches key on
            # the file stem, so a bare bonafide__0.flac would collide across
            # corpora. Keep the label prefix so _label_for() still works.
            fname = f"{tag}__{ckey}__{i}.flac"
            tasks.append((src, os.path.join(cache_dir, fname), lab))

    out: List[Tuple[str, int]] = []
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        for res in pool.map(_hf_download, tasks):
            if res:
                out.append(res)
    return out or cached


@st.cache_data(show_spinner=False)
def hf_eval_samples(corpus: str,
                    n_per_class: int = HF_EVAL_PER_CLASS) -> List[Tuple[str, int]]:
    """A small, balanced eval sample streamed from the public HF dataset for
    ``corpus`` and cached under samples/_hf_cache/. Returns [(local_path, label)]
    (empty if the corpus has no HF mapping or HF is unreachable and nothing is
    cached). Cached per session by Streamlit and on disk across reruns."""
    return _hf_eval_impl(corpus, n_per_class)


# ── Browseable HF index (list MANY clips, download only the chosen one) ─────── #
# hf_eval_samples downloads its whole balanced set up front — fine for scoring,
# but it means the Signal Explorer only ever shows ~50 clips per class and pays
# the download cost immediately. For browsing we instead pull a large INDEX of
# rows (label + presigned audio URL, NO audio) and fetch a single clip lazily
# when the user actually selects it. Much faster, far more files to pick from.
HF_BROWSE_PER_CLASS = 300        # how many clips per class to list for browsing


def _hf_listing_impl(corpus: str, max_per_class: int):
    import random
    from urllib.parse import quote

    spec = HF_EVAL_DATASETS.get(corpus)
    if not spec:
        return []
    base = (f"{_DS_ROWS_URL}?dataset={quote(spec['id'], safe='')}"
            f"&config={spec['config']}&split={spec['split']}")
    try:
        first = _hf_get_json(base + "&offset=0&length=100")
    except Exception:                                    # noqa: BLE001 — HF unreachable
        return []
    total = int(first.get("num_rows_total", 100))
    rng = random.Random(7)
    pages = max(4, (2 * max_per_class) // 100 + 4)
    offsets = [0] + [rng.randint(0, max(0, total - 100)) for _ in range(pages - 1)]

    collected: Dict[int, List[Tuple[int, str, str]]] = {
        LABEL_BONAFIDE: [], LABEL_SPOOF: []}
    seen = set()

    def _ingest(payload):
        for row in payload.get("rows", []):
            r   = row.get("row", {})
            lab = r.get(spec["label_col"])
            if lab not in (LABEL_BONAFIDE, LABEL_SPOOF):
                continue
            if len(collected[lab]) >= max_per_class:
                continue
            audio = r.get("audio")
            src   = (audio[0].get("src") if isinstance(audio, list) and audio
                     else None)
            if not src or src in seen:
                continue
            seen.add(src)
            stem  = os.path.basename(src.split("?")[0]) or f"row{row.get('row_idx')}"
            collected[lab].append((lab, src, stem))

    _ingest(first)
    for off in offsets[1:]:
        if all(len(collected[c]) >= max_per_class for c in collected):
            break
        try:
            _ingest(_hf_get_json(base + f"&offset={off}&length=100"))
        except Exception:                                # noqa: BLE001 — skip bad page
            continue
    return collected[LABEL_BONAFIDE] + collected[LABEL_SPOOF]


@st.cache_data(show_spinner=False)
def hf_eval_listing(corpus: str,
                    max_per_class: int = HF_BROWSE_PER_CLASS
                    ) -> List[Tuple[int, str, str]]:
    """A large browseable index of eval clips for ``corpus``: [(label, src_url,
    fname)] read from the HF datasets-server WITHOUT downloading any audio. The
    Signal Explorer lists these; hf_fetch_clip() fetches only the selected clip.
    Empty if the corpus has no HF mapping or HF is unreachable."""
    return _hf_listing_impl(corpus, max_per_class)


def hf_fetch_clip(corpus: str, src: str, fname: str, label: int) -> Optional[str]:
    """Download ONE listed clip into the browse cache and return its local path
    (or None on failure). Idempotent: reuses the file if already fetched."""
    ckey      = _SAMPLE_KEYS.get(corpus, corpus)
    cache_dir = os.path.join(_HF_CACHE_DIR, ckey, "browse")
    os.makedirs(cache_dir, exist_ok=True)
    tag  = "bonafide" if label == LABEL_BONAFIDE else "spoof"
    safe = "".join(c for c in fname if c.isalnum() or c in "._-")[-48:] or "clip"
    dst  = os.path.join(cache_dir, f"{tag}__{ckey}__{safe}")
    if not dst.lower().endswith((".flac", ".wav")):
        dst += ".flac"
    res = _hf_download((src, dst, label))
    return res[0] if res else None


def _download_if_missing(url: str, path: str, label: str) -> None:
    if os.path.isfile(path):
        return
    if not _url_set(url):
        raise FileNotFoundError(
            f"{label}: weights not found in models/. Run Benchmark → Full "
            "comparison → Train all locally to generate every model file, then "
            "commit models/ to the repo (or set HF_BASE_URL to stream them)."
        )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import urllib.request
    with st.spinner(f"Calibrating the kyber crystals — fetching {label} (first run only)…"):
        urllib.request.urlretrieve(url, path)


@st.cache_resource(show_spinner=False)
def load_pretrained_torch(file: str, url: str, name: str):
    """Load a torch CNN checkpoint on CPU (downloading it first if needed).
    Returns (model_in_eval_mode, checkpoint_meta). Cached per (file)."""
    import torch

    from src.models import AudioDeepfakeCNN, ResNetCNN

    path = file if os.path.isabs(file) else os.path.join(MODELS_DIR, file)
    _download_if_missing(url, path, name)
    with st.spinner(f"Consulting the Jedi Archives — loading {name} on CPU…"):
        ckpt = torch.load(path, map_location=torch.device("cpu"))
        model_cls = ResNetCNN if ckpt.get("arch") == "resnet" else AudioDeepfakeCNN
        model = model_cls(dropout=float(ckpt.get("dropout", 0.3)))
        model.load_state_dict(ckpt["state_dict"])
        model.to("cpu").eval()
    return model, ckpt


@st.cache_resource(show_spinner=False)
def load_pretrained_classic(file: str, url: str, name: str):
    """Load a joblib-dumped classic estimator (downloading it first if needed)."""
    import joblib

    path = os.path.join(MODELS_DIR, file)
    _download_if_missing(url, path, name)
    with st.spinner(f"Consulting the Jedi Archives — loading {name}…"):
        return joblib.load(path)


@st.cache_resource(show_spinner=False)
def load_pretrained_raw(file: str, url: str, name: str,
                        hf_repo: str = "", hf_file: str = ""):
    """Load a raw-waveform torch model (wav2vec 2.0 SSL detector) on CPU.

    Weight resolution order: a local ``models/`` file (used on the GPU machine),
    then an explicit direct ``url``, then a PUBLIC Hugging Face repo (``hf_repo`` /
    ``hf_file``) streamed via ``hf_hub_download`` — that last path is what makes the
    cloud demo self-contained without committing the 469 MB checkpoint to GitHub.

    The checkpoint stores the weights under ``model_state_dict`` (a training
    checkpoint, not the plain ``state_dict`` the CNNs use). ``transformers`` is only
    needed here; if it is missing the ImportError propagates and the caller skips
    this model (same as a missing weight file)."""
    import torch

    from src.models import Wav2Vec2Classifier

    path = file if os.path.isabs(file) else os.path.join(MODELS_DIR, file)
    if not os.path.isfile(path):
        if _url_set(url):
            _download_if_missing(url, path, name)
        elif hf_repo:
            from huggingface_hub import hf_hub_download
            with st.spinner(f"Calibrating the kyber crystals — fetching {name} "
                            "from Hugging Face (first run only)…"):
                path = hf_hub_download(repo_id=hf_repo, filename=hf_file)
        else:
            _download_if_missing(url, path, name)     # raises the helpful error
    with st.spinner(f"Consulting the Jedi Archives — loading {name} on CPU…"):
        ckpt = torch.load(path, map_location=torch.device("cpu"))
        state = ckpt.get("model_state_dict", ckpt)
        model = Wav2Vec2Classifier()
        model.load_state_dict(state)
        model.to("cpu").eval()
    return model


def load_pretrained_model(entry: Dict):
    """Load one registry entry (torch CNN, raw-waveform model or classic estimator) on CPU."""
    if entry["kind"] == "cnn":
        model, _ = load_pretrained_torch(entry["file"], entry["url"], entry["name"])
        return model
    if entry["kind"] == "raw":
        return load_pretrained_raw(entry["file"], entry.get("url", ""), entry["name"],
                                   entry.get("hf_repo", ""), entry.get("hf_file", ""))
    return load_pretrained_classic(entry["file"], entry["url"], entry["name"])


def load_pretrained_cnn():
    """Backward-compatible helper: load the first available CNN entry (or the
    legacy single checkpoint) and return (model, meta)."""
    for entry in available_pretrained_models():
        if entry["kind"] == "cnn":
            return load_pretrained_torch(entry["file"], entry["url"], entry["name"])
    if os.path.isfile(CHECKPOINT_PATH):
        return load_pretrained_torch(CHECKPOINT_PATH, "", "Pretrained CNN")
    raise FileNotFoundError("No pretrained CNN available.")


def demo_corpus_notice(title: str = "Disabled in the web demo",
                       body: Optional[str] = None) -> None:
    """Styled notice replacing the bare 'corpus not found' error: explains why a
    corpus-dependent section is unavailable in the public CPU demo."""
    body = body or (
        "This section runs on the full ASVspoof corpus, which is not bundled in "
        "the public CPU demo (it is several GB). Clone the repository and run "
        "the app locally with the dataset — and a GPU for training — to use it."
        '<br><span style="opacity:0.65;font-style:italic;">'
        "“If an item does not appear in our records… it does not exist.”"
        "</span>"
    )
    st.markdown(
        f'<div class="info-card" style="border-left:3px solid #4F8BF9;">'
        f'<div class="ic-title">{title}</div>'
        f'<p class="ic-body">{body}</p></div>',
        unsafe_allow_html=True,
    )


def test_audio_cta(
    text: str = "Instead, hear the state of the art in real time: upload your own "
                "clip and watch every pretrained model judge it side by side.",
) -> None:
    """Attractive redirect from a corpus-only section to the multi-model file
    analysis that DOES work in the web demo."""
    st.markdown(
        f'<p style="margin:0.9rem 0 1.15rem;opacity:0.8;">{text}</p>',
        unsafe_allow_html=True,
    )
    try:
        st.page_link("app_pages/3_Detection_Analysis.py",
                     label="Try the live multi-model analysis",
                     icon=":material/hearing:")
    except Exception:  # noqa: BLE001 — page_link needs st.navigation context
        st.caption("Open **Detection Analysis → Test an audio** from the sidebar.")


# ===========================================================================
# Reusable UI components
# ===========================================================================

def show_empty_state(title: str, message: str, icon: str = "◌") -> None:
    """Centred empty-state card: sober glyph in a dashed ring, heading, text."""
    st.markdown(
        f'<div class="empty-state">'
        f'<div class="empty-icon">{icon}</div>'
        f'<h3 style="color:#82B1FF;margin:0 0 .45rem;font-weight:700;">{title}</h3>'
        f'<p style="max-width:480px;margin:.3rem auto;line-height:1.65;opacity:0.65;">{message}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def label_badge(label: str) -> str:
    """Return an HTML colour badge (dot + text) for a class label string."""
    is_spoof = "spoof" in label.lower()
    color = SPOOF_COLOR if is_spoof else BONAFIDE_COLOR
    return (
        f'<span style="color:{color};font-weight:600;">'
        f'<span class="dot" style="background:{color};'
        f'box-shadow:0 0 6px {color};"></span>{label}</span>'
    )


def section_header(num: str, title: str, caption: Optional[str] = None) -> None:
    """Editorial section header: index number, title, fading rule, caption."""
    st.markdown(
        f'<div class="sec-head"><span class="sh-num">{num}</span>'
        f'<h3 class="sh-title">{title}</h3><span class="sh-rule"></span></div>'
        + (f'<p class="sec-sub">{caption}</p>' if caption else ""),
        unsafe_allow_html=True,
    )


def sidebar_panel(
    title: str,
    rows: Optional[List[Tuple[str, str]]] = None,
    text: Optional[str] = None,
) -> None:
    """Compact sidebar block: optional key/value rows and/or a short text.

    rows: list of (label, value) pairs rendered right-aligned.
    text: free-form sentence rendered below the rows.
    """
    body = "".join(
        f'<div class="ss-row"><span class="ss-k">{k}</span>'
        f'<span class="ss-v">{v}</span></div>'
        for k, v in (rows or [])
    )
    if text:
        body += f'<div style="margin-top:.3rem;">{text}</div>'
    st.markdown(
        f'<div class="side-status"><div class="ss-title">{title}</div>{body}</div>',
        unsafe_allow_html=True,
    )


def mini_note(text: str, warn: bool = False) -> None:
    """Fixed-height inline notice — never resizes the surrounding panel."""
    cls = "mini-note warn" if warn else "mini-note"
    st.markdown(f'<div class="{cls}"><span>{text}</span></div>',
                unsafe_allow_html=True)


def app_footer(left: str, right: str) -> None:
    """Closed page ending: top rule + colophon, no trailing scroll space."""
    st.markdown(
        f'<div class="app-footer"><span class="af-left">{left}</span>'
        f'<span class="af-right">{right}</span></div>',
        unsafe_allow_html=True,
    )


# Evaluation corpus options. Only 2019 LA has a dev split (seen attacks); the
# 2021 corpora are eval-only, so the available "Score on" options DEPEND on the
# corpus (see score_options_for / dev_corpus).
EVAL_CORPUS_CHOICES = ["2019 LA", "2021 LA", "2021 DF"]


def eval_corpora_for(choice: str):
    """Return [(label, eval_samples)] for the chosen eval corpus (a 1-item list,
    or empty if that corpus is unavailable).

    On the corpus-less web demo the local splits are empty, so we fall back to a
    small balanced eval set streamed from the public Hugging Face dataset — this
    is what lets the benchmark modes evaluate the pretrained models on the cloud
    exactly as they would locally (only training is off)."""
    if choice == "2019 LA":
        samples = get_samples("eval")
    elif choice == "2021 LA":
        samples = get_samples_2021_la()
    else:
        samples = get_samples_2021_df()
    if not samples and not corpus_available() and choice in HF_EVAL_DATASETS:
        samples = hf_eval_samples(choice, HF_EVAL_PER_CLASS)
    return [(choice, samples)] if samples else []


def score_options_for(corpus: str):
    """The valid 'Score on' options for a corpus. 2019 LA has dev + eval; the
    2021 corpora are eval-only."""
    return ["Dev", "Eval", "Dev + Eval"] if corpus == "2019 LA" else ["Eval"]


def eval_score_controls(
    prefix: str,
    disabled: bool = False,
    train_label: str = "Train on",
):
    """Unified 'Evaluation' group: the eval-corpus picker and the dependent
    score-on picker inside one framed block (used by every benchmark mode so
    they look consistent). Returns (corpus, score_split).

    Keys are f'{prefix}_corpus' and f'{prefix}_split'.
    Pass train_label="Trained on" when a model is already loaded.
    """
    ck, sk = f"{prefix}_corpus", f"{prefix}_split"
    _train_key = f"{prefix}_train_on"
    # Defaults (and self-heal any stale/invalid persisted value).
    if st.session_state.get(ck) not in EVAL_CORPUS_CHOICES:
        st.session_state[ck] = "2019 LA"
    if st.session_state.get(sk) not in ("Dev", "Eval", "Dev + Eval"):
        st.session_state[sk] = "Dev + Eval"
    if st.session_state.get(_train_key) is None:
        st.session_state[_train_key] = "2019 LA"

    def _keep_train_on():
        if st.session_state.get(_train_key) is None:
            st.session_state[_train_key] = "2019 LA"

    # The two controls live in one flex-row, fit-content frame (CSS) so the
    # "Score on" sits right NEXT TO "Evaluate on" and the blue box stays short.
    with st.container(key=f"evalgrp_{prefix}"):
        # Fixed "Train on" / "Trained on" — same segmented_control format as the
        # other two. on_change prevents the user from deselecting the single option.
        st.segmented_control(
            train_label, ["2019 LA"],
            key=_train_key, disabled=disabled, on_change=_keep_train_on,
        )
        corpus = st.segmented_control(
            "Evaluate on", EVAL_CORPUS_CHOICES, key=ck, disabled=disabled)
        corpus = corpus or "2019 LA"

        # "Score on" always renders the same 3 options so the box never shifts.
        # When the selected corpus has no dev split, Dev/Dev+Eval are visually
        # disabled via injected CSS (pointer-events:none) — the style element is
        # collapsed to zero height by PAGE_CSS but its rules still apply globally.
        # Session state is also reset to "Eval" so no stale selection leaks through.
        if corpus != "2019 LA":
            if st.session_state.get(sk) in ("Dev", "Dev + Eval"):
                st.session_state[sk] = "Eval"
            st.markdown(
                "<style>"
                "[class*='st-key-evalgrp_']>[data-testid='stElementContainer']:last-child "
                "button:nth-child(1),"
                "[class*='st-key-evalgrp_']>[data-testid='stElementContainer']:last-child "
                "button:nth-child(3)"
                "{pointer-events:none!important;opacity:0.28!important;}"
                "</style>",
                unsafe_allow_html=True,
            )
        score = st.segmented_control(
            "Score on", ["Dev", "Eval", "Dev + Eval"], key=sk, disabled=disabled)
        score = score or "Dev + Eval"
        if corpus != "2019 LA":
            score = "Eval"   # enforce even if CSS failed and user clicked Dev
    return corpus, score


def op_in_progress() -> bool:
    """True while ANY background job (full comparison OR CNN training) runs."""
    for key in ("bench_future", "cnn_future"):
        fut = st.session_state.get(key)
        if fut is not None and not fut.done():
            return True
    return False


def op_status():
    """Return (kind, label) of the running background job, or (None, None).

    kind is 'full' or 'cnn'; label is a short human description for the banner.
    """
    fut = st.session_state.get("bench_future")
    if fut is not None and not fut.done():
        return "full", "Full comparison running"
    fut = st.session_state.get("cnn_future")
    if fut is not None and not fut.done():
        return "cnn", "Training CNN"
    return None, None


def op_busy_notice() -> bool:
    """Return True while a background job runs (the running banner now lives in
    the sidebar, so this no longer renders anything in the page body — pages
    just use the return value to disable their run/train controls)."""
    return op_in_progress()


# Geometric symbols for the banner (no emoji): a small node-graph for the CNN
# and a bar-chart for the full comparison.
_OP_ICON_CNN = ('<svg viewBox="0 0 24 24" width="14" height="14" fill="none" '
                'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
                'stroke-linejoin="round"><circle cx="5" cy="6" r="1.8"/>'
                '<circle cx="5" cy="18" r="1.8"/><circle cx="12.5" cy="12" r="1.8"/>'
                '<circle cx="20" cy="6" r="1.8"/><circle cx="20" cy="18" r="1.8"/>'
                '<path d="M6.6 6.8 11 11M6.6 17.2 11 13M14 11 18.4 6.8M14 13 18.4 17.2"/>'
                '</svg>')
_OP_ICON_FULL = ('<svg viewBox="0 0 24 24" width="14" height="14" fill="none" '
                 'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
                 'stroke-linejoin="round"><path d="M4 20V11M10 20V4M16 20v-6M3 20h18"/></svg>')


def _op_banner_render() -> None:
    """Render the banner (or trigger the completion rerun). Shared by the live and
    idle fragment wrappers below."""
    for key in ("bench_future", "cnn_future"):
        fut = st.session_state.get(key)
        if fut is not None and fut.done():
            st.rerun(scope="app")
            return

    kind, label = op_status()
    if kind is None:
        return

    from src.jobs import progress as _progress
    pr  = _progress()
    pct = int(round(pr["frac"] * 100))
    sym = _OP_ICON_CNN if kind == "cnn" else _OP_ICON_FULL
    # Called inside a `with st.sidebar:` block (see app.py) — a fragment may only
    # write to its own parent container, so we do NOT open st.sidebar here. The
    # whole banner is a click target (invisible overlay button) that jumps to the
    # page where the job runs.
    with st.container(key="opbanner"):
        st.markdown(
            f'<div class="op-banner"><div class="ob-head">'
            f'<span class="ob-dot"></span><span class="ob-ic">{sym}</span>'
            f'<span>{label}</span><span class="ob-go">open ›</span></div>'
            f'<div class="ob-sub">{pr["label"]}</div>'
            f'<div class="ob-track"><span style="width:{pct}%"></span></div>'
            f'<div class="ob-pct">{pct}% · stage {pr["done"]}/{max(pr["total"], 1)}'
            '</div></div>',
            unsafe_allow_html=True,
        )
        if st.button("open", key="opbanner_go", width="stretch"):
            st.session_state["bench_choice"] = "cnn" if kind == "cnn" else "full"
            if kind == "cnn":
                st.session_state["cnn_focus_curves"] = True   # open Training curves
            st.switch_page("app_pages/2_Benchmark.py")


@st.fragment(run_every=2.0)
def _op_banner_live() -> None:
    _op_banner_render()


@st.fragment
def _op_banner_idle() -> None:
    _op_banner_render()


def op_banner_fragment() -> None:
    """Global background-job banner, pinned to the bottom of the sidebar.

    Appears on EVERY page (rendered before the page script runs, so it survives
    st.stop()). It auto-refreshes every 2 s ONLY while a job is running; when the
    job finishes it triggers ONE full app rerun so app.py collects the result.

    The 2 s timer (run_every) is attached only in the running state: an idle app
    would otherwise keep a live timer that, after the rapid reruns of startup or
    page navigation, fires against a container that no longer exists — logging
    'The fragment ... does not exist anymore' warnings on every rerun."""
    if op_in_progress():
        _op_banner_live()
    else:
        _op_banner_idle()


def launch_full_comparison(classic_subset: int = 4000, include_cnn: bool = True) -> None:
    """Submit a full comparison with sensible defaults (2019 LA, dev + eval).

    Used by the sidebar quick-launch button so the headline benchmark — the base
    for everything else in the app — is one click away from any page."""
    from src.jobs import submit_benchmark
    ext = get_extractor()
    st.session_state["bench_future"] = submit_benchmark(
        ext=ext, feat_labels=FeatureExtractor.OPTION_NAMES,
        base_params=dict(load_config()["train_params"]),
        train=get_samples("train"), primary=get_samples("dev"), pname="dev",
        eval_corpora=eval_corpora_for("2019 LA"),
        classic_subset=int(classic_subset), cnn_subset=0,
        include_cnn=include_cnn, seed=42,
    )
    st.session_state["bench_score"] = "Dev + Eval"
    st.session_state["op_running"] = True


def render_full_cta() -> None:
    """Sidebar quick-launch for the full comparison, pinned to the same bottom
    spot the running banner uses (shown only when nothing is running). Once a
    full comparison has finished, it turns into a shortcut to its leaderboard."""
    with st.sidebar:
        with st.container(key="opcta"):
            if st.session_state.get("bench_done"):
                if st.button("See full comparison", key="cta_see_full",
                             type="primary", width="stretch",
                             icon=":material/leaderboard:"):
                    st.session_state["bench_choice"] = "full"
                    st.switch_page("app_pages/2_Benchmark.py")
            elif st.button("Run full comparison", key="cta_full_cmp",
                           type="primary", width="stretch",
                           icon=":material/playlist_play:",
                           disabled=not corpus_available()):
                launch_full_comparison()
                # Land on the full-comparison page so its live progress is visible.
                st.session_state["bench_choice"] = "full"
                st.switch_page("app_pages/2_Benchmark.py")


# ===========================================================================
# Signal statistics
# ===========================================================================

def compute_signal_stats(y: np.ndarray, sr: int) -> Dict:
    """Return a dict with key audio statistics for display as metric cards."""
    duration   = len(y) / sr
    rms        = float(np.sqrt(np.mean(y ** 2)))
    rms_db     = float(20 * np.log10(rms + _EPS))   # dBFS: ~-30 to -6 for speech
    zcr        = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    centroid   = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    return {
        "duration_s":   duration,
        "rms":          rms,
        "rms_db":       rms_db,
        "zcr":          zcr,
        "centroid_hz":  centroid,
    }


# ===========================================================================
# Plotting helpers — all return a plt.Figure for st.pyplot()
# ===========================================================================

def _fig_style(ax: plt.Axes) -> None:
    """Apply a clean, dark-theme-consistent style to any axis."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(_FIG_EDGE)
    ax.grid(True, alpha=0.4, linewidth=0.5, color=_FIG_GRID)
    ax.tick_params(labelsize=8, colors=_FIG_TEXT)


def fig_waveform(
    y: np.ndarray,
    sr: int,
    title: str = "Waveform",
    label: Optional[str] = None,
    figsize: Tuple[float, float] = (9, 2.5),
) -> plt.Figure:
    """Time-domain amplitude plot, colour-coded by class label.

    ``figsize`` is exposed so callers that place the waveform next to another
    figure (e.g. the CNN input) can match heights exactly."""
    if label and "spoof" in str(label).lower():
        color = SPOOF_COLOR
    elif label and "bonafide" in str(label).lower():
        color = BONAFIDE_COLOR
    else:
        color = NEUTRAL_COLOR

    fig, ax = plt.subplots(figsize=figsize)
    t = np.arange(len(y)) / sr
    ax.fill_between(t, y, alpha=0.22, color=color)
    ax.plot(t, y, linewidth=0.6, color=color)
    ax.axhline(0, color="#aaa", linewidth=0.4, linestyle="--", alpha=0.6)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Amplitude", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.margins(x=0)
    _fig_style(ax)
    fig.tight_layout()
    return fig


def _specshow(
    matrix: np.ndarray,
    sr: int,
    hop_length: int,
    title: str,
    y_axis: Optional[str] = None,
    cmap: str = "magma",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 3.2))
    img = librosa.display.specshow(
        matrix, sr=sr, hop_length=hop_length,
        x_axis="time", y_axis=y_axis, ax=ax, cmap=cmap,
    )
    fig.colorbar(img, ax=ax, format="%+.0f", pad=0.01)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def fig_stft_db(y: np.ndarray, extractor: FeatureExtractor) -> plt.Figure:
    mag = np.abs(librosa.stft(y, n_fft=extractor.n_fft,
                              hop_length=extractor.hop_length, window="hann"))
    db  = librosa.amplitude_to_db(mag, ref=np.max)
    return _specshow(db, extractor.sample_rate, extractor.hop_length,
                     "STFT Magnitude (dB)", y_axis="hz")


def fig_cnn_input(y: np.ndarray, extractor: FeatureExtractor) -> plt.Figure:
    matrix = extractor.get_spectrogram_matrix(y)
    # constrained_layout packs the axes + colorbar to fill the figure width (plain
    # tight_layout left a right-hand gap with the colorbar); the figure is rendered
    # with bbox_inches=None by the caller, so its 9×3.2 canvas matches the waveform's
    # exactly → the two panels end up the same height with no empty strip.
    fig, ax = plt.subplots(figsize=(9, 3.2), layout="constrained")
    img = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
    fig.colorbar(img, ax=ax, pad=0.01)
    ax.set_title(
        f"CNN Input — z-scored STFT-dB  "
        f"({extractor.freq_bins} freq bins × {extractor.time_frames} time frames)",
        fontsize=10, fontweight="bold",
    )
    ax.set_xlabel("Time frames", fontsize=9)
    ax.set_ylabel("Frequency bins", fontsize=9)
    ax.tick_params(labelsize=8)
    return fig


def fig_mfcc(y: np.ndarray, extractor: FeatureExtractor) -> plt.Figure:
    mfcc = librosa.feature.mfcc(
        y=y, sr=extractor.sample_rate, n_mfcc=extractor.n_mfcc,
        n_fft=extractor.n_fft, hop_length=extractor.hop_length,
        n_mels=extractor.n_mels, window="hann",
    )
    return _specshow(mfcc, extractor.sample_rate, extractor.hop_length,
                     f"MFCC  ({extractor.n_mfcc} coefficients)", cmap="coolwarm")


def fig_lfcc(y: np.ndarray, extractor: FeatureExtractor) -> plt.Figure:
    power      = extractor._stft_magnitude(y) ** 2
    band_energy = extractor._linear_filterbank @ power
    log_energy  = np.log(band_energy + _EPS)
    cepstrum    = dct(log_energy, type=2, axis=0, norm="ortho")[: extractor.n_lfcc]
    return _specshow(cepstrum, extractor.sample_rate, extractor.hop_length,
                     f"LFCC  ({extractor.n_lfcc} linear cepstral coefficients)",
                     cmap="coolwarm")


def fig_cqcc(y: np.ndarray, extractor: FeatureExtractor) -> plt.Figure:
    cqt        = librosa.cqt(y, sr=extractor.sample_rate,
                             hop_length=extractor.hop_length,
                             n_bins=extractor.cqcc_n_bins,
                             bins_per_octave=extractor.cqcc_bins_per_octave)
    log_energy = np.log(np.abs(cqt) ** 2 + _EPS)
    cepstrum   = dct(log_energy, type=2, axis=0, norm="ortho")[: extractor.n_cqcc]
    return _specshow(cepstrum, extractor.sample_rate, extractor.hop_length,
                     f"CQCC  ({extractor.n_cqcc} constant-Q cepstral coefficients)",
                     cmap="coolwarm")


def fig_activation_grid(
    activation: np.ndarray,
    title: str,
    max_maps: int = 16,
) -> plt.Figure:
    """Grid of feature maps from one CNN convolutional block.

    Args:
        activation: shape (channels, H, W) for a single sample.
        title:      Block title.
        max_maps:   Maximum channels to display.
    """
    n    = min(activation.shape[0], max_maps)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i in range(len(axes)):
        ax = axes[i]
        if i < n:
            ax.imshow(activation[i], aspect="auto", origin="lower", cmap="inferno")
            ax.set_title(f"ch {i}", fontsize=7)
        ax.axis("off")
    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def fig_activation_evolution(
    acts: List,
    title: str,
    max_blocks: int = 4,
) -> plt.Figure:
    """Compact one-row summary of how a CNN transforms the spectrogram block by
    block: each convolutional block is reduced to a single heatmap (the mean over
    its channels) and laid left→right, so the overall evolution is visible at a
    glance without scrolling through every feature map.

    Args:
        acts: list of per-block activation tensors, each shaped (1, C, H, W) or
              (C, H, W) for a single sample.
        title: figure title (model name).
    """
    blocks = list(acts)[:max_blocks]
    n = max(1, len(blocks))
    # FIXED figure size regardless of block count: ResNet (4 blocks) and the
    # 3-Block CNN (3 blocks) must render at the same aspect ratio so that, placed
    # in equal-width columns, they end up exactly the same height.
    fig, axes = plt.subplots(1, n, figsize=(7.2, 2.1))
    axes = np.atleast_1d(axes).ravel()
    for i in range(len(axes)):
        ax = axes[i]
        if i < len(blocks):
            act = blocks[i]
            arr = act.numpy() if hasattr(act, "numpy") else np.asarray(act)
            if arr.ndim == 4:        # (1, C, H, W) → drop batch
                arr = arr[0]
            heat = arr.mean(axis=0)  # mean over channels → (H, W)
            ax.imshow(heat, aspect="auto", origin="lower", cmap="inferno")
            ax.set_title(f"Block {i + 1}", fontsize=8)
        ax.axis("off")
    # Title INSIDE the canvas (rendered with bbox_inches=None, anything at y>1 is
    # clipped). Keep a small top margin so it isn't cut, but sit it just above the
    # heatmaps (rect top close to the title) so it hugs the images, not floats away.
    fig.suptitle(title, fontsize=9.5, fontweight="bold", y=0.90)
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    return fig


def fig_overall_split_bar(n_bonafide: int, n_spoof: int) -> plt.Figure:
    """Tiny 100%-stacked horizontal bar of the overall bonafide/spoof share.

    Only short "NN%" labels go inside the segments (a long word would overflow
    the narrow bonafide slice); a legend below names the colours.
    """
    total = max(n_bonafide + n_spoof, 1)
    pb = 100 * n_bonafide / total
    ps = 100 * n_spoof / total
    fig, ax = plt.subplots(figsize=(4.8, 1.25))
    ax.barh([0], [pb], color=BONAFIDE_COLOR, edgecolor=_FIG_BG,
            height=0.5, label="Bonafide")
    ax.barh([0], [ps], left=[pb], color=SPOOF_COLOR, edgecolor=_FIG_BG,
            height=0.5, label="Spoof")
    ax.text(pb / 2, 0, f"{pb:.0f}%", ha="center", va="center",
            color="white", fontsize=9, fontweight="bold")
    ax.text(pb + ps / 2, 0, f"{ps:.0f}%", ha="center", va="center",
            color="white", fontsize=9, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 0.6)
    ax.axis("off")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2,
              frameon=False, fontsize=8, handlelength=1.1, columnspacing=1.6)
    ax.set_title("Overall class share · train + dev", fontsize=8.5,
                 fontweight="bold", color=_FIG_TEXT, pad=10)
    fig.tight_layout()
    return fig


def fig_corpus_overview(
    train_samples: List[Tuple[str, int]],
    dev_samples:   List[Tuple[str, int]],
) -> plt.Figure:
    """Stacked bar showing bonafide / spoof split for train and dev subsets."""
    labels = ["Train", "Dev"]
    bon = [sum(1 for _, l in s if l == LABEL_BONAFIDE)
           for s in (train_samples, dev_samples)]
    spo = [sum(1 for _, l in s if l == LABEL_SPOOF)
           for s in (train_samples, dev_samples)]
    totals = [b + s for b, s in zip(bon, spo)]

    x   = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.bar(x, bon, label="Bonafide (real)",    color=BONAFIDE_COLOR, alpha=0.9, width=0.6)
    ax.bar(x, spo, bottom=bon, label="Spoof (deepfake)", color=SPOOF_COLOR, alpha=0.9, width=0.6)

    for i, (b, s, t) in enumerate(zip(bon, spo, totals)):
        ax.text(i, t + max(totals) * 0.02, f"{t:,}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=_FIG_TEXT)
        ax.text(i, b / 2, f"{b:,}", ha="center", va="center",
                fontsize=8, color="white", fontweight="600")
        ax.text(i, b + s / 2, f"{s:,}", ha="center", va="center",
                fontsize=8, color="white", fontweight="600")

    # Headroom so the total labels never collide with the top spine.
    ax.set_ylim(0, max(totals) * 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Audio files", fontsize=9)
    ax.set_title("Class distribution by subset", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    # Legend BELOW the axes — never overlaps the bars.
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1),
              ncol=2, frameon=False, handlelength=1.2, columnspacing=1.4)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    return fig
