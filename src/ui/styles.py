# -*- coding: utf-8 -*-
"""src/ui/styles.py — CSS, the Light/Dark Side theme and HTML-wrapping helpers.

Split out of the former monolithic src/ui_helpers.py and re-exported there for
backward compatibility, so existing `from src.ui_helpers import ...` call sites
are unaffected. Contains the page stylesheet, the lightsaber/theme CSS, the
dark->light colour swap, and build_page_css / themed / inject_css.
"""

from pathlib import Path
from typing import Tuple

import streamlit as st

# ── Colour palette ────────────────────────────────────────────────────────── #
BONAFIDE_COLOR = "#42A5F5"   # blue  — real voice  (lighter for dark bg)
SPOOF_COLOR    = "#EF5350"   # red   — deepfake
NEUTRAL_COLOR  = "#78909C"   # slate — unknown / upload
# ── Shared CSS injected at the top of every page ──────────────────────────── #
# Page stylesheet externalised to static/styles.css (IDE highlighting + browser
# caching); loaded once at import. _light_swap() still rewrites its colour
# literals at build time for the Light Side, so behaviour is unchanged.
PAGE_CSS = (Path(__file__).resolve().parents[2] / "static" / "styles.css").read_text(encoding="utf-8")


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
