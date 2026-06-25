# -*- coding: utf-8 -*-
"""
app_pages/5_Settings.py — Appearance & accessibility control panel.

Persistence note: Streamlit discards a widget's state once its page stops
rendering, so binding settings straight to widget keys made them reset when you
navigated away. Instead every control writes to a PLAIN session key (sw_theme,
sw_bg, …) through an on_change callback; app.py reads those plain keys on every
rerun. The widget keys (…_ctl) are just the live control state for this page.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st  # noqa: E402

from src.ui_helpers import (  # noqa: E402
    app_footer, sidebar_panel, theme_mode, themed,
)

_SIDE_LABELS = {"dark": "Dark Side", "light": "Light Side"}
_DEFAULTS = {
    "sw_theme": "dark", "sw_saber": "Auto", "sw_bg": "Star Wars",
    "sw_bg_intensity": "Normal", "sw_show_ships": True,
    "sw_show_deathstar": True, "sw_reduced_motion": False,
    "sw_contrast": False, "sw_text_scale": "Normal",
}


# ── Callbacks: copy the live widget value into the persistent plain key ────────
def _sync(plain: str, ctl: str) -> None:
    st.session_state[plain] = st.session_state[ctl]


def _sync_side() -> None:
    st.session_state["sw_theme"] = (
        "light" if st.session_state.get("sw_side_ctl") == "Light Side" else "dark"
    )


def _reset() -> None:
    for k, v in _DEFAULTS.items():
        st.session_state[k] = v
    st.session_state["sw_side_ctl"]      = _SIDE_LABELS["dark"]
    st.session_state["sw_saber_ctl"]     = "Auto"
    st.session_state["sw_bg_ctl"]        = "Star Wars"
    st.session_state["sw_intensity_ctl"] = "Normal"
    st.session_state["sw_ships_ctl"]     = True
    st.session_state["sw_ds_ctl"]        = True
    st.session_state["sw_rm_ctl"]        = False
    st.session_state["sw_hc_ctl"]        = False
    st.session_state["sw_ts_ctl"]        = "Normal"


# Seed each widget key from its persistent value (so controls reflect the saved
# choice without passing default=+key=, which would warn). setdefault only fills
# the first time the widget is created.
st.session_state.setdefault("sw_side_ctl", _SIDE_LABELS[theme_mode()])
st.session_state.setdefault("sw_saber_ctl", st.session_state.get("sw_saber", "Auto"))
st.session_state.setdefault("sw_bg_ctl", st.session_state.get("sw_bg", "Star Wars"))
st.session_state.setdefault("sw_intensity_ctl", st.session_state.get("sw_bg_intensity", "Normal"))
st.session_state.setdefault("sw_ships_ctl", bool(st.session_state.get("sw_show_ships", True)))
st.session_state.setdefault("sw_ds_ctl", bool(st.session_state.get("sw_show_deathstar", True)))
st.session_state.setdefault("sw_rm_ctl", bool(st.session_state.get("sw_reduced_motion")))
st.session_state.setdefault("sw_hc_ctl", bool(st.session_state.get("sw_contrast")))
st.session_state.setdefault("sw_ts_ctl", st.session_state.get("sw_text_scale", "Normal"))


# ── Page-local styling (themed so it swaps cleanly on the Light Side) ──────────
st.markdown(themed("""
<style>
/* Section headers — NOT a glowing saber rule (too saturated on this dense page).
   Instead a compact saber-coloured number chip with a soft glow: eye-catching but
   far less invasive, and it still tracks the chosen lightsaber colour. */
.set-sec { display: flex; align-items: center; gap: 0.65rem; margin: 1.5rem 0 0.15rem; }
.set-sec .ss-num {
    display: inline-flex; align-items: center; justify-content: center;
    min-width: 1.75rem; height: 1.75rem; border-radius: 0.55rem;
    font-size: 0.82rem; font-weight: 800; color: #fff; background: var(--saber);
    box-shadow: 0 0 10px var(--saber-glow); letter-spacing: 0.02em;
}
.set-sec .ss-title {
    font-size: 1.2rem; font-weight: 750; color: #E8EDF8; letter-spacing: -0.01em;
}
.set-sub2 { font-size: 0.82rem; opacity: 0.6; margin: 0 0 0.7rem 2.4rem; }

/* Live lightsaber preview — a full blade + a properly modelled metal hilt that
   updates to whatever colour you pick. */
.saber-demo {
    position: relative; height: 9px; border-radius: 5px;
    margin: 1.25rem 0 0.5rem 38px; background: var(--saber);
    box-shadow: 0 0 12px var(--saber-glow), 0 0 26px var(--saber-glow),
                0 0 48px var(--saber-glow);
    animation: saberPulse 2.4s ease-in-out infinite;
}
.saber-demo::before {
    content: ""; position: absolute; top: 50%; left: -36px; transform: translateY(-50%);
    width: 36px; height: 17px; border-radius: 4px 2px 2px 4px;
    background:
        linear-gradient(90deg, transparent 0 79%, rgba(255,255,255,0.55) 79% 81%,
                        var(--saber) 81% 100%),
        repeating-linear-gradient(90deg, rgba(255,255,255,0.12) 0 1px,
                        rgba(0,0,0,0.22) 3px 4px, transparent 4px 7px),
        linear-gradient(180deg, #f1f3f7 0%, #c4cad4 20%, #889 50%,
                        #4d5362 80%, #2a2f3a 100%);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.5),
                inset 0 -1px 0 rgba(0,0,0,0.55), 0 1px 3px rgba(0,0,0,0.55);
}
.saber-hint { font-size: 0.74rem; opacity: 0.6; margin-top: 0.15rem; }

/* Cantina rumour (Konami) with little keycaps. */
.konami-hint {
    font-size: 0.76rem; color: #AFC3E8; line-height: 2.1; margin-top: 0.4rem;
}
.konami-hint .kbd {
    display: inline-block; min-width: 1.15em; text-align: center;
    padding: 0.06rem 0.4rem; margin: 0 0.08rem; font-size: 0.72rem; font-weight: 700;
    color: #C9D7F5; background: rgba(79,139,249,0.10);
    border: 1px solid rgba(79,139,249,0.32); border-radius: 0.35rem;
    box-shadow: 0 1px 0 rgba(0,0,0,0.3);
}
</style>
"""), unsafe_allow_html=True)


# ── Title (a normal page title, like every other page) ────────────────────────
st.title("Settings")
st.caption(
    "Choose your side of the Force, pick a lightsaber colour, command the viewport "
    "background and tune accessibility. Changes apply live across the whole app; "
    "press Save changes to confirm them for this session."
)


def _sec(num: str, title: str, sub: str) -> None:
    """Compact, low-saturation section header (no glowing saber rule)."""
    st.markdown(
        f'<div class="set-sec"><span class="ss-num">{num}</span>'
        f'<span class="ss-title">{title}</span></div>'
        f'<div class="set-sub2">{sub}</div>',
        unsafe_allow_html=True,
    )


# ── Appearance ────────────────────────────────────────────────────────────────
_sec("01", "Appearance", "Your side of the Force and a lightsaber to match.")

_c1, _c2 = st.columns(2, gap="large")
with _c1:
    with st.container(border=True):
        st.markdown('<div class="section-label">Side of the Force</div>',
                    unsafe_allow_html=True)
        st.segmented_control(
            "Side of the Force", ["Dark Side", "Light Side"],
            key="sw_side_ctl", on_change=_sync_side, label_visibility="collapsed",
            help="Dark Side = the deep-space dark theme. Light Side = a high-key "
                 "light theme for bright rooms or projectors.",
        )
        st.caption("Dark Side keeps the deep-space look · Light Side is the "
                   "accessibility light theme with a Tatooine twin-sun backdrop.")
with _c2:
    with st.container(border=True):
        st.markdown('<div class="section-label">Lightsaber colour</div>',
                    unsafe_allow_html=True)
        st.selectbox(
            "Lightsaber colour",
            ["Auto", "Blue", "Green", "Red", "Purple", "Amber"],
            key="sw_saber_ctl", on_change=_sync, args=("sw_saber", "sw_saber_ctl"),
            label_visibility="collapsed",
            help="Colour of every glowing accent blade across the app. Auto = a red "
                 "Sith blade on the Dark Side, a blue Jedi blade on the Light Side.",
        )
        st.markdown('<div class="saber-demo"></div>'
                    '<div class="saber-hint">Live preview — this is your blade.</div>',
                    unsafe_allow_html=True)


# ── Viewport / background ─────────────────────────────────────────────────────
_sec("02", "Viewport", "The space beyond the canopy.")

with st.container(border=True):
    _b1, _b2 = st.columns([3, 2], gap="large")
    with _b1:
        st.markdown('<div class="section-label">Background</div>',
                    unsafe_allow_html=True)
        st.segmented_control(
            "Background", ["Star Wars", "Particle network", "Off"],
            key="sw_bg_ctl", on_change=_sync, args=("sw_bg", "sw_bg_ctl"),
            label_visibility="collapsed",
            help="Star Wars = a starfield with a drifting Death Star, shooting stars "
                 "and ships flying wandering routes. Particle network = the original "
                 "connected-dots field. Off = a plain background.",
        )
    with _b2:
        st.markdown('<div class="section-label">Intensity</div>',
                    unsafe_allow_html=True)
        st.segmented_control(
            "Intensity", ["Subtle", "Normal", "Busy"],
            key="sw_intensity_ctl", on_change=_sync,
            args=("sw_bg_intensity", "sw_intensity_ctl"),
            label_visibility="collapsed",
            help="How many stars and how often ships fly across.",
        )

    st.markdown('<div class="section-label" style="margin-top:0.6rem;">Fleet</div>',
                unsafe_allow_html=True)
    _f1, _f2 = st.columns(2, gap="large")
    with _f1:
        st.toggle(
            "Passing ships", key="sw_ships_ctl",
            on_change=_sync, args=("sw_show_ships", "sw_ships_ctl"),
            help="TIEs, X-wings, the Millennium Falcon and Star Destroyers crossing "
                 "the viewport on wandering flight paths.",
        )
    with _f2:
        st.toggle(
            "Death Star", key="sw_ds_ctl",
            on_change=_sync, args=("sw_show_deathstar", "sw_ds_ctl"),
            help="The drifting battle station in the upper field of the Star Wars "
                 "background.",
        )

    st.markdown(themed(
        '<div class="konami-hint">Rumour from the cantina: try '
        '<span class="kbd">&uarr;</span><span class="kbd">&uarr;</span>'
        '<span class="kbd">&darr;</span><span class="kbd">&darr;</span>'
        '<span class="kbd">&larr;</span><span class="kbd">&rarr;</span>'
        '<span class="kbd">&larr;</span><span class="kbd">&rarr;</span>'
        '<span class="kbd">B</span><span class="kbd">A</span> '
        'for a jump to lightspeed.</div>'),
        unsafe_allow_html=True,
    )


# ── Accessibility ─────────────────────────────────────────────────────────────
_sec("03", "Accessibility", "Make the app easier to read and calmer.")

with st.container(border=True):
    _a1, _a2, _a3 = st.columns(3, gap="large")
    with _a1:
        st.toggle(
            "Reduce motion", key="sw_rm_ctl",
            on_change=_sync, args=("sw_reduced_motion", "sw_rm_ctl"),
            help="Stops the ambient animations (saber glow, drifting background, "
                 "passing ships) for a calmer, distraction-free interface.",
        )
    with _a2:
        st.toggle(
            "High contrast", key="sw_hc_ctl",
            on_change=_sync, args=("sw_contrast", "sw_hc_ctl"),
            help="Stronger text and thicker borders for better legibility.",
        )
    with _a3:
        st.markdown('<div class="section-label">Text size</div>',
                    unsafe_allow_html=True)
        st.segmented_control(
            "Text size", ["Normal", "Large", "Larger"],
            key="sw_ts_ctl", on_change=_sync, args=("sw_text_scale", "sw_ts_ctl"),
            label_visibility="collapsed",
        )


# ── Save / reset ──────────────────────────────────────────────────────────────
st.markdown('<div style="height:0.4rem;"></div>', unsafe_allow_html=True)
_r1, _r2, _r3 = st.columns([2.4, 1, 1], vertical_alignment="center")
with _r1:
    st.markdown(
        '<div class="info-card">'
        '<div class="ic-title">About these settings</div>'
        '<p class="ic-body">Preferences live in your browser session only and '
        'reset when you reload the public demo. The Dark Side is the default so '
        'first-time visitors always see the intended look.</p></div>',
        unsafe_allow_html=True,
    )
with _r2:
    st.button("Restore defaults", icon=":material/restart_alt:",
              on_click=_reset, width="stretch")
with _r3:
    if st.button("Save changes", type="primary", icon=":material/save:",
                 width="stretch"):
        st.toast("Settings saved for this session.", icon=":material/check_circle:")

sidebar_panel(
    "Current setup",
    rows=[
        ("Side", _SIDE_LABELS[theme_mode()]),
        ("Saber", st.session_state.get("sw_saber", "Auto")),
        ("Background", st.session_state.get("sw_bg", "Star Wars")),
        ("Intensity", st.session_state.get("sw_bg_intensity", "Normal")),
        ("Ships", "On" if st.session_state.get("sw_show_ships", True) else "Off"),
        ("Death Star", "On" if st.session_state.get("sw_show_deathstar", True) else "Off"),
        ("Reduced motion", "On" if st.session_state.get("sw_reduced_motion") else "Off"),
        ("Text size", st.session_state.get("sw_text_scale", "Normal")),
    ],
)

app_footer("Settings", "May the Force be with you.")
