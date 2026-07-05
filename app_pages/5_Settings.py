# -*- coding: utf-8 -*-
"""
app_pages/5_Settings.py, Appearance & accessibility control panel.

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
    ACCENT_COLORS, app_footer, sidebar_panel, theme_mode, themed,
)

_SIDE_LABELS = {"dark": "Dark Side", "light": "Light Side"}
_DEFAULTS = {
    "sw_theme": "dark", "sw_saber": "Red", "sw_bg": "Star Wars",
    "sw_bg_intensity": "Normal", "sw_show_ships": True,
    "sw_show_deathstar": True, "sw_reduced_motion": False,
    "sw_contrast": False, "sw_text_scale": "Normal",
    "sw_underline": False, "sw_text_spacing": False, "sw_audio_color": "Red",
}

# Widget key → (persistent plain key, plain→widget transform). The SINGLE wiring
# table used both to seed the controls and to reset them, so a new setting only
# has to be added here (the old hand-maintained lists drifted apart).
_CTL_WIRING = {
    "sw_side_ctl":      ("sw_theme",          lambda v: _SIDE_LABELS[v]),
    # Guard legacy values (the saber key once allowed "Auto") so a stale session
    # can never seed the selectbox with an option it doesn't have.
    "sw_color_ctl":     ("sw_saber",
                         lambda v: v if v in ACCENT_COLORS else "Red"),
    "sw_bg_ctl":        ("sw_bg",             None),
    "sw_intensity_ctl": ("sw_bg_intensity",   None),
    "sw_ships_ctl":     ("sw_show_ships",     bool),
    "sw_ds_ctl":        ("sw_show_deathstar", bool),
    "sw_rm_ctl":        ("sw_reduced_motion", bool),
    "sw_hc_ctl":        ("sw_contrast",       bool),
    "sw_ul_ctl":        ("sw_underline",      bool),
    "sw_sp_ctl":        ("sw_text_spacing",   bool),
    "sw_ts_ctl":        ("sw_text_scale",     None),
}


# ── Callbacks: copy the live widget value into the persistent plain key ────────
def _sync(plain: str, ctl: str) -> None:
    st.session_state[plain] = st.session_state[ctl]


def _sync_color() -> None:
    """One colour drives both the saber accents and the Home audio bars."""
    color_val = st.session_state.get("sw_color_ctl")
    st.session_state["sw_saber"] = color_val
    st.session_state["sw_audio_color"] = color_val


def _sync_side() -> None:
    light = st.session_state.get("sw_side_ctl") == "Light Side"
    st.session_state["sw_theme"] = "light" if light else "dark"
    # Each side ignites its signature blade: Jedi blue on the Light Side, Sith
    # red on the Dark Side. It keeps the light palette coherently blue; the
    # colour picker below can still override it afterwards.
    accent = "Blue" if light else "Red"
    for _k in ("sw_saber", "sw_audio_color", "sw_color_ctl"):
        st.session_state[_k] = accent


def _reset() -> None:
    for k, v in _DEFAULTS.items():
        st.session_state[k] = v
    for ctl, (plain, conv) in _CTL_WIRING.items():
        v = _DEFAULTS[plain]
        st.session_state[ctl] = conv(v) if conv else v
    st.toast("Defaults restored.", icon=":material/restart_alt:")


# Seed each widget key from its persistent value (so controls reflect the saved
# choice without passing default=+key=, which would warn). setdefault only fills
# the first time the widget is created.
for _ctl, (_plain, _conv) in _CTL_WIRING.items():
    _v = st.session_state.get(_plain, _DEFAULTS[_plain])
    st.session_state.setdefault(_ctl, _conv(_v) if _conv else _v)


# ── Page-local styling (themed so it swaps cleanly on the Light Side) ──────────
st.markdown(themed("""
<style>
/* Section headers, NOT a glowing saber rule (too saturated on this dense page).
   Instead a compact saber-coloured number chip with a soft glow: eye-catching but
   far less invasive, and it still tracks the chosen lightsaber colour.
   The chip darkens the saber colour under the white numeral (color-mix) so the
   pairing meets AA contrast even for the bright Amber/Green blades. */
.set-sec { display: flex; align-items: center; gap: 0.65rem; margin: 1.15rem 0 0.15rem; }
.set-sec .ss-num {
    display: inline-flex; align-items: center; justify-content: center;
    min-width: 1.75rem; height: 1.75rem; border-radius: 0.55rem;
    font-size: 0.82rem; font-weight: 800; color: #fff;
    background: color-mix(in srgb, var(--saber) 58%, #000);
    border: 1px solid var(--saber);
    box-shadow: 0 0 10px var(--saber-glow); letter-spacing: 0.02em;
}
.set-sec .ss-title {
    font-size: 1.2rem; font-weight: 750; color: #E8EDF8; letter-spacing: -0.01em;
}
/* Quiet rule filling the rest of the row: ties each section to the saber tint
   and gives the page the same editorial rhythm as the other pages' sec-heads,
   without the animated blade (too loud on a dense control page). */
.set-sec .ss-rule {
    flex: 1 1 auto; min-width: 2rem; height: 1px; align-self: center;
    background: linear-gradient(90deg, var(--saber-glow), transparent);
    opacity: 0.45;
}
/* Real colour instead of opacity: translucent text fell below AA contrast. */
.set-sub2 { font-size: 0.82rem; color: #8FA3CE; margin: 0 0 0.55rem 2.4rem; }

/* A little air between stacked toggles inside the accessibility grid. */
[class*="st-key-a11ygrid"] [data-testid="stToggle"] { margin-bottom: 0.15rem; }
.set-hint { font-size: 0.74rem; color: #8FA3CE; margin: -0.35rem 0 0.6rem; }

/* Live lightsaber preview: the blade only, the metal hilt comes from the shared
   saber CSS (same ::before as the title rules), one hilt definition app-wide. */
.saber-demo {
    position: relative; height: 9px; border-radius: 5px; overflow: visible;
    margin: 1.25rem 0 0.5rem 38px; background: var(--saber);
    box-shadow: 0 0 12px var(--saber-glow), 0 0 26px var(--saber-glow),
                0 0 48px var(--saber-glow);
    animation: saberPulse 2.4s ease-in-out infinite;
}
.saber-hint { font-size: 0.74rem; color: #8FA3CE; margin-top: 0.15rem; }

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
        f'<span class="ss-title">{title}</span><span class="ss-rule"></span></div>'
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
        st.caption("Dark Side keeps the deep-space look with a red blade · "
                   "Light Side switches to a blue-on-light theme (blue blade) "
                   "with a Tatooine twin-sun backdrop.")
with _c2:
    with st.container(border=True):
        st.markdown('<div class="section-label">Accent & Audio colour</div>',
                    unsafe_allow_html=True)
        # nosearch_* container → CSS makes this a pure dropdown: no text caret, no
        # type-to-filter, click to open and pick only (there are just 5 named
        # colours, typing to search adds nothing but a stray focus caret).
        with st.container(key="nosearch_color"):
            st.selectbox(
                "Accent & Audio colour",
                list(ACCENT_COLORS),
                key="sw_color_ctl", on_change=_sync_color,
                label_visibility="collapsed",
                help="Colour of the lightsaber blade accents across the app and the dancing audio bars on the home page.",
            )
        st.markdown('<div class="saber-demo" aria-hidden="true"></div>'
                    '<div class="saber-hint">Live preview, this is your blade and audio colour.</div>',
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
_sec("03", "Accessibility", "Make the app easier to read and calmer. "
     "Everything applies instantly, across every page.")

with st.container(border=True, key="a11ygrid"):
    _a1, _a2 = st.columns(2, gap="large")
    with _a1:
        st.markdown('<div class="section-label">Comfort</div>',
                    unsafe_allow_html=True)
        st.toggle(
            "Reduce motion", key="sw_rm_ctl",
            on_change=_sync, args=("sw_reduced_motion", "sw_rm_ctl"),
            help="Stops the ambient animations (saber glow, drifting background, "
                 "passing ships) for a calmer, distraction-free interface.",
        )
        st.toggle(
            "Comfortable text spacing", key="sw_sp_ctl",
            on_change=_sync, args=("sw_text_spacing", "sw_sp_ctl"),
            help="Looser line height, letter and word spacing on running text "
                 "(in the spirit of WCAG 1.4.12), easier reading, e.g. for "
                 "dyslexic users.",
        )
    with _a2:
        st.markdown('<div class="section-label">Legibility</div>',
                    unsafe_allow_html=True)
        st.toggle(
            "High contrast", key="sw_hc_ctl",
            on_change=_sync, args=("sw_contrast", "sw_hc_ctl"),
            help="Stronger text and thicker borders for better legibility.",
        )
        st.toggle(
            "Underline links", key="sw_ul_ctl",
            on_change=_sync, args=("sw_underline", "sw_ul_ctl"),
            help="Links stay underlined everywhere, so colour is never the only "
                 "cue that something is clickable (WCAG 1.4.1).",
        )

    st.markdown('<div class="section-label" style="margin-top:0.6rem;">Text size</div>',
                unsafe_allow_html=True)
    st.segmented_control(
        "Text size", ["Normal", "Large", "Larger"],
        key="sw_ts_ctl", on_change=_sync, args=("sw_text_scale", "sw_ts_ctl"),
        label_visibility="collapsed",
        help="Scales the content text only; the layout keeps its proportions.",
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

# Reading aids summarised in one row so the panel stays scannable.
_aids = [label for key, label in (
    ("sw_reduced_motion", "motion"), ("sw_contrast", "contrast"),
    ("sw_underline", "links"), ("sw_text_spacing", "spacing"),
) if st.session_state.get(key)]
sidebar_panel(
    "Current setup",
    rows=[
        ("Side", _SIDE_LABELS[theme_mode()]),
        ("Colour", st.session_state.get("sw_saber", "Red")),
        ("Background", st.session_state.get("sw_bg", "Star Wars")),
        ("Intensity", st.session_state.get("sw_bg_intensity", "Normal")),
        ("Ships", "On" if st.session_state.get("sw_show_ships", True) else "Off"),
        ("Death Star", "On" if st.session_state.get("sw_show_deathstar", True) else "Off"),
        ("Reading aids", ", ".join(_aids) if _aids else "Off"),
        ("Text size", st.session_state.get("sw_text_scale", "Normal")),
    ],
)

app_footer("Settings", "May the Force be with you.")
