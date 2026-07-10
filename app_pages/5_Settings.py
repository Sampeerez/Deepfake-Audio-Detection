# -*- coding: utf-8 -*-
"""
app_pages/5_Settings.py, Appearance & accessibility control panel.

Staged-apply model
------------------
Picking an option NEVER changes the app on its own: every control stages its
value in a private widget key (…_ctl) and nothing else. The app reads the
PERSISTENT plain keys (sw_theme, sw_saber, …), which are only rewritten when the
user presses "Save changes" (or "Restore defaults"). So the two commit buttons
are the ONLY things that change the live appearance, exactly what "Save" implies.
The lightsaber colour is previewed live on THIS page only (a scoped --saber
override), so you can see a blade colour before committing it everywhere.

Persistence note: Streamlit discards a widget's state once its page stops
rendering, so unsaved edits are dropped on navigation (expected for a form); the
saved plain keys survive because app.py reseeds them every rerun.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st  # noqa: E402

from src.ui_helpers import ACCENT_COLORS, themed  # noqa: E402

_SIDE_LABELS = {"dark": "Dark Side", "light": "Light Side"}
_DEFAULTS = {
    "sw_theme": "dark", "sw_saber": "Red", "sw_bg": "Star Wars",
    "sw_bg_intensity": "Normal", "sw_show_ships": True,
    "sw_show_deathstar": True, "sw_reduced_motion": False,
    "sw_contrast": False, "sw_text_scale": "Normal",
    "sw_underline": False, "sw_text_spacing": False, "sw_audio_color": "Red",
}


def _side_to_plain(w):   return "light" if w == "Light Side" else "dark"
def _side_seed(p):       return _SIDE_LABELS.get(p, "Dark Side")
def _color_norm(v):      return v if v in ACCENT_COLORS else "Red"
def _ident(v):           return v


# One row per control: (widget key, [plain keys it commits to], widget→plain,
# plain→widget seed). The FIRST plain key seeds the widget. This single table
# drives seeding, the dirty check, Save and Restore, so a new setting is added
# in exactly one place.
_CONTROLS = [
    ("sw_side_ctl",      ["sw_theme"],                   _side_to_plain, _side_seed),
    ("sw_color_ctl",     ["sw_saber", "sw_audio_color"], _color_norm,    _color_norm),
    ("sw_bg_ctl",        ["sw_bg"],                       _ident,         _ident),
    ("sw_intensity_ctl", ["sw_bg_intensity"],             _ident,         _ident),
    ("sw_ships_ctl",     ["sw_show_ships"],               bool,           bool),
    ("sw_ds_ctl",        ["sw_show_deathstar"],           bool,           bool),
    ("sw_rm_ctl",        ["sw_reduced_motion"],           bool,           bool),
    ("sw_hc_ctl",        ["sw_contrast"],                 bool,           bool),
    ("sw_ul_ctl",        ["sw_underline"],                bool,           bool),
    ("sw_sp_ctl",        ["sw_text_spacing"],             bool,           bool),
    ("sw_ts_ctl",        ["sw_text_scale"],               _ident,         _ident),
]


# ── Staged-apply helpers ──────────────────────────────────────────────────────
def _has_unsaved() -> bool:
    """True when any staged widget value differs from the committed plain key."""
    for ctl, plains, to_plain, _seed in _CONTROLS:
        wv = st.session_state.get(ctl)
        if any(st.session_state.get(pk) != to_plain(wv) for pk in plains):
            return True
    return False


def _save() -> None:
    """Commit every staged widget value into its persistent plain key(s). app.py
    reads those at the top of the (post-callback) rerun, so the whole app updates."""
    for ctl, plains, to_plain, _seed in _CONTROLS:
        wv = st.session_state.get(ctl)
        for pk in plains:
            st.session_state[pk] = to_plain(wv)
    st.toast("Settings saved for this session.", icon=":material/check_circle:")


def _reset() -> None:
    """Restore defaults AND apply them (reset the staged widgets and commit)."""
    for ctl, plains, _to_plain, seed in _CONTROLS:
        st.session_state[ctl] = seed(_DEFAULTS[plains[0]])
        for pk in plains:
            st.session_state[pk] = _DEFAULTS[pk]
    st.toast("Defaults restored.", icon=":material/restart_alt:")


# Seed each widget key from its saved value so the controls open on the saved
# choice (setdefault only fills the first time the widget is created).
for _ctl, _plains, _to_plain, _seed in _CONTROLS:
    _v = st.session_state.get(_plains[0], _DEFAULTS[_plains[0]])
    st.session_state.setdefault(_ctl, _seed(_v))


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

/* Appearance row: the two cards share one height regardless of their content
   (the colour card is taller with its live blade preview), so their bottom
   edges line up. The two columns are already equal height; force the bordered
   card AND Streamlit's layout wrapper between it and the column to fill that
   height (that wrapper was the ~4px gap). */
[data-testid="stLayoutWrapper"]:has(> [class*="st-key-appbox_"]) { height: 100% !important; }
[class*="st-key-appbox_"] { height: 100% !important; }

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

/* Cantina rumour (Konami) with little keycaps. The extra bottom margin keeps
   the wrapped line off the viewport card's bottom border. */
.konami-hint {
    font-size: 0.76rem; color: #AFC3E8; line-height: 1.95;
    margin: 0.4rem 0 0.35rem;
}
.konami-hint .kbd {
    display: inline-block; min-width: 1.15em; text-align: center;
    padding: 0.06rem 0.4rem; margin: 0 0.08rem; font-size: 0.72rem; font-weight: 700;
    color: #C9D7F5; background: rgba(79,139,249,0.10);
    border: 1px solid rgba(79,139,249,0.32); border-radius: 0.35rem;
    box-shadow: 0 1px 0 rgba(0,0,0,0.3);
}

/* Save row: an inline saved / unsaved status next to the two commit buttons. */
.save-status { font-size: 0.82rem; font-weight: 600; display: flex;
    align-items: center; gap: 0.4rem; }
.save-status .sdot { width: 8px; height: 8px; border-radius: 50%; flex: 0 0 auto; }
.save-status.unsaved { color: #D9BC8A; }
.save-status.unsaved .sdot { background: #E0A93B; box-shadow: 0 0 8px rgba(224,169,59,0.6); }
.save-status.saved { color: #8FA3CE; }
.save-status.saved .sdot { background: #66BB6A; box-shadow: 0 0 8px rgba(102,187,106,0.5); }
</style>
"""), unsafe_allow_html=True)


# Live blade preview: override --saber on THIS page's main area only, driven by
# the STAGED colour, so the saber demo + section chips show the colour you are
# about to save (the rest of the app keeps the saved colour until you Save).
_staged_hex = ACCENT_COLORS.get(_color_norm(st.session_state.get("sw_color_ctl")), "#FF3B3B")
_r, _g, _b = (int(_staged_hex[i:i + 2], 16) for i in (1, 3, 5))
st.markdown(
    f"<style>[data-testid='stMain']{{--saber:{_staged_hex} !important;"
    f"--saber-glow:rgba({_r},{_g},{_b},0.65) !important;}}</style>",
    unsafe_allow_html=True,
)


# ── Title (a normal page title, like every other page) ────────────────────────
st.title("Settings")
st.caption(
    "Choose your side of the Force, pick a lightsaber colour, command the viewport "
    "background and tune accessibility. Nothing changes until you press "
    "Save changes, the lightsaber colour previews live on this page."
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
    with st.container(border=True, key="appbox_side"):
        st.markdown('<div class="section-label">Side of the Force</div>',
                    unsafe_allow_html=True)
        st.segmented_control(
            "Side of the Force", ["Dark Side", "Light Side"],
            key="sw_side_ctl", label_visibility="collapsed",
            help="Dark Side = the deep-space dark theme. Light Side = a high-key "
                 "light theme for bright rooms or projectors.",
        )
        st.caption("Dark Side keeps the deep-space look · Light Side switches to a "
                   "blue-on-light theme with a Tatooine twin-sun backdrop.")
with _c2:
    with st.container(border=True, key="appbox_color"):
        st.markdown('<div class="section-label">Accent & Audio colour</div>',
                    unsafe_allow_html=True)
        # nosearch_* container → CSS makes this a pure dropdown: no text caret, no
        # type-to-filter, click to open and pick only (there are just 5 named
        # colours, typing to search adds nothing but a stray focus caret).
        with st.container(key="nosearch_color"):
            st.selectbox(
                "Accent & Audio colour",
                list(ACCENT_COLORS),
                key="sw_color_ctl",
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
            key="sw_bg_ctl", label_visibility="collapsed",
            help="Star Wars = a starfield with a drifting Death Star, shooting stars "
                 "and ships flying wandering routes. Particle network = the original "
                 "connected-dots field. Off = a plain background.",
        )
    with _b2:
        st.markdown('<div class="section-label">Intensity</div>',
                    unsafe_allow_html=True)
        st.segmented_control(
            "Intensity", ["Subtle", "Normal", "Busy"],
            key="sw_intensity_ctl", label_visibility="collapsed",
            help="How many stars and how often ships fly across.",
        )

    st.markdown('<div class="section-label" style="margin-top:0.6rem;">Fleet</div>',
                unsafe_allow_html=True)
    _f1, _f2 = st.columns(2, gap="large")
    with _f1:
        st.toggle(
            "Passing ships", key="sw_ships_ctl",
            help="TIEs, X-wings, the Millennium Falcon and Star Destroyers crossing "
                 "the viewport on wandering flight paths.",
        )
    with _f2:
        st.toggle(
            "Death Star", key="sw_ds_ctl",
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
     "Applies across every page when you save.")

with st.container(border=True, key="a11ygrid"):
    _a1, _a2 = st.columns(2, gap="large")
    with _a1:
        st.markdown('<div class="section-label">Comfort</div>',
                    unsafe_allow_html=True)
        st.toggle(
            "Reduce motion", key="sw_rm_ctl",
            help="Stops the ambient animations (saber glow, drifting background, "
                 "passing ships) for a calmer, distraction-free interface.",
        )
        st.toggle(
            "Comfortable text spacing", key="sw_sp_ctl",
            help="Looser line height, letter and word spacing on running text "
                 "(in the spirit of WCAG 1.4.12), easier reading, e.g. for "
                 "dyslexic users.",
        )
    with _a2:
        st.markdown('<div class="section-label">Legibility</div>',
                    unsafe_allow_html=True)
        st.toggle(
            "High contrast", key="sw_hc_ctl",
            help="Stronger text and thicker borders for better legibility.",
        )
        st.toggle(
            "Underline links", key="sw_ul_ctl",
            help="Links stay underlined everywhere, so colour is never the only "
                 "cue that something is clickable (WCAG 1.4.1).",
        )

    st.markdown('<div class="section-label" style="margin-top:0.6rem;">Text size</div>',
                unsafe_allow_html=True)
    st.segmented_control(
        "Text size", ["Normal", "Large", "Larger"],
        key="sw_ts_ctl", label_visibility="collapsed",
        help="Scales the content text only; the layout keeps its proportions.",
    )


# ── Save / reset ──────────────────────────────────────────────────────────────
st.markdown('<div style="height:0.6rem;"></div>', unsafe_allow_html=True)
_dirty = _has_unsaved()
_s1, _s2, _s3 = st.columns([2.4, 1, 1], vertical_alignment="center")
with _s1:
    if _dirty:
        st.markdown('<div class="save-status unsaved"><span class="sdot"></span>'
                    'Unsaved changes, press Save to apply.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="save-status saved"><span class="sdot"></span>'
                    'All changes saved.</div>', unsafe_allow_html=True)
with _s2:
    st.button("Restore defaults", icon=":material/restart_alt:",
              on_click=_reset, width="stretch")
with _s3:
    st.button("Save changes", type="primary", icon=":material/save:",
              on_click=_save, width="stretch", disabled=not _dirty)
