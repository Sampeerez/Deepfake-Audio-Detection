# -*- coding: utf-8 -*-
"""
app_pages/0_Home.py — Landing page: editorial hero, a compact stat strip and
the navigation cards to the three tools. The dense data/methodology detail
lives on the Methodology page so this landing stays light and inviting.
set_page_config and PAGE_CSS are applied in app.py.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st  # noqa: E402

from src.ui_helpers import (  # noqa: E402
    app_footer, corpus_available, demo_mode, get_samples, load_config,
    sidebar_panel, themed,
)

st.markdown(themed("""
<style>
/* Separate the "Open …" page links from the panel card above them. */
[data-testid="stPageLink"] { margin-top: 0.7rem !important; }

/* ── Audio equaliser strip (under the hero) ──────────────────────────────── */
.audio-eq {
    display: flex; align-items: center; justify-content: space-between;
    height: 26px; margin: 2.1rem 0 1.5rem;   /* clear separation from the hero above */
    width: 100%;                              /* span the full header width */
    opacity: 0.82;
}
.audio-eq span {
    display: block; width: 4px; border-radius: 2px;
    background: linear-gradient(180deg, #E0AAFF 0%, #9D4EDD 55%, #7B2CBF 100%);
    animation: eqBounce 1.2s ease-in-out infinite;
    transition: filter .6s ease;          /* smooth glow in/out */
    will-change: transform;
}
@keyframes eqBounce {
    0%, 100% { transform: scaleY(0.18); }
    50%      { transform: scaleY(1); }
}
/* Glow ramps in/out smoothly (the SPEED ramp is handled in JS via playbackRate,
   which CSS cannot transition). */
.audio-eq:hover span {
    filter: brightness(1.35) drop-shadow(0 0 7px rgba(157,78,221,0.75));
}

/* ── Tech stack pills ────────────────────────────────────────────────────── */
.tech-stack {
    display: flex; flex-wrap: wrap; justify-content: center;
    gap: .3rem; padding: .6rem 0 .4rem;
}
.tech-pill {
    display: inline-block;
    background: rgba(79,139,249,0.1);
    border: 1px solid rgba(79,139,249,0.22);
    border-radius: 20px;
    padding: .22rem .8rem;
    font-size: .76rem; color: #90CAF9; cursor: default;
    transition: background 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
}
.tech-pill:hover {
    background: rgba(79,139,249,0.2);
    transform: translateY(-2px);
    box-shadow: 0 4px 10px rgba(79,139,249,0.18);
}

/* ── Navigation tiles — the WHOLE tile is clickable (invisible button over it,
   driving st.switch_page) ────────────────────────────────────────────────── */
.nav-tile {
    display: flex; flex-direction: column;
    background: linear-gradient(150deg, rgba(10,22,62,0.5), rgba(14,28,75,0.28));
    border: 1px solid rgba(79,139,249,0.16);
    border-radius: 0.9rem;
    padding: 1.05rem 1.2rem 1.1rem;
    min-height: 9rem;
    position: relative; overflow: hidden; cursor: pointer;
    transition: transform .25s cubic-bezier(0.34,1.56,0.64,1),
                border-color .25s ease, box-shadow .25s ease;
}
/* Equal-height row tiles regardless of description length. */
.nav-tile:not(.nt-wide) { height: 9.5rem; min-height: 9.5rem; }
.nav-tile::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #4F8BF9 40%, #00BCD4 60%, transparent);
    background-size: 220% 100%;
    animation: gradientShift 6s ease infinite;
    opacity: 0.5;
}
.nav-tile .nt-icon {
    width: 42px; height: 42px; border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    background: rgba(79,139,249,0.12);
    border: 1px solid rgba(79,139,249,0.25);
    color: #82B1FF; margin-bottom: 0.75rem; flex: 0 0 auto;
    transition: transform .3s cubic-bezier(0.34,1.56,0.64,1), background .25s ease;
}
.nav-tile .nt-icon svg { width: 22px; height: 22px; }
.nav-tile .nt-title { font-size: 1rem; font-weight: 700; color: #82B1FF; margin-bottom: .3rem; }
.nav-tile .nt-desc  { font-size: .82rem; opacity: .68; line-height: 1.5; }
.nav-tile.nt-ref .nt-icon { color: #B39DDB; background: rgba(156,39,176,0.12); border-color: rgba(156,39,176,0.28); }
.nav-tile.nt-ref::before { background: linear-gradient(90deg, transparent, #9C27B0 40%, #4F8BF9 60%, transparent); }
.nav-tile.nt-ref .nt-title { color: #C9A6E8; }
.nav-tile.nt-wide { flex-direction: row; align-items: center; gap: 1.1rem; min-height: auto; }
.nav-tile.nt-wide .nt-icon { margin-bottom: 0; }

/* Invisible full-tile click target. The button lives in the LAST element
   container of the tile; that container is the button's positioning context,
   so we must lift THAT container (not just the button) out of flow and stretch
   it over the whole card — otherwise only the little strip beneath the card is
   clickable. */
[class*="st-key-navt_"] { position: relative; }
[class*="st-key-navt_"] [data-testid="stElementContainer"]:last-child {
    position: absolute !important; inset: 0 !important;
    margin: 0 !important; z-index: 4;
}
[class*="st-key-navt_"] [data-testid="stElementContainer"]:last-child [data-testid="stButton"],
[class*="st-key-navt_"] [data-testid="stElementContainer"]:last-child [data-testid="stButton"] button {
    width: 100% !important; height: 100% !important;
    margin: 0 !important;
}
[class*="st-key-navt_"] [data-testid="stElementContainer"]:last-child [data-testid="stButton"] button {
    display: block !important; min-height: 0 !important; opacity: 0 !important;
    border: none !important; padding: 0 !important;
}
[class*="st-key-navt_"]:hover .nav-tile {
    transform: translateY(-4px); border-color: rgba(79,139,249,0.42);
    box-shadow: 0 12px 30px rgba(9,28,78,0.45);
}
[class*="st-key-navt_"]:hover .nav-tile .nt-icon { transform: scale(1.08) rotate(-4deg); background: rgba(79,139,249,0.22); }
[class*="st-key-navt_"]:hover .nav-tile.nt-ref .nt-icon { background: rgba(156,39,176,0.2); }
</style>
"""), unsafe_allow_html=True)

# ── Minimal data needed for the stat strip ──────────────────────────────── #
config    = load_config()
corpus_ok = corpus_available()
n_tr      = len(get_samples("train")) if corpus_ok else 0

try:
    import torch
    torch_ver  = torch.__version__
    cuda_ok    = torch.cuda.is_available()
    device_str = torch.cuda.get_device_name(0) if cuda_ok else "CPU (no GPU)"
    cuda_badge = "CUDA" if cuda_ok else "CPU"
except Exception:
    torch_ver, device_str, cuda_badge, cuda_ok = "N/A", "PyTorch not installed", "—", False

# ── Hero — flush at the very top ────────────────────────────────────────── #
st.markdown("""
<div class="hero-banner">
    <div class="hero-overline">TFG · Ingeniería Informática · Universidad de La Laguna</div>
    <h1>Deepfake Audio Detection</h1>
    <p>
        An interactive benchmark for synthetic-speech detection. Three model
        families go head to head across three ASVspoof corpora: classical DSP
        front-ends (RMS, MFCC, LFCC, DWT, CQCC) with traditional classifiers,
        2-D CNNs on STFT spectrograms — including a
        <strong>ResNet + Squeeze-and-Excitation</strong> — and a self-supervised
        <strong>wav2vec 2.0</strong> transformer working on the raw waveform.
        Explore each feature representation, train models live, run the full
        head-to-head comparison and analyse the decision threshold, all measured
        with the standard EER and minDCF metrics.
    </p>
    <div class="hero-author">
        <span class="ha-name">Samuel Pérez López</span>
        <a href="https://github.com/Sampeerez/Deepfake-detection.git" target="_blank" rel="noopener">GitHub</a>
        <a href="https://www.linkedin.com/in/samuel-pérez-lópez/" target="_blank" rel="noopener">LinkedIn</a>
        <a href="https://www.instagram.com/sampeerez_" target="_blank" rel="noopener">Instagram</a>
    </div>
    <div class="hero-wave" aria-hidden="true">
        <svg class="hw hw1" viewBox="0 0 2400 240" preserveAspectRatio="none"><path d="M0 120 Q75 55 150 120 T300 120 T450 120 T600 120 T750 120 T900 120 T1050 120 T1200 120 T1350 120 T1500 120 T1650 120 T1800 120 T1950 120 T2100 120 T2250 120 T2400 120"/></svg>
        <svg class="hw hw2" viewBox="0 0 2400 240" preserveAspectRatio="none"><path d="M0 135 Q100 205 200 135 T400 135 T600 135 T800 135 T1000 135 T1200 135 T1400 135 T1600 135 T1800 135 T2000 135 T2200 135 T2400 135"/></svg>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Audio equaliser strip — animated "spectrum" bars replacing the old glowing
#    accent line under the hero. A travelling wave (per-bar phase delay) makes it
#    read as live audio. ────────────────────────────────────────────────────── #
_N_BARS = 96
_eq_spans = "".join(
    '<span style="height:{h}px;animation-delay:{d:.2f}s"></span>'.format(
        h=5 + int(16 * (0.5 - 0.5 * math.cos(2 * math.pi * (i % 11) / 11))),
        d=(i % 11) * 0.10,
    )
    for i in range(_N_BARS)
)
st.markdown(f'<div class="audio-eq" aria-hidden="true">{_eq_spans}</div>',
            unsafe_allow_html=True)

# Smooth hover SPEED ramp for the equaliser (CSS can't transition animation
# speed, so we ease each bar's Web-Animation playbackRate toward a target that
# climbs while hovered and decays back to rest when the pointer leaves).
# Hosted in a collapsed container so the height=1 iframe never shows as a line.
with st.container(key="eqramp_host"):
  st.iframe(
    """
<script>
(function(){
  var doc; try { doc = window.parent.document; } catch(e) { return; }
  var win = window.parent;
  var eq = doc.querySelector('.audio-eq');
  if (!eq || eq.__eqRamp) { return; }
  eq.__eqRamp = true;
  if (win.__eqRAF) { win.cancelAnimationFrame(win.__eqRAF); }
  var spans = eq.querySelectorAll('span');
  var hovered = false, cur = 1, target = 1, last = 0;
  var BASE = 1, MAX = 3.2;
  eq.addEventListener('mouseenter', function(){ hovered = true; });
  eq.addEventListener('mouseleave', function(){ hovered = false; });
  function frame(t){
    if (!last) last = t;
    var dt = Math.min((t - last) / 1000, 0.05); last = t;
    // While hovered the target accelerates upward; on leave it eases back down.
    target = hovered ? Math.min(MAX, target + dt * 1.5)
                     : Math.max(BASE, target - dt * 1.1);
    cur += (target - cur) * Math.min(1, dt * 4);   // smooth follow
    for (var i = 0; i < spans.length; i++){
      var a = spans[i].getAnimations ? spans[i].getAnimations() : [];
      if (a && a[0]) { a[0].playbackRate = cur; }
    }
    win.__eqRAF = win.requestAnimationFrame(frame);
  }
  win.__eqRAF = win.requestAnimationFrame(frame);
})();
</script>
""",
    height=1,
)

# ── Stat strip — editorial numerals (full official corpus sizes) ────────── #
st.markdown(
    '<div class="stat-strip">'
    '<div class="stat-cell"><div class="st-num">25,380</div>'
    '<div class="st-lbl">2019 LA train files</div></div>'
    '<div class="stat-cell"><div class="st-num">181,566</div>'
    '<div class="st-lbl">2021 LA eval files</div></div>'
    '<div class="stat-cell"><div class="st-num">458,871</div>'
    '<div class="st-lbl">2021 DF eval files</div></div>'
    f'<div class="stat-cell"><div class="st-num" style="font-size:1.1rem;'
    f'padding-top:.3rem;">{device_str}</div>'
    f'<div class="st-lbl">{cuda_badge} · PyTorch {torch_ver}</div></div>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown("<div style='height:1.6rem;'></div>", unsafe_allow_html=True)

# ── Explore — the whole tile is a real clickable link ───────────────────── #
_ICON_SE = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M3 12h2l2-6 3 14 3-11 2 5h6"/></svg>')
_ICON_BENCH = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
               'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
               '<path d="M9 3h6M10 3v5L5 19a1.5 1.5 0 0 0 1.4 2h11.2a1.5 1.5 0 0 0 '
               '1.4-2L14 8V3"/><path d="M8.5 14h7"/></svg>')
_ICON_DET = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
             'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
             '<path d="M3 3v18h18"/><path d="M7 14l3-4 3 3 4-6"/></svg>')
_ICON_MET = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
             'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
             '<path d="M5 4.5A1.5 1.5 0 0 1 6.5 3H20v15H6.5A1.5 1.5 0 0 0 5 19.5z"/>'
             '<path d="M5 19.5A1.5 1.5 0 0 0 6.5 21H20"/></svg>')
_ICON_LD = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            '<rect x="9" y="2" width="6" height="12" rx="3"/>'
            '<path d="M5 11a7 7 0 0 0 14 0M12 18v3"/></svg>')

st.markdown(
    '<div class="sec-head"><h3 class="sh-title">Explore the Benchmark</h3>'
    '<span class="sh-rule"></span></div>',
    unsafe_allow_html=True,
)


def _nav_tile(col, key, cls, icon, title, desc, page, extra=""):
    with col:
        with st.container(key=f"navt_{key}"):
            st.markdown(
                f'<div class="nav-tile {cls}"><div class="nt-icon">{icon}</div>'
                f'<div{extra}><div class="nt-title">{title}</div>'
                f'<div class="nt-desc">{desc}</div></div></div>',
                unsafe_allow_html=True,
            )
            if st.button(title, key=f"navbtn_{key}", width="stretch"):
                st.switch_page(page)


t1, t2, t3 = st.columns(3, gap="medium")
_nav_tile(t1, "se", "", _ICON_SE, "Signal Explorer",
          "See &amp; compare audio at the feature level — waveform, STFT, "
          "MFCC, LFCC, CQCC.", "app_pages/1_Signal_Explorer.py")
_nav_tile(t2, "bench", "", _ICON_BENCH, "Benchmark",
          "Classic models, the CNN, or run everything at once — ranked by "
          "minDCF &middot; EER.", "app_pages/2_Benchmark.py")
_nav_tile(t3, "det", "", _ICON_DET, "Detection Analysis",
          "ROC &amp; DET curves and the decision threshold — or drop your own "
          "clip for a live verdict.", "app_pages/3_Detection_Analysis.py")

st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
(m_col,) = st.columns(1)
_nav_tile(m_col, "met", "nt-ref nt-wide", _ICON_MET, "Methodology",
          "Corpora, methods &amp; metrics — the full picture behind the "
          "benchmark.", "app_pages/4_Methodology.py")

# ── Tech stack + footer (closed ending — no scroll past this point) ─────── #
st.markdown("""
<div class="tech-stack" style="margin-top:1.2rem;">
<span class="tech-pill">Python 3.12</span>
<span class="tech-pill">PyTorch 2.x</span>
<span class="tech-pill">librosa</span>
<span class="tech-pill">PyWavelets</span>
<span class="tech-pill">scikit-learn</span>
<span class="tech-pill">XGBoost</span>
<span class="tech-pill">Transformers &middot; wav2vec 2.0</span>
<span class="tech-pill">NumPy &middot; SciPy</span>
<span class="tech-pill">pandas</span>
<span class="tech-pill">Matplotlib &middot; Altair</span>
<span class="tech-pill">Streamlit</span>
<span class="tech-pill">ASVspoof 2019 / 2021</span>
</div>
""", unsafe_allow_html=True)

app_footer(
    "TFG · Ingeniería Informática · Samuel Pérez López",
    "Universidad de La Laguna · ASVspoof 2019 LA / 2021 LA / 2021 DF · "
    "Anti-spoofing binario (bonafide vs. spoof)",
)

# ── Public-demo notice — in the SIDEBAR (only on the corpus-less deployment) ─ #
if demo_mode():
    with st.sidebar:
        sidebar_panel(
            "Web demo (CPU)",
            text=("Live training needs the corpus and a GPU, so the pretrained "
                  "models run here on CPU. Grab them all in <b>Benchmark → Full "
                  "comparison</b> and test your own clip in <b>Detection "
                  "Analysis → Test an audio</b>."),
        )
