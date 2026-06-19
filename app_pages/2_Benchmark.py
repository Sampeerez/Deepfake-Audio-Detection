# -*- coding: utf-8 -*-
"""
app_pages/2_Benchmark.py — Benchmark launcher.

A small "home" screen that explains the three ways to compare detectors; you
pick one (card → opens that mode) and can go ← Back any time. Each mode is the
existing workflow, dispatched via runpy so its rich behaviour is unchanged:
  • Classic models  → modes/_mode_classic.py
  • CNN             → modes/_mode_cnn.py
  • Full comparison → modes/_mode_full.py
"""

import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st  # noqa: E402

_MODES = {"classic": "_mode_classic.py", "cnn": "_mode_cnn.py", "full": "_mode_full.py"}

_ICON_CLASSIC = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                 'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                 '<path d="M9 3h6M10 3v5L5 19a1.5 1.5 0 0 0 1.4 2h11.2a1.5 1.5 0 0 0 '
                 '1.4-2L14 8V3"/><path d="M8.5 14h7"/></svg>')
_ICON_CNN = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
             'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
             '<path d="M12 3l9 5-9 5-9-5 9-5z"/><path d="M3 12l9 5 9-5"/>'
             '<path d="M3 16.5l9 5 9-5"/></svg>')
_ICON_FULL = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
              'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
              '<path d="M4 20V11M10 20V4M16 20v-6M3 20h18"/></svg>')

_OPTIONS = [
    ("classic", "Classic models", _ICON_CLASSIC,
     "Tune one DSP extractor × classifier by hand and accumulate the results."),
    ("cnn", "CNN", _ICON_CNN,
     "Train the 2-D CNN / ResNet + SE and inspect how it learns."),
    ("full", "Full comparison", _ICON_FULL,
     "Run every extractor, classifier and the CNN at once — the ranked, "
     "scientific comparison."),
]

st.markdown("""
<style>
.mode-card {
    background: linear-gradient(150deg, rgba(10,22,62,0.5), rgba(14,28,75,0.28));
    border: 1px solid rgba(79,139,249,0.16);
    border-radius: 1rem; padding: 1.4rem 1.3rem 1.2rem;
    /* Fixed height (not just min-height) so all three cards line up exactly,
       no matter how many lines each description wraps to. */
    height: 13rem; box-sizing: border-box;
    text-align: center; cursor: default;
    transition: transform .25s cubic-bezier(0.34,1.56,0.64,1),
                border-color .25s ease, box-shadow .25s ease;
}
.mode-card:hover { transform: translateY(-4px); border-color: rgba(79,139,249,0.42);
    box-shadow: 0 12px 30px rgba(9,28,78,0.45); }
.mode-card .mc-ic {
    width: 52px; height: 52px; border-radius: 14px; margin: 0 auto 0.85rem;
    display: flex; align-items: center; justify-content: center;
    background: rgba(79,139,249,0.12); border: 1px solid rgba(79,139,249,0.25);
    color: #82B1FF;
}
.mode-card .mc-ic svg { width: 26px; height: 26px; }
.mode-card .mc-title { font-size: 1.1rem; font-weight: 750; color: #82B1FF; margin-bottom: 0.4rem; }
.mode-card .mc-desc  { font-size: 0.84rem; opacity: 0.72; line-height: 1.55; }

/* ── "How the comparison works" — vertical numbered timeline ──────────────── */
@keyframes hcwIn { from { opacity: 0; transform: translateY(14px); }
                   to   { opacity: 1; transform: translateY(0); } }
/* Original hover effect: a diagonal light sweeps across the whole card. */
@keyframes hcwShine { 0%   { left: -70%; opacity: 0; }
                      12%  { opacity: 1; }
                      100% { left: 130%; opacity: 0; } }
.hcw-step {
    display: grid; grid-template-columns: 60px 1fr; gap: 1.1rem;
    cursor: default; user-select: none;            /* normal cursor, no text I-beam */
    animation: hcwIn .55s cubic-bezier(0.22,1,0.36,1) both;
}
.hcw-step:nth-child(1) { animation-delay: .04s; }
.hcw-step:nth-child(2) { animation-delay: .14s; }
.hcw-step:nth-child(3) { animation-delay: .24s; }
.hcw-step + .hcw-step { margin-top: 1.5rem; }     /* more air between the steps */
.hcw-num {
    position: relative; display: flex; align-items: center; justify-content: center;
    font-size: 1.45rem; font-weight: 800; color: #82B1FF;
    background: rgba(79,139,249,0.10); border: 1px solid rgba(79,139,249,0.28);
    border-radius: 16px; min-height: 56px;
    transition: transform .3s cubic-bezier(0.34,1.56,0.64,1),
                background .25s ease, color .25s ease, box-shadow .3s ease;
}
/* connecting segment between the numbered nodes (replaces the old arrows) */
.hcw-step:not(:last-child) .hcw-num::after {
    content: ""; position: absolute; top: 100%; left: 50%;
    width: 2px; height: 1.5rem; transform: translateX(-50%);   /* longer link = steps further apart */
    background: linear-gradient(180deg, rgba(79,139,249,0.55), rgba(0,188,212,0.15));
}
.hcw-body {
    position: relative; overflow: hidden;
    background: linear-gradient(150deg, rgba(10,22,62,0.45), rgba(14,28,75,0.2));
    border: 1px solid rgba(79,139,249,0.14); border-left: 3px solid #4F8BF9;
    border-radius: 0.8rem; padding: 0.8rem 1.1rem;
    transition: transform .3s cubic-bezier(0.34,1.56,0.64,1),
                border-color .3s ease, box-shadow .3s ease;
}
/* the moving light band (idle, fired on hover) */
.hcw-body::after {
    content: ""; position: absolute; top: 0; left: -70%; width: 55%; height: 100%;
    background: linear-gradient(110deg, transparent 0%, rgba(0,229,255,0.10) 38%,
                rgba(140,185,255,0.26) 50%, rgba(0,229,255,0.10) 62%, transparent 100%);
    transform: skewX(-16deg); opacity: 0; pointer-events: none;
}
.hcw-step:hover .hcw-body {
    transform: translateY(-3px); border-left-color: #00BCD4;
    box-shadow: 0 12px 28px rgba(9,28,78,0.45);
}
.hcw-step:hover .hcw-body::after { animation: hcwShine .9s ease; }
.hcw-step:hover .hcw-num {
    transform: translateY(-3px) scale(1.07);
    background: rgba(79,139,249,0.2); color: #BFE0FF;
    box-shadow: 0 0 0 4px rgba(79,139,249,0.12), 0 0 18px rgba(0,229,255,0.32);
}
.hcw-kicker { font-size: 0.66rem; font-weight: 800; letter-spacing: 0.14em;
    text-transform: uppercase; color: #64B5F6; margin-bottom: 0.2rem; }
.hcw-title { font-size: 1.02rem; font-weight: 700; color: #E8EDF8; margin-bottom: 0.18rem; }
.hcw-desc  { font-size: 0.84rem; opacity: 0.74; line-height: 1.5; }
.hcw-tags  { margin-top: 0.55rem; display: flex; flex-wrap: wrap; gap: 0.35rem; }
.hcw-tag   { font-size: 0.69rem; color: #90CAF9; background: rgba(79,139,249,0.1);
    border: 1px solid rgba(79,139,249,0.2); border-radius: 20px; padding: 0.1rem 0.6rem; }
</style>
""", unsafe_allow_html=True)

choice = st.session_state.get("bench_choice")

if choice not in _MODES:
    st.title("Benchmark")
    st.caption("Three ways to compare deepfake detectors — pick one to start. "
               "You can go back to switch any time.")
    cols = st.columns(3, gap="medium")
    for col, (key, label, icon, desc) in zip(cols, _OPTIONS):
        with col:
            st.markdown(
                f'<div class="mode-card"><div class="mc-ic">{icon}</div>'
                f'<div class="mc-title">{label}</div>'
                f'<div class="mc-desc">{desc}</div></div>'
                "<div style='height:0.7rem'></div>",   # space before the button
                unsafe_allow_html=True,
            )
            if st.button(f"Open {label}", key=f"bench_open_{key}", width="stretch",
                         type="primary" if key == "full" else "secondary"):
                st.session_state["bench_choice"] = key
                st.rerun()

    # ── Fill the page: how a comparison works + what you have so far ───────── #
    st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-head"><h3 class="sh-title">How the comparison works</h3>'
        '<span class="sh-rule"></span></div>',
        unsafe_allow_html=True,
    )
    def _hcw_step(num, kicker, title, desc, tags):
        tag_html = "".join(f'<span class="hcw-tag">{t}</span>' for t in tags)
        return (
            f'<div class="hcw-step"><div class="hcw-num">{num}</div>'
            f'<div class="hcw-body"><div class="hcw-kicker">{kicker}</div>'
            f'<div class="hcw-title">{title}</div>'
            f'<div class="hcw-desc">{desc}</div>'
            f'<div class="hcw-tags">{tag_html}</div></div></div>'
        )

    st.markdown(
        _hcw_step(
            "1", "Extract · front-end",
            "Turn raw audio into a compact representation",
            "Every signal is decoded once and passed through a DSP front-end or "
            "an STFT spectrogram. Features are cached, so re-runs are instant.",
            ["RMS", "MFCC", "LFCC", "DWT", "CQCC", "Fusion", "STFT-dB"],
        )
        + _hcw_step(
            "2", "Train · classifier",
            "Learn to separate bonafide speech from deepfakes",
            "A classic model fits the DSP features, or a convolutional network "
            "learns directly from the spectrogram — both run in parallel "
            "(CPU ∥ GPU) during a full comparison.",
            ["Logistic Reg.", "SVM (RBF)", "XGBoost", "2-D CNN", "ResNet + SE"],
        )
        + _hcw_step(
            "3", "Score · metrics",
            "Measure detection quality on seen and unseen attacks",
            "Each model is scored on the dev split (attacks A01–A06, seen in "
            "training) and the eval split (A07–A19, completely unseen) — the "
            "real test of generalisation.",
            ["minDCF", "EER", "dev split", "eval split"],
        ),
        unsafe_allow_html=True,
    )
else:
    c_back, _ = st.columns([1, 5])
    with c_back:
        if st.button("← Modes", key="bench_back", width="stretch"):
            del st.session_state["bench_choice"]
            st.rerun()
    runpy.run_path(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "modes", _MODES[choice]),
        run_name="__main__",
    )
