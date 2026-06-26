# -*- coding: utf-8 -*-
"""
app_pages/4_Methodology.py — Reference page: the corpora and data context, plus
the full benchmark methodology (DSP front-ends, classifiers, CNN, metrics).

Kept off the Home page so the landing stays light; this is where the detail
lives for whoever wants it. set_page_config and PAGE_CSS are applied in app.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st  # noqa: E402

from src.ui_helpers import (  # noqa: E402
    corpus_available, corpus_configured_2021_la, corpus_configured_2021_df,
    fig_corpus_overview, fig_overall_split_bar, get_samples, load_config,
    section_header, themed,
)

st.markdown(themed("""
<style>
/* ── Corpus status cards ─────────────────────────────────────────────────── */
.corpus-card {
    background: linear-gradient(145deg,
        rgba(10,22,62,0.65) 0%, rgba(14,28,78,0.4) 100%);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(79,139,249,0.16);
    border-radius: 1rem;
    padding: 1.15rem 1.35rem 1.1rem;
    height: 100%;
    position: relative;
    overflow: hidden;
    cursor: default;
    transition: transform 0.28s cubic-bezier(0.34,1.56,0.64,1),
                box-shadow 0.28s ease, border-color 0.28s ease;
}
.corpus-card::after {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg,
        transparent, rgba(79,139,249,0.4), transparent);
}
.corpus-card:hover {
    transform: translateY(-5px);
    border-color: rgba(79,139,249,0.32);
    box-shadow: 0 14px 36px rgba(9,28,78,0.55), 0 0 0 1px rgba(79,139,249,0.2);
}
.corpus-card .cc-label {
    font-size: 0.68rem; text-transform: uppercase; letter-spacing: .1em;
    opacity: .45; margin-bottom: .25rem; font-weight: 600;
}
.corpus-card .cc-value {
    font-size: 1.35rem; font-weight: 800;
    background: linear-gradient(135deg, #82B1FF, #4FC3F7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.2; margin-bottom: .35rem;
}
.corpus-card .cc-sub { font-size: 0.77rem; opacity: .62; line-height: 1.65; }
.cc-badge-ok  { color:#66BB6A; font-weight:700; }
.cc-badge-err { color:#EF5350; font-weight:700; }

/* ── Methodology cards ───────────────────────────────────────────────────── */
.method-card {
    background: linear-gradient(145deg,
        rgba(8,18,52,0.55) 0%, rgba(12,24,65,0.35) 100%);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(79,139,249,0.13);
    border-radius: 0.85rem;
    padding: 1.1rem 1.2rem 1.15rem;
    height: 100%;
    cursor: default;
    transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    position: relative;
    overflow: hidden;
}
.method-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #4F8BF9, #9C27B0);
    opacity: 0; transition: opacity 0.25s ease;
}
.method-card:hover {
    transform: translateY(-4px);
    border-color: rgba(79,139,249,0.28);
    box-shadow: 0 10px 28px rgba(9,28,78,0.4);
}
.method-card:hover::before { opacity: 1; }
.method-card .mc-num {
    font-size: 0.66rem; font-weight: 800; color: #4F8BF9;
    letter-spacing: 0.12em; margin-bottom: 0.3rem;
}
.method-card h5 {
    margin: 0 0 .6rem; font-size: .9rem; font-weight: 700;
    color: #82B1FF; letter-spacing: 0.01em;
}
.method-card ul { margin: 0; padding-left: 1.05rem; }
.method-card li {
    font-size: .8rem; opacity: .7; margin-bottom: .28rem; line-height: 1.45;
    transition: opacity 0.2s ease;
}
.method-card:hover li { opacity: .85; }

/* ── Class-distribution cards (expander) ─────────────────────────────────── */
.dist-cards { display: flex; flex-direction: column; gap: 1rem; margin-top: .2rem; }
.dist-row {
    background: linear-gradient(145deg,
        rgba(10,22,62,0.55) 0%, rgba(14,28,75,0.3) 100%);
    border: 1px solid rgba(79,139,249,0.14);
    border-radius: 0.8rem;
    padding: 0.85rem 1.05rem 0.95rem;
    cursor: default;
}
.dist-row .dist-name {
    font-size: 0.72rem; font-weight: 800; text-transform: uppercase;
    letter-spacing: 0.1em; color: #82B1FF; margin-bottom: 0.5rem;
}
.dist-bar {
    display: flex; height: 12px; border-radius: 6px; overflow: hidden;
    margin-bottom: 0.55rem; box-shadow: inset 0 0 0 1px rgba(255,255,255,0.04);
}
.dist-bar .dist-bon { background: linear-gradient(90deg, #42A5F5, #64B5F6); }
.dist-bar .dist-spo { background: linear-gradient(90deg, #EF5350, #E57373); }
.dist-nums { font-size: 0.8rem; color: #C2CFEC; }
.dist-nums b { color: #E8EDF8; font-weight: 700; }
.dist-ratio { color: #FFB74D; font-weight: 700; font-variant-numeric: tabular-nums; }
.dist-note {
    font-size: 0.82rem; line-height: 1.6; opacity: 0.78;
    margin: 1.6rem 0 0; padding-top: 1rem;
    border-top: 1px solid rgba(79,139,249,0.12); cursor: default;
}
.dist-note b { color: #82B1FF; }
</style>
"""), unsafe_allow_html=True)

# ── Data gathered up front ───────────────────────────────────────────────── #
config    = load_config()
corpus_ok = corpus_available()
la21_ok   = corpus_configured_2021_la()
df21_ok   = corpus_configured_2021_df()

if corpus_ok:
    train_s  = get_samples("train")
    dev_s    = get_samples("dev")
    n_tr     = len(train_s)
    n_dev    = len(dev_s)
    n_bon_tr = sum(1 for _, l in train_s if l == 0)
    n_spo_tr = n_tr - n_bon_tr
    n_bon_dv = sum(1 for _, l in dev_s if l == 0)
    n_spo_dv = n_dev - n_bon_dv
else:
    train_s = dev_s = []
    n_tr = n_dev = n_bon_tr = n_spo_tr = n_bon_dv = n_spo_dv = 0

st.title("Methodology")
st.caption(
    "How the benchmark works — the corpora and their class balance, the DSP "
    "front-ends and classifiers, the CNN architectures, the self-supervised "
    "wav2vec 2.0 transformer, and the detection metrics."
)

# ── 01 · Data & corpora ──────────────────────────────────────────────────── #
section_header("01", "Data & Corpora",
               "Three corpora with increasing difficulty — from the official "
               "2019 benchmark to in-the-wild 2021 deepfakes.")

if corpus_ok:
    _card19 = (
        f'<div class="corpus-card">'
        f'<div class="cc-label">Training corpus</div>'
        f'<div class="cc-value">ASVspoof 2019 LA</div>'
        f'<div class="cc-sub"><span class="cc-badge-ok">● Available</span><br>'
        f'Train: <strong>{n_tr:,}</strong> &middot; {n_bon_tr:,} bonafide / {n_spo_tr:,} spoof<br>'
        f'Dev: <strong>{n_dev:,}</strong> &middot; {n_bon_dv:,} bonafide / {n_spo_dv:,} spoof<br>'
        f'Imbalance 1:{round(n_spo_tr/max(n_bon_tr,1),0):.0f} &mdash; EER &amp; minDCF recommended'
        f'</div></div>'
    )
else:
    _card19 = (
        '<div class="corpus-card">'
        '<div class="cc-label">Training corpus</div>'
        '<div class="cc-value">ASVspoof 2019 LA</div>'
        '<div class="cc-sub"><span class="cc-badge-err">● Not found</span><br>'
        'Check <code>dataset.path_la2019</code> in config.yaml</div></div>'
    )

_la21b = '<span class="cc-badge-ok">● Configured</span>' if la21_ok else '<span class="cc-badge-err">● Not configured</span>'
_df21b = '<span class="cc-badge-ok">● Configured</span>' if df21_ok else '<span class="cc-badge-err">● Not configured</span>'

_corpus_grid = (
    '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin:.4rem 0 .5rem;">'
    + _card19
    + f'<div class="corpus-card"><div class="cc-label">Advanced evaluation</div>'
    f'<div class="cc-value">ASVspoof 2021 LA</div>'
    f'<div class="cc-sub">{_la21b}<br>181,566 eval files &middot; telephone channel<br>'
    f'No official train split &mdash; 80/20 internal split used<br>'
    f'Different acoustic conditions from 2019</div></div>'
    + f'<div class="corpus-card"><div class="cc-label">Deepfakes in-the-wild</div>'
    f'<div class="cc-value">ASVspoof 2021 DF</div>'
    f'<div class="cc-sub">{_df21b}<br>458,871 eval files &middot; 3 partitions<br>'
    f'Modern TTS/VC attacks (2020&ndash;2021)<br>'
    f'Most challenging corpus in the benchmark</div></div>'
    + '</div>'
)
st.markdown(_corpus_grid, unsafe_allow_html=True)

if corpus_ok and train_s:
    with st.expander("Class distribution — ASVspoof 2019 LA", expanded=False):
        ratio_tr = round(n_spo_tr / max(n_bon_tr, 1), 1)
        ratio_dv = round(n_spo_dv / max(n_bon_dv, 1), 1)

        col_chart, col_txt = st.columns([1, 1.05], gap="large")
        with col_chart:
            st.pyplot(fig_corpus_overview(train_s, dev_s), clear_figure=True)
        with col_txt:
            st.markdown(
                '<div class="dist-cards">'
                '<div class="dist-row">'
                '<div class="dist-name">Train</div>'
                f'<div class="dist-bar"><span class="dist-bon" style="flex:{n_bon_tr};"></span>'
                f'<span class="dist-spo" style="flex:{n_spo_tr};"></span></div>'
                f'<div class="dist-nums"><b>{n_tr:,}</b> files · '
                f'{n_bon_tr:,} bonafide / {n_spo_tr:,} spoof · '
                f'<span class="dist-ratio">1 : {ratio_tr}</span></div>'
                '</div>'
                '<div class="dist-row">'
                '<div class="dist-name">Dev</div>'
                f'<div class="dist-bar"><span class="dist-bon" style="flex:{n_bon_dv};"></span>'
                f'<span class="dist-spo" style="flex:{n_spo_dv};"></span></div>'
                f'<div class="dist-nums"><b>{n_dev:,}</b> files · '
                f'{n_bon_dv:,} bonafide / {n_spo_dv:,} spoof · '
                f'<span class="dist-ratio">1 : {ratio_dv}</span></div>'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<p class="dist-note">A roughly <b>1 : {ratio_tr:.0f}</b> '
                'bonafide-to-spoof imbalance makes <b>raw accuracy misleading</b> '
                '— a trivial all-spoof classifier already scores ~90%. This is '
                'why <b>EER</b> and <b>minDCF</b> are the primary metrics: both '
                'weigh false alarms and misses fairly.</p>',
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:0.7rem'></div>", unsafe_allow_html=True)
            st.pyplot(
                fig_overall_split_bar(n_bon_tr + n_bon_dv, n_spo_tr + n_spo_dv),
                clear_figure=True,
            )

st.markdown("<div style='height:1.6rem;'></div>", unsafe_allow_html=True)

# ── 02 · Pipeline & methods ──────────────────────────────────────────────── #
section_header("02", "Pipeline & Methods",
               "Full pipeline: feature extraction → classification → detection metrics.")

st.markdown(
    '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:1rem;margin:.4rem 0;">'
    '<div class="method-card"><div class="mc-num">I</div><h5>DSP Front-ends</h5><ul>'
    '<li><strong>RMS</strong> — frame-level temporal energy</li>'
    '<li><strong>MFCC</strong> — spectral envelope (Mel scale)</li>'
    '<li><strong>LFCC</strong> — linear-frequency cepstrum</li>'
    '<li><strong>DWT</strong> — multi-resolution wavelet energy (db4)</li>'
    '<li><strong>CQCC</strong> — Constant-Q cepstral coefficients</li>'
    '</ul></div>'
    '<div class="method-card"><div class="mc-num">II</div><h5>Classical Classifiers</h5><ul>'
    '<li><strong>LR</strong> — Logistic Regression (baseline)</li>'
    '<li><strong>SVM</strong> — RBF kernel, probability-calibrated</li>'
    '<li><strong>XGBoost</strong> — gradient boosting on tabular features</li>'
    '<li>Stratified subsampling (bonafide/spoof ratio preserved)</li>'
    '<li>Feature cache for fast re-runs</li>'
    '<li>Export results table to CSV</li>'
    '</ul></div>'
    '<div class="method-card"><div class="mc-num">III</div><h5>Deep Learning (CNN)</h5><ul>'
    '<li><strong>2D CNN</strong> — 3&times; Conv&rarr;BN&rarr;ReLU&rarr;MaxPool</li>'
    '<li><strong>ResNet + SE</strong> — residual + Squeeze-and-Excitation</li>'
    '<li><strong>SpecAugment</strong> — time &amp; frequency masking</li>'
    '<li>Best-checkpoint restore (early stopping)</li>'
    '<li>BCEWithLogitsLoss + pos_weight (class imbalance)</li>'
    '<li>ReduceLROnPlateau scheduler</li>'
    '</ul></div>'
    '<div class="method-card"><div class="mc-num">IV</div><h5>Self-Supervised (Transformer)</h5><ul>'
    '<li><strong>wav2vec 2.0</strong> — 12-layer transformer encoder (base, hidden 768)</li>'
    '<li>Self-supervised pretraining + fine-tuned spoof head</li>'
    '<li>Raw 16&nbsp;kHz waveform — no DSP, no spectrogram</li>'
    '<li>Mean-pooled embedding &rarr; linear 2-class head</li>'
    '<li>Temperature-calibrated softmax (T=2) tames overconfidence</li>'
    '<li>Inference-only; weighted late-fusion member (Test an audio)</li>'
    '</ul></div>'
    '<div class="method-card"><div class="mc-num">V</div><h5>Evaluation Metrics</h5><ul>'
    '<li><strong>EER</strong> — point where FAR = FRR</li>'
    '<li><strong>minDCF</strong> — NIST min cost (C<sub>miss</sub>=1, C<sub>fa</sub>=10)</li>'
    '<li><strong>Accuracy</strong> — informative, not primary</li>'
    '<li>Score distribution histogram</li>'
    '<li>Inference latency per audio</li>'
    '<li>Post-training eval on 2021 LA / DF</li>'
    '</ul></div>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown("<div style='height:1.6rem;'></div>", unsafe_allow_html=True)

# ── 03 · Spoofing attacks ────────────────────────────────────────────────── #
section_header("03", "Spoofing attacks (logical access)",
               "How the fakes are made — and why the eval set is so much harder "
               "than dev.")
st.markdown(
    '<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:1rem;margin:.4rem 0;">'
    '<div class="method-card"><div class="mc-num">A01–A06</div>'
    '<h5>Known attacks · train + dev (&ldquo;seen&rdquo;)</h5><ul>'
    '<li><strong>2 voice-conversion (VC)</strong> systems — one neural-network-based, '
    'one spectral-/transfer-function-based</li>'
    '<li><strong>4 text-to-speech (TTS)</strong> systems — waveform concatenation, and '
    'neural parametric synthesis with a source&ndash;filter vocoder or a '
    '<strong>WaveNet</strong> vocoder</li>'
    '<li>The <em>same six</em> algorithms appear in both training and development, so a '
    'model can learn them directly</li>'
    '</ul></div>'
    '<div class="method-card"><div class="mc-num">A07–A19</div>'
    '<h5>Unknown attacks · eval only (&ldquo;unseen&rdquo;)</h5><ul>'
    '<li><strong>13 attacks</strong>: 2 reused known + <strong>11 brand-new</strong> '
    '(2 VC, 6 TTS, 3 hybrid TTS&ndash;VC)</li>'
    '<li>Waveform generation never seen in training: classical vocoding, Griffin&ndash;Lim, '
    '<strong>GANs</strong>, neural waveform models, concatenation, waveform &amp; spectral '
    'filtering</li>'
    '<li>Designed to be <em>different</em> from A01&ndash;A06 — the genuine generalisation '
    'test</li>'
    '</ul></div>'
    '</div>',
    unsafe_allow_html=True,
)
st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)
st.markdown(
    '<div class="info-card"><div class="ic-title">Why dev EER &laquo; eval EER'
    '<span class="ic-tag">generalisation</span></div>'
    '<p class="ic-body">The logical-access data is built from the <strong>VCTK</strong> '
    'corpus (107 speakers), with <strong>no speaker overlap</strong> across train / dev / '
    'eval and utterances of ~1&ndash;2&nbsp;s. Because train and dev share attacks '
    'A01&ndash;A06 but eval swaps in 11 unseen systems (A07&ndash;A19), a detector that '
    'merely memorises the six known attacks collapses on eval. This is exactly what the '
    '<em>Evaluate on</em> selector exposes across the benchmark: dev = seen, eval = '
    'unseen, 2021&nbsp;LA/DF = unseen + new channel/compression.</p></div>',
    unsafe_allow_html=True,
)

st.markdown("<div style='height:1.6rem;'></div>", unsafe_allow_html=True)

# ── 04 · Evaluation protocol ─────────────────────────────────────────────── #
section_header("04", "Evaluation protocol",
               "Exactly how ASVspoof scores a countermeasure (ASVspoof 2019 "
               "evaluation plan).")
st.markdown(
    '<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:1rem;margin:.4rem 0;">'
    '<div class="method-card"><div class="mc-num">§</div><h5>The task &amp; scores</h5><ul>'
    '<li>Two-class: <strong>bonafide</strong> (accept) vs <strong>spoof</strong> (reject)</li>'
    '<li>One real-valued score per file — <strong>high = bonafide, low = spoof</strong> '
    '(log-likelihood-ratio convention)</li>'
    '<li>Hard binary decisions are <em>not</em> allowed; thresholds are set by the metric</li>'
    '<li>Scores are <strong>pooled</strong> across all trials; LA and PA are scored '
    'independently</li>'
    '</ul></div>'
    '<div class="method-card"><div class="mc-num">EER</div>'
    '<h5>Equal Error Rate · secondary</h5><ul>'
    '<li>Threshold s where the miss rate equals the false-alarm rate: '
    'P<sub>miss</sub>(s) = P<sub>fa</sub>(s)</li>'
    '<li>Application-independent &amp; threshold-free — the headline metric this app '
    'reports</li>'
    '<li>Retained from ASVspoof 2015/2017 for backwards comparison</li>'
    '</ul></div>'
    '<div class="method-card"><div class="mc-num">t-DCF</div>'
    '<h5>min tandem-DCF · primary</h5><ul>'
    '<li>Cost of a <em>tandem</em> system: the countermeasure feeding a fixed ASV '
    'verifier</li>'
    '<li>t-DCF = &beta;&middot;P<sub>miss</sub>(s) + P<sub>fa</sub>(s), minimised over s '
    '(oracle calibration)</li>'
    '<li>&beta; comes from the costs/priors <em>and</em> the ASV&rsquo;s own error rates</li>'
    '</ul></div>'
    '<div class="method-card"><div class="mc-num">C/&pi;</div>'
    '<h5>Costs &amp; priors (LA)</h5><ul>'
    '<li>&pi;<sub>tar</sub>=0.9405 &middot; &pi;<sub>non</sub>=0.0095 &middot; '
    '&pi;<sub>spoof</sub>=0.05</li>'
    '<li>C<sub>miss</sub>=1 &middot; C<sub>fa</sub>=10 → a <strong>false accept costs 10&times;'
    '</strong> a miss</li>'
    '<li>That heavy FA penalty is why minDCF stays near 1.0 until false acceptances are '
    'genuinely rare</li>'
    '</ul></div>'
    '</div>',
    unsafe_allow_html=True,
)
st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)
st.markdown(
    '<div class="info-card"><div class="ic-title">What this app computes'
    '<span class="ic-tag">scope</span></div>'
    '<p class="ic-body">ASVspoof ships no ASV scores with the public data, so this TFG '
    'reports the <strong>EER</strong> plus a <strong>simplified minDCF</strong> '
    '(C<sub>miss</sub>=1, C<sub>fa</sub>=10, P<sub>tar</sub>=0.05) — the stand-alone '
    'countermeasure cost, not the full tandem t-DCF. For 2021, <strong>LA</strong> adds '
    'telephone codec &amp; transmission variability and <strong>DF</strong> adds lossy '
    'compression / &ldquo;in-the-wild&rdquo; audio; both are eval-only and scored by EER, '
    'making them the toughest cross-domain test of a detector trained on 2019.</p></div>',
    unsafe_allow_html=True,
)
