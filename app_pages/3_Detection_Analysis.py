# -*- coding: utf-8 -*-
"""
app_pages/3_Detection_Analysis.py — Two complementary views:

  • "Test an audio" — drop your own .flac / .wav and have EVERY pretrained model
    (ResNet, 3-Block CNN and the classic XGBoost × DSP detectors) score it in
    parallel on CPU: a live, side-by-side comparison with a consensus verdict.
    This is the star of the public web demo (no corpus required).
  • "Analyse on a split" — WHY a detector scores the EER / minDCF it does: score
    distributions, ROC and DET curves and an interactive decision threshold
    (needs the local ASVspoof corpus).

set_page_config and PAGE_CSS are applied in app.py.
"""

import concurrent.futures as cf
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import altair as alt  # noqa: E402
import librosa  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data_loader import (  # noqa: E402
    LABEL_BONAFIDE, LABEL_SPOOF, ASVspoofTorchDataset, stratified_subsample,
)
from src.features import FeatureExtractor  # noqa: E402
from src.metrics import calculate_eer, calculate_min_dcf  # noqa: E402
from src.models import get_classic_model  # noqa: E402
from src.pipeline import extract_feature_matrix  # noqa: E402
from src.ui_helpers import (  # noqa: E402
    BONAFIDE_COLOR, SPOOF_COLOR, HF_EVAL_DATASETS, HF_EVAL_PER_CLASS,
    available_pretrained_models, corpus_available,
    demo_corpus_notice, demo_mode, fig_activation_grid, fig_cnn_input,
    fig_waveform, get_extractor, get_samples, hf_eval_samples, load_demo_leaderboard,
    load_pretrained_cnn, load_pretrained_model, op_busy_notice, pretrained_available,
    show_empty_state, sidebar_panel,
)

FEATURE_LABELS = FeatureExtractor.OPTION_NAMES
FEATURE_ORDER  = ["1", "2", "3", "4", "6", "5"]
CLASSIFIERS    = {
    "Logistic Regression": "logistic_regression",
    "SVM (RBF)":           "svm_lineal",
    "XGBoost":             "xgboost",
}

extractor = get_extractor()
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dev_label = "CUDA GPU" if device.type == "cuda" else "CPU"

corpus_ok  = corpus_available()
pre_models = available_pretrained_models()

st.title("Detection Analysis")
st.caption(
    "Drop your own clip to have every pretrained model score it side by side, "
    "or analyse how one detector separates bonafide from spoof on a corpus split."
)

# Nothing to run at all → a single clear notice.
if not corpus_ok and not pre_models:
    demo_corpus_notice(
        "Detection Analysis unavailable",
        "This page needs either the local ASVspoof corpus or at least one "
        "pretrained model. Set HF_BASE_URL in ui_helpers to your Hugging Face "
        "folder to enable the live multi-model analysis, or run the app locally "
        "with the dataset.",
    )
    st.stop()


# ── Split-analysis scorers (local; need the corpus) ──────────────────────── #
@st.cache_resource(show_spinner=False)
def _classic_model(feat_key, clf_name, subset, seed=42):
    """Fit (and cache) a classic DSP × classifier model on a train subset."""
    train = stratified_subsample(get_samples("train"), subset, seed)
    x_tr, y_tr, _ = extract_feature_matrix(train, extractor, feat_key, "train",
                                           n_workers=4, use_cache=True)
    imbalance = float(np.sum(y_tr == LABEL_BONAFIDE)) / max(np.sum(y_tr == LABEL_SPOOF), 1)
    model = get_classic_model(clf_name, seed=seed, scale_pos_weight=imbalance)
    model.fit(x_tr, y_tr)
    return model


def _score_classic(feat_key, clf_name, split, subset, seed=42):
    model   = _classic_model(feat_key, clf_name, subset, seed)
    evalset = stratified_subsample(get_samples(split), subset, seed + 1)
    x_ev, y_ev, _ = extract_feature_matrix(evalset, extractor, feat_key, split,
                                           n_workers=4, use_cache=True)
    return model.predict_proba(x_ev)[:, 1].tolist(), y_ev.tolist()


_SPEC_TAG = (f"{int(extractor.sample_rate)}_{int(extractor.n_fft)}_"
             f"{int(extractor.hop_length)}_"
             f"{int(extractor.freq_bins)}x{int(extractor.time_frames)}")


def _score_cnn_on_samples(model, mdev, samples):
    """Score a CNN on an explicit (path, label) list (local or HF-cached)."""
    loader = DataLoader(
        ASVspoofTorchDataset(samples, extractor.get_spectrogram_matrix,
                             extractor.sample_rate, augment=False,
                             cache_tag=_SPEC_TAG),
        batch_size=32, shuffle=False, num_workers=0,
    )
    scores, labels = [], []
    model.eval()
    with torch.no_grad():
        for tensors, lbls in loader:
            logits = model(tensors.to(mdev))
            scores.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(lbls.long().tolist())
    return scores, labels


def _score_cnn_on_split(model, mdev, split, subset, seed=42):
    samples = stratified_subsample(get_samples(split), subset, seed)
    return _score_cnn_on_samples(model, mdev, samples)


# ── Web-demo scorers: pretrained registry models on HF-streamed eval clips ─── #
def _score_classic_on_samples(entry, samples):
    """Load a pretrained classic estimator and score it on (path, label) clips."""
    model = load_pretrained_model(entry)
    x_ev, y_ev, _ = extract_feature_matrix(samples, extractor, entry["feat"],
                                           f"hf_{entry['key']}", n_workers=4,
                                           use_cache=True)
    return model.predict_proba(x_ev)[:, 1].tolist(), y_ev.tolist()


def _render_split_results():
    """Shared rendering of a scored split: metrics, threshold slider and the
    distribution / ROC / DET plots. Reads da_scores / da_labels / da_name from
    session state (used by both the local-corpus and web-demo split analysis)."""
    if "da_scores" not in st.session_state:
        show_empty_state(
            "No detector analysed yet",
            "Pick a detector and press Analyze. You will get its score "
            "distribution, ROC and DET curves, and an interactive decision "
            "threshold to explore the false-alarm vs miss trade-off behind the EER.",
        )
        return

    scores = np.asarray(st.session_state["da_scores"], dtype=float)
    labels = np.asarray(st.session_state["da_labels"], dtype=int)
    bona   = scores[labels == LABEL_BONAFIDE]
    spoof  = scores[labels == LABEL_SPOOF]

    eer, eer_thr = calculate_eer(scores.tolist(), labels.tolist())
    mindcf       = calculate_min_dcf(scores.tolist(), labels.tolist())

    grid = np.linspace(0.0, 1.0, 501)
    far  = np.array([(spoof < t).mean() if len(spoof) else 0.0 for t in grid])
    frr  = np.array([(bona >= t).mean() if len(bona) else 0.0 for t in grid])
    tpr  = 1.0 - far
    fpr  = frr
    order = np.argsort(fpr)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc   = float(_trapz(tpr[order], fpr[order]))

    st.markdown(f"**{st.session_state['da_name']}** &nbsp;·&nbsp; "
                f"{len(scores):,} trials &nbsp;·&nbsp; "
                f"{len(bona):,} bonafide / {len(spoof):,} spoof",
                unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("minDCF", f"{mindcf:.3f}", help="Primary ASVspoof cost metric.")
    m2.metric("EER", f"{100 * eer:.2f} %")
    m3.metric("AUC-ROC", f"{auc:.3f}")
    m4.metric("EER threshold", f"{eer_thr:.3f}")

    st.divider()
    st.markdown('<div class="section-label">Decision threshold</div>',
                unsafe_allow_html=True)
    thr = st.slider("Threshold on p(spoof)", 0.0, 1.0, float(round(eer_thr, 3)), 0.005,
                    help="score ≥ threshold → declared spoof.")
    t_far = (spoof < thr).mean() if len(spoof) else 0.0
    t_frr = (bona >= thr).mean() if len(bona) else 0.0
    preds = (scores >= thr).astype(int)
    t_acc = float((preds == labels).mean())
    tc1, tc2, tc3 = st.columns(3)
    tc1.metric("False acceptance (spoof let through)", f"{100 * t_far:.2f} %")
    tc2.metric("False rejection (bonafide blocked)",   f"{100 * t_frr:.2f} %")
    tc3.metric("Accuracy @ threshold",                 f"{100 * t_acc:.2f} %")

    st.divider()
    g1, g2 = st.columns(2, gap="large")
    with g1:
        st.markdown("**Score distribution**")
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        ax.hist(bona,  bins=40, alpha=0.7, label="Bonafide", color=BONAFIDE_COLOR)
        ax.hist(spoof, bins=40, alpha=0.7, label="Spoof",    color=SPOOF_COLOR)
        ax.axvline(thr, color="#FFD54F", lw=1.6, ls="--", label=f"threshold {thr:.2f}")
        ax.axvline(eer_thr, color="#9E9E9E", lw=1.1, ls=":", label=f"EER thr {eer_thr:.2f}")
        ax.set_xlabel("p(spoof)"); ax.set_ylabel("count")
        ax.legend(fontsize=7.5, loc="upper center")
        fig.tight_layout(); st.pyplot(fig, clear_figure=True)

        st.markdown("**ROC curve**")
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        ax.plot(fpr[order], tpr[order], color="#4FC3F7", lw=1.8)
        ax.plot([0, 1], [0, 1], color="#5C6B8A", lw=0.8, ls="--")
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_title(f"AUC = {auc:.3f}", fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        fig.tight_layout(); st.pyplot(fig, clear_figure=True)

    with g2:
        st.markdown("**DET curve** (miss vs false-alarm)")
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        ax.plot(100 * far, 100 * frr, color="#AB47BC", lw=1.8)
        lim = max(1.0, 100 * max(far.max(), frr.max()))
        ax.plot([0, lim], [0, lim], color="#5C6B8A", lw=0.8, ls="--", label="EER line")
        ax.scatter([100 * eer], [100 * eer], color="#FFD54F", zorder=5,
                   label=f"EER {100 * eer:.1f}%")
        ax.scatter([100 * t_far], [100 * t_frr], color="#66BB6A", zorder=5,
                   label="threshold")
        ax.set_xlabel("False acceptance (%)"); ax.set_ylabel("False rejection (%)")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.legend(fontsize=7.5); fig.tight_layout(); st.pyplot(fig, clear_figure=True)

        st.markdown(
            '<div class="info-card"><div class="ic-title">Reading it</div>'
            '<p class="ic-body">The <b>EER</b> is where the two error rates meet '
            '(the diagonal). <b>minDCF</b> weighs a false accept 10× a miss at a '
            '5% prior, so it only drops below 1.0 once false acceptances are very '
            'rare. Slide the threshold to walk along the DET curve.</p></div>',
            unsafe_allow_html=True,
        )


# ===========================================================================
# Single-clip inference shared by the multi-model tab
# ===========================================================================
def _load_signal(uploaded):
    signal, _ = librosa.load(uploaded, sr=extractor.sample_rate, mono=True)
    if len(signal) < extractor.n_fft:
        signal = np.pad(signal, (0, extractor.n_fft - len(signal)))
    return signal


def _load_all_models(entries):
    """Load every pretrained model AT ONCE (parallel threads) instead of one by
    one — the first-run Hugging Face downloads then overlap, which is the slow
    part. Each worker thread inherits the Streamlit script context so the cached
    loaders behave exactly as on the main thread. Returns (loaded, failed)."""
    import threading

    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

    ctx = get_script_run_ctx()

    def _init():
        add_script_run_ctx(threading.current_thread(), ctx)

    loaded, failed = {}, []
    with cf.ThreadPoolExecutor(max_workers=min(8, max(1, len(entries))),
                               initializer=_init) as pool:
        futs = {pool.submit(load_pretrained_model, e): e for e in entries}
        for fut in cf.as_completed(futs):
            e = futs[fut]
            try:
                loaded[e["key"]] = fut.result()
            except Exception as exc:                 # noqa: BLE001 — report, continue
                failed.append((e["name"], str(exc)))
    return loaded, failed


def _analyse_all_models(signal, entries, loaded, threshold):
    """Score one clip with every pretrained model. DSP front-ends are extracted
    once each in parallel threads (librosa releases the GIL), then each model
    predicts on its precomputed representation. Returns a list of result dicts."""
    need_spec  = any(e["kind"] == "cnn" for e in entries)
    uniq_feats = sorted({e["feat"] for e in entries if e["kind"] == "classic"})

    feat_vecs, spec_np = {}, None
    with cf.ThreadPoolExecutor(max_workers=min(6, len(uniq_feats) + 1)) as pool:
        fut_feat = {fk: pool.submit(extractor.get_flat_vector, signal, fk)
                    for fk in uniq_feats}
        fut_spec = pool.submit(extractor.get_spectrogram_matrix, signal) if need_spec else None
        for fk, fut in fut_feat.items():
            feat_vecs[fk] = fut.result().reshape(1, -1)
        if fut_spec is not None:
            spec_np = fut_spec.result()

    spec_tensor = (torch.from_numpy(spec_np).unsqueeze(0).unsqueeze(0).float()
                   if spec_np is not None else None)

    rows = []
    for e in entries:
        model = loaded[e["key"]]
        acts  = None
        if e["kind"] == "cnn":
            model.eval()
            with torch.no_grad():
                logit, acts = model.forward_with_activations(spec_tensor)
            prob = float(torch.sigmoid(logit).item())
        else:
            prob = float(model.predict_proba(feat_vecs[e["feat"]])[0, 1])
        rows.append({
            "key":     e["key"],
            "Model":   e["name"],
            "Front-end": e["front"],
            "p(spoof)": prob,
            "Verdict": "SPOOF" if prob >= threshold else "BONAFIDE",
            "_acts":   acts,
        })
    return rows


# ===========================================================================
# Tabs — the multi-model "Test an audio" leads on BOTH the web demo and locally,
# so the layout is identical everywhere (Test an audio first, split second).
# ===========================================================================
tab_test, tab_analyse = st.tabs(["Test an audio", "Analyse on a split"])

# These are only meaningful in the (corpus-backed) split analysis below; pre-set
# so the sidebar never trips over them in the demo.
source = feat_key = clf_disp = None


# ===========================================================================
# Multi-model file analysis — works fully on CPU, no corpus needed
# ===========================================================================
with tab_test:
  if not pre_models:
    demo_corpus_notice(
        "No pretrained models configured",
        "Set HF_BASE_URL in ui_helpers to your Hugging Face folder, or drop the "
        "weight files into <code>models/</code>, to enable the live multi-model "
        "analysis. Run <b>Benchmark → Full comparison</b> locally to train and "
        "export the model files.",
    )
  else:
    st.markdown(
        f"Your clip is scored by **all {len(pre_models)} pretrained models** at "
        "once — the two CNNs on the STFT-dB spectrogram and the classic XGBoost "
        "detectors on their DSP front-ends — every one on CPU."
    )
    tc1, tc2 = st.columns([3, 1], vertical_alignment="bottom")
    with tc2:
        threshold = st.slider("Threshold", 0.05, 0.95, 0.50, 0.01, key="da_test_thr",
                              help="p(spoof) ≥ threshold → declared spoof.")
    with tc1:
        uploaded = st.file_uploader("Upload a .flac / .wav file", type=["flac", "wav"],
                                    key="da_test_upload")

    if uploaded is None:
        show_empty_state(
            "No audio uploaded",
            "Drop a .flac or .wav clip to see how the whole model zoo — ResNet, "
            "3-Block CNN and the classic XGBoost detectors — judges it in real time.",
        )
    else:
        signal = _load_signal(uploaded)
        with st.spinner("Consulting the Jedi Archives (loading every model in parallel)…"):
            loaded, failed = _load_all_models(pre_models)
        for _name, _err in failed:
            st.warning(f"{_name} unavailable: {_err}")
        entries_ok = [e for e in pre_models if e["key"] in loaded]
        if not entries_ok:
            st.error("No model could be loaded — check the download URLs.")
            st.stop()
        with st.spinner("Running it past the astromech (scoring every model)…"):
            rows = _analyse_all_models(signal, entries_ok, loaded, threshold)

        probs   = [r["p(spoof)"] for r in rows]
        mean_p  = float(np.mean(probs))
        n_spoof = int(sum(p >= threshold for p in probs))
        n_total = len(rows)

        # ── Final verdict comes from the BEST trained model (lowest eval minDCF in
        #    the committed leaderboard); fall back to the headline CNNs, then the
        #    first model, when no metrics are available. ──────────────────────── #
        _board = load_demo_leaderboard()

        def _best_row(rows, board):
            rated = [(board.get(r["key"], {}).get("mindcf_eval"), r) for r in rows]
            rated = [(md, r) for md, r in rated if isinstance(md, (int, float))]
            if rated:
                return min(rated, key=lambda t: t[0])[1]
            for key in ("resnet", "cnn3x3"):       # headline CNNs preferred
                for r in rows:
                    if r["key"] == key:
                        return r
            return rows[0]

        best = _best_row(rows, _board)
        b_md = _board.get(best["key"], {}).get("mindcf_eval")
        v_color = SPOOF_COLOR if best["Verdict"] == "SPOOF" else BONAFIDE_COLOR
        v_text  = ("SPOOF — deepfake" if best["Verdict"] == "SPOOF"
                   else "BONAFIDE — real speech")
        _rank_txt = (f" &nbsp;·&nbsp; eval minDCF {b_md:.3f}"
                     if isinstance(b_md, (int, float)) else "")

        # Primary card: the best detector's call (the authoritative final verdict),
        # with an outcome-coloured saber glow (the glow colour IS the data colour).
        st.markdown(
            f"<div style='text-align:center;margin:0.6rem auto 0.3rem;"
            f"max-width:560px;padding:1rem 1.3rem;border-radius:0.9rem;"
            f"border:1px solid {v_color}59;"
            f"box-shadow:0 0 20px {v_color}55, inset 0 0 24px {v_color}1f;'>"
            f"<span style='display:block;font-size:0.66rem;font-weight:800;"
            f"letter-spacing:0.16em;text-transform:uppercase;color:#9EA8C0;"
            f"margin-bottom:0.15rem;'>Final verdict</span>"
            f"<span style='font-size:1.8rem;font-weight:700;color:{v_color};"
            f"text-shadow:0 0 14px {v_color}aa;'>{v_text}</span><br>"
            f"<span style='color:#9EA8C0;font-size:0.9rem;'>"
            f"best detector: <b style='color:#C9D7F5;'>{best['Model']}</b> "
            f"&nbsp;·&nbsp; p(spoof) = {best['p(spoof)']:.3f}{_rank_txt}"
            f"</span></div>",
            unsafe_allow_html=True,
        )
        # Secondary line: the panel of models behind the verdict.
        _cons = ("majority flags spoof" if n_spoof * 2 > n_total
                 else ("majority says bonafide" if n_spoof * 2 < n_total
                       else "the panel is split"))
        st.markdown(
            f"<div style='text-align:center;color:#8A95AE;font-size:0.85rem;"
            f"margin:0 auto 0.4rem;'>Panel of {n_total} models — {_cons}: "
            f"<b>{n_spoof}/{n_total}</b> flag spoof &nbsp;·&nbsp; "
            f"mean p(spoof) = {mean_p:.3f} &nbsp;·&nbsp; threshold = {threshold:.2f}"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.audio(signal, sample_rate=extractor.sample_rate)
        st.divider()

        # ── Every model's individual verdict (who flagged what) ─────────── #
        st.markdown("**Every model's verdict**")
        _sorted = sorted(rows, key=lambda r: r["p(spoof)"], reverse=True)
        _chips = []
        for r in _sorted:
            c = SPOOF_COLOR if r["Verdict"] == "SPOOF" else BONAFIDE_COLOR
            _chips.append(
                f'<div style="border:1px solid {c}55;border-left:3px solid {c};'
                f'border-radius:0.6rem;padding:0.5rem 0.7rem;background:{c}14;">'
                f'<div style="font-size:0.8rem;font-weight:700;color:#C9D7F5;'
                f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" '
                f'title="{r["Model"]} · {r["Front-end"]}">{r["Model"]}</div>'
                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:baseline;margin-top:0.25rem;">'
                f'<span style="color:{c};font-weight:800;font-size:0.82rem;">'
                f'{r["Verdict"]}</span>'
                f'<span style="color:#8A95AE;font-size:0.74rem;">'
                f'p={r["p(spoof)"]:.2f}</span></div></div>'
            )
        st.markdown(
            '<div style="display:grid;grid-template-columns:'
            'repeat(auto-fill,minmax(168px,1fr));gap:0.5rem;margin:0.3rem 0 0.5rem;">'
            + "".join(_chips) + "</div>",
            unsafe_allow_html=True,
        )
        st.caption("Each pretrained model's own call on your clip — green = "
                   "bonafide, red = spoof. The full sortable table and chart are below.")
        st.divider()

        # ── Per-model comparison table + bar chart ──────────────────────── #
        df = (pd.DataFrame([{k: r[k] for k in ("Model", "Front-end", "p(spoof)", "Verdict")}
                            for r in rows])
              .sort_values("p(spoof)", ascending=False).reset_index(drop=True))

        gcol1, gcol2 = st.columns([1.05, 1], gap="large")
        with gcol1:
            st.markdown("**Per-model scores**")

            def _vcolor(v):
                c = SPOOF_COLOR if v == "SPOOF" else BONAFIDE_COLOR
                return f"color:{c};font-weight:700;"

            st.dataframe(
                df.style.format({"p(spoof)": "{:.3f}"})
                        .applymap(_vcolor, subset=["Verdict"]),
                width="stretch", hide_index=True,
            )
            st.caption("Higher p(spoof) → more confident the clip is synthetic. "
                       "Disagreement across front-ends is itself informative.")
        with gcol2:
            st.markdown("**Probability per model**")
            bar = (alt.Chart(df).mark_bar().encode(
                        x=alt.X("p(spoof):Q", scale=alt.Scale(domain=[0, 1]),
                                title="p(spoof)"),
                        y=alt.Y("Model:N", sort="-x", title=None),
                        color=alt.Color("Verdict:N", legend=None,
                                        scale=alt.Scale(domain=["BONAFIDE", "SPOOF"],
                                                        range=[BONAFIDE_COLOR, SPOOF_COLOR])),
                        tooltip=["Model", "Front-end", "p(spoof)", "Verdict"])
                   .properties(height=max(150, 42 * len(df))))
            rule = (alt.Chart(pd.DataFrame({"t": [threshold]}))
                    .mark_rule(color="#FFD54F", strokeDash=[4, 4]).encode(x="t:Q"))
            st.altair_chart(bar + rule, width="stretch")

        # ── Signal views + CNN activation maps ──────────────────────────── #
        st.divider()
        pv1, pv2 = st.columns(2)
        with pv1:
            st.pyplot(fig_waveform(signal, extractor.sample_rate, title="Waveform",
                                   label=None), clear_figure=True)
        with pv2:
            st.pyplot(fig_cnn_input(signal, extractor), clear_figure=True)

        _cnn_rows = [r for r in rows if r["_acts"] is not None]
        if _cnn_rows:
            with st.expander("Convolutional activation maps — how each CNN reacts"):
                for r in _cnn_rows:
                    st.markdown(f"**{r['Model']}** — p(spoof) = {r['p(spoof)']:.3f}")
                    for i, act in enumerate(r["_acts"], start=1):
                        n_ch = act.shape[1]
                        st.pyplot(
                            fig_activation_grid(
                                act[0].numpy(),
                                f"Block {i} — {n_ch} feature maps {tuple(act.shape[1:])}"),
                            clear_figure=True,
                        )


# ===========================================================================
# Analyse one detector on a corpus split (local; needs the dataset)
# ===========================================================================
with tab_analyse:
  if corpus_ok:
    has_cnn = "cnn_model" in st.session_state
    with st.container(border=True):
        st.markdown('<div class="section-label">Detector</div>', unsafe_allow_html=True)
        src_opts = ["Classic model"]
        if pretrained_available():
            src_opts.append("CNN")
        if has_cnn:
            src_opts.append("CNN (this session)")
        source = st.segmented_control("Source", src_opts, default="Classic model",
                                      key="da_source", label_visibility="collapsed")
        source = source or "Classic model"
        _busy = op_busy_notice()

        if source == "Classic model":
            c1, c2 = st.columns(2, vertical_alignment="bottom")
            with c1:
                with st.container(key="nosearch_da_feat"):
                    feat_key = st.selectbox("Feature extractor", FEATURE_ORDER,
                                            format_func=lambda k: FEATURE_LABELS[k],
                                            key="da_feat")
            with c2:
                with st.container(key="nosearch_da_clf"):
                    clf_disp = st.selectbox("Classifier", list(CLASSIFIERS), key="da_clf")
        elif source == "CNN":
            st.caption("Pretrained CNN checkpoint, served on CPU.")
        else:
            st.caption("Scores the CNN trained in Benchmark · CNN this session.")

    a1, a2, a3 = st.columns([1, 1, 1.4], vertical_alignment="bottom")
    with a1:
        split = st.selectbox("Split", ["dev", "eval"], key="da_split")
    with a2:
        subset = st.number_input("Files", 100, 25400, 800, step=100, key="da_subset")
    with a3:
        analyze = st.button("Analyze", type="primary", disabled=_busy,
                            icon=":material/insights:", width="stretch")

    if analyze:
        with st.spinner("Awaiting the Council's verdict (scoring the detector)…"):
            if source == "Classic model":
                sc, lb = _score_classic(feat_key, CLASSIFIERS[clf_disp],
                                        split, int(subset))
                name = f"{FEATURE_LABELS[feat_key]} × {clf_disp} · {split}"
            elif source == "CNN":
                _m, _ = load_pretrained_cnn()
                sc, lb = _score_cnn_on_split(_m, "cpu", split, int(subset))
                name = f"CNN · {split}"
            else:
                sc, lb = _score_cnn_on_split(st.session_state["cnn_model"], device,
                                             split, int(subset))
                name = f"CNN · {split}"
        st.session_state["da_scores"] = sc
        st.session_state["da_labels"] = lb
        st.session_state["da_name"]   = name

    _render_split_results()

  elif pre_models:
    # Web demo: score a PRETRAINED registry model on eval clips streamed from the
    # public Hugging Face datasets (no corpus, no training — cross-dataset eval).
    classic_entries = [e for e in pre_models if e["kind"] == "classic"]
    cnn_entries     = [e for e in pre_models if e["kind"] == "cnn"]

    with st.container(border=True):
        st.markdown('<div class="section-label">Detector</div>', unsafe_allow_html=True)
        src_opts = (["Classic model"] if classic_entries else []) + \
                   (["CNN"] if cnn_entries else [])
        source = st.segmented_control("Source", src_opts, default=src_opts[0],
                                      key="da_source_hf", label_visibility="collapsed")
        source = source or src_opts[0]
        _busy = op_busy_notice()

        cnn_name = None
        if source == "Classic model":
            feats = sorted({e["feat"] for e in classic_entries},
                           key=lambda f: FEATURE_ORDER.index(f)
                           if f in FEATURE_ORDER else 99)
            c1, c2 = st.columns(2, vertical_alignment="bottom")
            with c1:
                with st.container(key="nosearch_da_feat_hf"):
                    feat_key = st.selectbox("Feature extractor", feats,
                                            format_func=lambda k: FEATURE_LABELS[k],
                                            key="da_feat_hf")
            with c2:
                clf_choices = [d for d, n in CLASSIFIERS.items()
                               if any(e["clf"] == n and e["feat"] == feat_key
                                      for e in classic_entries)]
                with st.container(key="nosearch_da_clf_hf"):
                    clf_disp = st.selectbox("Classifier", clf_choices, key="da_clf_hf")
        else:
            with st.container(key="nosearch_da_cnn_hf"):
                cnn_name = st.selectbox("CNN", [e["name"] for e in cnn_entries],
                                        key="da_cnn_hf")
            st.caption("CNN checkpoint, served on CPU.")

    a1, a2, a3 = st.columns([1.3, 1, 1.4], vertical_alignment="bottom")
    with a1:
        corpus = st.selectbox("Eval corpus", list(HF_EVAL_DATASETS), key="da_corpus_hf",
                              help="Eval split streamed from the public HF dataset.")
    with a2:
        nper = st.number_input("Clips / class", 5, 100, HF_EVAL_PER_CLASS, step=5,
                               key="da_nper_hf",
                               help="Balanced bonafide + spoof clips fetched from HF.")
    with a3:
        analyze = st.button("Analyze", type="primary", disabled=_busy,
                            icon=":material/insights:", width="stretch")

    if analyze:
        with st.spinner(f"Pulling {corpus} records from the Archives and scoring…"):
            samples = hf_eval_samples(corpus, int(nper))
            if not samples:
                st.warning("Could not fetch eval clips from Hugging Face — "
                           "try again or pick another corpus.")
            else:
                if source == "Classic model":
                    entry = next(e for e in classic_entries
                                 if e["feat"] == feat_key
                                 and e["clf"] == CLASSIFIERS[clf_disp])
                    sc, lb = _score_classic_on_samples(entry, samples)
                    name = f"{FEATURE_LABELS[feat_key]} × {clf_disp} · {corpus} eval"
                else:
                    entry = next(e for e in cnn_entries if e["name"] == cnn_name)
                    sc, lb = _score_cnn_on_samples(load_pretrained_model(entry),
                                                   "cpu", samples)
                    name = f"{cnn_name} · {corpus} eval"
                st.session_state["da_scores"] = sc
                st.session_state["da_labels"] = lb
                st.session_state["da_name"]   = name

    _render_split_results()

  else:
    demo_corpus_notice(
        "Split analysis needs the corpus or pretrained models",
        "Scoring a detector on a split needs either the local ASVspoof dataset "
        "or at least one pretrained model. Set HF_BASE_URL in ui_helpers to your "
        "Hugging Face weights folder, or run the app locally with the dataset.",
    )

# ── Sidebar ──────────────────────────────────────────────────────────────── #
with st.sidebar:
    _rows = [("Device", dev_label), ("Models", f"{len(pre_models)} pretrained")]
    if source == "Classic model" and feat_key is not None:
        _rows.append(("Features", FEATURE_LABELS[feat_key]))
    if "da_scores" in st.session_state:
        _sc = np.asarray(st.session_state["da_scores"], dtype=float)
        _lb = np.asarray(st.session_state["da_labels"], dtype=int)
        _eer, _ = calculate_eer(_sc.tolist(), _lb.tolist())
        _dcf = calculate_min_dcf(_sc.tolist(), _lb.tolist())
        _rows += [("Trials", f"{len(_sc):,}"), ("minDCF", f"{_dcf:.3f}"),
                  ("EER", f"{100 * _eer:.2f} %")]
    sidebar_panel("Detection", _rows)
