# -*- coding: utf-8 -*-
"""
modes/_mode_classic.py — "Classic models" mode of the Benchmark page: configure
and launch one DSP-extractor × classifier experiment (the CNN has its own mode).

Configuration lives in the MAIN area (a bordered panel), not the sidebar:
dropdown popovers detach from sidebars and the controls are central to the
page's purpose. The sidebar carries navigation + environment status only.

Dispatched from app_pages/2_Benchmark.py via runpy; set_page_config and PAGE_CSS
are applied once in app.py.
"""

import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import altair as alt
import pandas as pd
import streamlit as st

from src.data_loader import stratified_subsample
from src.features import FeatureExtractor
from src.pipeline import (
    MODEL_OPTIONS, extract_feature_matrix, run_classic_models, score_fitted_classic,
)
from src.reporting import (
    COL_ACCURACY, COL_EER, COL_FEATURES, COL_MIN_DCF, COL_MODEL,
    RESULT_COLUMNS,
)
from src.ui_helpers import (
    available_pretrained_models, corpus_available, demo_corpus_notice,
    eval_corpora_for, eval_score_controls,
    get_extractor, get_samples, load_config, load_pretrained_classic,
    op_busy_notice, op_in_progress, show_empty_state, sidebar_panel, test_audio_cta,
)

st.markdown("""
<style>
@keyframes bannerGlow {
    0%, 100% { box-shadow: 0 0 0 1px rgba(102,187,106,0.18); }
    50%      { box-shadow: 0 0 20px rgba(102,187,106,0.30); }
}
.best-banner {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 0.5rem 1.1rem;
    background: linear-gradient(135deg,
        rgba(102,187,106,0.1) 0%, rgba(102,187,106,0.03) 100%);
    border: 1px solid rgba(102,187,106,0.3);
    border-left: 3px solid #66BB6A;
    border-radius: 0.8rem;
    padding: 0.8rem 1.15rem;
    margin: 0.2rem 0 0.9rem;
    /* Continuous soft glow — draws the eye to the headline result. */
    animation: bannerGlow 3.5s ease-in-out infinite;
}
.best-banner .bb-tag {
    font-size: 0.64rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #81C784;
}
.best-banner .bb-combo {
    font-size: 1.02rem;
    font-weight: 750;
    color: #E8EDF8;
}
.best-banner .bb-metric {
    font-size: 0.8rem;
    color: #AFC3E8;
}
.best-banner .bb-metric b { color: #A5D6A7; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

st.title("Classic Models")
st.caption(
    "DSP extractor + classic classifier on ASVspoof 2019 LA. "
    "Results accumulate across runs so you can compare configurations."
)

if not corpus_available():
    demo_corpus_notice(
        "Classic models — disabled in the web demo",
        "Fitting the DSP × classifier models needs the full ASVspoof corpus, "
        "which is not available on the public cloud.",
    )
    test_audio_cta()
    st.stop()

config    = load_config()
extractor = get_extractor()

FEATURE_LABELS = FeatureExtractor.OPTION_NAMES
MODEL_LABELS = {
    "1": "Logistic Regression",
    "2": "SVM (RBF)",
    "3": "XGBoost",
    "4": "All classic models",
}

FEATURE_BLURBS = {
    "1": "Frame-level energy envelope — the simplest temporal baseline.",
    "2": "Mel-scale cepstrum: the classic speech front-end, tuned to human hearing.",
    "3": "Linear-frequency cepstrum — keeps high-band detail where TTS artefacts live.",
    "4": "Multi-resolution wavelet energies (db4) — transient and band-energy cues.",
    "6": "Constant-Q cepstrum — log-spaced bins, the strongest classic anti-spoofing front-end.",
}

FEATURE_ORDER = ["1", "2", "3", "4", "6"]
MODEL_BLURBS = {
    "1": "Fast linear baseline; well-calibrated scores out of the box.",
    "2": "Maximum-margin linear separator; robust on small subsets.",
    "3": "Gradient-boosted trees; captures non-linear feature interactions.",
    "4": "Runs LR, SVM and XGBoost back-to-back on the same features.",
}

SPLIT_DEV, SPLIT_EVAL, SPLIT_BOTH = "Dev", "Eval", "Dev + Eval"
SPLIT_HELP = {
    SPLIT_DEV:  "Score on the dev split — same attacks (A01–A06) as training.",
    SPLIT_EVAL: "Score on the eval split — 13 unseen attacks (A07–A19): the generalisation test.",
    SPLIT_BOTH: "Score on dev and eval — adds an extra eval row per model.",
}

# Right column tweaks: the Evaluate/Score box fills the full column width (its
# two groups pushed to the edges), and the tall Run-experiment button makes the
# right rows line up with Classifier / Advanced on the left.
st.markdown("""
<style>
[class*="st-key-evalgrp_exp"] { width: 100% !important; justify-content: space-between; }
[class*="st-key-runbtn_exp"] button { min-height: 3.4rem; }
[class*="st-key-advbtn_exp"] button { min-height: 3.4rem; }
</style>
""", unsafe_allow_html=True)

# ── Configuration panel (main area) ──────────────────────────────────────── #
with st.container(border=True):
    st.markdown('<div class="section-label">Experiment configuration</div>',
                unsafe_allow_html=True)

    _is_busy = op_in_progress()

    # Row 1: Feature+Classifier | Train on / Evaluate on / Score on controls
    _r1l, _r1r = st.columns(2, gap="large")
    with _r1l:
        # nosearch_* container → CSS blocks type-to-filter (pure dropdown).
        _fc, _mc = st.columns(2, gap="small")
        with _fc:
            with st.container(key="nosearch_feature"):
                feature_option = st.selectbox(
                    "Feature extractor",
                    FEATURE_ORDER,
                    format_func=lambda k: FEATURE_LABELS[k],
                    key="exp_feature",
                )
        with _mc:
            with st.container(key="nosearch_model"):
                model_option = st.selectbox(
                    "Classifier",
                    list(MODEL_LABELS.keys()),
                    format_func=lambda k: MODEL_LABELS[k],
                    key="exp_model",
                )
    with _r1r:
        _train_lbl = "Trained on" if st.session_state.get("fitted_classic_models") else "Train on"
        corpus, score_split = eval_score_controls("exp", train_label=_train_lbl)
        _busy = op_busy_notice()

    # Row 2: Advanced + Clear | Train models + Evaluate (same row → natural alignment)
    _r2l, _r2r = st.columns(2, gap="large")
    with _r2l:
        with st.container(key="advbtn_exp"):
            with st.popover("Advanced", icon=":material/tune:", width="stretch"):
                _a1, _a2 = st.columns(2)
                with _a1:
                    subset = st.number_input(
                        "Files per subset", min_value=0, value=300, step=100,
                        help="0 = full dataset. Stratified subsampling preserves the "
                             "bonafide/spoof ratio.",
                    )
                    use_cache = st.checkbox(
                        "Use DSP feature cache", value=True,
                        help="Cache extracted vectors to disk, keyed by config hash.")
                with _a2:
                    seed = st.number_input("Seed", min_value=0, value=42, step=1)
                    workers = st.slider("DSP extraction threads", 1, 8, 4)
        clear_btn = st.button("Clear results", icon=":material/delete:",
                              width="stretch", disabled=_is_busy)

    with _r2r:
        # Check if evaluation is possible: session-fitted models OR HF pretrained.
        _needed = list(MODEL_OPTIONS[model_option])
        _fitted_info = st.session_state.get("fitted_classic_models", {})
        _fitted_match = (
            _fitted_info.get("feature") == feature_option
            and _fitted_info.get("model_option") == model_option
        )
        _hf_ok = all(
            any(e["kind"] == "classic" and e["clf"] == n and e["feat"] == feature_option
                for e in available_pretrained_models())
            for n in _needed
        )
        _can_eval = _fitted_match or _hf_ok
        with st.container(key="runbtn_exp"):
            run_btn = st.button("Train models", type="primary",
                                icon=":material/play_arrow:", width="stretch",
                                disabled=_busy)
        eval_btn = st.button(
            "Evaluate", icon=":material/query_stats:", width="stretch",
            disabled=_busy or not _can_eval,
            help=None if _can_eval else
                 "Run experiment first, or configure HF_BASE_URL for pretrained weights.",
        )

if clear_btn:
    st.session_state["experiment_rows"] = []
    st.session_state.pop("run_signatures", None)
    st.session_state.pop("fitted_classic_models", None)
    st.rerun()

# The 3-step pipeline overview lives on the Benchmark home page now; here we
# only keep the config summaries used by the run caption and the sidebar.
_subset_txt = "full dataset" if subset == 0 else f"{int(subset):,} files / subset"
_split_txt  = {SPLIT_DEV: "dev split", SPLIT_EVAL: "eval split",
               SPLIT_BOTH: "dev + eval splits"}[score_split]

COL_SPLIT = "Split"


def _tag_rows(results: list, base_split: str) -> list:
    """Attach the scored split (with eval corpus) to each row; strip markers."""
    tagged = []
    for row in results:
        row = dict(row)
        if "[EVAL]" in row.get(COL_MODEL, ""):
            row[COL_MODEL]    = row[COL_MODEL].replace("[EVAL]", "").strip()
            row[COL_FEATURES] = row[COL_FEATURES].replace("[EVAL]", "").strip()
            _c = str(row.get("Corpus", "")).strip()
            row[COL_SPLIT]    = f"eval · {_c}" if _c else "eval"
        else:
            row[COL_SPLIT] = base_split
        tagged.append(row)
    return tagged


# ── Execution ────────────────────────────────────────────────────────────── #
# Run experiment: train classifiers + score on 2019 LA dev (always).
# A run is fully determined by (extractor, classifier, subset, seed).
_run_sig = (feature_option, model_option, int(subset), int(seed))
if run_btn and _run_sig in st.session_state.get("run_signatures", set()):
    st.info(
        f"**{FEATURE_LABELS[feature_option]} × {MODEL_LABELS[model_option]}** "
        f"(subset {int(subset) or 'full'}, seed {int(seed)}) has already been "
        "trained — the dev results are unchanged. Press Evaluate for eval corpus "
        "scoring, or change the configuration to re-train.",
        icon=":material/info:",
    )
    run_btn = False

if run_btn:
    train_samples   = get_samples("train")
    primary_samples, primary_name = get_samples("dev"), "dev"
    if subset > 0:
        train_samples   = stratified_subsample(train_samples,   int(subset), int(seed))
        primary_samples = stratified_subsample(primary_samples, int(subset), int(seed) + 1)

    log      = io.StringIO()
    progress = st.progress(0.0, text="Extracting train features...")
    _fitted: dict = {}

    try:
        with contextlib.redirect_stdout(log):
            x_train, y_train, _ = extract_feature_matrix(
                train_samples, extractor, feature_option, "train",
                n_workers=int(workers), use_cache=use_cache,
            )
            progress.progress(0.5, text="Extracting dev features...")
            x_primary, y_primary, ms_dsp = extract_feature_matrix(
                primary_samples, extractor, feature_option, "dev",
                n_workers=int(workers), use_cache=use_cache,
            )
            progress.progress(0.8, text="Training and scoring on dev...")
            results = run_classic_models(
                MODEL_OPTIONS[model_option],
                x_train, y_train, x_primary, y_primary,
                FEATURE_LABELS[feature_option], ms_dsp, int(seed),
                eval_sets=[],           # no eval corpus here; use Evaluate button
                model_sink=lambda n, m: _fitted.update({n: m}),
            )
        progress.progress(1.0)
    except Exception as err:
        progress.empty()
        st.exception(err)
        st.stop()

    progress.empty()
    results = _tag_rows(results, primary_name)
    st.session_state.setdefault("experiment_rows", []).extend(results)
    st.session_state.setdefault("run_signatures", set()).add(_run_sig)
    # Store fitted models so the Evaluate button can score them on eval corpora.
    st.session_state["fitted_classic_models"] = {
        "models":       _fitted,
        "feature":      feature_option,
        "model_option": model_option,
    }

if eval_btn:
    # Resolve models: prefer session-fitted, fall back to HF pretrained.
    _fitted_info = st.session_state.get("fitted_classic_models", {})
    _fitted_match = (
        _fitted_info.get("feature") == feature_option
        and _fitted_info.get("model_option") == model_option
    )
    if _fitted_match:
        _models_to_eval = _fitted_info["models"]
    else:
        _models_to_eval = {}
        for _name in _needed:
            _entry = next(
                (e for e in available_pretrained_models()
                 if e["kind"] == "classic" and e["clf"] == _name
                 and e["feat"] == feature_option),
                None,
            )
            if _entry:
                _models_to_eval[_name] = load_pretrained_classic(
                    _entry["file"], _entry["url"], _entry["name"]
                )
        if not _models_to_eval:
            st.error("No models available for evaluation.")
            st.stop()

    _eval_corpora = eval_corpora_for(corpus)
    if not _eval_corpora:
        st.warning("Selected eval corpus has no samples available.")
    else:
        _prog = st.progress(0.0, text="Preparing eval...")
        try:
            _eval_results = []
            for _i, (_lbl, _samps) in enumerate(_eval_corpora):
                if subset > 0:
                    _samps = stratified_subsample(_samps, int(subset), int(seed) + 2)
                if not _samps:
                    continue
                _prog.progress(
                    _i / len(_eval_corpora),
                    text=f"Extracting features for {_lbl}…",
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    _x_ev, _y_ev, _ = extract_feature_matrix(
                        _samps, extractor, feature_option, f"eval[{_lbl}]",
                        n_workers=int(workers), use_cache=use_cache,
                    )
                _prog.progress(
                    (_i + 0.7) / len(_eval_corpora),
                    text=f"Scoring on {_lbl}…",
                )
                _eval_results += score_fitted_classic(
                    _models_to_eval, _x_ev, _y_ev,
                    FEATURE_LABELS[feature_option],
                    corpus_label=_lbl,
                )
            _prog.progress(1.0)
        except Exception as _err:
            _prog.empty()
            st.exception(_err)
            st.stop()
        _prog.empty()
        _eval_results = _tag_rows(_eval_results, "eval")
        st.session_state.setdefault("experiment_rows", []).extend(_eval_results)
        st.rerun()

# ── Results area ─────────────────────────────────────────────────────────── #
rows = st.session_state.get("experiment_rows", [])

if not rows:
    show_empty_state(
        "No experiments yet",
        "Set the configuration above and press Run experiment. Each run is "
        "added to a shared results table so you can benchmark configurations "
        "against each other — and they also feed the Leaderboard.",
    )
else:
    df = pd.DataFrame(rows)
    if COL_SPLIT not in df.columns:
        df[COL_SPLIT] = "dev"
    df = df[[COL_SPLIT] + [c for c in RESULT_COLUMNS if c in df.columns]]

    df_num = df.copy()
    for col in (COL_ACCURACY, COL_EER, COL_MIN_DCF):
        df_num[col] = pd.to_numeric(df_num[col], errors="coerce")
    valid = df_num.dropna(subset=[COL_EER])

    # ── Best configuration banner (by minDCF — primary metric — then EER) ──── #
    best_idx = None
    if not valid.empty:
        best_idx = valid.sort_values([COL_MIN_DCF, COL_EER],
                                     na_position="last").index[0]
        best     = df.loc[best_idx]
        st.markdown(
            '<div class="best-banner">'
            '<span class="bb-tag">Best configuration</span>'
            f'<span class="bb-combo">{best[COL_FEATURES]} × '
            f'{best[COL_MODEL].replace("[CPU]", "").strip()}</span>'
            f'<span class="bb-metric">minDCF <b>{best[COL_MIN_DCF]}</b></span>'
            f'<span class="bb-metric">EER <b>{best[COL_EER]} %</b></span>'
            f'<span class="bb-metric">accuracy <b>{best[COL_ACCURACY]}</b></span>'
            f'<span class="bb-metric">split <b>{best[COL_SPLIT]}</b></span>'
            '</div>',
            unsafe_allow_html=True,
        )

    # No "last run" metric strip — those figures are columns of the results
    # table below (and the best run is highlighted), so it would be redundant.
    tab_res, tab_charts = st.tabs(["Results table", "Charts"])

    with tab_res:
        def _highlight_best(row):
            style = ("background-color: rgba(102,187,106,0.14);"
                     "font-weight: 600;") if row.name == best_idx else ""
            return [style] * len(row)

        st.dataframe(df.style.apply(_highlight_best, axis=1),
                     width="stretch", hide_index=True)
        if best_idx is not None:
            st.caption("The highlighted row is the best configuration by minDCF "
                       "(primary metric), EER as tiebreaker.")
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results as CSV",
            data=csv_bytes,
            file_name="experiment_results.csv",
            mime="text/csv",
            icon=":material/download:",
        )

    with tab_charts:
        if valid.empty:
            st.info("No numeric results available to chart yet.")
        else:
            plot_df = valid.copy()
            plot_df["Classifier"] = (
                plot_df[COL_MODEL].str.split("(").str[0]
                .str.replace("[CPU]", "", regex=False).str.strip()
            )
            plot_df["Config"] = (
                plot_df[COL_FEATURES].str.split("(").str[0].str.strip()
                + " · " + plot_df["Classifier"]
                + " · " + plot_df[COL_SPLIT]
            )
            tooltip = [
                alt.Tooltip(COL_FEATURES, title="Features"),
                alt.Tooltip("Classifier"),
                alt.Tooltip(COL_SPLIT, title="Split"),
                alt.Tooltip(COL_EER, title="EER (%)"),
                alt.Tooltip(COL_MIN_DCF, title="minDCF"),
                alt.Tooltip(COL_ACCURACY, title="Accuracy"),
            ]

            def _hbar(metric: str, title: str):
                return (
                    alt.Chart(plot_df)
                    .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
                    .encode(
                        y=alt.Y("Config:N", title=None,
                                sort=alt.EncodingSortField(field=metric,
                                                           order="ascending"),
                                axis=alt.Axis(labelLimit=240)),
                        x=alt.X(f"{metric}:Q", title=title),
                        color=alt.Color("Classifier:N",
                                        scale=alt.Scale(range=["#4F8BF9", "#AB47BC",
                                                               "#26C6DA", "#EF5350"]),
                                        legend=alt.Legend(orient="bottom",
                                                          title=None)),
                        opacity=alt.condition(
                            alt.datum[COL_SPLIT] == "eval",
                            alt.value(0.65), alt.value(1.0),
                        ),
                        tooltip=tooltip,
                    )
                    .properties(height=alt.Step(28))
                )

            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown("**EER (%) — lower is better**")
                st.altair_chart(_hbar(COL_EER, "EER (%)"), width="stretch")
            with cc2:
                st.markdown("**minDCF — lower is better**")
                st.altair_chart(_hbar(COL_MIN_DCF, "minDCF"), width="stretch")

            st.markdown("**Accuracy — informative only**")
            st.altair_chart(_hbar(COL_ACCURACY, "Accuracy"), width="stretch")
            st.caption(
                "Bars are coloured by classifier; eval-split rows are drawn "
                "slightly translucent. Hover any bar for the full metric set."
            )

# ── Sidebar: live session summary (rendered last, when results exist) ───── #

with st.sidebar:
    # Single panel: the current configuration plus how many runs are collected.
    sidebar_panel("Experiment", [
        ("Corpus",   corpus),
        ("Scoring",  _split_txt),
        ("Sampling", _subset_txt),
        ("Seed",     str(int(seed))),
        ("Cache",    "on" if use_cache else "off"),
        ("Threads",  str(int(workers))),
        ("Runs",     str(len(rows))),
    ])
