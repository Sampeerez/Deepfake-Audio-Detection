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
from src.pipeline import MODEL_OPTIONS, extract_feature_matrix, run_classic_models
from src.reporting import (
    COL_ACCURACY, COL_EER, COL_FEATURES, COL_MIN_DCF, COL_MODEL,
    RESULT_COLUMNS,
)
from src.ui_helpers import (
    corpus_available, demo_corpus_notice, eval_corpora_for, eval_score_controls,
    get_extractor, get_samples, load_config, op_busy_notice, show_empty_state,
    sidebar_panel, test_audio_cta,
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

st.title("Run Experiment")
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
    "5": "Early fusion of every descriptor — RMS + MFCC + LFCC + DWT + CQCC — in one vector.",
    "6": "Constant-Q cepstrum — log-spaced bins, the strongest classic anti-spoofing front-end.",
}

# Display order in the dropdown: CQCC (6) is shown right before Fusion (5),
# since the fusion now includes it. Underlying option keys are unchanged.
FEATURE_ORDER = ["1", "2", "3", "4", "6", "5"]
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

# ── Configuration panel (main area) ──────────────────────────────────────── #
with st.container(border=True):
    st.markdown('<div class="section-label">Experiment configuration</div>',
                unsafe_allow_html=True)

    # Feature extractor, classifier and Advanced share one row; the less-used
    # numeric knobs (files/subset, seed) live inside Advanced to save space.
    c_feat, c_model, c_adv = st.columns([1.4, 1.4, 0.9])
    with c_feat:
        # nosearch_* container → CSS blocks type-to-filter (pure dropdown).
        with st.container(key="nosearch_feature"):
            feature_option = st.selectbox(
                "Feature extractor",
                FEATURE_ORDER,
                format_func=lambda k: FEATURE_LABELS[k],
                key="exp_feature",
            )
    with c_model:
        with st.container(key="nosearch_model"):
            model_option = st.selectbox(
                "Classifier",
                list(MODEL_LABELS.keys()),
                format_func=lambda k: MODEL_LABELS[k],
                key="exp_model",
            )
    with c_adv:
        st.markdown('<div style="height:1.75rem;"></div>', unsafe_allow_html=True)
        with st.popover("Advanced", icon=":material/tune:", width="stretch"):
            subset = st.number_input(
                "Files per subset", min_value=0, value=300, step=100,
                help="0 = full dataset. Stratified subsampling preserves the "
                     "bonafide/spoof ratio.",
            )
            seed = st.number_input("Seed", min_value=0, value=42, step=1)
            use_cache = st.checkbox(
                "Use DSP feature cache", value=True,
                help="Cache extracted vectors to disk, keyed by config hash.")
            workers = st.slider("DSP extraction threads", 1, 8, 4)

    corpus, score_split = eval_score_controls("exp")

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    _busy = op_busy_notice()
    c_run, c_clear = st.columns([3, 1])
    with c_run:
        run_btn = st.button("Run experiment", type="primary",
                            icon=":material/play_arrow:", width="stretch",
                            disabled=_busy)
    with c_clear:
        clear_btn = st.button("Clear results", icon=":material/delete:",
                              width="stretch", disabled=_busy)

if clear_btn:
    st.session_state["experiment_rows"] = []
    st.session_state.pop("run_signatures", None)
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
# A run is fully determined by (extractor, classifier, split, subset, seed) — so
# the same combination always yields the same result. Skip identical re-runs
# (only the training time would differ) instead of duplicating rows.
_run_sig = (feature_option, model_option, corpus, score_split, int(subset), int(seed))
if run_btn and _run_sig in st.session_state.get("run_signatures", set()):
    st.info(
        f"**{FEATURE_LABELS[feature_option]} × {MODEL_LABELS[model_option]}** on "
        f"the **{_split_txt}** (subset {int(subset) or 'full'}, seed {int(seed)}) "
        "has already been run — the result is unchanged, so nothing was "
        "re-trained. Change the configuration to add a new row.",
        icon=":material/info:",
    )
    run_btn = False

if run_btn:
    # Training + the dev row always use the 2019 LA train/dev splits; eval rows
    # come from the chosen corpus. "Eval"-only drops the dev row afterwards.
    train_samples   = get_samples("train")
    primary_samples, primary_name = get_samples("dev"), "dev"
    score_eval = score_split in (SPLIT_EVAL, SPLIT_BOTH)
    eval_corpora = eval_corpora_for(corpus) if score_eval else []

    if subset > 0:
        train_samples   = stratified_subsample(train_samples,   int(subset), int(seed))
        primary_samples = stratified_subsample(primary_samples, int(subset), int(seed) + 1)
        eval_corpora = [(lbl, stratified_subsample(s, int(subset), int(seed) + 2))
                        for lbl, s in eval_corpora]

    log      = io.StringIO()
    progress = st.progress(0.0, text="Extracting train features...")

    try:
        with contextlib.redirect_stdout(log):
            x_train, y_train, _ = extract_feature_matrix(
                train_samples, extractor, feature_option, "train",
                n_workers=int(workers), use_cache=use_cache,
            )
            progress.progress(0.4, text="Extracting dev features...")
            x_primary, y_primary, ms_dsp = extract_feature_matrix(
                primary_samples, extractor, feature_option, "dev",
                n_workers=int(workers), use_cache=use_cache,
            )
            eval_sets = []
            for lbl, s in eval_corpora:
                if s:
                    progress.progress(0.6, text=f"Extracting eval [{lbl}] features...")
                    x_ev, y_ev, _ = extract_feature_matrix(
                        s, extractor, feature_option, f"eval[{lbl}]",
                        n_workers=int(workers), use_cache=use_cache,
                    )
                    eval_sets.append((lbl, x_ev, y_ev))
            progress.progress(0.8, text="Training and evaluating...")
            results = run_classic_models(
                MODEL_OPTIONS[model_option],
                x_train, y_train, x_primary, y_primary,
                FEATURE_LABELS[feature_option], ms_dsp, int(seed),
                eval_sets=eval_sets,
            )
        progress.progress(1.0)
    except Exception as err:
        progress.empty()
        st.exception(err)
        st.stop()

    # No "Done." text or success banner — the new rows speak for themselves in
    # the results table below.
    progress.empty()
    # "Eval"-only: keep just the eval rows (drop the 2019 dev row).
    if score_split == SPLIT_EVAL:
        results = [r for r in results if "[EVAL]" in str(r.get(COL_MODEL, ""))]
    results = _tag_rows(results, primary_name)
    st.session_state.setdefault("experiment_rows", [])
    st.session_state["experiment_rows"].extend(results)
    st.session_state.setdefault("run_signatures", set()).add(_run_sig)

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
