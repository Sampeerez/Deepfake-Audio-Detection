# -*- coding: utf-8 -*-
"""
modes/_mode_full.py — "Full comparison" mode of the Benchmark page.

Locally (corpus present): one click sweeps every DSP×classifier and both CNNs in
the background, then ranks them by minDCF (EER tiebreaker) — the headline TFG
question, do the CNNs beat the classic front-ends?

On the corpus-less web demo: the same entry point becomes a MODEL HUB that
downloads all pretrained models at once and shows their precomputed results.

Dispatched from app_pages/2_Benchmark.py via runpy; set_page_config and PAGE_CSS
are applied in app.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.features import FeatureExtractor  # noqa: E402
from src.jobs import (  # noqa: E402
    cnn_epochs as job_cnn_epochs, progress as job_progress,
    request_cancel, submit_benchmark,
)
from src.reporting import (  # noqa: E402
    COL_ACCURACY, COL_EER, COL_FEATURES, COL_MIN_DCF, COL_MODEL,
)
from src.ui_helpers import (  # noqa: E402
    EVAL_CORPUS_CHOICES, available_pretrained_models, corpus_available,
    corpus_configured_2021_df, corpus_configured_2021_la, demo_corpus_notice,
    eval_corpora_for, get_extractor, get_samples,
    load_config, load_leaderboard_models, load_leaderboard_rows,
    load_pretrained_model, model_downloaded,
    op_in_progress, show_empty_state, sidebar_panel, test_audio_cta,
    themed,
)

COL_SPLIT = "Split"   # added to classic rows by Run Experiment
FEATURE_LABELS = FeatureExtractor.OPTION_NAMES
FEATURE_ORDER  = ["1", "2", "3", "4", "6"]

st.markdown(themed("""
<style>
@keyframes champGlow {
    0%, 100% { box-shadow: 0 0 0 1px rgba(255,193,7,0.18); }
    50%      { box-shadow: 0 0 20px rgba(255,193,7,0.32); }
}
.champ-banner {
    display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.5rem 1.2rem;
    background: linear-gradient(135deg, rgba(255,193,7,0.12) 0%, rgba(255,193,7,0.03) 100%);
    border: 1px solid rgba(255,193,7,0.32);
    border-left: 3px solid #FFC107;
    border-radius: 0.8rem;
    padding: 0.85rem 1.2rem; margin: 0.2rem 0 1rem;
    animation: champGlow 3.5s ease-in-out infinite;
}
.champ-banner .cb-tag {
    font-size: 0.64rem; font-weight: 800; text-transform: uppercase;
    letter-spacing: 0.12em; color: #FFD454;
}
.champ-banner .cb-combo { font-size: 1.05rem; font-weight: 750; color: #E8EDF8; }
.champ-banner .cb-metric { font-size: 0.82rem; color: #AFC3E8; }
.champ-banner .cb-metric b { color: #FFE08A; font-weight: 700; }

/* The Train/Evaluate/Score box is informational: every option is pre-selected
   and the whole box is LOCKED so the selection can't be changed (it just shows
   what "Train and evaluate all" will do). pointer-events:none keeps the selected
   segments vivid instead of greying them out the way `disabled` would. */
[class*="st-key-evalgrp_lb"] [data-baseweb="button-group"] {
    pointer-events: none !important;
}
[class*="st-key-evalgrp_lb"] [data-testid="stBaseButton-segmented_control"],
[class*="st-key-evalgrp_lb"] [data-testid="stBaseButton-segmented_controlActive"] {
    opacity: 1 !important;
}

/* Box + button on ONE row. The box keeps EXACTLY its normal look (same width/
   spacing/height as the other modes — no internal overrides); the button takes
   ALL the remaining width to the box's right and matches its height. */
[class*="st-key-lb_row"] {
    display: flex !important;
    flex-direction: row !important;
    align-items: stretch !important;   /* button matches the box's natural height */
    gap: 0.5rem !important;
    flex-wrap: nowrap !important;       /* keep box + button on ONE row */
}
/* The box is sized to its CONTENT (max-content, single line) so its flex item is
   not inflated by the wrap reserve — that hidden width was the gap that kept the
   button from reaching the box. The box drives the row height, so it is never
   stretched taller than its content → no empty strip below the chips. */
[class*="st-key-lb_row"] > [class*="st-key-evalgrp_lb"] {
    flex: 0 0 auto !important;
    width: max-content !important;
    flex-wrap: nowrap !important;
}
/* Target the row's two flex CELLS positionally (Streamlit may wrap each nested
   container in an element-block, so the cells aren't the keyed divs themselves):
   shrink the FIRST cell (box) to its content — closing the hidden gap — and let
   the SECOND cell (button) grow to fill everything to the right. */
[class*="st-key-lb_row"] > *:first-child { flex: 0 0 auto !important; width: max-content !important; }
[class*="st-key-lb_row"] > *:last-child  { flex: 1 1 0 !important; min-width: 0 !important; }
/* Make the button span the FULL box height. The percentage/flex chain kept leaving
   it at half height because some intermediate Streamlit wrapper collapsed to its
   content size. Instead, collapse EVERY wrapper between the row cell and the actual
   <button> with display:contents (they vanish from layout), so the <button> itself
   becomes a direct flex child of the row and stretches to the box height. */
[class*="st-key-lb_row"] > *:last-child,
[class*="st-key-lb_run_btn"],
[class*="st-key-lb_run_btn"] > div,
[class*="st-key-lb_run_btn"] [data-testid="stElementContainer"],
[class*="st-key-lb_run_btn"] [data-testid="stButton"] {
    display: contents !important;
}
[class*="st-key-lb_run_btn"] button {
    flex: 1 1 0 !important;            /* take all the width to the box's right */
    align-self: stretch !important;   /* match the box's height in the flex row */
    height: auto !important;
    min-height: 100% !important;
    width: 100% !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
}
</style>
"""), unsafe_allow_html=True)

st.title("Full Comparison")
st.caption(
    "Every run this session — classic DSP pipelines and trained CNNs — ranked by "
    "**minDCF** (primary), EER as tiebreaker. Classic (CPU) and CNN (GPU) run **in "
    "parallel in the background**, so you can browse other pages while it works. "
    "**Train and evaluate all** fits and saves all 17 trainable models (LR / SVM / "
    "XGBoost × 5 DSP front-ends + both CNNs) and additionally **evaluates** the "
    "pretrained wav2vec 2.0 SSL detector (inference-only), then scores every one on "
    "the 2019 LA dev split "
    "and every available eval corpus (2019 LA, 2021 LA, 2021 DF) — capped to a "
    "stratified 8,000 clips each — and writes models/*.pth, *.joblib and "
    "leaderboard.json."
)


def _prefetch_models(entries, bar):
    """Download every pretrained model AT ONCE (parallel threads) so the Hugging
    Face fetches overlap instead of running one by one. Each worker inherits the
    Streamlit script context; the progress bar is advanced from the main thread as
    each download finishes. Returns (loaded, failed)."""
    import concurrent.futures as cf
    import threading

    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

    ctx = get_script_run_ctx()

    def _init():
        add_script_run_ctx(threading.current_thread(), ctx)

    loaded, failed, done = {}, [], 0
    total = max(1, len(entries))
    with cf.ThreadPoolExecutor(max_workers=min(8, total), initializer=_init) as pool:
        futs = {pool.submit(load_pretrained_model, e): e for e in entries}
        for fut in cf.as_completed(futs):
            e = futs[fut]
            try:
                loaded[e["key"]] = fut.result()
            except Exception as exc:                 # noqa: BLE001 — report, continue
                failed.append((e["name"], str(exc)))
            done += 1
            bar.progress(done / total, f"{done}/{total} models ready")
    return loaded, failed


# Model-family palette — Classic (blue), CNN (purple), SSL/transformer (cyan).
_LB_TYPE_COLORS = {"Classic": "#4F8BF9", "CNN": "#AB47BC", "SSL": "#26C6DA"}


def _render_leaderboard(rows: list) -> None:
    """Render the leaderboard (champion banner, headline metrics, filters, ranking
    table + chart, sidebar) from a list of result rows. Shared by the LOCAL live
    view (rows from this session) and the WEB demo (rows from leaderboard.json), so
    both show the IDENTICAL filterable table. Assumes `rows` is non-empty."""
    def _clean(s, *markers):
        s = str(s)
        for m in markers:
            s = s.replace(m, "")
        return s.split("(")[0].strip()

    def _dataset_of(split) -> str:
        s = str(split).strip()
        if s in ("", "dev"):
            return "2019 LA"
        return s.split("·")[-1].strip() or "2019 LA"

    df = pd.DataFrame(rows)
    df[COL_EER]     = pd.to_numeric(df.get(COL_EER), errors="coerce")
    df[COL_MIN_DCF] = pd.to_numeric(df.get(COL_MIN_DCF), errors="coerce")
    df["Type"]      = df.get("Type", "Classic")
    df["Features"]  = df.get(COL_FEATURES, "").map(lambda s: _clean(s, "[EVAL]"))
    df["Model"]     = df.get(COL_MODEL, "").map(lambda s: _clean(s, "[CPU]", "[CUDA]", "[EVAL]"))
    df["Config"]    = df["Features"] + " · " + df["Model"]
    df["Dataset"]   = df[COL_SPLIT].map(_dataset_of)
    df["SplitKind"] = df[COL_SPLIT].map(lambda s: "Dev" if str(s).strip() == "dev" else "Eval")

    # One row per (type, features, model, split): best by minDCF, EER as tiebreaker.
    _SORT = [COL_MIN_DCF, COL_EER]
    df = (df.sort_values(_SORT, na_position="last")
            .drop_duplicates(subset=["Type", "Features", "Model", COL_SPLIT], keep="first")
            .reset_index(drop=True))
    ranked = df.dropna(subset=[COL_EER]).sort_values(_SORT, na_position="last").reset_index(drop=True)

    # Model families actually present, in a stable display order.
    _fam_order = ["Classic", "CNN", "SSL"]
    _fams = [t for t in _fam_order if (df["Type"] == t).any()] + \
            [t for t in df["Type"].unique() if t not in _fam_order]

    if not ranked.empty:
        best = ranked.iloc[0]
        st.markdown(
            '<div class="champ-banner">'
            '<span class="cb-tag">★ Best run</span>'
            f'<span class="cb-combo">{best["Config"]}</span>'
            f'<span class="cb-metric">type <b>{best["Type"]}</b></span>'
            f'<span class="cb-metric">minDCF <b>{best[COL_MIN_DCF]:.3f}</b></span>'
            f'<span class="cb-metric">EER <b>{best[COL_EER]:.2f} %</b></span>'
            f'<span class="cb-metric">split <b>{best[COL_SPLIT]}</b></span>'
            '</div>',
            unsafe_allow_html=True,
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total runs", len(df))
    c2.metric("By family",
              " / ".join(str(int((df["Type"] == t).sum())) for t in _fams),
              help=" / ".join(_fams))
    c3.metric("Best minDCF",
              f'{ranked[COL_MIN_DCF].min():.3f}' if ranked[COL_MIN_DCF].notna().any() else "—")
    c4.metric("Best EER", f'{ranked[COL_EER].min():.2f} %' if not ranked.empty else "—")

    st.divider()
    st.markdown('<div class="section-label">Filter the leaderboard</div>',
                unsafe_allow_html=True)
    _ds_present = [c for c in EVAL_CORPUS_CHOICES if c in set(ranked["Dataset"])]
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        f_ds = st.segmented_control("Dataset", ["All"] + _ds_present,
                                    default="All", key="lb_f_ds")
    with fc2:
        f_sp = st.segmented_control("Split", ["All", "Dev", "Eval"],
                                    default="All", key="lb_f_sp")
    with fc3:
        f_ty = st.segmented_control("Model type", ["All"] + _fams,
                                    default="All", key="lb_f_ty")
    f_ds, f_sp, f_ty = (f_ds or "All"), (f_sp or "All"), (f_ty or "All")

    view = ranked
    if f_ds != "All":
        view = view[view["Dataset"] == f_ds]
    if f_sp != "All":
        view = view[view["SplitKind"] == f_sp]
    if f_ty != "All":
        view = view[view["Type"] == f_ty]
    view = view.sort_values(_SORT, na_position="last").reset_index(drop=True)

    _active = [f for f in (f_ds, f_sp, f_ty) if f != "All"]
    st.caption(("Showing **all** results — pick a Dataset, Split or Model type to "
                "narrow them." if not _active
                else f"Filtered to **{len(view)}** of {len(ranked)} results "
                     f"({' · '.join(_active)})."))

    tab_table, tab_chart = st.tabs(["Ranking", "Chart"])
    with tab_table:
        if view.empty:
            st.info("No results match the current filters.")
        else:
            show = view.copy()
            show.insert(0, "Rank", range(1, len(show) + 1))
            show = show[["Rank", "Type", "Features", "Model", COL_SPLIT,
                         COL_MIN_DCF, COL_EER, COL_ACCURACY]]
            show = show.rename(columns={COL_EER: "EER (%)", COL_MIN_DCF: "minDCF",
                                        COL_ACCURACY: "Accuracy", COL_SPLIT: "Split"})

            def _hl(row):
                base = ("background-color: rgba(255,193,7,0.14); font-weight:600;"
                        if row["Rank"] == 1 else "")
                return [base] * len(row)

            _rh = 36
            st.dataframe(
                show.style.apply(_hl, axis=1).format({"EER (%)": "{:.2f}", "minDCF": "{:.3f}"}),
                width="stretch", hide_index=True,
                row_height=_rh, height=int(_rh * (len(show) + 1) + 4),
            )
            st.caption("Ranked by **minDCF** (the primary ASVspoof cost — lower is "
                       "better), EER as tiebreaker. The leader is highlighted.")
            st.download_button(
                "Download leaderboard as CSV",
                data=show.to_csv(index=False).encode("utf-8"),
                file_name="leaderboard.csv", mime="text/csv",
                icon=":material/download:",
            )

    with tab_chart:
        if view.empty:
            st.info("No results match the current filters.")
        else:
            chart_df = view.copy()
            chart_df["Entry"] = chart_df["Config"] + "  [" + chart_df[COL_SPLIT] + "]"
            _dom = [t for t in _fams if (chart_df["Type"] == t).any()]
            chart = (
                alt.Chart(chart_df)
                .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
                .encode(
                    y=alt.Y("Entry:N", title=None,
                            sort=alt.EncodingSortField(field=COL_MIN_DCF, order="ascending"),
                            axis=alt.Axis(labelLimit=340)),
                    x=alt.X(f"{COL_MIN_DCF}:Q", title="minDCF — lower is better"),
                    color=alt.Color("Type:N",
                                    scale=alt.Scale(domain=_dom,
                                                    range=[_LB_TYPE_COLORS.get(t, "#90A4AE")
                                                           for t in _dom]),
                                    legend=alt.Legend(orient="bottom", title=None)),
                    opacity=alt.condition(alt.datum["SplitKind"] == "Eval",
                                          alt.value(0.6), alt.value(1.0)),
                    tooltip=["Type", "Features", "Model",
                             alt.Tooltip(COL_SPLIT, title="Split"),
                             alt.Tooltip(COL_MIN_DCF, title="minDCF", format=".3f"),
                             alt.Tooltip(COL_EER, title="EER (%)", format=".2f")],
                )
                .properties(height=alt.Step(28))
            )
            st.altair_chart(chart, width="stretch")
            st.caption("Coloured by model family; eval-split rows are translucent.")

    with st.sidebar:
        sb_rows = [("Total runs", str(len(df)))]
        for t in _fams:
            sb_rows.append((t, str(int((df["Type"] == t).sum()))))
        if not ranked.empty:
            sb_rows.append(("Best minDCF", f"{ranked[COL_MIN_DCF].min():.3f}"))
            sb_rows.append(("Best EER", f"{ranked[COL_EER].min():.2f} %"))
        sidebar_panel("Leaderboard", sb_rows)


# ── One-click full benchmark (runs in the BACKGROUND) ────────────────────── #
_running = op_in_progress()

# On the corpus-less web demo the live sweep is impossible (no dataset, no GPU),
# so the Full comparison becomes a MODEL HUB: one click fetches every pretrained
# model from Hugging Face and shows their head-to-head results, ready to test.
if not corpus_available():
    entries = available_pretrained_models()
    st.subheader("Model hub")
    st.caption(
        "Training and live scoring need the multi-GB corpus and a GPU, so on the "
        "web demo the comparison is precomputed. Download every pretrained model "
        "(trained locally with its best default configuration) in one click, then "
        "try them on your own audio in Detection Analysis."
    )

    if not entries:
        demo_corpus_notice(
            "No pretrained models configured",
            "Run <b>Full comparison</b> once on a machine with the corpus to train "
            "and export all 20 models, then set <b>HF_BASE_URL</b> in "
            "src/ui_helpers.py to your Hugging Face folder to populate the hub.",
        )
        st.stop()

    # Auto-download every pretrained model as soon as the page opens (once per
    # session), in parallel so the Hugging Face fetches overlap. After this the
    # whole zoo is warm and Detection Analysis runs instantly.
    if not st.session_state.get("hf_models_prefetched"):
        bar = st.progress(0.0, "Downloading all pretrained models…")
        _loaded, _failed = _prefetch_models(entries, bar)
        bar.empty()
        st.session_state["hf_models_prefetched"] = True
        for _nm, _err in _failed:
            st.warning(f"{_nm}: {_err}")

    n_ready = sum(model_downloaded(e) for e in entries)
    if n_ready == len(entries):
        st.success(f"All {n_ready} pretrained models downloaded and ready — open "
                   "Detection Analysis to test them on your own clip.",
                   icon=":material/cloud_done:")
    else:
        cdl, cinfo = st.columns([1, 2], vertical_alignment="center")
        with cdl:
            if st.button("Retry downloads", type="primary",
                         icon=":material/refresh:", width="stretch"):
                st.session_state.pop("hf_models_prefetched", None)
                st.rerun()
        with cinfo:
            st.markdown(f"**{n_ready} / {len(entries)}** models cached on this "
                        "server. Models download once and stay warm for the session.")

    # The web demo shows the SAME full leaderboard as local — every model on every
    # split/corpus, filterable — read from the committed leaderboard.json rows.
    st.divider()
    _lb_rows = load_leaderboard_rows()
    if _lb_rows:
        _render_leaderboard(_lb_rows)
    else:
        show_empty_state(
            "Leaderboard not generated yet",
            "Run <b>Full comparison</b> once on a machine with the corpus to train, "
            "evaluate and write <code>leaderboard.json</code>, then commit it — the "
            "full ranking (classic, CNN and the wav2vec 2.0 SSL detector across every "
            "corpus) appears here.",
        )
    test_audio_cta("Pick any of these models and hear them judge your own clip:")
    st.stop()

# Fixed, best-effort training configuration — the Full comparison trains the
# WHOLE zoo with strong defaults (no per-run knobs here; tune individual models
# in the Classic / CNN modes instead). The CNN trains on the full train set.
_CLASSIC_SUBSET = 6000          # per classic model; RBF-SVM scales poorly beyond
_CNN_SUBSET     = 0             # 0 = full 2019 LA train set
_EVAL_SUBSET    = 8000          # stratified cap per eval corpus (2021 DF ≈ 600k)

# Which eval corpora are actually present on this machine. 2019 LA is implied by
# corpus_available() (we only reach here with the corpus on disk); the 2021
# tracks are optional downloads, checked with a single lightweight stat call.
_corpus_present = {
    "2019 LA": True,
    "2021 LA": corpus_configured_2021_la(),
    "2021 DF": corpus_configured_2021_df(),
}


with st.container(border=True):
    st.markdown('<div class="section-label">Generate the full leaderboard</div>',
                unsafe_allow_html=True)

    # Box + button live on ONE flex row (CSS) so the button hugs the right edge of
    # the "Score on" box and stretches to fill the rest of the width.
    with st.container(key="lb_row"):
        # LEFT — the familiar Train/Evaluate/Score selection box, but every option
        # is pre-selected and LOCKED (CSS pointer-events:none), so it reads as "all
        # of this will run" without the user having to pick anything.
        with st.container(key="evalgrp_lb"):
            st.segmented_control(
                "Train on", ["2019 LA"], default=["2019 LA"],
                selection_mode="multi", key="lb_train_on")
            _ev_default = [c for c in EVAL_CORPUS_CHOICES if _corpus_present[c]]
            st.segmented_control(
                "Evaluate on", EVAL_CORPUS_CHOICES, default=_ev_default,
                selection_mode="multi", key="lb_eval_on")
            st.segmented_control(
                "Score on", ["Dev", "Eval"], default=["Dev", "Eval"],
                selection_mode="multi", key="lb_score_on")

        # RIGHT — the action, deliberately large, filling the rest of the row.
        with st.container(key="lb_run_btn"):
            g_train = st.button("Train and evaluate all", type="primary",
                                icon=":material/playlist_play:", width="stretch",
                                disabled=_running)

if st.session_state.get("bench_error"):
    st.error(f"Full comparison failed: {st.session_state.pop('bench_error')}")
if st.session_state.pop("bench_cancelled", False):
    st.info("Full comparison cancelled.", icon=":material/cancel:")


@st.fragment(run_every=2.0)
def _live_progress():
    """Split view (classic ∥ CNN) of the running comparison, refreshing itself."""
    if not op_in_progress():
        return
    pr = job_progress()
    _epochs = job_cnn_epochs()
    st.markdown('<div class="section-label">Running — classic and CNN in '
                'parallel</div>', unsafe_allow_html=True)

    lc, rc = st.columns(2, gap="large")

    # ── Classic panel ─────────────────────────────────────────────────────── #
    s_cl = pr["streams"]["classic"]
    with lc:
        with st.container(border=True):
            st.markdown("**Classic models · CPU**")
            tot = max(int(s_cl["total"]), 1)
            st.progress(min(1.0, s_cl["done"] / tot),
                        text=f"{s_cl['done']}/{s_cl['total']} · {s_cl['label']}")
            for it in s_cl["items"][-9:]:
                st.markdown(f"<div style='font-size:0.8rem;opacity:0.82;"
                            f"color:#82B1FF'>✓ {it}</div>", unsafe_allow_html=True)
            if not s_cl["items"]:
                st.caption("…")
            st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)

    # ── CNN panel — shows a live loss curve when epoch data is available ───── #
    s_cn = pr["streams"]["cnn"]
    with rc:
        with st.container(border=True):
            st.markdown("**CNN · GPU**")
            tot = max(int(s_cn["total"]), 1)
            st.progress(min(1.0, s_cn["done"] / tot),
                        text=f"{s_cn['done']}/{s_cn['total']} · {s_cn['label']}")
            if _epochs:
                _ldf = (pd.DataFrame(_epochs)
                        .melt(id_vars="epoch",
                              value_vars=["train_loss", "val_loss"],
                              var_name="curve", value_name="loss"))
                _chart = (
                    alt.Chart(_ldf).mark_line(point=True).encode(
                        x=alt.X("epoch:Q", title="epoch",
                                axis=alt.Axis(tickMinStep=1)),
                        y=alt.Y("loss:Q", title="loss"),
                        color=alt.Color(
                            "curve:N", title=None,
                            scale=alt.Scale(domain=["train_loss", "val_loss"],
                                            range=["#4F8BF9", "#EF5350"])),
                    ).properties(height=180)
                )
                st.altair_chart(_chart, width="stretch")
                _last = _epochs[-1]
                st.caption(
                    f"Epoch {_last['epoch']} · "
                    f"train {_last['train_loss']:.4f} · "
                    f"val {_last['val_loss']:.4f}"
                )
            else:
                for it in s_cn["items"][-9:]:
                    st.markdown(f"<div style='font-size:0.8rem;opacity:0.82;"
                                f"color:#C9A6FF'>✓ {it}</div>", unsafe_allow_html=True)
                if not s_cn["items"]:
                    st.caption("…")
            st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)

    if st.button("Cancel run", icon=":material/cancel:", key="full_cancel"):
        request_cancel()
        st.toast("Cancelling… stops at the next checkpoint.")
        st.rerun(scope="app")


if _running:
    _live_progress()

# Launch the background sweep (the running indicator also lives in the sidebar
# banner). Reaching here implies the corpus is present — the web demo returns
# above via the model hub.
if g_train and not _running:
    ext = get_extractor()
    train = get_samples("train")
    primary, pname = get_samples("dev"), "dev"
    # Train AND evaluate all: train on 2019 LA, score the dev split, and evaluate
    # every model on every available eval corpus (2019 LA + 2021 LA + 2021 DF eval).
    eval_corpora = []
    for _c in EVAL_CORPUS_CHOICES:
        eval_corpora.extend(eval_corpora_for(_c))
    st.session_state["bench_future"] = submit_benchmark(
        ext=ext, feat_labels=FEATURE_LABELS,
        base_params=dict(load_config()["train_params"]),
        train=train, primary=primary, eval_corpora=eval_corpora, pname=pname,
        classic_subset=_CLASSIC_SUBSET, cnn_subset=_CNN_SUBSET,
        eval_subset=_EVAL_SUBSET, include_cnn=True, seed=42,
    )
    st.session_state["bench_score"] = "Dev + Eval"
    st.session_state["op_running"] = True
    st.rerun()


def _collect():
    rows = []
    for r in st.session_state.get("experiment_rows", []):
        d = dict(r)
        d["Type"] = "Classic"
        d.setdefault(COL_SPLIT, "dev")
        rows.append(d)
    for r in st.session_state.get("cnn_runs", []):
        d = dict(r)
        # Respect a family tag already set on the row (e.g. "SSL" for wav2vec2);
        # default the rest of the cnn_runs stream to "CNN".
        d["Type"] = d.get("Type") or "CNN"
        # Full-comparison rows already carry a resolved "Split" (the job strips
        # the [EVAL] marker when tagging). Manual CNN Learning rows don't, so
        # derive it from the marker. RESPECT an existing split — re-deriving it
        # here would mislabel eval rows as dev and the dedup would drop them.
        if not d.get(COL_SPLIT):
            _corpus = str(d.get("Corpus", "")).strip()
            d[COL_SPLIT] = ((f"eval · {_corpus}" if _corpus else "eval")
                            if "[EVAL]" in str(d.get(COL_MODEL, "")) else "dev")
        rows.append(d)
    return rows


rows = _collect()

# After a full comparison, confirm the deployment assets were written from here.
if st.session_state.get("bench_done"):
    from src.ui_helpers import LEADERBOARD_PATH  # noqa: E402
    _models = available_pretrained_models()
    _ready  = sum(model_downloaded(e) for e in _models)
    _json   = "written" if os.path.isfile(LEADERBOARD_PATH) else "pending"
    st.success(f"Deployment assets exported: {_ready}/{len(_models)} model files "
               f"in models/ · leaderboard.json {_json}.",
               icon=":material/cloud_done:")

if not rows:
    if not _running:
        # No session rows yet — show the committed leaderboard.json if available.
        _lb_rows = load_leaderboard_rows()
        if _lb_rows:
            st.info(
                "Showing results from the last saved run (`leaderboard.json`). "
                "Click **Train and evaluate all** to refresh with this session's results.",
                icon=":material/info:",
            )
            _render_leaderboard(_lb_rows)
        else:
            show_empty_state(
                "No runs yet",
                "Run a few configurations in the Classic models and CNN modes — every "
                "result is collected here and ranked by minDCF so you can compare the "
                "classic front-ends, the CNNs and the wav2vec 2.0 SSL detector.",
            )
    st.stop()

# Render the leaderboard (the SAME view the web demo shows, from session rows).
_render_leaderboard(rows)
