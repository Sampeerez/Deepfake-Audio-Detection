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
    COL_ACCURACY, COL_EER, COL_FEATURES, COL_FEAT_TIME, COL_INFER_TIME,
    COL_MIN_DCF, COL_MODEL, COL_TRAIN_TIME,
)
from src.ui_helpers import (  # noqa: E402
    EVAL_CORPUS_CHOICES, available_pretrained_models, corpus_available,
    corpus_configured_2021_df, corpus_configured_2021_la, demo_corpus_notice,
    eval_corpora_for, get_extractor, get_samples,
    load_config, load_leaderboard_models, load_leaderboard_rows,
    load_pretrained_model, model_downloaded,
    op_in_progress, show_empty_state, sidebar_panel,
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
    "**Train and evaluate all** fits and saves all 20 trainable models (LR / SVM / "
    "XGBoost × 5 DSP front-ends + 5 CNN architectures) and additionally **evaluates** the "
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


def _aggregate_by_model(frame: "pd.DataFrame") -> "pd.DataFrame":
    """Collapse a leaderboard frame to ONE row per model, averaging its metrics
    and timings across EVERY split/corpus present.

    The headline picks ("best model", "most efficient") and the Efficiency tab
    use this so they reflect overall behaviour across all corpora — not just the
    easy 2019 dev split, which flatters the deep nets. Training time only exists
    on the dev row, so its mean is simply that single value."""
    g = frame.copy()
    for c in (COL_TRAIN_TIME, COL_FEAT_TIME, COL_INFER_TIME,
              COL_EER, COL_MIN_DCF, COL_ACCURACY):
        g[c] = pd.to_numeric(g[c], errors="coerce") if c in g.columns else float("nan")
    agg = g.groupby(["Type", "Features", "Model", "Config"], as_index=False).agg(
        Train_s=(COL_TRAIN_TIME, "mean"),
        Extract_ms=(COL_FEAT_TIME, "mean"),
        Infer_ms=(COL_INFER_TIME, "mean"),
        EER=(COL_EER, "mean"),
        minDCF=(COL_MIN_DCF, "mean"),
        Accuracy=(COL_ACCURACY, "mean"),
        Corpora=(COL_EER, "count"),
    )
    # End-to-end per-clip latency = feature extraction + model forward.
    agg["Latency_ms"] = agg["Extract_ms"].fillna(0) + agg["Infer_ms"].fillna(0)
    agg.loc[agg["Extract_ms"].isna() & agg["Infer_ms"].isna(),
            "Latency_ms"] = float("nan")
    return agg


def _render_leaderboard(rows: list) -> None:
    """Render the leaderboard (champion banner, headline metrics, filters, ranking
    table + chart, sidebar) from a list of result rows. Shared by the LOCAL live
    view (rows from this session) and the WEB demo (rows from leaderboard.json), so
    both show the IDENTICAL filterable table. Assumes `rows` is non-empty."""
    def _clean(s, *markers):
        s = str(s)
        for m in markers:
            s = s.replace(m, "")
        # Collapse any whitespace left by removing mid-string markers (e.g. "PyTorch").
        return " ".join(s.split("(")[0].split())

    def _dataset_of(split) -> str:
        s = str(split).strip()
        if s in ("", "dev"):
            return "2019 LA"
        return s.split("·")[-1].strip() or "2019 LA"

    df = pd.DataFrame(rows)
    df[COL_EER]     = pd.to_numeric(df.get(COL_EER), errors="coerce")
    df[COL_MIN_DCF] = pd.to_numeric(df.get(COL_MIN_DCF), errors="coerce")
    # Cross-seed variance (only present on multi-seed CNN rows; "—"/absent else).
    # Guard column presence: older leaderboards / single-run rows lack these keys.
    df["EER std"]    = (pd.to_numeric(df["EER std"], errors="coerce")
                        if "EER std" in df.columns else float("nan"))
    df["minDCF std"] = (pd.to_numeric(df["minDCF std"], errors="coerce")
                        if "minDCF std" in df.columns else float("nan"))
    df["Seeds"]      = (pd.to_numeric(df["Seeds"], errors="coerce").fillna(1).astype(int)
                        if "Seeds" in df.columns else 1)
    df["Type"]      = df.get("Type", "Classic")
    df["Features"]  = df.get(COL_FEATURES, "").map(lambda s: _clean(s, "[EVAL]"))
    df["Model"]     = df.get(COL_MODEL, "").map(
        lambda s: _clean(s, "[CPU]", "[CUDA]", "[EVAL]", "PyTorch"))
    df["Config"]    = df["Features"] + " · " + df["Model"]
    df["Dataset"]   = df[COL_SPLIT].map(_dataset_of)
    df["SplitKind"] = df[COL_SPLIT].map(lambda s: "Dev" if str(s).strip() == "dev" else "Eval")
    # User-facing split label: tag the bare "dev" with its corpus (always 2019 LA)
    # so it reads like the eval rows ("dev · 2019 LA"). The raw COL_SPLIT value is
    # kept untouched — SplitKind and the dedup key still rely on it.
    df["SplitDisplay"] = df.apply(
        lambda r: (f"dev · {r['Dataset']}" if str(r[COL_SPLIT]).strip() == "dev"
                   else str(r[COL_SPLIT])), axis=1)

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

    # Two headline picks, both judged ACROSS ALL CORPORA (not the easy dev split):
    #   • Best model      — lowest mean minDCF (the model that detects best overall).
    #   • Most efficient  — best accuracy×speed trade-off (mean EER × per-clip
    #     latency, lower is better), so a fast detector with decent accuracy can win.
    _by_all = _aggregate_by_model(ranked)
    _det = _by_all.dropna(subset=["minDCF"]).sort_values(["minDCF", "EER"])
    if not _det.empty:
        b = _det.iloc[0]
        st.markdown(
            '<div class="champ-banner">'
            '<span class="cb-tag">★ Best model</span>'
            f'<span class="cb-combo">{b["Config"]}</span>'
            f'<span class="cb-metric">type <b>{b["Type"]}</b></span>'
            f'<span class="cb-metric">minDCF <b>{b["minDCF"]:.3f}</b></span>'
            f'<span class="cb-metric">EER <b>{b["EER"]:.2f} %</b></span>'
            f'<span class="cb-metric">avg over <b>{int(b["Corpora"])} corpora</b></span>'
            '</div>',
            unsafe_allow_html=True,
        )

    _effp = _by_all.dropna(subset=["Latency_ms", "EER"]).copy()
    if not _effp.empty:
        # "Efficient" must reward QUALITY, not just raw speed — a fast but
        # inaccurate classic isn't efficient. Min-max normalise EER and latency
        # across the field and score 0.65·EER + 0.35·latency (lower = better), so
        # the winner is the model with near-best accuracy at low cost (a light deep
        # net), not the cheapest mediocre one nor the most accurate but slowest.
        def _norm(s):
            lo, hi = float(s.min()), float(s.max())
            return (s - lo) / (hi - lo) if hi > lo else s * 0.0
        _effp["score"] = 0.65 * _norm(_effp["EER"]) + 0.35 * _norm(_effp["Latency_ms"])
        e = _effp.sort_values("score").iloc[0]
        _train = f'{e["Train_s"]:.0f}s' if pd.notna(e["Train_s"]) else "—"
        st.markdown(
            '<div class="champ-banner" style="background:linear-gradient(135deg,'
            'rgba(38,198,218,0.12) 0%,rgba(38,198,218,0.03) 100%);'
            'border:1px solid rgba(38,198,218,0.32);border-left:3px solid #26C6DA;'
            'animation:none;">'
            '<span class="cb-tag" style="color:#7FE0EC">&#8623; Most efficient</span>'
            f'<span class="cb-combo">{e["Config"]}</span>'
            f'<span class="cb-metric">type <b>{e["Type"]}</b></span>'
            f'<span class="cb-metric">latency <b>{e["Latency_ms"]:.2f} ms/clip</b></span>'
            f'<span class="cb-metric">EER <b>{e["EER"]:.2f} %</b></span>'
            f'<span class="cb-metric">train <b>{_train}</b></span>'
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
    # Hide the filters when the Efficiency (last) tab is active — it always averages
    # every corpus, so Dataset/Split/Model-type don't apply there; they stay on the
    # Ranking and Chart tabs. (:has fires when the last tab button is the selected one.)
    st.markdown(
        "<style>.st-key-lb_filterwrap:has(div[data-baseweb='tab-list'] "
        "button[data-baseweb='tab']:last-of-type[aria-selected='true']) "
        ".st-key-lb_filters{display:none}</style>",
        unsafe_allow_html=True,
    )

    # Filters + tabs share the `lb_filterwrap` container so the :has rule above can
    # reach the tab state. st.tabs() is CALLED here (nesting the tab DOM inside the
    # wrapper); the `with tab_*` blocks below stay at the outer indent — content is
    # routed to each panel regardless of where the block lives.
    with st.container(key="lb_filterwrap"):
        with st.container(key="lb_filters"):
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
            st.caption(("Showing **all** results — pick a Dataset, Split or Model type "
                        "to narrow them." if not _active
                        else f"Filtered to **{len(view)}** of {len(ranked)} results "
                             f"({' · '.join(_active)})."))

        tab_table, tab_chart, tab_eff = st.tabs(["Ranking", "Chart", "Efficiency"])

    with tab_table:
        if view.empty:
            st.info("No results match the current filters.")
        else:
            show = view.copy()
            show.insert(0, "Rank", range(1, len(show) + 1))
            # Compact ranking: just the (seed-averaged) metrics, one row per model
            # × split. The per-seed spread stays in leaderboard.json for anyone who
            # wants it; the table keeps the headline numbers readable.
            show = show[["Rank", "Type", "Features", "Model", "SplitDisplay",
                         COL_MIN_DCF, COL_EER, COL_ACCURACY]]
            show = show.rename(columns={COL_EER: "EER (%)", COL_MIN_DCF: "minDCF",
                                        COL_ACCURACY: "Accuracy", "SplitDisplay": "Split"})

            def _hl(row):
                base = ("background-color: rgba(255,193,7,0.14); font-weight:600;"
                        if row["Rank"] == 1 else "")
                return [base] * len(row)

            _rh = 36
            st.dataframe(
                show.style.apply(_hl, axis=1).format({"EER (%)": "{:.2f}",
                                                      "minDCF": "{:.3f}"}),
                width="stretch", hide_index=True,
                row_height=_rh, height=int(_rh * (len(show) + 1) + 4),
            )
            _seeds = int(df["Seeds"].max()) if "Seeds" in df.columns else 1
            _seed_note = (f" CNN metrics are averaged over {_seeds} seeds."
                          if _seeds > 1 else "")
            st.caption("Ranked by **minDCF** (the primary ASVspoof cost — lower is "
                       "better), EER as tiebreaker. The leader is highlighted."
                       + _seed_note + " Hover the table and use its toolbar to "
                       "download or copy it.")

    with tab_eff:
        # Cost-vs-benefit view AVERAGED ACROSS ALL CORPORA: training cost, the
        # per-clip inference latency (feature extraction + forward) and the mean
        # accuracy a model achieves over every split/corpus — so the trade-off
        # reflects overall behaviour, not just the easy 2019 dev split. It ALWAYS
        # aggregates the full set (`ranked`, not the filtered `view`): this tab is
        # the overall-efficiency picture, so the Dataset/Split filters don't apply.
        eff = _aggregate_by_model(ranked)

        if eff[["Train_s", "Latency_ms"]].isna().all().all():
            st.info("No timing data yet — run **Train and evaluate all** to record "
                    "training time and inference latency for every model. (Older "
                    "saved leaderboards predate these metrics.)")
        else:
            # Keep the Features front-end so a classic classifier on five DSP
            # front-ends (each with a different extraction cost) stays distinct.
            etab = (eff[["Type", "Features", "Model", "Train_s", "Extract_ms",
                         "Infer_ms", "Latency_ms", "minDCF", "EER"]]
                    .sort_values("Latency_ms", na_position="last")
                    .rename(columns={"Train_s": "Train (s)",
                                     "Extract_ms": "Extract (ms/clip)",
                                     "Infer_ms": "Infer (ms/clip)",
                                     "Latency_ms": "Latency (ms)",
                                     "EER": "EER (%)"}))
            _erh = 36   # fixed row height → size the frame to fit every row (no scroll)
            st.dataframe(
                etab.style.format({"Train (s)": "{:.1f}", "Extract (ms/clip)": "{:.2f}",
                                   "Infer (ms/clip)": "{:.3f}", "Latency (ms)": "{:.2f}",
                                   "minDCF": "{:.3f}", "EER (%)": "{:.2f}"}, na_rep="—"),
                width="stretch", hide_index=True,
                row_height=_erh, height=int(_erh * (len(etab) + 1) + 4),
            )
            st.caption("Per-clip **Latency = Extract + Infer** (the cost a fresh "
                       "audio pays end to end). minDCF / EER are the **mean across "
                       "all four corpora** (2019 dev + 2019/2021 LA + 2021 DF). "
                       "wav2vec 2.0 reads the raw waveform, so it has no separate "
                       "extraction stage.")
            st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)

            # Cost-vs-benefit scatter: latency (x) against overall EER (y). The
            # sweet spot is the lower-left — fast AND accurate across all corpora.
            sc = eff.dropna(subset=["Latency_ms", "EER"]).copy()
            if not sc.empty:
                _dom = [t for t in _fams if (sc["Type"] == t).any()]
                # Pre-format EVERY tooltip field as a plain string in a cleanly named
                # column. This sidesteps the two things that silently kill Vega-Lite
                # hover here: numeric `format` on NaN cells (wav2vec 2.0 has no train
                # time) serialising as invalid JSON, and field names Vega parses as
                # shorthand. Strings → the tooltip always resolves.
                sc["tt_train"]   = sc["Train_s"].map(
                    lambda v: "—" if pd.isna(v) else f"{v:.0f} s")
                sc["tt_latency"] = sc["Latency_ms"].map(lambda v: f"{v:.2f} ms/clip")
                sc["tt_eer"]     = sc["EER"].map(lambda v: f"{v:.2f} %")
                sc["tt_dcf"]     = sc["minDCF"].map(
                    lambda v: "—" if pd.isna(v) else f"{v:.3f}")
                _sel = alt.selection_point(name="effsel", fields=["Model"],
                                           on="click", clear="dblclick", empty=False)
                scatter = (
                    alt.Chart(sc).mark_circle(size=170).encode(
                        x=alt.X("Latency_ms:Q",
                                title="Latency per clip (ms) — lower is faster",
                                scale=alt.Scale(zero=False)),
                        y=alt.Y("EER:Q", title="EER (%), avg over corpora — lower is better",
                                scale=alt.Scale(zero=False)),
                        color=alt.Color("Type:N",
                                        scale=alt.Scale(domain=_dom,
                                                        range=[_LB_TYPE_COLORS.get(t, "#90A4AE")
                                                               for t in _dom]),
                                        legend=alt.Legend(orient="bottom", title=None)),
                        opacity=alt.condition(_sel, alt.value(1.0), alt.value(0.85)),
                        tooltip=[alt.Tooltip("Model", title="Model"),
                                 alt.Tooltip("Type", title="Family"),
                                 alt.Tooltip("Features", title="Front-end"),
                                 alt.Tooltip("tt_train", title="Train"),
                                 alt.Tooltip("tt_latency", title="Latency"),
                                 alt.Tooltip("tt_eer", title="EER"),
                                 alt.Tooltip("tt_dcf", title="minDCF")],
                    ).add_params(_sel).properties(height=340)
                )
                _ev = st.altair_chart(scatter, width="stretch", key="eff_scatter",
                                      on_select="rerun")
                # Hover tooltips can stop firing after a Streamlit rerun (a known Vega
                # quirk), so ALSO support click-to-inspect: clicking a dot pins its
                # full numbers below, which never depends on the hover handler.
                _picked = None
                try:
                    _state = (_ev.selection if hasattr(_ev, "selection")
                              else _ev.get("selection", {}))
                    _hits = _state.get("effsel", []) if hasattr(_state, "get") else []
                    if _hits:
                        _picked = _hits[0].get("Model")
                except Exception:
                    _picked = None
                if _picked is not None and (sc["Model"] == _picked).any():
                    _pr = sc[sc["Model"] == _picked].iloc[0]
                    st.caption(
                        f"**{_picked}** ({_pr['Type']}) — latency "
                        f"**{_pr['Latency_ms']:.2f} ms/clip** · EER "
                        f"**{_pr['EER']:.2f} %** · minDCF **{_pr['minDCF']:.3f}** · "
                        f"train {_pr['tt_train']}  ·  double-click to clear.")
                else:
                    st.caption("Each dot is a model, averaged over all corpora. "
                               "**Lower-left = the best trade-off** (fast *and* "
                               "accurate). **Hover** a point, or **click** it to pin "
                               "its numbers here.")

            # Second efficiency angle — TRAINING cost: wall-clock seconds to fit each
            # model (one run; CNNs averaged over seeds). The latency scatter shows
            # inference speed; this shows the very different up-front compute, where
            # the grouped-convolution ResNeXt stands out as the most expensive.
            tc = eff.dropna(subset=["Train_s"]).copy()
            tc = tc[tc["Train_s"] > 0]
            if not tc.empty:
                st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
                tc["tt_train"] = tc["Train_s"].map(lambda v: f"{v:.0f} s")
                tc["tt_eer"]   = tc["EER"].map(lambda v: f"{v:.2f} %")
                _domt = [t for t in _fams if (tc["Type"] == t).any()]
                train_bar = (
                    alt.Chart(tc)
                    .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
                    .encode(
                        y=alt.Y("Config:N", title=None,
                                sort=alt.EncodingSortField(field="Train_s",
                                                           order="descending"),
                                axis=alt.Axis(labelLimit=300)),
                        x=alt.X("Train_s:Q", title="Training time (s) — lower is cheaper"),
                        color=alt.Color("Type:N",
                                        scale=alt.Scale(domain=_domt,
                                                        range=[_LB_TYPE_COLORS.get(t, "#90A4AE")
                                                               for t in _domt]),
                                        legend=alt.Legend(orient="bottom", title=None)),
                        tooltip=[alt.Tooltip("Model", title="Model"),
                                 alt.Tooltip("Features", title="Front-end"),
                                 alt.Tooltip("tt_train", title="Train"),
                                 alt.Tooltip("tt_eer", title="EER")],
                    )
                    .properties(height=alt.Step(24))
                )
                st.altair_chart(train_bar, width="stretch", key="eff_train_bar")
                st.caption("Deep nets cost orders of magnitude more to train than the "
                           "classic DSP detectors (near-zero here); among them the "
                           "grouped-convolution **ResNeXt** is the most expensive. "
                           "wav2vec 2.0 is inference-only, so it has no bar.")

    with tab_chart:
        if view.empty:
            st.info("No results match the current filters.")
        else:
            chart_df = view.copy()
            chart_df["Entry"] = chart_df["Config"] + "  [" + chart_df["SplitDisplay"] + "]"
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
                             alt.Tooltip("SplitDisplay", title="Split"),
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

    if not entries:
        demo_corpus_notice(
            "No pretrained models configured",
            "Run <b>Full comparison</b> once on a machine with the corpus to train "
            "and export all 21 models, then set <b>HF_BASE_URL</b> in "
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
    if n_ready != len(entries):
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


# The leaderboard is ALWAYS the committed leaderboard.json. Manual evaluations run
# in the Classic / CNN modes (kept in session for THOSE pages) are deliberately NOT
# shown here. A full "Train and evaluate all" rewrites leaderboard.json when it
# finishes, so its results appear in the table once the run completes.
_lb_rows = load_leaderboard_rows()
if _lb_rows:
    _render_leaderboard(_lb_rows)
elif not _running:
    show_empty_state(
        "Leaderboard not generated yet",
        "Run <b>Train and evaluate all</b> once to train every model and write "
        "<code>leaderboard.json</code> — the full ranking (classic, CNN and the "
        "wav2vec 2.0 SSL detector across every corpus) appears here.",
    )
