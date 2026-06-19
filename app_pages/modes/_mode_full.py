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
from src.jobs import progress as job_progress, request_cancel, submit_benchmark  # noqa: E402
from src.reporting import (  # noqa: E402
    COL_ACCURACY, COL_EER, COL_FEATURES, COL_MIN_DCF, COL_MODEL,
)
from src.ui_helpers import (  # noqa: E402
    available_pretrained_models, corpus_available, demo_corpus_notice,
    eval_corpora_for, eval_score_controls, get_extractor, get_samples,
    load_config, load_demo_leaderboard, load_pretrained_model, model_downloaded,
    op_in_progress, show_empty_state, sidebar_panel, test_audio_cta,
)

COL_SPLIT = "Split"   # added to classic rows by Run Experiment
FEATURE_LABELS = FeatureExtractor.OPTION_NAMES
FEATURE_ORDER  = ["1", "2", "3", "4", "6", "5"]   # CQCC before Fusion

st.markdown("""
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
</style>
""", unsafe_allow_html=True)

st.title("Leaderboard")
st.caption(
    "Every run this session — classic DSP pipelines and trained CNNs — ranked by "
    "**minDCF** (primary), EER as tiebreaker. The central question: does a CNN on "
    "spectrograms beat the classic front-ends? Classic (CPU) and CNN (GPU) run "
    "**in parallel in the background**, so you can browse other pages while it works."
)


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
            "Add the Hugging Face download links in src/ui_helpers.py (RESNET_URL, "
            "CNN3X3_URL, XGB_*_URL), or run export_models.py locally and host the "
            "generated files, to populate the hub.",
        )
        st.stop()

    n_ready = sum(model_downloaded(e) for e in entries)
    cdl, cinfo = st.columns([1, 2], vertical_alignment="center")
    with cdl:
        if st.button("Download all models", type="primary",
                     icon=":material/cloud_download:", width="stretch"):
            bar = st.progress(0.0, "Starting…")
            for i, e in enumerate(entries, 1):
                bar.progress((i - 1) / len(entries), f"Fetching {e['name']}…")
                try:
                    load_pretrained_model(e)            # downloads + caches on CPU
                except Exception as exc:                # noqa: BLE001 — report, continue
                    st.warning(f"{e['name']}: {exc}")
                bar.progress(i / len(entries), f"{e['name']} ready")
            bar.empty()
            st.success("All models initialised — open Detection Analysis to test them.")
            st.rerun()
    with cinfo:
        st.markdown(f"**{n_ready} / {len(entries)}** models cached on this server. "
                    "Models download once and stay warm for the session.")

    # Leaderboard from the committed metrics (empty until export_models.py runs).
    board = load_demo_leaderboard()
    rows = []
    for e in entries:
        m = board.get(e["key"], {})
        rows.append({
            "Model":      e["name"],
            "Front-end":  e["front"],
            "EER dev (%)":  m.get("eer_dev"),
            "minDCF dev":   m.get("mindcf_dev"),
            "EER eval (%)": m.get("eer_eval"),
            "minDCF eval":  m.get("mindcf_eval"),
            "Status":     "ready" if model_downloaded(e) else "downloads on use",
        })
    df = pd.DataFrame(rows)
    if df["minDCF eval"].notna().any():
        df = df.sort_values("minDCF eval", na_position="last").reset_index(drop=True)
    st.dataframe(
        df.style.format({"EER dev (%)": "{:.2f}", "minDCF dev": "{:.3f}",
                         "EER eval (%)": "{:.2f}", "minDCF eval": "{:.3f}"},
                        na_rep="—"),
        width="stretch", hide_index=True,
    )
    if not board:
        st.caption("Metrics appear here once demo_leaderboard.json is generated by "
                   "export_models.py and committed.")
    test_audio_cta("Pick any of these models and hear them judge your own clip:")
    st.stop()

with st.container(border=True):
    st.markdown('<div class="section-label">Generate the full leaderboard</div>',
                unsafe_allow_html=True)
    g1, _gmid, g2 = st.columns([1.1, 0.3, 1.7], vertical_alignment="bottom")
    with g1:
        # Classic models use this subset (RBF-SVM scales poorly with n); the CNN
        # always trains on the FULL train set for its best result.
        g_subset = st.number_input("Classic files / subset", 0, 25400, 4000,
                                   step=500, disabled=_running,
                                   help="Per classic model, and per scored split. "
                                        "The CNN trains on the full train set "
                                        "regardless.")
    with g2:
        g_corpus, g_split = eval_score_controls("lb", disabled=_running)
    g_cnn = st.toggle("Include CNN (ResNet + 3-Block, full train) — slow",
                      value=True, disabled=_running)
    g_run = st.button("Run all", type="primary", icon=":material/playlist_play:",
                      width="stretch", disabled=_running)

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
    st.markdown('<div class="section-label">Running — classic and CNN in '
                'parallel</div>', unsafe_allow_html=True)

    def _panel(col, title, s, accent):
        with col:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                tot = max(int(s["total"]), 1)
                st.progress(min(1.0, s["done"] / tot),
                            text=f"{s['done']}/{s['total']} · {s['label']}")
                for it in s["items"][-9:]:
                    st.markdown(f"<div style='font-size:0.8rem;opacity:0.82;"
                                f"color:{accent}'>✓ {it}</div>",
                                unsafe_allow_html=True)
                if not s["items"]:
                    st.caption("…")
                st.markdown("<div style='height:0.35rem'></div>",
                            unsafe_allow_html=True)   # breathing room at the bottom

    lc, rc = st.columns(2, gap="large")
    _panel(lc, "Classic models · CPU", pr["streams"]["classic"], "#82B1FF")
    _panel(rc, "CNN · GPU",            pr["streams"]["cnn"],     "#C9A6FF")
    if st.button("Cancel run", icon=":material/cancel:", key="full_cancel"):
        request_cancel()
        st.toast("Cancelling… stops at the next checkpoint.")
        st.rerun(scope="app")


if _running:
    _live_progress()

# Launch the background sweep (the running indicator also lives in the sidebar
# banner). Reaching here implies the corpus is present — the web demo returns
# above via the model hub.
if g_run and not _running:
    ext = get_extractor()
    train = get_samples("train")
    primary, pname = get_samples("dev"), "dev"
    eval_corpora = (eval_corpora_for(g_corpus)
                    if g_split in ("Eval", "Dev + Eval") else [])
    st.session_state["bench_future"] = submit_benchmark(
        ext=ext, feat_labels=FEATURE_LABELS,
        base_params=dict(load_config()["train_params"]),
        train=train, primary=primary, eval_corpora=eval_corpora, pname=pname,
        classic_subset=int(g_subset), cnn_subset=0, include_cnn=g_cnn, seed=42,
    )
    # "Eval" = eval-only: drop the dev rows in app.py when collecting.
    st.session_state["bench_score"] = g_split
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
        d["Type"] = "CNN"
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

if not rows:
    # While a run is in progress the live split-view above already shows what is
    # happening, so don't also show the "No runs yet" empty state.
    if not _running:
        show_empty_state(
            "No runs yet",
            "Run a few configurations in Run Experiment and train a model or two "
            "in CNN Learning — every result is collected here and ranked by minDCF "
            "so you can compare the classic front-ends against the CNNs.",
        )
    st.stop()


def _clean(s, *markers):
    s = str(s)
    for m in markers:
        s = s.replace(m, "")
    return s.split("(")[0].strip()


df = pd.DataFrame(rows)
df[COL_EER]      = pd.to_numeric(df.get(COL_EER), errors="coerce")
df[COL_MIN_DCF]  = pd.to_numeric(df.get(COL_MIN_DCF), errors="coerce")
df["Features"]   = df.get(COL_FEATURES, "").map(lambda s: _clean(s, "[EVAL]"))
df["Model"]      = df.get(COL_MODEL, "").map(lambda s: _clean(s, "[CPU]", "[CUDA]", "[EVAL]"))
df["Config"]     = df["Features"] + " · " + df["Model"]

# Same configuration only differs in training time → keep one row per
# (type, features, model, split), the best by the PRIMARY metric. No dupes.
# ASVspoof's primary metric is the (min t-)DCF; EER is secondary. We therefore
# rank by minDCF and break ties with EER (minDCF saturates near 1.0 for weak
# detectors, so EER orders those).
_SORT = [COL_MIN_DCF, COL_EER]
df = (df.sort_values(_SORT, na_position="last")
        .drop_duplicates(subset=["Type", "Features", "Model", COL_SPLIT], keep="first")
        .reset_index(drop=True))

ranked = df.dropna(subset=[COL_EER]).sort_values(_SORT, na_position="last").reset_index(drop=True)

# ── Champion banner ──────────────────────────────────────────────────────── #
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

# ── Headline metrics (minDCF first — it is the primary metric) ───────────────── #
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total runs", len(df))
c2.metric("Classic / CNN",
          f'{(df["Type"] == "Classic").sum()} / {(df["Type"] == "CNN").sum()}')
c3.metric("Best minDCF",
          f'{ranked[COL_MIN_DCF].min():.3f}' if ranked[COL_MIN_DCF].notna().any() else "—")
c4.metric("Best EER", f'{ranked[COL_EER].min():.2f} %' if not ranked.empty else "—")

st.divider()

tab_table, tab_chart = st.tabs(["Ranking", "Chart"])

with tab_table:
    show = ranked.copy()
    show.insert(0, "Rank", range(1, len(show) + 1))
    show = show[["Rank", "Type", "Features", "Model", COL_SPLIT,
                 COL_MIN_DCF, COL_EER, COL_ACCURACY]]
    show = show.rename(columns={COL_EER: "EER (%)", COL_MIN_DCF: "minDCF",
                                COL_ACCURACY: "Accuracy", COL_SPLIT: "Split"})

    def _hl(row):
        base = ("background-color: rgba(255,193,7,0.14); font-weight:600;"
                if row["Rank"] == 1 else "")
        return [base] * len(row)

    st.dataframe(show.style.apply(_hl, axis=1).format({"EER (%)": "{:.2f}",
                 "minDCF": "{:.3f}"}), width="stretch", hide_index=True)
    st.caption("Ranked by **minDCF** (the primary ASVspoof cost — lower is "
               "better), EER as tiebreaker. The leader is highlighted.")
    st.download_button(
        "Download leaderboard as CSV",
        data=show.to_csv(index=False).encode("utf-8"),
        file_name="leaderboard.csv", mime="text/csv",
        icon=":material/download:",
    )

with tab_chart:
    if ranked.empty:
        st.info("No numeric results to chart yet.")
    else:
        chart = (
            alt.Chart(ranked)
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(
                y=alt.Y("Config:N", title=None,
                        sort=alt.EncodingSortField(field=COL_MIN_DCF, order="ascending"),
                        axis=alt.Axis(labelLimit=300)),
                x=alt.X(f"{COL_MIN_DCF}:Q", title="minDCF — lower is better"),
                color=alt.Color("Type:N",
                                scale=alt.Scale(domain=["Classic", "CNN"],
                                                range=["#4F8BF9", "#AB47BC"]),
                                legend=alt.Legend(orient="bottom", title=None)),
                opacity=alt.condition(alt.datum[COL_SPLIT] == "eval",
                                      alt.value(0.6), alt.value(1.0)),
                tooltip=["Type", "Features", "Model",
                         alt.Tooltip(COL_SPLIT, title="Split"),
                         alt.Tooltip(COL_MIN_DCF, title="minDCF", format=".3f"),
                         alt.Tooltip(COL_EER, title="EER (%)", format=".2f")],
            )
            .properties(height=alt.Step(30))
        )
        st.altair_chart(chart, width="stretch")
        st.caption("Coloured by model family; eval-split rows are translucent.")

# ── Sidebar summary ──────────────────────────────────────────────────────── #
with st.sidebar:
    sb_rows = [("Total runs", str(len(df))),
               ("Classic", str(int((df["Type"] == "Classic").sum()))),
               ("CNN", str(int((df["Type"] == "CNN").sum())))]
    if not ranked.empty:
        sb_rows.append(("Best minDCF", f"{ranked[COL_MIN_DCF].min():.3f}"))
        sb_rows.append(("Best EER", f"{ranked[COL_EER].min():.2f} %"))
    sidebar_panel("Leaderboard", sb_rows)
