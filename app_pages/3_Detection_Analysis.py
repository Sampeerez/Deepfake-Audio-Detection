# -*- coding: utf-8 -*-
"""
app_pages/3_Detection_Analysis.py — Two complementary views:

  • "Test an audio" — drop your own .flac / .wav and have EVERY pretrained model
    (wav2vec 2.0, the deep spectrogram nets — 5-Block CNN ±SE, ResNet+SE,
    ResNeXt+SE, CRNN — and the classic ML × DSP detectors) score it in
    parallel on CPU: a live, side-by-side comparison with a consensus verdict.
    This is the star of the public web demo (no corpus required).
  • "Analyse on a split" — WHY a detector scores the EER / minDCF it does: score
    distributions, ROC and DET curves and an interactive decision threshold
    (needs the local ASVspoof corpus).

set_page_config and PAGE_CSS are applied in app.py.
"""

import concurrent.futures as cf
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import altair as alt  # noqa: E402
import librosa  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data_loader import (  # noqa: E402
    LABEL_BONAFIDE, LABEL_SPOOF, ASVspoofRawWaveDataset, ASVspoofTorchDataset,
    stratified_subsample,
)
from src.features import FeatureExtractor  # noqa: E402
from src.metrics import calculate_eer, calculate_min_dcf  # noqa: E402
from src.models import get_classic_model  # noqa: E402
from src.pipeline import extract_feature_matrix  # noqa: E402
from src.ui_helpers import (  # noqa: E402
    BONAFIDE_COLOR, SPOOF_COLOR, EVAL_CORPUS_CHOICES, HF_EVAL_DATASETS,
    HF_EVAL_PER_CLASS, available_pretrained_models, corpus_available,
    demo_corpus_notice, eval_corpora_for, fig_activation_evolution,
    fig_cnn_input, fig_waveform, get_extractor, get_samples, hf_eval_samples,
    load_leaderboard_models, load_pretrained_model, op_busy_notice,
    show_empty_state, sidebar_panel,
)

FEATURE_LABELS = FeatureExtractor.OPTION_NAMES
# Front-ends offered in the split analysis — the five trained ones (RMS, MFCC,
# LFCC, DWT, CQCC). "Full Fusion" (option "5") is intentionally left out: it is not
# part of the model zoo / leaderboard.
FEATURE_ORDER  = ["1", "2", "3", "4", "6"]
CLASSIFIERS    = {
    "Logistic Regression": "logistic_regression",
    "SVM (RBF)":           "svm_lineal",
    "XGBoost":             "xgboost",
}

# CSS: explicit height on Analyze/Clear buttons to match the file-uploader drop
# zone height (label is collapsed, leaving only the drop zone ~56 px).
st.markdown("""
<style>
[class*="st-key-da_analyze_btn"] button,
[class*="st-key-da_clear_btn"] button {
    min-height: 56px !important;
    height: 56px !important;
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)

# Weighted late-fusion that produces the headline verdict on "Test an audio".
# The cross-domain benchmark (2019 eval + 2021 LA/DF) is unambiguous: the
# self-supervised wav2vec 2.0 is by far the most reliable detector (~10.5% mean
# eval EER, and the only model with a useful cross-domain minDCF), and among the
# spectrogram nets the grouped-convolution ResNeXt+SE is the strongest AND the
# most stable across seeds (best dev minDCF 0.24, EER std 0.04). The classic DSP
# detectors collapse out of domain (minDCF ≈ 1.0), so they are NOT trusted for the
# verdict — they stay visible in the full panel below. The verdict therefore fuses
# just those two complementary views: raw-waveform SSL + the best spectrogram CNN.
# Weights are renormalised at fusion time, and any member that fails to load is
# skipped (so the verdict degrades gracefully to whatever loaded).
FUSION_WEIGHTS = {"wav2vec2": 0.65, "resnext": 0.35}


def _fusion_verdict(rows):
    """Weighted late-fusion → headline verdict: a renormalised weighted average of
    the trusted members' p(spoof), compared against the weighted average of THEIR
    OWN best thresholds (so the fused cut is consistent with the per-model operating
    points). Any member missing from ``rows`` is skipped and the weights renormalised."""
    by_key = {r["key"]: r for r in rows}
    members = []
    for key, w in FUSION_WEIGHTS.items():
        if key in by_key:
            r = by_key[key]
            members.append({"key": key, "name": r["Model"], "weight": w,
                            "p": r["p(spoof)"], "thr": float(r.get("thr", 0.5))})
    wsum  = sum(m["weight"] for m in members)
    fused = (sum(m["weight"] * m["p"]   for m in members) / wsum) if wsum else float("nan")
    fthr  = (sum(m["weight"] * m["thr"] for m in members) / wsum) if wsum else 0.5
    for m in members:
        m["wnorm"] = (m["weight"] / wsum) if wsum else 0.0
    return {"members": members, "fused": fused, "fused_thr": fthr,
            "verdict": "SPOOF" if (wsum and fused >= fthr) else "BONAFIDE"}

extractor = get_extractor()
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dev_label = "CUDA GPU" if device.type == "cuda" else "CPU"

corpus_ok  = corpus_available()
pre_models = available_pretrained_models()

st.title("Detection Analysis")
st.markdown(
    "Drop a clip to score it across every model in parallel — "
    "or pick a detector and explore how it separates bonafide from spoof on a corpus split."
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
# Train subset size for locally-trained classics in the eval-corpus split tab
# (eval clip count is chosen separately now, so training size is fixed here).
_CLASSIC_TRAIN_SUBSET = 800


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


def _score_classic_trained_on_samples(feat_key, clf_name, samples, tag):
    """Score a locally-trained classic model on an explicit (path, label) list —
    lets the local split tab pick a corpus (2019/2021) instead of a dev/eval
    split. The model is still trained on the local 2019 train split via the
    cached _classic_model()."""
    model = _classic_model(feat_key, clf_name, _CLASSIC_TRAIN_SUBSET)
    x_ev, y_ev, _ = extract_feature_matrix(samples, extractor, feat_key, tag,
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


def _score_raw_on_samples(model, mdev, samples, max_samples=64000):
    """Score a raw-waveform model (wav2vec 2.0) on a (path, label) list. p(spoof)
    is the model's own softmax; clips are cropped/padded to a fixed window so they
    batch. Clips are fixed-shape, so the CUDA cache is flushed every 8th batch
    (plus once at the end) to spare the 6 GB GPU without a per-batch sync."""
    loader = DataLoader(
        ASVspoofRawWaveDataset(samples, extractor.sample_rate, max_samples),
        batch_size=8, shuffle=False, num_workers=0,
    )
    scores, labels = [], []
    model.to(mdev).eval()
    is_cuda = getattr(mdev, "type", mdev) == "cuda"
    with torch.no_grad():
        for i, (waves, lbls) in enumerate(loader):
            probs = model.prob_spoof(waves.to(mdev))
            scores.extend(probs.float().cpu().tolist())
            labels.extend(lbls.long().tolist())
            if is_cuda and i % 8 == 0:
                torch.cuda.empty_cache()
    if is_cuda:
        torch.cuda.empty_cache()
    return scores, labels


# ── Web-demo scorers: pretrained registry models on HF-streamed eval clips ─── #
def _score_classic_on_samples(entry, samples):
    """Load a pretrained classic estimator and score it on (path, label) clips."""
    model = load_pretrained_model(entry)
    x_ev, y_ev, _ = extract_feature_matrix(samples, extractor, entry["feat"],
                                           f"hf_{entry['key']}", n_workers=4,
                                           use_cache=True)
    return model.predict_proba(x_ev)[:, 1].tolist(), y_ev.tolist()


@st.fragment
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
    name   = st.session_state["da_name"]

    # The EER / minDCF / ROC-DET curves and the binned histogram do NOT depend on
    # the threshold, so compute them ONCE per analysed split and stash them in
    # session state. Moving the slider then only recomputes the three cheap
    # operating-point numbers, keeping the redraw snappy. Pre-binning the histogram
    # (~80 rows instead of up to 10 000 raw points) is the other big win.
    _fp = (name, int(scores.size), float(np.round(scores.sum(), 4)))
    if st.session_state.get("_da_static_fp") != _fp:
        _bona  = scores[labels == LABEL_BONAFIDE]
        _spoof = scores[labels == LABEL_SPOOF]
        _eer, _eer_thr = calculate_eer(scores.tolist(), labels.tolist())
        _mindcf        = calculate_min_dcf(scores.tolist(), labels.tolist())
        _grid = np.linspace(0.0, 1.0, 401)
        _far  = np.array([(_spoof < t).mean() if len(_spoof) else 0.0 for t in _grid])
        _frr  = np.array([(_bona >= t).mean() if len(_bona)  else 0.0 for t in _grid])
        _tpr, _fpr = 1.0 - _far, _frr
        _order = np.argsort(_fpr)
        _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
        _auc  = float(_trapz(_tpr[_order], _fpr[_order]))
        _edges = np.linspace(0.0, 1.0, 41)
        _bc, _ = np.histogram(_bona,  bins=_edges)
        _sc, _ = np.histogram(_spoof, bins=_edges)
        _hist = pd.DataFrame({
            "b0": np.concatenate([_edges[:-1], _edges[:-1]]),
            "b1": np.concatenate([_edges[1:],  _edges[1:]]),
            "count": np.concatenate([_bc, _sc]).astype(int),
            "cls": ["Bonafide"] * (len(_edges) - 1) + ["Spoof"] * (len(_edges) - 1)})
        st.session_state["_da_static"] = {
            "bona": _bona, "spoof": _spoof, "eer": float(_eer),
            "eer_thr": float(_eer_thr), "mindcf": float(_mindcf), "auc": _auc,
            "hist": _hist,
            "curve": pd.DataFrame({"TPR": _tpr, "FPR": _fpr,
                                   "FAR_pct": _far * 100.0, "FRR_pct": _frr * 100.0})}
        st.session_state["_da_static_fp"] = _fp

    S = st.session_state["_da_static"]
    bona, spoof = S["bona"], S["spoof"]
    eer, eer_thr, mindcf, auc = S["eer"], S["eer_thr"], S["mindcf"], S["auc"]
    curve, hist_df = S["curve"], S["hist"]

    st.markdown(f"**{name}** &nbsp;·&nbsp; "
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
    thr = st.slider(
        "A clip is declared SPOOF when its score  p(spoof) ≥ threshold",
        0.0, 1.0, float(round(eer_thr, 3)), 0.005)
    t_far = float((spoof < thr).mean()) if len(spoof) else 0.0
    t_frr = float((bona >= thr).mean()) if len(bona)  else 0.0
    t_acc = float(((scores >= thr).astype(int) == labels).mean())

    c1, c2, c3 = st.columns(3)
    c1.metric("False acceptance", f"{100 * t_far:.2f} %",
              help="Spoof clips let through (scored below the threshold).")
    c2.metric("False rejection", f"{100 * t_frr:.2f} %",
              help="Bonafide clips wrongly blocked (scored at/above the threshold).")
    c3.metric("Accuracy", f"{100 * t_acc:.2f} %",
              help="Correct decisions at this threshold.")

    st.divider()
    # Only the threshold-dependent marks are rebuilt each move (tiny frames).
    op = pd.DataFrame({"FPR": [t_frr], "TPR": [1 - t_far],
                       "FAR_pct": [100 * t_far], "FRR_pct": [100 * t_frr]})
    thr_df  = pd.DataFrame({"x": [float(thr)]})
    eerx_df = pd.DataFrame({"x": [float(eer_thr)]})

    # Fixed plot height; clip=True on the moving marks so a point at a scale edge
    # never makes Vega re-pad (which is what resized the charts as the slider moved).
    _H = 380
    g1, g2, g3 = st.columns(3, gap="medium")
    with g1:
        st.markdown(
            f"**Score distribution** &nbsp;&nbsp;"
            f"<span style='color:{BONAFIDE_COLOR};font-weight:700;'>● Bonafide</span> "
            f"&nbsp;<span style='color:{SPOOF_COLOR};font-weight:700;'>● Spoof</span>",
            unsafe_allow_html=True)
        # Two single-colour layers (one per class) so each bar is anchored to a
        # zero baseline and they OVERLAY (translucent) instead of stacking — a
        # colour-encoded bar with stack=None loses its 0-baseline and renders as
        # thin floating dashes, which is what broke before.
        def _hbar(_d, _c):
            return alt.Chart(_d).mark_bar(opacity=0.6, color=_c).encode(
                x=alt.X("b0:Q", bin="binned", title="p(spoof)",
                        scale=alt.Scale(domain=[0, 1])),
                x2="b1:Q",
                y=alt.Y("count:Q", title="count"))
        hist = (_hbar(hist_df[hist_df["cls"] == "Bonafide"], BONAFIDE_COLOR)
                + _hbar(hist_df[hist_df["cls"] == "Spoof"], SPOOF_COLOR))
        eer_rule = (alt.Chart(eerx_df).mark_rule(color="#9E9E9E", strokeDash=[2, 2],
                    clip=True).encode(x="x:Q"))
        thr_rule = (alt.Chart(thr_df).mark_rule(color="#FFD54F", strokeDash=[5, 4],
                    size=2, clip=True).encode(x="x:Q"))
        st.altair_chart((hist + eer_rule + thr_rule).properties(height=_H),
                        width="stretch", key="da_dist")
    with g2:
        st.markdown(f"**ROC curve** &nbsp;·&nbsp; AUC = **{auc:.3f}**")
        roc_line = alt.Chart(curve).mark_line(color="#4FC3F7", size=2, clip=True).encode(
            x=alt.X("FPR:Q", title="False positive rate", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("TPR:Q", title="True positive rate", scale=alt.Scale(domain=[0, 1])))
        roc_diag = (alt.Chart(pd.DataFrame({"a": [0.0, 1.0]}))
                    .mark_line(color="#5C6B8A", strokeDash=[4, 4]).encode(x="a:Q", y="a:Q"))
        roc_eer = (alt.Chart(pd.DataFrame({"x": [float(eer)], "y": [float(1 - eer)]}))
                   .mark_point(color="#FFD54F", size=95, filled=True, clip=True)
                   .encode(x="x:Q", y="y:Q"))
        roc_thr = (alt.Chart(op).mark_point(color="#66BB6A", size=160, filled=True,
                   clip=True).encode(x="FPR:Q", y="TPR:Q"))
        st.altair_chart((roc_line + roc_diag + roc_eer + roc_thr).properties(height=_H),
                        width="stretch", key="da_roc")
    with g3:
        st.markdown("**DET curve** &nbsp;·&nbsp; miss vs false-alarm")
        det_line = alt.Chart(curve).mark_line(color="#AB47BC", size=2, clip=True).encode(
            x=alt.X("FAR_pct:Q", title="False acceptance (%)", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("FRR_pct:Q", title="False rejection (%)", scale=alt.Scale(domain=[0, 100])))
        det_diag = (alt.Chart(pd.DataFrame({"a": [0.0, 100.0]}))
                    .mark_line(color="#5C6B8A", strokeDash=[4, 4]).encode(x="a:Q", y="a:Q"))
        det_eer = (alt.Chart(pd.DataFrame({"x": [float(eer * 100)], "y": [float(eer * 100)]}))
                   .mark_point(color="#FFD54F", size=95, filled=True, clip=True)
                   .encode(x="x:Q", y="y:Q"))
        det_thr = (alt.Chart(op).mark_point(color="#66BB6A", size=160, filled=True,
                   clip=True).encode(x="FAR_pct:Q", y="FRR_pct:Q"))
        st.altair_chart((det_line + det_diag + det_eer + det_thr).properties(height=_H),
                        width="stretch", key="da_det")

    st.caption("Yellow dashed line / green dot = the current **threshold**; grey "
               "dotted line + yellow dot = the **EER** operating point. Move the "
               "slider to walk the green dot along the ROC and DET curves. minDCF "
               "weighs a false accept 10× a miss at a 5% prior, so it only drops "
               "below 1.0 once false acceptances are rare.")


# ===========================================================================
# Single-clip inference shared by the multi-model tab
# ===========================================================================
def _audio_suffix(name, raw: bytes) -> str:
    """Best temp-file suffix for an upload: trust the filename extension first,
    then sniff the container magic bytes. Determines which decoder librosa's
    audioread fallback selects, so getting it right is what lets mp3/ogg/m4a
    (and odd wav/flac) decode."""
    if name and "." in name:
        ext = "." + name.rsplit(".", 1)[1].lower()
        if ext in (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"):
            return ext
    if raw[:4] == b"RIFF":
        return ".wav"
    if raw[:4] == b"fLaC":
        return ".flac"
    if raw[:4] == b"OggS":
        return ".ogg"
    if raw[:3] == b"ID3" or raw[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return ".mp3"
    if raw[4:8] == b"ftyp":
        return ".m4a"
    return ".flac"


def _decode_with_ffmpeg(raw: bytes, sr: int) -> np.ndarray:
    """Decode arbitrary audio bytes to mono float32 via the ffmpeg binary bundled
    by ``imageio-ffmpeg``. This needs NO system ffmpeg, so it works on hosts
    (Streamlit Cloud) where soundfile's libsndfile chokes on a FLAC and audioread
    has no backend. Pipes the raw container in, reads f32le PCM out."""
    import subprocess
    import imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.run(
        [exe, "-nostdin", "-loglevel", "quiet", "-i", "pipe:0",
         "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sr), "pipe:1"],
        input=raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def _load_signal(uploaded, name=None):
    try:
        signal, _ = librosa.load(uploaded, sr=extractor.sample_rate, mono=True)
    except Exception:
        # soundfile can't read compressed formats (mp3/m4a/ogg) or some FLAC from
        # a BytesIO. Read the bytes and decode through the bundled ffmpeg binary,
        # which is far more permissive and needs no system install.
        if hasattr(uploaded, "seek"):
            uploaded.seek(0)
        raw = (uploaded.read() if hasattr(uploaded, "read")
               else bytes(uploaded) if isinstance(uploaded, (bytes, bytearray))
               else b"")
        try:
            signal = _decode_with_ffmpeg(raw, extractor.sample_rate)
        except Exception as _ex:                          # final, clearer failure
            suffix = _audio_suffix(name, raw)
            raise RuntimeError(
                f"unsupported or corrupt audio ({suffix}). [{_ex}]"
            ) from _ex
    if len(signal) < extractor.n_fft:
        signal = np.pad(signal, (0, extractor.n_fft - len(signal)))
    return signal


def _open_in_signal_explorer(blob: bytes, fname: str) -> None:
    """Drop everything currently loaded in Signal Explorer and place THIS uploaded
    clip as its single source, then jump to that page so the user can browse every
    representation of the same audio. Mirrors Signal Explorer's session contract
    (slot prefixes a/b/c, plain upload keys, the cross-page `_se_memory` dict)."""
    for p in ("a", "b", "c"):                       # wipe all existing slots
        for k in [key for key in list(st.session_state) if key.startswith(f"{p}_")]:
            del st.session_state[k]
    # Pre-select EVERY representation so the user sees them all without ticking
    # any (mirrors Signal Explorer's ALL_VIEWS / se_views pills key).
    _all_views = ["Waveform", "STFT", "CNN Input", "MFCC", "LFCC", "CQCC"]
    st.session_state["se_n"]            = 1
    st.session_state["se_views"]        = list(_all_views)
    st.session_state["a_source"]        = "Upload"
    st.session_state["a_upload_name"]   = fname or "uploaded.wav"
    st.session_state["a_upload_bytes"]  = blob
    # Seed the cross-page memory too, so Signal Explorer's _restore() keeps it.
    mem = st.session_state.get("_se_memory")
    mem = mem if isinstance(mem, dict) else {}
    for k in [key for key in list(mem) if key[:2] in ("a_", "b_", "c_")]:
        del mem[k]
    mem.update({"se_n": 1, "se_views": list(_all_views), "a_source": "Upload",
                "a_upload_name": fname or "uploaded.wav", "a_upload_bytes": blob})
    st.session_state["_se_memory"] = mem
    st.switch_page("app_pages/1_Signal_Explorer.py")


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


def _analyse_all_models(signal, entries, loaded, thresholds):
    """Score one clip with every pretrained model. DSP front-ends are extracted
    once each in parallel threads (librosa releases the GIL), then each model
    predicts on its precomputed representation. Each model is judged against ITS
    OWN best threshold (``thresholds[key]``, the dev EER operating point from the
    leaderboard; 0.5 if unknown). Returns a list of result dicts."""
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
        elif e["kind"] == "raw":
            # wav2vec 2.0 eats the full raw waveform directly (no crop for a
            # single clip); p(spoof) is its own softmax, not a sigmoid. The cached
            # model may live on GPU (e.g. after a split analysis), so send the input
            # to wherever its weights are instead of assuming CPU.
            _mdev = next(model.parameters()).device
            wave = torch.from_numpy(signal).unsqueeze(0).float().to(_mdev)
            with torch.no_grad():
                prob = float(model.prob_spoof(wave).item())
        else:
            prob = float(model.predict_proba(feat_vecs[e["feat"]])[0, 1])
        _thr = float(thresholds.get(e["key"], 0.5))
        rows.append({
            "key":     e["key"],
            "Model":   e["name"],
            "Front-end": e["front"],
            "p(spoof)": prob,
            "thr":     _thr,
            "Verdict": "SPOOF" if prob >= _thr else "BONAFIDE",
            "_acts":   acts,
        })
    return rows


def _render_test_results(rows, signal, blob, fname):
    """Render the full Test-an-audio results panel: fusion card, model chips,
    table/chart, signal views, and CNN activation maps. Each model's Verdict was
    already decided at its OWN best threshold in _analyse_all_models."""
    probs   = [r["p(spoof)"] for r in rows]
    mean_p  = float(np.mean(probs))
    n_spoof = int(sum(r["Verdict"] == "SPOOF" for r in rows))
    n_total = len(rows)

    fusion  = _fusion_verdict(rows)
    _members = fusion["members"]
    fused_p  = fusion["fused"]
    v_color  = SPOOF_COLOR if fusion["verdict"] == "SPOOF" else BONAFIDE_COLOR
    v_text   = ("SPOOF — deepfake" if fusion["verdict"] == "SPOOF"
                else "BONAFIDE — real speech")

    _contrib = " &nbsp;·&nbsp; ".join(
        f"<b style='color:#C9D7F5;'>{m['name']}</b>"
        f"<span style='color:#7C88A3;'> (p={m['p']:.2f}, "
        f"{m['wnorm']*100:.0f}%)</span>"
        for m in _members)
    st.markdown(
        f"<div style='text-align:center;margin:0.6rem auto 0.3rem;"
        f"max-width:620px;padding:1rem 1.3rem;border-radius:0.9rem;"
        f"border:1px solid {v_color}59;"
        f"box-shadow:0 0 20px {v_color}55, inset 0 0 24px {v_color}1f;'>"
        f"<span style='display:block;font-size:0.66rem;font-weight:800;"
        f"letter-spacing:0.16em;text-transform:uppercase;color:#9EA8C0;"
        f"margin-bottom:0.15rem;'>Final verdict — weighted fusion</span>"
        f"<span style='font-size:1.8rem;font-weight:700;color:{v_color};"
        f"text-shadow:0 0 14px {v_color}aa;'>{v_text}</span><br>"
        f"<span style='color:#9EA8C0;font-size:0.9rem;'>"
        f"fused p(spoof) = <b style='color:#C9D7F5;'>{fused_p:.3f}</b></span>"
        f"<div style='color:#8A95AE;font-size:0.8rem;margin-top:0.4rem;'>"
        f"{_contrib}</div></div>",
        unsafe_allow_html=True,
    )
    _cons = ("majority flags spoof" if n_spoof * 2 > n_total
             else ("majority says bonafide" if n_spoof * 2 < n_total
                   else "the panel is split"))
    st.markdown(
        f"<div style='text-align:center;color:#8A95AE;font-size:0.85rem;"
        f"margin:1.6rem auto 1.6rem;'>Panel of {n_total} models — {_cons}: "
        f"<b>{n_spoof}/{n_total}</b> flag spoof &nbsp;·&nbsp; "
        f"mean p(spoof) = {mean_p:.3f}</div>",
        unsafe_allow_html=True,
    )
    st.audio(signal, sample_rate=extractor.sample_rate)
    st.divider()

    st.markdown("**Every other model's verdict**")
    _fusion_keys = {m["key"] for m in _members}
    _sorted = [r for r in sorted(rows, key=lambda r: r["p(spoof)"], reverse=True)
               if r["key"] not in _fusion_keys]
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
    st.divider()

    df = (pd.DataFrame([{"Model": r["Model"], "Front-end": r["Front-end"],
                         "p(spoof)": r["p(spoof)"], "threshold": r.get("thr", 0.5),
                         "Verdict": r["Verdict"]} for r in rows])
          .sort_values("p(spoof)", ascending=False).reset_index(drop=True))

    gcol1, gcol2 = st.columns([1.05, 1], gap="large")
    with gcol1:
        st.markdown("**Per-model scores**")

        def _vcolor(v):
            c = SPOOF_COLOR if v == "SPOOF" else BONAFIDE_COLOR
            return f"color:{c};font-weight:700;"

        _rh = 37
        st.dataframe(
            df.style.format({"p(spoof)": "{:.3f}", "threshold": "{:.3f}"})
                    .map(_vcolor, subset=["Verdict"]),
            width="stretch", hide_index=True,
            row_height=_rh, height=int(_rh * (len(df) + 1) + 2),
        )
    with gcol2:
        st.markdown("**Probability per model**")
        bar = (alt.Chart(df).mark_bar().encode(
                    x=alt.X("p(spoof):Q", scale=alt.Scale(domain=[0, 1]),
                            title="p(spoof)"),
                    y=alt.Y("Model:N", sort="-x", title=None),
                    color=alt.Color("Verdict:N", legend=None,
                                    scale=alt.Scale(domain=["BONAFIDE", "SPOOF"],
                                                    range=[BONAFIDE_COLOR, SPOOF_COLOR])),
                    tooltip=["Model", "Front-end", "p(spoof)", "threshold", "Verdict"])
               .properties(height=max(150, 42 * len(df))))
        # Each model's OWN threshold, as a yellow tick on its bar (no single line —
        # every model now decides at a different cut).
        ticks = (alt.Chart(df).mark_tick(color="#FFD54F", thickness=2, size=22)
                 .encode(x="threshold:Q", y=alt.Y("Model:N", sort="-x")))
        st.altair_chart(bar + ticks, width="stretch")
        st.caption("Yellow tick = each model's own decision threshold (dev EER "
                   "operating point).")

    st.divider()
    _sh1, _sh2 = st.columns([3, 1.4], vertical_alignment="center")
    with _sh1:
        st.markdown("**Signal views**")
    with _sh2:
        if st.button("Open in Signal Explorer", icon=":material/graphic_eq:",
                     width="stretch", key="da_to_se",
                     help="Load this clip into Signal Explorer to see every "
                          "representation (replaces whatever is loaded there)."):
            _open_in_signal_explorer(blob, fname)
    pv1, pv2 = st.columns(2)
    with pv1:
        st.pyplot(fig_waveform(signal, extractor.sample_rate, title="Waveform",
                               label=None, figsize=(9, 3.2)), clear_figure=True,
                  bbox_inches=None)
    with pv2:
        st.pyplot(fig_cnn_input(signal, extractor), clear_figure=True,
                  bbox_inches=None)

    # Canonical display order: 5-Block CNN, +SE, ResNet+SE, ResNeXt+SE, CRNN — so
    # the 2-per-row grid here AND the dropdown in CNN Learning both read row 1
    # (cnn, cnn+se) · row 2 (resnet, resnext) · row 3 (crnn).
    _MAP_ORDER = {"cnn5": 0, "cnn5_se": 1, "resnet": 2, "resnext": 3, "crnn": 4}
    _cnn_rows = sorted([r for r in rows if r["_acts"] is not None],
                       key=lambda r: _MAP_ORDER.get(r["key"], 99))
    if _cnn_rows:
        st.divider()
        _ah1, _ah2 = st.columns([3, 1.4], vertical_alignment="center")
        with _ah1:
            st.markdown("**Convolutional activation maps**")
        with _ah2:
            if st.button("See full maps in CNN Learning",
                         icon=":material/account_tree:", width="stretch",
                         key="da_to_cnn",
                         help="Open the CNN Learning page with every feature "
                              "map of every block for this clip."):
                st.session_state["cnn_handoff"] = {
                    "fname": fname,
                    "items": [{"name": r["Model"], "p": r["p(spoof)"],
                               "acts": r["_acts"]} for r in _cnn_rows],
                }
                st.session_state["bench_choice"] = "cnn"
                st.switch_page("app_pages/2_Benchmark.py")
        # Two maps per row (instead of cramming all five into one) so each renders
        # large enough to read; the last row may hold a single map.
        _per_row = 2
        for _start in range(0, len(_cnn_rows), _per_row):
            _cols = st.columns(_per_row)
            for _col, r in zip(_cols, _cnn_rows[_start:_start + _per_row]):
                with _col:
                    st.pyplot(
                        fig_activation_evolution(
                            r["_acts"],
                            f"{r['Model']} · p(spoof) = {r['p(spoof)']:.2f}"),
                        clear_figure=True, bbox_inches=None,
                    )


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
        "once — the self-supervised **wav2vec 2.0** transformer on the raw "
        "waveform, the deep spectrogram nets (5-Block CNN ±SE, ResNet+SE, "
        "ResNeXt+SE, CRNN) and the classic detectors (LR / SVM / XGBoost) on "
        "their DSP front-ends. Each model decides at **its own best threshold** "
        "(the dev EER operating point), not a flat 0.5. The **final verdict** is a "
        "weighted late-fusion of the two most reliable cross-domain detectors — "
        "**wav2vec 2.0** (0.65) and the best spectrogram net, **ResNeXt+SE** (0.35); "
        "the full panel of every model is shown below."
    )
    # Per-model decision thresholds — each model's dev EER operating point, recorded
    # in leaderboard.json by the full sweep (0.5 fallback when not present yet).
    _board = load_leaderboard_models() or {}
    thresholds = {k: float(m["thr_dev"]) for k, m in _board.items()
                  if isinstance(m, dict) and isinstance(m.get("thr_dev"), (int, float))}

    # ── Upload / Analyze / Clear row ─────────────────────────────────────── #
    _c_up, _c_an, _c_cl = st.columns([5, 1.5, 1], vertical_alignment="center")
    with _c_up:
        uploaded = st.file_uploader(
            "Upload an audio clip (flac / wav / mp3 / ogg / m4a)",
            type=["flac", "wav", "mp3", "ogg", "m4a"],
            key="da_test_upload", label_visibility="collapsed")
    with _c_an:
        _has_audio = (st.session_state.get("da_test_bytes") is not None
                      or uploaded is not None)
        do_analyze = st.button("Analyze", type="primary", icon=":material/radar:",
                               width="stretch", key="da_analyze_btn",
                               disabled=not _has_audio)
    with _c_cl:
        do_clear = st.button("Clear", icon=":material/close:", width="stretch",
                             key="da_clear_btn")

    if do_clear:
        for _k in ["da_test_bytes", "da_test_name", "da_test_rows"]:
            st.session_state.pop(_k, None)
        st.rerun()

    # Persist bytes across page navigation; invalidate cached rows when file changes.
    if uploaded is not None:
        _new = uploaded.getvalue()
        if _new != st.session_state.get("da_test_bytes"):
            st.session_state.pop("da_test_rows", None)
        st.session_state["da_test_bytes"] = _new
        st.session_state["da_test_name"]  = uploaded.name
    _blob  = st.session_state.get("da_test_bytes")
    _fname = st.session_state.get("da_test_name")

    if _blob is None:
        show_empty_state(
            "No audio uploaded",
            "Drop an audio clip (flac / wav / mp3 / ogg / m4a) above, then click "
            "Analyze to score it "
            "with every pretrained model.",
        )
    else:
        if uploaded is None and _fname:
            st.caption(f"Loaded: **{_fname}** — click Analyze to score, or Clear to discard.")

        # Load signal (needed for both analysis and display). Errors surface here.
        try:
            signal = _load_signal(io.BytesIO(_blob), name=_fname)
        except Exception as _ex:
            st.error(f"Could not decode the audio file — {_ex}")
            st.stop()

        if do_analyze:
            with st.spinner("Consulting the Jedi Archives (loading every model in parallel)…"):
                loaded, failed = _load_all_models(pre_models)
            for _name, _err in failed:
                st.warning(f"{_name} unavailable: {_err}")
            entries_ok = [e for e in pre_models if e["key"] in loaded]
            if not entries_ok:
                st.error("No model could be loaded — check the download URLs.")
                st.stop()
            with st.spinner("Running it past the astromech (scoring every model)…"):
                rows = _analyse_all_models(signal, entries_ok, loaded, thresholds)
            st.session_state["da_test_rows"] = rows

        _cached_rows = st.session_state.get("da_test_rows")
        if _cached_rows is None:
            show_empty_state(
                "Ready to analyze",
                "Click Analyze above to score this clip with all pretrained models.",
            )
        else:
            _render_test_results(_cached_rows, signal, _blob, _fname)


# ===========================================================================
# Analyse one detector on a corpus split (local; needs the dataset)
# ===========================================================================
def _clear_split_state() -> None:
    """Discard the current split-analysis result and rerun. Shared verbatim by
    the Clear buttons of both the local-corpus and the web/HF split branches."""
    for _k in ["da_scores", "da_labels", "da_name", "da_detector"]:
        st.session_state.pop(_k, None)
    st.rerun()


with tab_analyse:
  if corpus_ok:
    has_cnn = "cnn_model" in st.session_state
    with st.container(border=True):
        st.markdown('<div class="section-label">Detector</div>', unsafe_allow_html=True)
        src_opts = ["Classic models"]
        if any(e["kind"] == "cnn" for e in pre_models):
            src_opts.append("CNN")
        if any(e["kind"] == "raw" for e in pre_models):
            src_opts.append("SSL")
        if has_cnn:
            src_opts.append("CNN (this session)")
        source = st.segmented_control("Source", src_opts, default="Classic models",
                                      key="da_source", label_visibility="collapsed")
        source = source or "Classic models"
        _busy = op_busy_notice()

        # Bind all three up front: the store block below reads feat_key/clf_disp/
        # cnn_name regardless of branch, so selecting CNN/SSL must not leave the
        # classic-only names unbound (that would NameError on Analyze).
        feat_key = clf_disp = cnn_name = None
        if source == "Classic models":
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
            # Pick WHICH pretrained deep model (ResNet+SE, 5-Block CNN, CRNN, …) to score.
            _cnn_entries = [e for e in pre_models if e["kind"] == "cnn"]
            with st.container(key="nosearch_da_cnn"):
                cnn_name = st.selectbox("CNN", [e["name"] for e in _cnn_entries],
                                        key="da_cnn")
        elif source == "SSL":
            # The self-supervised transformer family (wav2vec 2.0), kept distinct
            # from the CNNs for methodological clarity.
            _ssl_entries = [e for e in pre_models if e["kind"] == "raw"]
            with st.container(key="nosearch_da_ssl"):
                cnn_name = st.selectbox("SSL model", [e["name"] for e in _ssl_entries],
                                        format_func=lambda n: n.replace(" (SSL)", ""),
                                        key="da_ssl")
        else:
            st.caption("Scores the CNN trained in Benchmark · CNN this session.")

    a1, a2, a3, a4 = st.columns([1.3, 1, 1.4, 0.7], vertical_alignment="bottom")
    with a1:
        with st.container(key="nosearch_da_corpus"):
            corpus = st.selectbox("Eval corpus", EVAL_CORPUS_CHOICES, key="da_corpus",
                                  help="Evaluate on this corpus' eval split "
                                       "(2019 in-domain, 2021 = generalization).")
    with a2:
        nper = st.number_input("Clips / class", 50, 5000, 400, step=50, key="da_nper",
                               help="Balanced bonafide + spoof eval clips.")
    with a3:
        analyze = st.button("Analyze", type="primary", disabled=_busy,
                            icon=":material/insights:", width="stretch")
    with a4:
        if st.button("Clear", icon=":material/close:", width="stretch",
                     key="da_split_clear"):
            _clear_split_state()

    if analyze:
        with st.spinner("Awaiting the Council's verdict (scoring the detector)…"):
            _resolved = eval_corpora_for(corpus)
            samples = (stratified_subsample(_resolved[0][1], 2 * int(nper), 42)
                       if _resolved else [])
            if not samples:
                st.warning(f"No {corpus} eval clips available locally — pick "
                           "another corpus or download that eval set.")
            else:
                _tag = f"da_local_{corpus}"
                if source == "Classic models":
                    sc, lb = _score_classic_trained_on_samples(
                        feat_key, CLASSIFIERS[clf_disp], samples, _tag)
                    name = f"{FEATURE_LABELS[feat_key]} × {clf_disp} · {corpus} eval"
                elif source == "CNN":
                    _entry = next(e for e in pre_models
                                  if e["kind"] == "cnn" and e["name"] == cnn_name)
                    sc, lb = _score_cnn_on_samples(load_pretrained_model(_entry),
                                                   "cpu", samples)
                    name = f"{cnn_name} · {corpus} eval"
                elif source == "SSL":
                    _entry = next(e for e in pre_models
                                  if e["kind"] == "raw" and e["name"] == cnn_name)
                    sc, lb = _score_raw_on_samples(load_pretrained_model(_entry),
                                                   device, samples)
                    name = f"{cnn_name.replace(' (SSL)', '')} · {corpus} eval"
                else:
                    sc, lb = _score_cnn_on_samples(st.session_state["cnn_model"],
                                                   device, samples)
                    name = f"CNN · {corpus} eval"
                st.session_state["da_scores"] = sc
                st.session_state["da_labels"] = lb
                st.session_state["da_name"]   = name
                st.session_state["da_detector"] = {
                    "source": source, "feat": feat_key,
                    "clf": clf_disp, "cnn": cnn_name}

    _render_split_results()

  elif pre_models:
    # Web demo: score a PRETRAINED registry model on eval clips streamed from the
    # public Hugging Face datasets (no corpus, no training — cross-dataset eval).
    classic_entries = [e for e in pre_models if e["kind"] == "classic"]
    cnn_entries     = [e for e in pre_models if e["kind"] == "cnn"]
    ssl_entries     = [e for e in pre_models if e["kind"] == "raw"]

    with st.container(border=True):
        st.markdown('<div class="section-label">Detector</div>', unsafe_allow_html=True)
        src_opts = (["Classic models"] if classic_entries else []) + \
                   (["CNN"] if cnn_entries else []) + \
                   (["SSL"] if ssl_entries else [])
        source = st.segmented_control("Source", src_opts, default=src_opts[0],
                                      key="da_source_hf", label_visibility="collapsed")
        source = source or src_opts[0]
        _busy = op_busy_notice()

        # Bind all three up front: the store block below reads feat_key/clf_disp/
        # cnn_name regardless of branch, so selecting CNN/SSL must not leave the
        # classic-only names unbound (that would NameError on Analyze).
        feat_key = clf_disp = cnn_name = None
        if source == "Classic models":
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
        elif source == "SSL":
            with st.container(key="nosearch_da_ssl_hf"):
                cnn_name = st.selectbox("SSL model", [e["name"] for e in ssl_entries],
                                        format_func=lambda n: n.replace(" (SSL)", ""),
                                        key="da_ssl_hf")
            st.caption("Self-supervised wav2vec 2.0, served on CPU.")
        else:
            with st.container(key="nosearch_da_cnn_hf"):
                cnn_name = st.selectbox("CNN", [e["name"] for e in cnn_entries],
                                        key="da_cnn_hf")
            st.caption("CNN checkpoint, served on CPU.")

    a1, a2, a3, a4 = st.columns([1.3, 1, 1.4, 0.7], vertical_alignment="bottom")
    with a1:
        with st.container(key="nosearch_da_corpus_hf"):
            corpus = st.selectbox("Eval corpus", list(HF_EVAL_DATASETS),
                                  key="da_corpus_hf",
                                  help="Eval split streamed from the public HF dataset.")
    with a2:
        nper = st.number_input("Clips / class", 5, 500, HF_EVAL_PER_CLASS, step=5,
                               key="da_nper_hf",
                               help="Balanced bonafide + spoof clips fetched from HF.")
    with a3:
        analyze = st.button("Analyze", type="primary", disabled=_busy,
                            icon=":material/insights:", width="stretch")
    with a4:
        if st.button("Clear", icon=":material/close:", width="stretch",
                     key="da_hf_clear"):
            _clear_split_state()

    if analyze:
        with st.spinner(f"Pulling {corpus} records from the Archives and scoring…"):
            samples = hf_eval_samples(corpus, int(nper))
            if not samples:
                st.warning("Could not fetch eval clips from Hugging Face — "
                           "try again or pick another corpus.")
            else:
                if source == "Classic models":
                    entry = next(e for e in classic_entries
                                 if e["feat"] == feat_key
                                 and e["clf"] == CLASSIFIERS[clf_disp])
                    sc, lb = _score_classic_on_samples(entry, samples)
                    name = f"{FEATURE_LABELS[feat_key]} × {clf_disp} · {corpus} eval"
                elif source == "SSL":
                    entry = next(e for e in ssl_entries if e["name"] == cnn_name)
                    sc, lb = _score_raw_on_samples(load_pretrained_model(entry),
                                                   "cpu", samples)
                    name = f"{cnn_name.replace(' (SSL)', '')} · {corpus} eval"
                else:
                    entry = next(e for e in cnn_entries if e["name"] == cnn_name)
                    sc, lb = _score_cnn_on_samples(load_pretrained_model(entry),
                                                   "cpu", samples)
                    name = f"{cnn_name} · {corpus} eval"
                st.session_state["da_scores"] = sc
                st.session_state["da_labels"] = lb
                st.session_state["da_name"]   = name
                st.session_state["da_detector"] = {
                    "source": source, "feat": feat_key,
                    "clf": clf_disp, "cnn": cnn_name}

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
    # Detector details belong to the "Analyse on a split" tab, so only show them
    # once a split has actually been scored (never on the "Test an audio" tab,
    # where the feature/classifier selection is meaningless). For a classic model
    # we surface its front-end + classifier; for a CNN, the CNN name.
    if "da_scores" in st.session_state:
        _det = st.session_state.get("da_detector", {})
        _src = _det.get("source")
        if _src == "Classic models":
            if _det.get("feat") is not None:
                _rows.append(("Front-end", FEATURE_LABELS[_det["feat"]]))
            if _det.get("clf"):
                _rows.append(("Classifier", _det["clf"]))
        elif _src == "SSL":
            _rows.append(("SSL", _det.get("cnn") or "wav2vec 2.0"))
        elif _src == "CNN (this session)":
            _rows.append(("CNN", "this session"))
        elif _det.get("cnn"):
            _rows.append(("CNN", _det["cnn"]))
        _sc = np.asarray(st.session_state["da_scores"], dtype=float)
        _lb = np.asarray(st.session_state["da_labels"], dtype=int)
        _eer, _ = calculate_eer(_sc.tolist(), _lb.tolist())
        _dcf = calculate_min_dcf(_sc.tolist(), _lb.tolist())
        _rows += [("Trials", f"{len(_sc):,}"), ("minDCF", f"{_dcf:.3f}"),
                  ("EER", f"{100 * _eer:.2f} %")]
    sidebar_panel("Detection", _rows)
