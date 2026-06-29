# -*- coding: utf-8 -*-
"""
modes/_mode_cnn.py — "CNN" mode of the Benchmark page: train the 2-D CNN live and
visualise how it learns — per-epoch loss curves, LR scheduler, and convolutional
activation maps.

Training configuration lives in the MAIN area (bordered panel) — dropdowns and
sliders are awkward in a narrow sidebar, and the configuration IS the page's
primary content before a model exists. The sidebar carries navigation +
environment status only.

Dispatched from app_pages/2_Benchmark.py via runpy; set_page_config and PAGE_CSS
are applied in app.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402
import librosa  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from sklearn.model_selection import train_test_split  # noqa: E402
from src.data_loader import (  # noqa: E402
    LABEL_BONAFIDE, LABEL_SPOOF, ASVspoofTorchDataset, stratified_subsample,
)
from src.metrics import calculate_eer, calculate_min_dcf  # noqa: E402
from src.models import model_for_arch  # noqa: E402
from src.jobs import cnn_epochs, request_cancel, submit_cnn_training  # noqa: E402
from src.pipeline import evaluate_cnn_on_set  # noqa: E402
from src.reporting import (  # noqa: E402
    COL_ACCURACY, COL_EER, COL_MIN_DCF, COL_MODEL,
)
from src.ui_helpers import (  # noqa: E402
    BONAFIDE_COLOR, EVAL_CORPUS_CHOICES, SPOOF_COLOR,
    available_pretrained_models, corpus_available, demo_corpus_notice,
    eval_corpora_for, eval_score_controls,
    fig_activation_grid, fig_cnn_input, fig_waveform, get_extractor, get_samples,
    load_config, load_pretrained_torch, model_downloaded, op_busy_notice, op_in_progress,
    running_on_gpu, show_empty_state, sidebar_panel, test_audio_cta,
)

config    = load_config()
extractor = get_extractor()
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dev_label = "CUDA GPU" if device.type == "cuda" else "CPU"

st.title("CNN Learning")
st.caption(
    "Train the 2-D convolutional network on STFT-dB spectrograms and inspect "
    "its learning dynamics and internal representations in real time."
)

# ── Activation-map handoff from Detection Analysis ─────────────────────────── #
# When the user clicks "See full maps in CNN Learning" on an uploaded clip, the
# per-block activations (already computed there) are stashed in session_state;
# render the FULL feature-map grids for that clip and short-circuit the page.
_handoff = st.session_state.get("cnn_handoff")
if _handoff:
    if st.button("← Back to CNN training", key="cnn_handoff_back"):
        st.session_state.pop("cnn_handoff", None)
        st.rerun()
    st.markdown(
        f"Full convolutional activation maps for your uploaded clip "
        f"**{_handoff.get('fname') or 'audio'}** — every feature map of every "
        f"block, for each CNN that judged it in Detection Analysis."
    )
    _items = _handoff.get("items", [])
    _names = [it["name"] for it in _items]
    # Pick WHICH model's maps to show, instead of stacking all of them down the
    # page. Dropdown-only (no typing, via the nosearch container); the option
    # order is the canonical 5-Block → +SE → ResNet → ResNeXt → CRNN set upstream.
    with st.container(key="nosearch_cnn_handoff"):
        _pick = (st.selectbox("Show maps for", _names, key="cnn_handoff_pick")
                 if len(_items) > 1 else (_names[0] if _names else None))
    for _item in [it for it in _items if it["name"] == _pick]:
        _col = SPOOF_COLOR if _item["p"] >= 0.5 else BONAFIDE_COLOR
        _pred = "spoof" if _item["p"] >= 0.5 else "bonafide"
        st.markdown(
            f"### {_item['name']} — <span style='color:{_col};font-weight:700;'>"
            f"p(spoof) = {_item['p']:.3f} ({_pred} predicted)</span>",
            unsafe_allow_html=True,
        )
        for _i, _act in enumerate(_item["acts"], start=1):
            _nch = _act.shape[1]
            st.pyplot(
                fig_activation_grid(
                    _act[0].numpy(),
                    f"Conv Block {_i} — {_nch} feature maps {tuple(_act.shape[1:])}"),
                clear_figure=True,
            )
        st.divider()
    st.stop()

# On the corpus-less web demo we DON'T bail out: the page renders like local but
# training is off — you can still evaluate the pretrained CNNs on eval clips
# streamed from Hugging Face (see eval_corpora_for).
_web = not corpus_available()
if _web:
    demo_corpus_notice(
        "CNN — evaluation only in the web demo",
        "Training the network needs a GPU and the full ASVspoof corpus, so on the "
        "public cloud <b>training is disabled</b>. You can still <b>Evaluate</b> "
        "the pretrained deep models (ResNet + SE, 5-Block CNN, CRNN, …) on eval clips "
        "streamed from Hugging Face — the same view you get locally.",
    )

# ===========================================================================
# Training configuration — main-area panel
# ===========================================================================

ARCH_CNN    = "5-Block CNN"
ARCH_RESNET = "ResNet + SE"
ARCH_CRNN   = "CRNN"

# Right column tweaks: the Evaluate/Score box fills the full column width and
# the Train CNN button matches the height of the classifier row (like Classic).
st.markdown("""
<style>
[class*="st-key-evalgrp_cnn"] { width: 100% !important; justify-content: space-between; }
[class*="st-key-trainbtn_cnn"] button { min-height: 3.4rem; }
[class*="st-key-advbtn_cnn"]   button { min-height: 3.4rem; }
</style>
""", unsafe_allow_html=True)

with st.container(border=True):
    st.markdown('<div class="section-label">Training configuration</div>',
                unsafe_allow_html=True)

    _is_busy = op_in_progress()

    # Row 1: Architecture | Train on / Evaluate on / Score on controls
    _r1l, _r1r = st.columns(2, gap="large")
    with _r1l:
        with st.container(key="nosearch_arch"):
            arch = st.selectbox(
                "Architecture", [ARCH_CNN, ARCH_RESNET, ARCH_CRNN], key="cnn_arch",
            )
    with _r1r:
        _train_lbl = "Trained on" if "cnn_history" in st.session_state else "Train on"
        eval_corpus, score_split = eval_score_controls("cnn", train_label=_train_lbl)
        _busy = op_busy_notice()

    # Row 2: Advanced + Clear | Train CNN + Evaluate (same row → natural alignment)
    _r2l, _r2r = st.columns(2, gap="large")
    with _r2l:
        with st.container(key="advbtn_cnn"):
            with st.popover("Advanced", icon=":material/tune:", width="stretch"):
                _a1, _a2 = st.columns(2)
                with _a1:
                    subset = st.number_input(
                        "Files / subset", 20, 25400, 2000, step=100,
                        help="2019 LA train has ~25 380 files. MORE DATA = LOWER error. "
                             "On CPU keep ≤ 1 000; on GPU the full set is fine.")
                    epochs = st.number_input("Max epochs", 1, 30,
                                             int(config["train_params"]["epochs"]))
                    batch_size = st.selectbox(
                        "Batch size", [8, 16, 32, 64],
                        index=[8, 16, 32, 64].index(int(config["train_params"]["batch_size"])))
                with _a2:
                    seed = st.number_input("Seed", 0, value=42, step=1)
                    patience = st.number_input(
                        "Patience", 1, 10,
                        int(config["train_params"].get("early_stopping_patience", 3)),
                        help="Early-stopping patience in epochs.")
                    _lr_opts = [0.0001, 0.0005, 0.001, 0.005, 0.01]
                    lr = st.selectbox(
                        "Learning rate", _lr_opts,
                        index=_lr_opts.index(float(config["train_params"]["lr"])),
                        format_func=lambda v: f"{v:g}")
                augment = st.toggle(
                    "SpecAugment (recommended)", value=True,
                    help="Random time + frequency masking on each training "
                         "spectrogram. Strongly reduces overfitting on small subsets.")
                # Architecture-specific switches.
                use_se = use_resnext = False
                if arch == ARCH_CNN:
                    use_se = st.toggle(
                        "Enable Squeeze-and-Excitation (SE) block", value=False,
                        help="Inserts an SE channel-attention gate after the ReLU of "
                             "every convolutional block, re-weighting each frequency "
                             "channel by its discriminative importance.")
                elif arch == ARCH_RESNET:
                    use_resnext = st.toggle(
                        "ResNeXt mode (grouped convolution)", value=False,
                        help="Splits the residual convolutions into 32 independent "
                             "groups (cardinality), an advanced ResNet variant that "
                             "decorrelates feature paths.")
        clear_btn = st.button("Clear results", icon=":material/delete:",
                              width="stretch", disabled=_is_busy)

    # Resolve the canonical arch key from the selector + advanced toggles.
    if arch == ARCH_RESNET:
        arch_key = "resnext" if use_resnext else "resnet"
    elif arch == ARCH_CRNN:
        arch_key = "crnn"
    else:
        arch_key = "cnn_se" if use_se else "cnn"

    with _r2r:
        # Evaluate is enabled only when the selected architecture is actually on disk.
        arch_hf_key = {"cnn": "cnn5", "cnn_se": "cnn5_se", "resnet": "resnet",
                       "resnext": "resnext", "crnn": "crnn"}.get(arch_key, "cnn5")
        _hf_cnn = [e for e in available_pretrained_models() if e["kind"] == "cnn"]
        # Locally we require the checkpoint to be on disk; on the web demo the
        # registry entries are downloadable, so allow eval (it fetches on click).
        _hf_for_arch = [e for e in _hf_cnn if e["key"] == arch_hf_key
                        and (model_downloaded(e) or _web)]
        _has_model_for_eval = "cnn_model" in st.session_state or bool(_hf_for_arch)
        with st.container(key="trainbtn_cnn"):
            train_btn = st.button(
                "Train CNN", type="primary",
                icon=":material/play_arrow:", width="stretch",
                disabled=_busy or not running_on_gpu(),
                help=None if running_on_gpu() else "GPU required for CNN training.",
            )
        eval_btn = st.button(
            "Evaluate", icon=":material/query_stats:", width="stretch",
            disabled=_busy or not _has_model_for_eval,
            help=None if _has_model_for_eval else
                 "Train a model first, or configure HF_BASE_URL for pretrained weights.",
        )

train_corpus = "2019 LA"

if clear_btn:
    for _k in ["cnn_history", "cnn_model", "cnn_dev", "cnn_results",
               "cnn_runs", "cnn_arch_trained", "cnn_train_corpus"]:
        st.session_state.pop(_k, None)
    st.rerun()

# ===========================================================================
# Architecture & design panels — defined as reusable renderers so they stay
# visible ALWAYS (before training AND after, alongside the result tabs).
# ===========================================================================

def _render_architecture(arch_key):
    if True:
        _is_resid = arch_key in ("resnet", "resnext")
        col_desc, col_table = st.columns([1, 2], gap="medium")
        with col_desc:
            if _is_resid:
                st.markdown("##### ResNet + SE"
                            + (" → ResNeXt" if arch_key == "resnext" else ""))
                st.markdown(
                    "Four **residual blocks** with **Squeeze-and-Excitation "
                    "channel attention**. Residual connections prevent gradient "
                    "vanishing; SE gates re-weight each frequency channel by its "
                    "discriminative importance — critical for generalising to "
                    "unseen TTS/VC attacks (ASVspoof 2021+)."
                )
                if arch_key == "resnext":
                    st.markdown(
                        "**ResNeXt mode** is an advanced ResNet variant that "
                        "introduces **cardinality**, splitting the channels of "
                        "each residual convolution into 32 independent groups "
                        "(grouped convolutions) to decorrelate feature paths."
                    )
            elif arch_key == "crnn":
                st.markdown("##### CRNN")
                st.markdown(
                    "A 5-block convolutional extractor learns local time-frequency "
                    "patterns; instead of pooling the time axis away, the per-frame "
                    "feature vectors feed a **bidirectional GRU** that models their "
                    "**temporal** evolution (forward + backward context) before the "
                    "final logit — capturing how artefacts unfold over time."
                )
            else:
                st.markdown("##### 5-Block CNN"
                            + (" + SE" if arch_key == "cnn_se" else ""))
                st.markdown(
                    "Five convolutional blocks (channels 16 → 32 → 64 → 128 → 256) "
                    "progressively extract discriminative time-frequency patterns "
                    "from fixed-size **128 × 300** STFT-dB spectrograms (z-score "
                    "normalised per sample)."
                )
                if arch_key == "cnn_se":
                    st.markdown(
                        "**SE enabled:** a Squeeze-and-Excitation gate after every "
                        "block re-weights each channel by its global importance."
                    )

            try:
                _m = model_for_arch(arch_key,
                                    dropout=float(config["train_params"]["dropout"]))
                n_params = sum(p.numel() for p in _m.parameters())
                st.caption(f"Trainable parameters: **{n_params:,}**")
                del _m
            except Exception:
                pass

        with col_table:
            _se = " → SE" if arch_key == "cnn_se" else ""
            _grp = " (grouped, 32 groups)" if arch_key == "resnext" else ""
            if _is_resid:
                arch_rows = [
                    {"Block / Layer": "Input",            "Output shape": "(1, 1, 128, 300)",   "Description": "z-scored STFT-dB spectrogram"},
                    {"Block / Layer": "Res Block 1",      "Output shape": "(1, 32, 64, 150)",   "Description": f"Conv3×3{_grp} → BN → ReLU → Conv3×3 → BN + skip → SE → MaxPool"},
                    {"Block / Layer": "Res Block 2",      "Output shape": "(1, 64, 32, 75)",    "Description": f"Conv3×3{_grp} → BN → ReLU → Conv3×3 → BN + proj-skip → SE → MaxPool"},
                    {"Block / Layer": "Res Block 3",      "Output shape": "(1, 128, 16, 37)",   "Description": f"Conv3×3{_grp} → BN → ReLU → Conv3×3 → BN + proj-skip → SE → MaxPool"},
                    {"Block / Layer": "Res Block 4",      "Output shape": "(1, 128, 8, 18)",    "Description": f"Conv3×3{_grp} → BN → ReLU → Conv3×3 → BN + identity → SE → MaxPool"},
                    {"Block / Layer": "AdaptiveAvgPool",  "Output shape": "(1, 128, 4, 8)",     "Description": "Adaptive average pooling to fixed spatial size"},
                    {"Block / Layer": "Dropout + Linear", "Output shape": "(1,)",               "Description": "BCEWithLogitsLoss → p(spoof) via sigmoid"},
                ]
            elif arch_key == "crnn":
                arch_rows = [
                    {"Block / Layer": "Input",            "Output shape": "(1, 1, 128, 300)",  "Description": "z-scored STFT-dB spectrogram"},
                    {"Block / Layer": "Conv Blocks 1-5",  "Output shape": "(1, 256, 4, 9)",    "Description": "5× (Conv2d 3×3 → BatchNorm → ReLU → MaxPool2d 2×2)"},
                    {"Block / Layer": "Freq pool + reshape", "Output shape": "(1, 9, 1024)",   "Description": "AdaptiveAvgPool freq→4, per-frame vectors over time"},
                    {"Block / Layer": "BiGRU",            "Output shape": "(1, 9, 256)",       "Description": "Bidirectional GRU (hidden 128 × 2) over the time axis"},
                    {"Block / Layer": "Temporal mean",    "Output shape": "(1, 256)",          "Description": "Mean-pool recurrent states over time"},
                    {"Block / Layer": "Dropout + Linear", "Output shape": "(1,)",              "Description": "BCEWithLogitsLoss → p(spoof) via sigmoid"},
                ]
            else:
                arch_rows = [
                    {"Block / Layer": "Input",            "Output shape": "(1, 1, 128, 300)",  "Description": "z-scored STFT-dB spectrogram"},
                    {"Block / Layer": "Conv Block 1",     "Output shape": "(1, 16, 64, 150)",  "Description": f"Conv2d 3×3 → BatchNorm → ReLU{_se} → MaxPool2d 2×2"},
                    {"Block / Layer": "Conv Block 2",     "Output shape": "(1, 32, 32, 75)",   "Description": f"Conv2d 3×3 → BatchNorm → ReLU{_se} → MaxPool2d 2×2"},
                    {"Block / Layer": "Conv Block 3",     "Output shape": "(1, 64, 16, 37)",   "Description": f"Conv2d 3×3 → BatchNorm → ReLU{_se} → MaxPool2d 2×2"},
                    {"Block / Layer": "Conv Block 4",     "Output shape": "(1, 128, 8, 18)",   "Description": f"Conv2d 3×3 → BatchNorm → ReLU{_se} → MaxPool2d 2×2"},
                    {"Block / Layer": "Conv Block 5",     "Output shape": "(1, 256, 4, 9)",    "Description": f"Conv2d 3×3 → BatchNorm → ReLU{_se} → MaxPool2d 2×2"},
                    {"Block / Layer": "AdaptiveAvgPool",  "Output shape": "(1, 256, 4, 8)",    "Description": "Adaptive average pooling to fixed spatial size"},
                    {"Block / Layer": "Dropout + Linear", "Output shape": "(1,)",              "Description": "BCEWithLogitsLoss → p(spoof) via sigmoid"},
                ]
            st.dataframe(
                pd.DataFrame(arch_rows), width="stretch", hide_index=True,
                column_config={
                    "Block / Layer": st.column_config.TextColumn(width="small"),
                    # Keep the shape column tight so the Description gets the room
                    # it needs and stops wrapping/clipping.
                    "Output shape": st.column_config.TextColumn(width="small"),
                    "Description": st.column_config.TextColumn(width="large"),
                },
            )


def _render_design(arch_key):
    if True:
        def _choice_card(title: str, body: str, tag: str = "") -> None:
            tag_html = f'<span class="ic-tag">{tag}</span>' if tag else ""
            st.markdown(
                f'<div class="info-card">'
                f'<div class="ic-title">{title}{tag_html}</div>'
                f'<p class="ic-body">{body}</p></div>',
                unsafe_allow_html=True,
            )

        dc1, dc2 = st.columns(2, gap="medium")
        with dc1:
            if arch_key in ("resnet", "resnext"):
                _choice_card(
                    "Residual connections",
                    "Skip connections add each block's input back to its output, "
                    "preventing vanishing gradients and letting every block learn "
                    "only the <em>residual</em> correction instead of the full mapping.",
                    tag="ResNet",
                )
                _choice_card(
                    "SE channel attention",
                    "A small MLP gates each frequency channel by its global "
                    "importance — the network suppresses codec roll-off and noise "
                    "bands while amplifying band-specific synthesis artefacts. Key "
                    "for 2021+ attacks.",
                    tag="ResNet",
                )
            if arch_key == "resnext":
                _choice_card(
                    "Cardinality (grouped conv)",
                    "Each residual convolution is split into 32 independent groups, "
                    "an extra design axis besides depth and width that decorrelates "
                    "feature paths and improves generalisation at a similar cost.",
                    tag="ResNeXt",
                )
            if arch_key == "cnn_se":
                _choice_card(
                    "SE channel attention",
                    "An SE gate after every convolutional block re-weights each "
                    "frequency channel by its global importance, amplifying "
                    "band-specific synthesis artefacts.",
                    tag="SE",
                )
            if arch_key == "crnn":
                _choice_card(
                    "Bidirectional GRU",
                    "A recurrent layer reads the per-frame convolutional features "
                    "forward and backward, modelling how synthesis artefacts evolve "
                    "<em>over time</em> instead of collapsing the time axis.",
                    tag="CRNN",
                )
            _choice_card(
                "STFT-dB input",
                "Captures both low-frequency prosodic cues and the high-frequency "
                "synthesis artefacts that 1-D features miss.",
            )
            _choice_card(
                "SpecAugment",
                "Random time and frequency strips of each training spectrogram are "
                "masked with the mean value, forcing globally robust patterns and "
                "cutting overfitting on small subsets.",
            )
        with dc2:
            _choice_card(
                "BatchNorm + Dropout",
                "Stabilises optimisation on small subsets and regularises against "
                "the heavily imbalanced ASVspoof corpus.",
            )
            _choice_card(
                "BCEWithLogitsLoss + pos_weight",
                "Up-weights the minority bonafide class in the gradient, "
                "compensating the 9:1 spoof/bonafide imbalance.",
            )
            _choice_card(
                "Best-checkpoint restore",
                "Weights from the epoch with the lowest validation loss are "
                "restored at the end — early stopping never returns a degraded model.",
            )
            _choice_card(
                "ReduceLROnPlateau",
                "Halves the learning rate whenever validation loss plateaus, "
                "enabling fine-grained convergence without manual tuning.",
            )

# ===========================================================================
# Training
# ===========================================================================

if train_btn:
    # Training always uses the official 2019 LA train/dev splits.
    train_samples = stratified_subsample(get_samples("train"), int(subset), int(seed))
    dev_samples   = stratified_subsample(get_samples("dev"),   int(subset), int(seed) + 1)

    params = dict(config["train_params"])
    params.update({
        "semilla":                 int(seed),
        "epochs":                  int(epochs),
        "batch_size":              int(batch_size),
        "lr":                      float(lr),
        "early_stopping_patience": int(patience),
        "augment":                 bool(augment),
        "arch":                    arch_key,
    })

    # Train ONLY — no eval corpus scoring (use Evaluate button for that).
    # The background worker still uses dev_samples for val loss / early stopping,
    # but we mark train_only so app.py skips adding rows to cnn_runs.
    st.session_state["cnn_train_only"] = True
    st.session_state["cnn_future"] = submit_cnn_training(
        train_samples=train_samples, dev_samples=dev_samples,
        extractor=extractor, params=params, eval_sets=[],
    )
    st.session_state["cnn_pending"] = {"dev": dev_samples, "arch": arch,
                                       "corpus": train_corpus}
    st.session_state["op_running"] = True
    st.session_state["cnn_focus_curves"] = True   # redirect to Training curves
    st.rerun()

if eval_btn:
    _eval_params = dict(config["train_params"])
    # Resolve which model to use: prefer session-trained, fall back to HF pretrained.
    _model = st.session_state.get("cnn_model")
    _arch_lbl = st.session_state.get("cnn_arch_trained", arch)
    if _model is None:
        _entry = next((e for e in _hf_for_arch), None)
        if _entry is None:
            st.error("No CNN model available. Train first or configure HF_BASE_URL.")
            st.stop()
        _model, _ = load_pretrained_torch(_entry["file"], _entry["url"], _entry["name"])
        _arch_lbl = _entry["name"]

    _new_rows = []
    if score_split in ("Dev", "Dev + Eval"):
        _dev_samps = get_samples("dev")
        if subset > 0:
            _dev_samps = stratified_subsample(_dev_samps, int(subset), int(seed) + 1)
        if _dev_samps:
            with st.spinner("Evaluating on 2019 LA dev…"):
                _new_rows += evaluate_cnn_on_set(
                    _model, _dev_samps, extractor, _eval_params,
                    corpus_label="", arch_label=_arch_lbl, suffix="",
                )
    if score_split in ("Eval", "Dev + Eval"):
        for _lbl, _samps in eval_corpora_for(eval_corpus):
            if subset > 0:
                _samps = stratified_subsample(_samps, int(subset), int(seed) + 2)
            if _samps:
                with st.spinner(f"Evaluating on {_lbl}…"):
                    _new_rows += evaluate_cnn_on_set(
                        _model, _samps, extractor, _eval_params,
                        corpus_label=_lbl, arch_label=_arch_lbl,
                    )
    if _new_rows:
        st.session_state.setdefault("cnn_runs", []).extend(_new_rows)
        st.session_state["cnn_results"] = _new_rows
    st.rerun()

# ── Live training view — a self-refreshing fragment that redraws the loss curve
#    from the background worker's epoch records every 2 s (only this block reruns,
#    not the whole app). The running MESSAGE lives in the sidebar banner. ─────── #
@st.fragment(run_every=2.0)
def _live_training_view(max_epochs):
    fut = st.session_state.get("cnn_future")
    if fut is None or fut.done():
        return                                   # banner fragment does the full rerun
    st.subheader("Live training")
    _epochs = cnn_epochs()
    if _epochs:
        _long = (pd.DataFrame(_epochs)
                 .melt(id_vars="epoch", value_vars=["train_loss", "val_loss"],
                       var_name="curve", value_name="loss"))
        _live = (alt.Chart(_long).mark_line(point=True).encode(
                    x=alt.X("epoch:Q", title="epoch", axis=alt.Axis(tickMinStep=1)),
                    y=alt.Y("loss:Q", title="loss"),
                    color=alt.Color("curve:N", title=None,
                                    scale=alt.Scale(domain=["train_loss", "val_loss"],
                                                    range=["#4F8BF9", "#EF5350"])))
                 .properties(height=300))
        st.altair_chart(_live, width="stretch")
        _last = _epochs[-1]
        st.caption(f"Epoch {_last['epoch']}/{int(max_epochs)}  "
                   f"train={_last['train_loss']:.4f}  val={_last['val_loss']:.4f}  "
                   f"lr={_last['lr']:.2e}")
    else:
        st.caption("Preparing data and starting the first epoch…")
    if st.button("Cancel training", icon=":material/cancel:", key="cnn_cancel"):
        request_cancel()
        st.toast("Cancelling… stops after the current epoch.")
        st.rerun(scope="app")


if st.session_state.get("cnn_error"):
    st.error(f"CNN training failed: {st.session_state.pop('cnn_error')}")
if st.session_state.pop("cnn_cancelled", False):
    st.info("CNN training cancelled.", icon=":material/cancel:")

# ===========================================================================
# Unified panels — Architecture & Design choices are ALWAYS shown. The live
# training curve and the results appear as ADDITIONAL tabs (they never push the
# overview down): switch tabs freely while a model trains in the background,
# watch the curve in "Training curves", and the final graphs stay there after.
# ===========================================================================

_cnn_fut   = st.session_state.get("cnn_future")
_training  = _cnn_fut is not None and not _cnn_fut.done()
_has_model = "cnn_history" in st.session_state
_cnn_runs  = st.session_state.get("cnn_runs", [])

# Stable order at all times — "Training curves" is always the 3rd tab (the panel
# order never reshuffles). When a training has just been launched we redirect to
# it with a tiny script (st.tabs has no programmatic selection), but we do NOT
# move it to the front.
_tab_names = ["Architecture", "Design choices"]
if _training or _has_model:
    _tab_names.append("Training curves")
if _cnn_runs:
    _tab_names.append("Results")
if _has_model:
    _tab_names.append("Activation maps")

_tabmap = dict(zip(_tab_names, st.tabs(_tab_names)))

# One-shot redirect to the live curve right after launching a training.
if _training and st.session_state.pop("cnn_focus_curves", False):
    st.iframe(
        """
<script>
(function(){
  var doc; try { doc = window.parent.document; } catch(e) { return; }
  function clickCurves(){
    var tabs = doc.querySelectorAll('button[role="tab"]');
    for (var i = 0; i < tabs.length; i++){
      if ((tabs[i].textContent || '').trim() === 'Training curves'){ tabs[i].click(); return true; }
    }
    return false;
  }
  var n = 0, iv = window.parent.setInterval(function(){
    if (clickCurves() || ++n > 25) window.parent.clearInterval(iv);
  }, 120);
})();
</script>
""",
        height=1,
    )

with _tabmap["Architecture"]:
    _render_architecture(arch_key)
with _tabmap["Design choices"]:
    _render_design(arch_key)

# ── Training curves — live while training, fixed (non-zoomable) afterwards ─── #
if "Training curves" in _tabmap:
    with _tabmap["Training curves"]:
        if _training:
            _live_training_view(int(epochs))
        else:
            _hist = st.session_state["cnn_history"]
            cc1, cc2 = st.columns([2, 1])
            with cc1:
                st.markdown("**Loss curves — train vs. validation**")
                _ldf = (pd.DataFrame(_hist)
                        .melt(id_vars="epoch", value_vars=["train_loss", "val_loss"],
                              var_name="curve", value_name="loss"))
                _lc = (alt.Chart(_ldf).mark_line(point=True).encode(
                        x=alt.X("epoch:Q", title="epoch", axis=alt.Axis(tickMinStep=1)),
                        y=alt.Y("loss:Q", title="loss"),
                        color=alt.Color("curve:N", title=None,
                                        scale=alt.Scale(domain=["train_loss", "val_loss"],
                                                        range=["#4F8BF9", "#EF5350"])))
                       .properties(height=300))
                st.altair_chart(_lc, width="stretch")
            with cc2:
                st.markdown("**Learning rate schedule**")
                _rdf = pd.DataFrame(_hist)[["epoch", "lr"]]
                _rc = (alt.Chart(_rdf).mark_line(point=True, color="#26C6DA").encode(
                        x=alt.X("epoch:Q", title="epoch", axis=alt.Axis(tickMinStep=1)),
                        y=alt.Y("lr:Q", title="learning rate"))
                       .properties(height=300))
                st.altair_chart(_rc, width="stretch")
                st.caption("Flat = stable LR. Drops = ReduceLROnPlateau halved "
                           "the rate after a validation-loss plateau.")

# ── Results — every CNN scored this session (both architectures, dev + eval) ─ #
if "Results" in _tabmap:
    with _tabmap["Results"]:
        st.markdown(
            "Every CNN scored this session — both architectures, on **dev** "
            "(seen attacks A01–A06) and **eval** (unseen A07–A19). Models come "
            "from this page and from the Benchmark's full comparison."
        )
        _rrows = []
        for _r in _cnn_runs:
            _name   = str(_r.get(COL_MODEL, ""))
            _corpus = str(_r.get("Corpus", "")).strip()
            _split  = _r.get("Split") or (
                (f"eval · {_corpus}" if _corpus else "eval")
                if "[EVAL]" in _name else "dev")
            for _m in ("[EVAL]", "[CPU]", "[CUDA]"):
                _name = _name.replace(_m, "")
            _rrows.append({
                "Model":    _name.strip(),
                "Split":    _split,
                "minDCF":   pd.to_numeric(_r.get(COL_MIN_DCF), errors="coerce"),
                "EER (%)":  pd.to_numeric(_r.get(COL_EER), errors="coerce"),
                "Accuracy": pd.to_numeric(_r.get(COL_ACCURACY), errors="coerce"),
            })
        _rdf = (pd.DataFrame(_rrows)
                .sort_values(["minDCF", "EER (%)"], na_position="last")
                .reset_index(drop=True))
        st.dataframe(
            _rdf.style.format({"EER (%)": "{:.2f}", "minDCF": "{:.3f}",
                               "Accuracy": "{:.4f}"}),
            width="stretch", hide_index=True,
        )
        st.caption("Ranked by minDCF (primary metric), EER as tiebreaker. After "
                   "a full comparison both CNNs appear here, dev and eval.")

# ── Activation maps (dev sample) — only when a model was trained this session ─ #
if "Activation maps" in _tabmap:
    model = st.session_state["cnn_model"]
    dev   = st.session_state["cnn_dev"]
    with _tabmap["Activation maps"]:
        st.markdown(
            "Each convolutional block transforms the input spectrogram into a "
            "stack of feature maps. **Block 1** detects local edges; "
            "the last block encodes globally discriminative patterns."
        )

        bonafide_pool = [p for p, e in dev if e == LABEL_BONAFIDE]
        spoof_pool    = [p for p, e in dev if e == LABEL_SPOOF]

        act_choice = st.radio(
            "Sample class to inspect",
            ["bonafide (real)", "spoof (deepfake)"],
            horizontal=True, key="cnn_act_class",
        )
        pool = bonafide_pool if act_choice.startswith("bonafide") else spoof_pool

        if not pool:
            st.warning("No sample of that class in the current dev subset.")
        else:
            path   = pool[0]
            signal = extractor.load_audio(path)
            matrix = extractor.get_spectrogram_matrix(signal)
            tensor = (
                torch.from_numpy(matrix)
                .unsqueeze(0).unsqueeze(0)
                .float().to(device)
            )

            with torch.no_grad():
                logit, activations = model.forward_with_activations(tensor)
            prob = torch.sigmoid(logit).item()

            color = SPOOF_COLOR if prob >= 0.5 else BONAFIDE_COLOR
            pred  = "spoof" if prob >= 0.5 else "bonafide"
            st.markdown(
                f"`{os.path.basename(path)}` → "
                f"<span style='color:{color};font-weight:600;'>"
                f"p(spoof) = {prob:.3f}  ({pred} predicted)</span>",
                unsafe_allow_html=True,
            )

            for i, act in enumerate(activations, start=1):
                n_ch = act.shape[1]
                st.pyplot(
                    fig_activation_grid(
                        act[0].numpy(),
                        f"Conv Block {i} — {n_ch} feature maps (shape {tuple(act.shape[1:])})",
                    ),
                    clear_figure=True,
                )


# ── Sidebar: live model state (rendered last, when results exist) ───────── #

with st.sidebar:
    _rows = [("Device", dev_label)]
    if "cnn_history" in st.session_state:
        _hist = st.session_state["cnn_history"]
        _res  = st.session_state.get("cnn_results", [])   # may be absent in train-only mode
        _last = _res[-1] if _res else {}
        _rows += [
            ("Status",   "Trained"),
            ("Arch",     st.session_state.get("cnn_arch_trained", "—")),
            ("Corpus",   st.session_state.get("cnn_train_corpus", "—")),
            ("Epochs",   str(len(_hist))),
            ("Best val", f"{min(r['val_loss'] for r in _hist):.4f}"),
            ("minDCF dev", f'{_last.get(COL_MIN_DCF, "—")}'),
            ("EER dev",  f'{_last.get(COL_EER, "—")} %'),
        ]
    else:
        _rows += [("Status", "Not trained yet")]
    sidebar_panel("Model state", _rows)
