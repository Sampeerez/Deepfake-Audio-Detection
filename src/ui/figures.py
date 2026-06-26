# -*- coding: utf-8 -*-
"""src/ui/figures.py — matplotlib figure builders and signal statistics.

Split out of src/ui_helpers.py and re-exported there for backward compatibility.
The active-theme palette globals (_FIG_*) and apply_mpl_theme live here together
with the fig_* builders that read them, so the Light/Dark swap stays intra-module
(no cross-module global mutation).
"""

from typing import Dict, List, Optional, Tuple

import librosa
import librosa.display
import matplotlib
import numpy as np
from scipy.fft import dct

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data_loader import LABEL_BONAFIDE, LABEL_SPOOF  # noqa: E402
from src.features import FeatureExtractor  # noqa: E402
from src.ui.styles import BONAFIDE_COLOR, NEUTRAL_COLOR, SPOOF_COLOR  # noqa: E402

_EPS = 1e-10

# ── matplotlib figure palettes ────────────────────────────────────────────── #
# Two themes: Dark Side (default, matches the dark chrome) and Light Side. The
# data-semantic colours (BONAFIDE/SPOOF above) stay constant across both so the
# charts read the same; only the canvas/text/grid flip.
_DARK_FIG  = {"bg": "#161C2D", "axes": "#1E2640", "grid": "#263050",
              "text": "#C5CDE8", "edge": "#2E3A58"}
_LIGHT_FIG = {"bg": "#FFFFFF", "axes": "#F4F7FD", "grid": "#D5DEEF",
              "text": "#27324C", "edge": "#B7C3DC"}

# Active palette: module globals so the fig_* helpers below pick up the current
# theme at call time (they read these names when a figure is built). app.py calls
# apply_mpl_theme() once per rerun before any page draws; default to dark so any
# figure built at import time still renders correctly.
_FIG_BG   = _DARK_FIG["bg"]
_FIG_AXES = _DARK_FIG["axes"]
_FIG_GRID = _DARK_FIG["grid"]
_FIG_TEXT = _DARK_FIG["text"]
_FIG_EDGE = _DARK_FIG["edge"]


def apply_mpl_theme(theme: str = "dark") -> None:
    """Restyle every matplotlib figure for the active side (Light/Dark). Called
    from app.py on each rerun before the page runs, so all fig_* helpers match
    the chosen theme with no per-figure changes."""
    global _FIG_BG, _FIG_AXES, _FIG_GRID, _FIG_TEXT, _FIG_EDGE
    pal = _LIGHT_FIG if theme == "light" else _DARK_FIG
    _FIG_BG, _FIG_AXES = pal["bg"], pal["axes"]
    _FIG_GRID, _FIG_TEXT, _FIG_EDGE = pal["grid"], pal["text"], pal["edge"]
    plt.rcParams.update({
        "figure.facecolor":  _FIG_BG,
        "axes.facecolor":    _FIG_AXES,
        "axes.edgecolor":    _FIG_EDGE,
        "axes.labelcolor":   _FIG_TEXT,
        "xtick.color":       _FIG_TEXT,
        "ytick.color":       _FIG_TEXT,
        "text.color":        _FIG_TEXT,
        "grid.color":        _FIG_GRID,
        "grid.alpha":        0.55,
        "font.size":         9,
        "legend.facecolor":  _FIG_AXES,
        "legend.edgecolor":  _FIG_EDGE,
        "figure.dpi":        110,
    })


apply_mpl_theme("dark")
# ===========================================================================
# Signal statistics
# ===========================================================================

def compute_signal_stats(y: np.ndarray, sr: int) -> Dict:
    """Return a dict with key audio statistics for display as metric cards."""
    duration   = len(y) / sr
    rms        = float(np.sqrt(np.mean(y ** 2)))
    rms_db     = float(20 * np.log10(rms + _EPS))   # dBFS: ~-30 to -6 for speech
    zcr        = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    centroid   = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    return {
        "duration_s":   duration,
        "rms":          rms,
        "rms_db":       rms_db,
        "zcr":          zcr,
        "centroid_hz":  centroid,
    }


# ===========================================================================
# Plotting helpers — all return a plt.Figure for st.pyplot()
# ===========================================================================

def _fig_style(ax: plt.Axes) -> None:
    """Apply a clean, dark-theme-consistent style to any axis."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(_FIG_EDGE)
    ax.grid(True, alpha=0.4, linewidth=0.5, color=_FIG_GRID)
    ax.tick_params(labelsize=8, colors=_FIG_TEXT)


def fig_waveform(
    y: np.ndarray,
    sr: int,
    title: str = "Waveform",
    label: Optional[str] = None,
    figsize: Tuple[float, float] = (9, 2.5),
) -> plt.Figure:
    """Time-domain amplitude plot, colour-coded by class label.

    ``figsize`` is exposed so callers that place the waveform next to another
    figure (e.g. the CNN input) can match heights exactly."""
    if label and "spoof" in str(label).lower():
        color = SPOOF_COLOR
    elif label and "bonafide" in str(label).lower():
        color = BONAFIDE_COLOR
    else:
        color = NEUTRAL_COLOR

    fig, ax = plt.subplots(figsize=figsize)
    t = np.arange(len(y)) / sr
    ax.fill_between(t, y, alpha=0.22, color=color)
    ax.plot(t, y, linewidth=0.6, color=color)
    ax.axhline(0, color="#aaa", linewidth=0.4, linestyle="--", alpha=0.6)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Amplitude", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.margins(x=0)
    _fig_style(ax)
    fig.tight_layout()
    return fig


def _specshow(
    matrix: np.ndarray,
    sr: int,
    hop_length: int,
    title: str,
    y_axis: Optional[str] = None,
    cmap: str = "magma",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 3.2))
    img = librosa.display.specshow(
        matrix, sr=sr, hop_length=hop_length,
        x_axis="time", y_axis=y_axis, ax=ax, cmap=cmap,
    )
    fig.colorbar(img, ax=ax, format="%+.0f", pad=0.01)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def fig_stft_db(y: np.ndarray, extractor: FeatureExtractor) -> plt.Figure:
    mag = np.abs(librosa.stft(y, n_fft=extractor.n_fft,
                              hop_length=extractor.hop_length, window="hann"))
    db  = librosa.amplitude_to_db(mag, ref=np.max)
    return _specshow(db, extractor.sample_rate, extractor.hop_length,
                     "STFT Magnitude (dB)", y_axis="hz")


def fig_cnn_input(y: np.ndarray, extractor: FeatureExtractor) -> plt.Figure:
    matrix = extractor.get_spectrogram_matrix(y)
    # constrained_layout packs the axes + colorbar to fill the figure width (plain
    # tight_layout left a right-hand gap with the colorbar); the figure is rendered
    # with bbox_inches=None by the caller, so its 9×3.2 canvas matches the waveform's
    # exactly → the two panels end up the same height with no empty strip.
    fig, ax = plt.subplots(figsize=(9, 3.2), layout="constrained")
    img = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
    fig.colorbar(img, ax=ax, pad=0.01)
    ax.set_title(
        f"CNN Input — z-scored STFT-dB  "
        f"({extractor.freq_bins} freq bins × {extractor.time_frames} time frames)",
        fontsize=10, fontweight="bold",
    )
    ax.set_xlabel("Time frames", fontsize=9)
    ax.set_ylabel("Frequency bins", fontsize=9)
    ax.tick_params(labelsize=8)
    return fig


def fig_mfcc(y: np.ndarray, extractor: FeatureExtractor) -> plt.Figure:
    mfcc = librosa.feature.mfcc(
        y=y, sr=extractor.sample_rate, n_mfcc=extractor.n_mfcc,
        n_fft=extractor.n_fft, hop_length=extractor.hop_length,
        n_mels=extractor.n_mels, window="hann",
    )
    return _specshow(mfcc, extractor.sample_rate, extractor.hop_length,
                     f"MFCC  ({extractor.n_mfcc} coefficients)", cmap="coolwarm")


def fig_lfcc(y: np.ndarray, extractor: FeatureExtractor) -> plt.Figure:
    power      = extractor._stft_magnitude(y) ** 2
    band_energy = extractor._linear_filterbank @ power
    log_energy  = np.log(band_energy + _EPS)
    cepstrum    = dct(log_energy, type=2, axis=0, norm="ortho")[: extractor.n_lfcc]
    return _specshow(cepstrum, extractor.sample_rate, extractor.hop_length,
                     f"LFCC  ({extractor.n_lfcc} linear cepstral coefficients)",
                     cmap="coolwarm")


def fig_cqcc(y: np.ndarray, extractor: FeatureExtractor) -> plt.Figure:
    cqt        = librosa.cqt(y, sr=extractor.sample_rate,
                             hop_length=extractor.hop_length,
                             n_bins=extractor.cqcc_n_bins,
                             bins_per_octave=extractor.cqcc_bins_per_octave)
    log_energy = np.log(np.abs(cqt) ** 2 + _EPS)
    cepstrum   = dct(log_energy, type=2, axis=0, norm="ortho")[: extractor.n_cqcc]
    return _specshow(cepstrum, extractor.sample_rate, extractor.hop_length,
                     f"CQCC  ({extractor.n_cqcc} constant-Q cepstral coefficients)",
                     cmap="coolwarm")


def fig_activation_grid(
    activation: np.ndarray,
    title: str,
    max_maps: int = 16,
) -> plt.Figure:
    """Grid of feature maps from one CNN convolutional block.

    Args:
        activation: shape (channels, H, W) for a single sample.
        title:      Block title.
        max_maps:   Maximum channels to display.
    """
    n    = min(activation.shape[0], max_maps)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i in range(len(axes)):
        ax = axes[i]
        if i < n:
            ax.imshow(activation[i], aspect="auto", origin="lower", cmap="inferno")
            ax.set_title(f"ch {i}", fontsize=7)
        ax.axis("off")
    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def fig_activation_evolution(
    acts: List,
    title: str,
    max_blocks: int = 4,
) -> plt.Figure:
    """Compact one-row summary of how a CNN transforms the spectrogram block by
    block: each convolutional block is reduced to a single heatmap (the mean over
    its channels) and laid left→right, so the overall evolution is visible at a
    glance without scrolling through every feature map.

    Args:
        acts: list of per-block activation tensors, each shaped (1, C, H, W) or
              (C, H, W) for a single sample.
        title: figure title (model name).
    """
    blocks = list(acts)[:max_blocks]
    n = max(1, len(blocks))
    # FIXED figure size regardless of block count: ResNet (4 blocks) and the
    # 3-Block CNN (3 blocks) must render at the same aspect ratio so that, placed
    # in equal-width columns, they end up exactly the same height.
    fig, axes = plt.subplots(1, n, figsize=(7.2, 2.1))
    axes = np.atleast_1d(axes).ravel()
    for i in range(len(axes)):
        ax = axes[i]
        if i < len(blocks):
            act = blocks[i]
            arr = act.numpy() if hasattr(act, "numpy") else np.asarray(act)
            if arr.ndim == 4:        # (1, C, H, W) → drop batch
                arr = arr[0]
            heat = arr.mean(axis=0)  # mean over channels → (H, W)
            ax.imshow(heat, aspect="auto", origin="lower", cmap="inferno")
            ax.set_title(f"Block {i + 1}", fontsize=8)
        ax.axis("off")
    # Title INSIDE the canvas (rendered with bbox_inches=None, anything at y>1 is
    # clipped). Keep a small top margin so it isn't cut, but sit it just above the
    # heatmaps (rect top close to the title) so it hugs the images, not floats away.
    fig.suptitle(title, fontsize=9.5, fontweight="bold", y=0.90)
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    return fig


def fig_overall_split_bar(n_bonafide: int, n_spoof: int) -> plt.Figure:
    """Tiny 100%-stacked horizontal bar of the overall bonafide/spoof share.

    Only short "NN%" labels go inside the segments (a long word would overflow
    the narrow bonafide slice); a legend below names the colours.
    """
    total = max(n_bonafide + n_spoof, 1)
    pb = 100 * n_bonafide / total
    ps = 100 * n_spoof / total
    fig, ax = plt.subplots(figsize=(4.8, 1.25))
    ax.barh([0], [pb], color=BONAFIDE_COLOR, edgecolor=_FIG_BG,
            height=0.5, label="Bonafide")
    ax.barh([0], [ps], left=[pb], color=SPOOF_COLOR, edgecolor=_FIG_BG,
            height=0.5, label="Spoof")
    ax.text(pb / 2, 0, f"{pb:.0f}%", ha="center", va="center",
            color="white", fontsize=9, fontweight="bold")
    ax.text(pb + ps / 2, 0, f"{ps:.0f}%", ha="center", va="center",
            color="white", fontsize=9, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 0.6)
    ax.axis("off")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2,
              frameon=False, fontsize=8, handlelength=1.1, columnspacing=1.6)
    ax.set_title("Overall class share · train + dev", fontsize=8.5,
                 fontweight="bold", color=_FIG_TEXT, pad=10)
    fig.tight_layout()
    return fig


def fig_corpus_overview(
    train_samples: List[Tuple[str, int]],
    dev_samples:   List[Tuple[str, int]],
) -> plt.Figure:
    """Stacked bar showing bonafide / spoof split for train and dev subsets."""
    labels = ["Train", "Dev"]
    bon = [sum(1 for _, l in s if l == LABEL_BONAFIDE)
           for s in (train_samples, dev_samples)]
    spo = [sum(1 for _, l in s if l == LABEL_SPOOF)
           for s in (train_samples, dev_samples)]
    totals = [b + s for b, s in zip(bon, spo)]

    x   = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.bar(x, bon, label="Bonafide (real)",    color=BONAFIDE_COLOR, alpha=0.9, width=0.6)
    ax.bar(x, spo, bottom=bon, label="Spoof (deepfake)", color=SPOOF_COLOR, alpha=0.9, width=0.6)

    for i, (b, s, t) in enumerate(zip(bon, spo, totals)):
        ax.text(i, t + max(totals) * 0.02, f"{t:,}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=_FIG_TEXT)
        ax.text(i, b / 2, f"{b:,}", ha="center", va="center",
                fontsize=8, color="white", fontweight="600")
        ax.text(i, b + s / 2, f"{s:,}", ha="center", va="center",
                fontsize=8, color="white", fontweight="600")

    # Headroom so the total labels never collide with the top spine.
    ax.set_ylim(0, max(totals) * 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Audio files", fontsize=9)
    ax.set_title("Class distribution by subset", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    # Legend BELOW the axes — never overlaps the bars.
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1),
              ncol=2, frameon=False, handlelength=1.2, columnspacing=1.4)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    return fig
