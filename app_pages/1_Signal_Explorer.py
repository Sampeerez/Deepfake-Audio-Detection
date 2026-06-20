# -*- coding: utf-8 -*-
"""
app_pages/1_Signal_Explorer.py — Visualise and compare audio signals at the
feature level: waveform, STFT-dB, CNN input, MFCC, LFCC, CQCC.

State model
-----------
- One to three audio sources, each a slot with a stable key prefix
  ("a_*", "b_*", "c_*"). The active count is `se_n` (default 1); "Add audio
  source" / "Remove last" change it. Removing a slot forgets its keys.
- Streamlit deletes the state of unmounted widgets, so picker keys are
  re-asserted at the top of every run to survive page navigation.
- Uploaded files are mirrored into plain session keys ("{p}_upload_name" /
  "{p}_upload_bytes") the moment they arrive: file_uploader state dies with
  the widget, plain keys do not.
- The representation picker uses ONE key ("se_views") in the main area.
"""

import io
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import librosa  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from src.features import AudioLoadError  # noqa: E402
from src.ui_helpers import (  # noqa: E402
    bundle_samples, bundled_samples, compute_signal_stats, hf_eval_samples,
    label_badge, mini_note, show_empty_state, sidebar_panel,
    fig_cnn_input, fig_cqcc, fig_lfcc, fig_mfcc, fig_stft_db, fig_waveform,
    get_extractor, get_samples, get_samples_2021_la, get_samples_2021_df,
    split_by_label, BONAFIDE_COLOR, SPOOF_COLOR,
)

extractor = get_extractor()

# Push the upload "Clear" button flush against the right edge (compact, not full
# width) — overrides the generic right-align helper that stretches widgets.
st.markdown(
    "<style>"
    "[class*='st-key-alignr_clear']{display:flex;justify-content:flex-end;}"
    "[class*='st-key-alignr_clear'] [data-testid='stElementContainer']{width:auto !important;}"
    "[class*='st-key-alignr_clear'] button{width:auto !important;}"
    "</style>",
    unsafe_allow_html=True,
)

_PLACEHOLDER = -1   # sentinel index meaning "no file selected"

# Up to three audio sources; each is a slot with a stable key prefix.
SLOTS           = ["a", "b", "c"]
MAX_SLOTS       = len(SLOTS)
SLOT_DOT_COLORS = [BONAFIDE_COLOR, SPOOF_COLOR, "#AB47BC"]

# view name → one-line description
VIEW_META = {
    "Waveform":  "Time-domain amplitude — envelope, silences, clipping",
    "STFT": "Full spectrogram — synthesis artefacts across all bands",
    "CNN Input": "z-scored STFT-dB, exactly as the CNN sees it (128×300)",
    "MFCC":      "Mel-scale cepstrum — perceptual spectral envelope",
    "LFCC":      "Linear-frequency cepstrum — strong anti-spoofing baseline",
    "CQCC":      "Constant-Q cepstrum — fine log-frequency resolution",
}
ALL_VIEWS = list(VIEW_META.keys())

_CORPUS_OPTIONS = ["2019 LA", "2021 LA", "2021 DF"]
_CORPUS_INFO = {
    "2019 LA": "Official corpus — train / dev / eval splits",
    "2021 LA": "Eval-only split · 181 k files · telephone codecs",
    "2021 DF": "Eval-only split · ≈459 k files · in-the-wild deepfakes",
}

# Widget keys re-asserted to survive unmount; upload payload lives in plain
# (non-widget) keys that persist on their own.
_PICKER_SUFFIXES   = ("_source", "_corpus", "_subset", "_class", "_file_idx")
_GLOBAL_KEYS       = ("se_views",)
# What to mirror into the cross-page memory dict (NOT the file_uploader widget
# object key, only its decoded payload).
_REMEMBER_SUFFIXES = _PICKER_SUFFIXES + ("_upload_name", "_upload_bytes")


def _persist_state() -> None:
    """Re-assert widget session values so they survive widget unmounting."""
    for key in list(st.session_state.keys()):
        is_picker = key[:2] in ("a_", "b_", "c_") and key.endswith(_PICKER_SUFFIXES)
        if is_picker or key in _GLOBAL_KEYS:
            st.session_state[key] = st.session_state[key]


def _remember() -> None:
    """Mirror selections into a plain dict that SURVIVES page navigation.

    The re-assert trick only works across reruns of THIS page; when you leave
    the page the widgets unmount and Streamlit garbage-collects their keys. So
    we also stash everything in a non-widget key (`_se_memory`) that persists,
    and reseed from it on return (`_restore`).
    """
    mem = {k: st.session_state[k] for k in list(st.session_state.keys())
           if k[:2] in ("a_", "b_", "c_") and k.endswith(_REMEMBER_SUFFIXES)}
    for g in ("se_n",) + _GLOBAL_KEYS:
        if g in st.session_state:
            mem[g] = st.session_state[g]
    st.session_state["_se_memory"] = mem


def _restore() -> None:
    """Reseed widget keys from `_se_memory` after coming back to the page."""
    for k, v in (st.session_state.get("_se_memory") or {}).items():
        if k not in st.session_state:
            st.session_state[k] = v


def _default(key: str, value):
    """Widget default only when the key has no persisted value.

    Avoids Streamlit's "created with a default value but also had its value
    set via Session State" warning after _persist_state().
    """
    return None if key in st.session_state else value


def _clear_slot(prefix: str) -> None:
    """Forget every session key belonging to one source slot."""
    for key in [k for k in st.session_state if k.startswith(f"{prefix}_")]:
        del st.session_state[key]


def _add_source() -> None:
    st.session_state["se_n"] = min(
        MAX_SLOTS, int(st.session_state.get("se_n", 1)) + 1)


# Keys that make up one source slot's DATA (everything except the raw
# file_uploader widget object, which Streamlit won't let us reassign).
_SLOT_DATA_SUFFIXES = _PICKER_SUFFIXES + ("_upload_name", "_upload_bytes")


def _slot_data(prefix: str) -> dict:
    return {suf: st.session_state[prefix + suf]
            for suf in _SLOT_DATA_SUFFIXES if (prefix + suf) in st.session_state}


def _set_slot_data(prefix: str, data: dict) -> None:
    for suf in _SLOT_DATA_SUFFIXES:
        st.session_state.pop(prefix + suf, None)
    st.session_state.pop(prefix + "_upload", None)   # forget any stale upload widget
    for suf, val in data.items():
        st.session_state[prefix + suf] = val


def _move_source(i: int, delta: int) -> None:
    """Swap source i with its neighbour — reorder the signals horizontally."""
    j = i + delta
    n = int(st.session_state.get("se_n", 1))
    if 0 <= j < n:
        di, dj = _slot_data(SLOTS[i]), _slot_data(SLOTS[j])
        _set_slot_data(SLOTS[i], dj)
        _set_slot_data(SLOTS[j], di)


def _remove_source(i: int) -> None:
    """Remove ANY source (not just the last): shift the rest left to close the gap."""
    n = int(st.session_state.get("se_n", 1))
    if n <= 1:
        _set_slot_data(SLOTS[0], {})          # clearing the only source resets it
        return
    for k in range(i, n - 1):                 # shift k+1 → k
        _set_slot_data(SLOTS[k], _slot_data(SLOTS[k + 1]))
    _set_slot_data(SLOTS[n - 1], {})          # empty the now-unused last slot
    st.session_state["se_n"] = n - 1


def _source_toolbar(i: int, n: int) -> None:
    """Small ◀ ▶ ✕ icon toolbar (reorder / remove) for a source box."""
    with st.container(key=f"setbar_{i}"):
        l, r, x = st.columns(3)
        l.button("◀", key=f"se_mvl_{i}", disabled=(i == 0),
                 on_click=_move_source, args=(i, -1), help="Move left")
        r.button("▶", key=f"se_mvr_{i}", disabled=(i >= n - 1),
                 on_click=_move_source, args=(i, +1), help="Move right")
        x.button("✕", key=f"se_rm_{i}",
                 on_click=_remove_source, args=(i,), help="Remove this source")


# ===========================================================================
# Audio loading helpers
# ===========================================================================

@st.cache_data(show_spinner=False, max_entries=32)
def _cached_corpus_audio(path: str) -> np.ndarray:
    return get_extractor().load_audio(path)


def _load_corpus_signal(path: str):
    """Load audio from corpus path, returning None on decode failure."""
    try:
        return _cached_corpus_audio(path)
    except AudioLoadError as exc:
        st.warning(
            f"**Cannot decode** `{os.path.basename(path)}` — "
            "this file has non-standard FLAC encoding that libsndfile cannot read.  "
            "Press **Random** to try another sample.  \n"
            f"*Details: {exc}*"
        )
        return None


@st.cache_data(show_spinner=False, max_entries=16)
def _decode_upload(name: str, blob: bytes) -> np.ndarray:
    ext = get_extractor()
    signal, _ = librosa.load(io.BytesIO(blob), sr=ext.sample_rate, mono=True)
    if len(signal) < ext.n_fft:
        signal = np.pad(signal, (0, ext.n_fft - len(signal)))
    return signal


# ===========================================================================
# Source picker
# ===========================================================================

def _corpus_picker(key_prefix: str, n: int = 1):
    """Render corpus file picker. Returns (signal, label_str, path) or Nones.

    `n` is the number of source panels shown side by side: with 3 the panels are
    narrow, so the file selectbox is given less width to keep Clear/Random intact.
    """
    corpus_key = f"{key_prefix}_corpus"
    subset_key = f"{key_prefix}_subset"

    # nosearch_* container → CSS turns these into pure dropdowns (no typing).
    with st.container(key=f"nosearch_{key_prefix}_meta"):
        c_corpus, c_subset, c_class = st.columns([1.1, 1, 1.2])
        with c_corpus:
            corpus = st.selectbox("Corpus", _CORPUS_OPTIONS, key=corpus_key)

        # 2019 has real splits; 2021 corpora are eval-only. Rendering a
        # single-option selectbox keeps the layout identical across corpora.
        subset_opts = ["train", "dev", "eval"] if corpus == "2019 LA" else ["eval"]
        if st.session_state.get(subset_key) not in subset_opts:
            st.session_state[subset_key] = subset_opts[0]
        with c_subset:
            subset = st.selectbox(
                "Subset", subset_opts, key=subset_key,
                help=None if corpus == "2019 LA"
                else "2021 corpora ship a single official eval split.",
            )
        with c_class:
            cls = st.selectbox("Class", ["bonafide (real)", "spoof (deepfake)"],
                               key=f"{key_prefix}_class")

    st.caption(_CORPUS_INFO[corpus])

    if corpus == "2019 LA":
        samples = get_samples(subset)
    elif corpus == "2021 LA":
        with st.spinner("Loading 2021 LA index…"):
            samples = get_samples_2021_la()
    else:
        with st.spinner("Loading 2021 DF index…"):
            samples = get_samples_2021_df()

    if samples:
        bundle_samples(corpus, samples, subset)   # cache a few clips for the web demo
    else:
        # No live corpus index (web demo). EVAL streams a balanced sample from the
        # public Hugging Face dataset; dev/train read the committed samples/ tree.
        # Either way fall back to the bundled clips if HF is unavailable.
        if subset == "eval":
            with st.spinner(f"Streaming {corpus} eval clips from Hugging Face…"):
                samples = hf_eval_samples(corpus)
        if not samples:
            samples = bundled_samples(corpus, subset)

    if not samples:
        mini_note(f"No {corpus} samples bundled yet — switch to Upload, or run "
                  "locally with the dataset to populate them.")
        return None, None, None

    bonafide, spoof = split_by_label(samples)
    pool = bonafide if cls.startswith("bonafide") else spoof

    if not pool:
        st.warning(f"No {cls.split()[0]} files in the '{subset}' subset.")
        return None, None, None

    pool_shown = pool[:500]
    names      = [os.path.basename(p) for p in pool_shown]

    # Keep widget state valid for the current corpus/subset/class pool.
    state_key = f"{key_prefix}_file_idx"
    stored    = st.session_state.get(state_key, _PLACEHOLDER)
    if not isinstance(stored, int) or (stored != _PLACEHOLDER and stored >= len(pool_shown)):
        st.session_state[state_key] = _PLACEHOLDER

    # Narrower file column when panels are squeezed (2–3 sources) so the
    # Clear/Random buttons keep their labels instead of being clipped.
    _ratios = [1.7, 1.05, 1.2] if n >= 3 else \
              ([2.4, 1.05, 1.1] if n == 2 else [3.0, 1.05, 1.05])
    col_file, col_clear, col_rand = st.columns(_ratios)
    with col_clear:
        st.markdown('<div style="height:1.75rem;"></div>', unsafe_allow_html=True)
        # Callback runs before the rerun → clears the selectbox cleanly.
        st.button("Clear", key=f"{key_prefix}_clearsrc", icon=":material/close:",
                  help="Clear the selected audio file", width="stretch",
                  on_click=lambda k=state_key: st.session_state.update({k: _PLACEHOLDER}))
    with col_rand:
        st.markdown('<div style="height:1.75rem;"></div>', unsafe_allow_html=True)
        # Setting state BEFORE the selectbox below renders — no extra rerun.
        if st.button("Random", key=f"{key_prefix}_rand", icon=":material/casino:",
                     help="Pick a random sample", width="stretch"):
            st.session_state[state_key] = random.randint(0, len(pool_shown) - 1)
    with col_file:
        options = [_PLACEHOLDER] + list(range(len(pool_shown)))
        sel_idx = st.selectbox(
            "Audio file",
            options,
            format_func=lambda i: "— select a file —" if i == _PLACEHOLDER else names[i],
            key=state_key,
            help=f"{len(pool):,} files available — showing the first 500",
        )

    if sel_idx == _PLACEHOLDER:
        return None, None, None

    path = pool_shown[sel_idx]
    return _load_corpus_signal(path), cls, path


def _upload_picker(key_prefix: str):
    """File uploader whose payload survives widget unmount.

    The uploaded bytes are mirrored into plain session keys immediately, so
    the file keeps working after mode switches (where the uploader widget is
    destroyed) and can be copied between picker slots.
    Returns (signal, 'uploaded', name) or Nones.
    """
    name_key, bytes_key = f"{key_prefix}_upload_name", f"{key_prefix}_upload_bytes"

    uploaded = st.file_uploader(
        "Upload a .flac / .wav file", type=["flac", "wav"],
        key=f"{key_prefix}_upload",
    )
    if uploaded is not None:
        st.session_state[name_key]  = uploaded.name
        st.session_state[bytes_key] = uploaded.getvalue()

    name = st.session_state.get(name_key)
    blob = st.session_state.get(bytes_key)
    if blob is None:
        return None, None, None

    # Show caption when the widget is gone (after navigation); the uploader
    # already shows the filename when the file is freshly selected.
    if uploaded is None:
        st.caption(f"Using uploaded file: **{name}**")
    # Always show Clear so the user can reset without having to click the X on
    # the native file_uploader widget (and so it's visible even after navigation).
    with st.container(key=f"alignr_clear_{key_prefix}"):
        if st.button("Clear", key=f"{key_prefix}_upload_clear",
                     icon=":material/close:",
                     help="Forget this uploaded file"):
            st.session_state.pop(name_key, None)
            st.session_state.pop(bytes_key, None)
            st.rerun()

    return _decode_upload(name, blob), "uploaded", name


def _audio_picker(key_prefix: str, title_html: str, idx: int = 0, n: int = 1):
    """Full source picker inside a bordered panel.

    Returns (signal, label_str, source_id) or Nones.
    """
    with st.container(border=True, key=f"srcbox_{key_prefix}"):
        # Title on the left, the ◀ ▶ ✕ icons on the SAME row to the right (only
        # when there is more than one source). Sharing the row avoids the empty
        # gap a separate top element would leave above the title.
        if n > 1:
            _t, _b = st.columns([2.4, 1], vertical_alignment="center")
            _t.markdown(f'<div class="section-label">{title_html}</div>',
                        unsafe_allow_html=True)
            with _b:
                _source_toolbar(idx, n)
        else:
            st.markdown(
                f'<div class="section-label" style="margin-bottom:.7rem;">{title_html}</div>',
                unsafe_allow_html=True,
            )
        source = st.segmented_control(
            "Source", ["Corpus", "Upload"],
            key=f"{key_prefix}_source",
            default=_default(f"{key_prefix}_source", "Corpus"),
            label_visibility="collapsed",
        )
        if source is None:          # segmented control allows deselection
            source = "Corpus"
        if source == "Upload":
            return _upload_picker(key_prefix)
        return _corpus_picker(key_prefix, n)


# ===========================================================================
# Rendering helpers
# ===========================================================================

@st.cache_data(show_spinner=False, max_entries=96)
def _view_png(view: str, source_id: str, label: str, y: np.ndarray) -> bytes:
    """Render one representation to PNG bytes (cached → no flicker on rerun)."""
    ext = get_extractor()
    builders = {
        "Waveform":   lambda: fig_waveform(y, ext.sample_rate,
                                           title="Waveform", label=label),
        "STFT":  lambda: fig_stft_db(y, ext),
        "CNN Input":  lambda: fig_cnn_input(y, ext),
        "MFCC":       lambda: fig_mfcc(y, ext),
        "LFCC":       lambda: fig_lfcc(y, ext),
        "CQCC":       lambda: fig_cqcc(y, ext),
    }
    fig = builders[view]()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


def _render_views(y: np.ndarray, label: str, views: list, source_id: str) -> None:
    """Render each selected visualisation for one audio signal."""
    for view in views:
        with st.spinner(f"Rendering {view}…"):
            st.image(_view_png(view, source_id, str(label), y), width="stretch")


def _show_signal_stats(y: np.ndarray, n_metrics: int = 5) -> None:
    """Metric card row with the first ``n_metrics`` signal statistics.

    Fewer metrics are shown as more sources share the row width.
    """
    stats = compute_signal_stats(y, extractor.sample_rate)
    items = [
        ("Duration",   f"{stats['duration_s']:.2f} s"),
        ("RMS (dBFS)", f"{stats['rms_db']:.1f} dB"),
        ("Centroid",   f"{stats['centroid_hz']:.0f} Hz"),
        ("ZCR",        f"{stats['zcr']:.4f}"),
        ("RMS",        f"{stats['rms']:.4f}"),
    ][:max(1, n_metrics)]
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)


def _audio_header(label: str, source: str) -> None:
    """Coloured class badge + filename."""
    st.markdown(
        f"{label_badge(label)} &nbsp; `{os.path.basename(source)}`",
        unsafe_allow_html=True,
    )


def _views_picker() -> list:
    """Representation pills — single key, single location for both modes."""
    st.markdown('<div class="section-label" style="margin-bottom:.7rem;">'
                'Representations</div>', unsafe_allow_html=True)
    views = st.pills(
        "Representations",
        ALL_VIEWS,
        selection_mode="multi",
        default=_default("se_views", ["Waveform", "STFT"]),
        key="se_views",
        label_visibility="collapsed",
    )
    return views or []


# ===========================================================================
# Page layout
# ===========================================================================

_restore()        # reseed from cross-page memory (after returning to the page)
_persist_state()

st.title("Signal Explorer")
st.caption(
    "Waveform and spectral representations of audio from the ASVspoof "
    "corpora — analyse one file, or add up to three sources to compare them "
    "side by side."
)

n_sources = max(1, min(MAX_SLOTS, int(st.session_state.get("se_n", 1))))
st.session_state["se_n"] = n_sources

st.markdown("""
<style>
/* Small ◀ ▶ ✕ icons on the title row, right-aligned and lowered to sit level
   with the "SOURCE n" label. */
[class*="st-key-setbar_"] { margin-top: 0.35rem; }
[class*="st-key-setbar_"] [data-testid="stHorizontalBlock"] {
    gap: 0.18rem !important; justify-content: flex-end; flex-wrap: nowrap; }
[class*="st-key-setbar_"] [data-testid="stColumn"] {
    min-width: 0 !important; flex: 0 0 auto !important; width: auto !important; }
[class*="st-key-setbar_"] [data-testid="stButton"] button {
    padding: 0 !important; min-height: 0 !important; height: 1.5rem;
    width: 1.5rem; line-height: 1 !important; font-size: 0.72rem !important;
    border-radius: 0.4rem !important;
    background: rgba(79,139,249,0.1) !important;
    border: 1px solid rgba(79,139,249,0.22) !important; }
[class*="st-key-setbar_"] [data-testid="stButton"] button:hover {
    background: rgba(79,139,249,0.24) !important; }
</style>
""", unsafe_allow_html=True)

# Equal-height source panels: stretch the flex column chain so the short Upload
# slot fills to the tallest Corpus slot. align-items:stretch on the horizontal
# block propagates the row height into each column; the chain of flex containers
# below carries it all the way down to srcbox_. The JS iframe below also runs
# setHeight imperatively on every render to handle cases where CSS alone isn't
# enough (e.g. first-render timing in Streamlit).
st.markdown("""
<style>
[class*="st-key-srccols"] [data-testid="stHorizontalBlock"] {
    align-items: stretch !important;
}
[class*="st-key-srccols"] [data-testid="stColumn"] {
    display: flex !important; flex-direction: column !important;
}
[class*="st-key-srccols"] [data-testid="stColumn"] > [data-testid="stVerticalBlock"] {
    flex: 1 1 auto !important; display: flex !important; flex-direction: column !important;
}
[class*="st-key-srccols"] [data-testid="stElementContainer"]:has(> [class*="st-key-srcbox_"]) {
    flex: 1 1 auto !important; display: flex !important; flex-direction: column !important;
}
[class*="st-key-srcbox_"] { flex: 1 1 auto !important; }
[class*="st-key-srcbox_"] > [data-testid="stVerticalBlockBorderWrapper"] {
    flex: 1 1 auto !important; display: flex !important; flex-direction: column !important;
}
[class*="st-key-srcbox_"] > [data-testid="stVerticalBlockBorderWrapper"] > div {
    flex: 1 1 auto !important;
}
</style>
""", unsafe_allow_html=True)

# ── Source panel(s) ──────────────────────────────────────────────────────── #
sigs, lbls, srcs = [], [], []
with st.container(key="srccols"):
    cfg_cols = st.columns(n_sources, gap="medium")
    for i in range(n_sources):
        with cfg_cols[i]:
            if n_sources == 1:
                title = "Audio source"
            else:
                dot = (f'<span class="dot" style="background:'
                       f'{SLOT_DOT_COLORS[i]};"></span>')
                title = f"{dot}Source {i + 1}"
            sig, lbl, src = _audio_picker(SLOTS[i], title, idx=i, n=n_sources)
            sigs.append(sig)
            lbls.append(lbl)
            srcs.append(src)

# ── JS height equalizer — runs on every render so the Upload panel matches the
#    Corpus panel height immediately without requiring a page navigation.
#    Uses setTimeout chains (not setInterval) to avoid accumulation across reruns.
with st.container(key="seqh_host"):
    st.iframe(
        """<script>
(function(){
  var doc;try{doc=window.parent.document;}catch(e){return;}
  function eq(){
    var c=doc.querySelector('[class*="st-key-srccols"]');
    if(!c)return;
    var bs=Array.from(c.querySelectorAll('[class*="st-key-srcbox_"]'));
    if(bs.length<2)return;
    bs.forEach(function(b){b.style.minHeight='';});
    var m=0;
    bs.forEach(function(b){m=Math.max(m,b.getBoundingClientRect().height);});
    if(m>0)bs.forEach(function(b){b.style.minHeight=m+'px';});
  }
  [50,150,300,600].forEach(function(d){window.parent.setTimeout(eq,d);});
})();
</script>""",
        height=1,
    )

# ── Add source — removal & reordering now live on each source's toolbar ───── #
st.button(
    "Add audio source", icon=":material/add:", type="primary",
    width="stretch", on_click=_add_source, disabled=n_sources >= MAX_SLOTS,
    help="Compare up to three sources side by side"
         if n_sources < MAX_SLOTS else "Maximum of three sources reached",
)

views = _views_picker()
st.divider()

# ── Results ──────────────────────────────────────────────────────────────── #
if not any(s is not None for s in sigs):
    show_empty_state(
        "No audio selected",
        "Pick a corpus file above — or upload your own — then choose the "
        "representations to explore. Press Random for an instant sample, or "
        "Add audio source to compare up to three signals side by side.",
    )
else:
    n_metrics = {1: 5, 2: 4, 3: 3}[n_sources]
    res_cols = st.columns(n_sources, gap="medium")
    for i in range(n_sources):
        with res_cols[i]:
            if sigs[i] is not None:
                _audio_header(lbls[i], srcs[i])
                st.audio(sigs[i], sample_rate=extractor.sample_rate)
                _show_signal_stats(sigs[i], n_metrics=n_metrics)
                if views:
                    st.divider()
                    _render_views(sigs[i], lbls[i], views, str(srcs[i]))
            else:
                st.info(f"Source {i + 1} — pick a file above.",
                        icon=":material/graphic_eq:")
    if not views:
        st.info("Select one or more representations above to visualise the signals.")

# ── Sidebar: live session summary (rendered last, when selections exist) ── #


def _slot_row(label: str, source_id, cls) -> tuple:
    if source_id is None:
        return (label, "—")
    name = os.path.basename(str(source_id))
    tag  = "spoof" if cls and "spoof" in str(cls) else \
           "bonafide" if cls and "bonafide" in str(cls) else "upload"
    return (label, f"{name} · {tag}")


with st.sidebar:
    rows = [("Sources", str(n_sources))]
    for i in range(n_sources):
        label = "File" if n_sources == 1 else f"Source {i + 1}"
        rows.append(_slot_row(label, srcs[i], lbls[i]))
    rows.append(("Views", ", ".join(views) if views else "—"))
    sidebar_panel("Session", rows)

# Persist selections so they survive leaving and re-entering the page.
_remember()
