# -*- coding: utf-8 -*-
"""src/ui/components.py — Reusable Streamlit UI components and controls.

Empty-state cards, badges, sidebar panels, the eval corpus/score pickers, demo
notices and the background-job banner. Split out of src/ui_helpers.py and
re-exported there for backward compatibility. Cross-module names (data/model
loaders, leaderboard, theme helpers) are imported from their new home modules.
"""

from typing import List, Optional, Tuple

import streamlit as st

from src.features import FeatureExtractor
from src.ui.styles import BONAFIDE_COLOR, SPOOF_COLOR
from src.model_registry import (
    HF_EVAL_DATASETS, HF_EVAL_PER_CLASS, corpus_available, get_extractor,
    get_samples, get_samples_2021_df, get_samples_2021_la, hf_eval_samples,
    load_config,
)

def demo_corpus_notice(title: str = "Disabled in the web demo",
                       body: Optional[str] = None) -> None:
    """Styled notice replacing the bare 'corpus not found' error: explains why a
    corpus-dependent section is unavailable in the public CPU demo."""
    body = body or (
        "This section runs on the full ASVspoof corpus, which is not bundled in "
        "the public CPU demo (it is several GB). Clone the repository and run "
        "the app locally with the dataset — and a GPU for training — to use it."
        '<br><span style="opacity:0.65;font-style:italic;">'
        "“If an item does not appear in our records… it does not exist.”"
        "</span>"
    )
    st.markdown(
        f'<div class="info-card" style="border-left:3px solid #4F8BF9;">'
        f'<div class="ic-title">{title}</div>'
        f'<p class="ic-body">{body}</p></div>',
        unsafe_allow_html=True,
    )


def test_audio_cta(
    text: str = "Instead, hear the state of the art in real time: upload your own "
                "clip and watch every pretrained model judge it side by side.",
) -> None:
    """Attractive redirect from a corpus-only section to the multi-model file
    analysis that DOES work in the web demo."""
    st.markdown(
        f'<p style="margin:0.9rem 0 1.15rem;opacity:0.8;">{text}</p>',
        unsafe_allow_html=True,
    )
    try:
        st.page_link("app_pages/3_Detection_Analysis.py",
                     label="Try the live multi-model analysis",
                     icon=":material/hearing:")
    except Exception:  # noqa: BLE001 — page_link needs st.navigation context
        st.caption("Open **Detection Analysis → Test an audio** from the sidebar.")


# ===========================================================================
# Reusable UI components
# ===========================================================================

def show_empty_state(title: str, message: str, icon: str = "◌") -> None:
    """Centred empty-state card: sober glyph in a dashed ring, heading, text."""
    st.markdown(
        f'<div class="empty-state">'
        f'<div class="empty-icon">{icon}</div>'
        f'<h3 style="color:#82B1FF;margin:0 0 .45rem;font-weight:700;">{title}</h3>'
        f'<p style="max-width:480px;margin:.3rem auto;line-height:1.65;opacity:0.65;">{message}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def label_badge(label: str) -> str:
    """Return an HTML colour badge (dot + text) for a class label string."""
    is_spoof = "spoof" in label.lower()
    color = SPOOF_COLOR if is_spoof else BONAFIDE_COLOR
    return (
        f'<span style="color:{color};font-weight:600;">'
        f'<span class="dot" style="background:{color};'
        f'box-shadow:0 0 6px {color};"></span>{label}</span>'
    )


def section_header(num: str, title: str, caption: Optional[str] = None) -> None:
    """Editorial section header: index number, title, fading rule, caption."""
    st.markdown(
        f'<div class="sec-head"><span class="sh-num">{num}</span>'
        f'<h3 class="sh-title">{title}</h3><span class="sh-rule"></span></div>'
        + (f'<p class="sec-sub">{caption}</p>' if caption else ""),
        unsafe_allow_html=True,
    )


def sidebar_panel(
    title: str,
    rows: Optional[List[Tuple[str, str]]] = None,
    text: Optional[str] = None,
) -> None:
    """Compact sidebar block: optional key/value rows and/or a short text.

    rows: list of (label, value) pairs rendered right-aligned.
    text: free-form sentence rendered below the rows.
    """
    body = "".join(
        f'<div class="ss-row"><span class="ss-k">{k}</span>'
        f'<span class="ss-v">{v}</span></div>'
        for k, v in (rows or [])
    )
    if text:
        body += f'<div style="margin-top:.3rem;">{text}</div>'
    st.markdown(
        f'<div class="side-status"><div class="ss-title">{title}</div>{body}</div>',
        unsafe_allow_html=True,
    )


def mini_note(text: str, warn: bool = False) -> None:
    """Fixed-height inline notice — never resizes the surrounding panel."""
    cls = "mini-note warn" if warn else "mini-note"
    st.markdown(f'<div class="{cls}"><span>{text}</span></div>',
                unsafe_allow_html=True)


def app_footer(left: str, right: str) -> None:
    """Closed page ending: top rule + colophon, no trailing scroll space."""
    st.markdown(
        f'<div class="app-footer"><span class="af-left">{left}</span>'
        f'<span class="af-right">{right}</span></div>',
        unsafe_allow_html=True,
    )


# Evaluation corpus options. Only 2019 LA has a dev split (seen attacks); the
# 2021 corpora are eval-only, so the available "Score on" options DEPEND on the
# corpus (see score_options_for / dev_corpus).
EVAL_CORPUS_CHOICES = ["2019 LA", "2021 LA", "2021 DF"]


def eval_corpora_for(choice: str):
    """Return [(label, eval_samples)] for the chosen eval corpus (a 1-item list,
    or empty if that corpus is unavailable).

    On the corpus-less web demo the local splits are empty, so we fall back to a
    small balanced eval set streamed from the public Hugging Face dataset — this
    is what lets the benchmark modes evaluate the pretrained models on the cloud
    exactly as they would locally (only training is off)."""
    if choice == "2019 LA":
        samples = get_samples("eval")
    elif choice == "2021 LA":
        samples = get_samples_2021_la()
    else:
        samples = get_samples_2021_df()
    if not samples and not corpus_available() and choice in HF_EVAL_DATASETS:
        samples = hf_eval_samples(choice, HF_EVAL_PER_CLASS)
    return [(choice, samples)] if samples else []


def score_options_for(corpus: str):
    """The valid 'Score on' options for a corpus. 2019 LA has dev + eval; the
    2021 corpora are eval-only."""
    return ["Dev", "Eval", "Dev + Eval"] if corpus == "2019 LA" else ["Eval"]


def eval_score_controls(
    prefix: str,
    disabled: bool = False,
    train_label: str = "Train on",
):
    """Unified 'Evaluation' group: the eval-corpus picker and the dependent
    score-on picker inside one framed block (used by every benchmark mode so
    they look consistent). Returns (corpus, score_split).

    Keys are f'{prefix}_corpus' and f'{prefix}_split'.
    Pass train_label="Trained on" when a model is already loaded.
    """
    ck, sk = f"{prefix}_corpus", f"{prefix}_split"
    _train_key = f"{prefix}_train_on"
    # Defaults (and self-heal any stale/invalid persisted value).
    if st.session_state.get(ck) not in EVAL_CORPUS_CHOICES:
        st.session_state[ck] = "2019 LA"
    if st.session_state.get(sk) not in ("Dev", "Eval", "Dev + Eval"):
        st.session_state[sk] = "Dev + Eval"
    if st.session_state.get(_train_key) is None:
        st.session_state[_train_key] = "2019 LA"

    def _keep_train_on():
        if st.session_state.get(_train_key) is None:
            st.session_state[_train_key] = "2019 LA"

    # The two controls live in one flex-row, fit-content frame (CSS) so the
    # "Score on" sits right NEXT TO "Evaluate on" and the blue box stays short.
    with st.container(key=f"evalgrp_{prefix}"):
        # Fixed "Train on" / "Trained on" — same segmented_control format as the
        # other two. on_change prevents the user from deselecting the single option.
        st.segmented_control(
            train_label, ["2019 LA"],
            key=_train_key, disabled=disabled, on_change=_keep_train_on,
        )
        corpus = st.segmented_control(
            "Evaluate on", EVAL_CORPUS_CHOICES, key=ck, disabled=disabled)
        corpus = corpus or "2019 LA"

        # "Score on" always renders the same 3 options so the box never shifts.
        # When the selected corpus has no dev split, Dev/Dev+Eval are visually
        # disabled via injected CSS (pointer-events:none) — the style element is
        # collapsed to zero height by PAGE_CSS but its rules still apply globally.
        # Session state is also reset to "Eval" so no stale selection leaks through.
        if corpus != "2019 LA":
            if st.session_state.get(sk) in ("Dev", "Dev + Eval"):
                st.session_state[sk] = "Eval"
            st.markdown(
                "<style>"
                "[class*='st-key-evalgrp_']>[data-testid='stElementContainer']:last-child "
                "button:nth-child(1),"
                "[class*='st-key-evalgrp_']>[data-testid='stElementContainer']:last-child "
                "button:nth-child(3)"
                "{pointer-events:none!important;opacity:0.28!important;}"
                "</style>",
                unsafe_allow_html=True,
            )
        score = st.segmented_control(
            "Score on", ["Dev", "Eval", "Dev + Eval"], key=sk, disabled=disabled)
        score = score or "Dev + Eval"
        if corpus != "2019 LA":
            score = "Eval"   # enforce even if CSS failed and user clicked Dev
    return corpus, score


def op_in_progress() -> bool:
    """True while ANY background job (full comparison OR CNN training) runs."""
    for key in ("bench_future", "cnn_future"):
        fut = st.session_state.get(key)
        if fut is not None and not fut.done():
            return True
    return False


def op_status():
    """Return (kind, label) of the running background job, or (None, None).

    kind is 'full' or 'cnn'; label is a short human description for the banner.
    """
    fut = st.session_state.get("bench_future")
    if fut is not None and not fut.done():
        return "full", "Full comparison running"
    fut = st.session_state.get("cnn_future")
    if fut is not None and not fut.done():
        return "cnn", "Training CNN"
    return None, None


def op_busy_notice() -> bool:
    """Return True while a background job runs (the running banner now lives in
    the sidebar, so this no longer renders anything in the page body — pages
    just use the return value to disable their run/train controls)."""
    return op_in_progress()


# Geometric symbols for the banner (no emoji): a small node-graph for the CNN
# and a bar-chart for the full comparison.
_OP_ICON_CNN = ('<svg viewBox="0 0 24 24" width="14" height="14" fill="none" '
                'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
                'stroke-linejoin="round"><circle cx="5" cy="6" r="1.8"/>'
                '<circle cx="5" cy="18" r="1.8"/><circle cx="12.5" cy="12" r="1.8"/>'
                '<circle cx="20" cy="6" r="1.8"/><circle cx="20" cy="18" r="1.8"/>'
                '<path d="M6.6 6.8 11 11M6.6 17.2 11 13M14 11 18.4 6.8M14 13 18.4 17.2"/>'
                '</svg>')
_OP_ICON_FULL = ('<svg viewBox="0 0 24 24" width="14" height="14" fill="none" '
                 'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
                 'stroke-linejoin="round"><path d="M4 20V11M10 20V4M16 20v-6M3 20h18"/></svg>')


def _op_banner_render() -> None:
    """Render the banner (or trigger the completion rerun). Shared by the live and
    idle fragment wrappers below."""
    for key in ("bench_future", "cnn_future"):
        fut = st.session_state.get(key)
        if fut is not None and fut.done():
            st.rerun(scope="app")
            return

    kind, label = op_status()
    if kind is None:
        return

    from src.jobs import progress as _progress
    pr  = _progress()
    pct = int(round(pr["frac"] * 100))
    sym = _OP_ICON_CNN if kind == "cnn" else _OP_ICON_FULL
    # Called inside a `with st.sidebar:` block (see app.py) — a fragment may only
    # write to its own parent container, so we do NOT open st.sidebar here. The
    # whole banner is a click target (invisible overlay button) that jumps to the
    # page where the job runs.
    with st.container(key="opbanner"):
        st.markdown(
            f'<div class="op-banner"><div class="ob-head">'
            f'<span class="ob-dot"></span><span class="ob-ic">{sym}</span>'
            f'<span>{label}</span><span class="ob-go">open ›</span></div>'
            f'<div class="ob-sub">{pr["label"]}</div>'
            f'<div class="ob-track"><span style="width:{pct}%"></span></div>'
            f'<div class="ob-pct">{pct}% · stage {pr["done"]}/{max(pr["total"], 1)}'
            '</div></div>',
            unsafe_allow_html=True,
        )
        if st.button("open", key="opbanner_go", width="stretch"):
            st.session_state["bench_choice"] = "cnn" if kind == "cnn" else "full"
            if kind == "cnn":
                st.session_state["cnn_focus_curves"] = True   # open Training curves
            st.switch_page("app_pages/2_Benchmark.py")


@st.fragment(run_every=2.0)
def _op_banner_live() -> None:
    _op_banner_render()


@st.fragment
def _op_banner_idle() -> None:
    _op_banner_render()


def op_banner_fragment() -> None:
    """Global background-job banner, pinned to the bottom of the sidebar.

    Appears on EVERY page (rendered before the page script runs, so it survives
    st.stop()). It auto-refreshes every 2 s ONLY while a job is running; when the
    job finishes it triggers ONE full app rerun so app.py collects the result.

    The 2 s timer (run_every) is attached only in the running state: an idle app
    would otherwise keep a live timer that, after the rapid reruns of startup or
    page navigation, fires against a container that no longer exists — logging
    'The fragment ... does not exist anymore' warnings on every rerun."""
    if op_in_progress():
        _op_banner_live()
    else:
        _op_banner_idle()


def launch_full_comparison(classic_subset: int = 4000, include_cnn: bool = True) -> None:
    """Submit a full comparison with sensible defaults (2019 LA, dev + eval).

    Used by the sidebar quick-launch button so the headline benchmark — the base
    for everything else in the app — is one click away from any page."""
    from src.jobs import submit_benchmark
    ext = get_extractor()
    st.session_state["bench_future"] = submit_benchmark(
        ext=ext, feat_labels=FeatureExtractor.OPTION_NAMES,
        base_params=dict(load_config()["train_params"]),
        train=get_samples("train"), primary=get_samples("dev"), pname="dev",
        eval_corpora=eval_corpora_for("2019 LA"),
        classic_subset=int(classic_subset), cnn_subset=0,
        include_cnn=include_cnn, seed=42,
    )
    st.session_state["bench_score"] = "Dev + Eval"
    st.session_state["op_running"] = True


def render_full_cta() -> None:
    """Sidebar quick-launch for the full comparison, pinned to the same bottom
    spot the running banner uses (shown only when nothing is running). Once a
    full comparison has finished, it turns into a shortcut to its leaderboard."""
    with st.sidebar:
        with st.container(key="opcta"):
            if st.session_state.get("bench_done"):
                if st.button("See full comparison", key="cta_see_full",
                             type="primary", width="stretch",
                             icon=":material/leaderboard:"):
                    st.session_state["bench_choice"] = "full"
                    st.switch_page("app_pages/2_Benchmark.py")
            elif st.button("Run full comparison", key="cta_full_cmp",
                           type="primary", width="stretch",
                           icon=":material/playlist_play:",
                           disabled=not corpus_available()):
                launch_full_comparison()
                # Land on the full-comparison page so its live progress is visible.
                st.session_state["bench_choice"] = "full"
                st.switch_page("app_pages/2_Benchmark.py")


