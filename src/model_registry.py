# -*- coding: utf-8 -*-
"""src/model_registry.py — Data & model layer.

Config + cached FeatureExtractor, corpus/sample loading, the pretrained model
REGISTRY, Hugging Face streaming/download, and the model loaders. Split out of
src/ui_helpers.py and re-exported there for backward compatibility.
"""

import os
from typing import Dict, List, Optional, Tuple

import streamlit as st
import yaml

from src.data_loader import (
    LABEL_BONAFIDE, LABEL_SPOOF, parse_protocol, parse_protocol_2021,
)
from src.features import FeatureExtractor

CONFIG_PATH = os.path.join("config", "config.yaml")

def _forced_demo() -> bool:
    """Env switch to simulate the corpus-less cloud deployment on a local machine
    (set DEEPFAKE_FORCE_DEMO=1): the dataset loaders report empty so every page
    renders exactly as it will in the public web demo."""
    return os.environ.get("DEEPFAKE_FORCE_DEMO") in ("1", "true", "True")
@st.cache_resource
def load_config() -> Dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_resource
def get_extractor() -> FeatureExtractor:
    return FeatureExtractor(CONFIG_PATH)


@st.cache_data(show_spinner=False)
def get_samples(subset: str) -> List[Tuple[str, int]]:
    if _forced_demo():
        return []
    config    = load_config()
    root_dir  = config["dataset"]["path_la2019"]
    proto_dir = os.path.join(root_dir, config["dataset"]["protocols_dir"])
    proto     = config["dataset"]["protocols"].get(subset)
    if not proto:
        return []
    try:
        return parse_protocol(os.path.join(proto_dir, proto), root_dir, subset)
    except (FileNotFoundError, ValueError):
        return []


@st.cache_data(show_spinner=False)
def get_samples_2021_la() -> List[Tuple[str, int]]:
    """Load and cache the full ASVspoof 2021 LA eval split."""
    if _forced_demo():
        return []
    config = load_config()
    cfg    = config.get("dataset_2021", {}).get("la", {})
    eval_dir = cfg.get("eval_dir", "")
    keys     = cfg.get("keys", "")
    try:
        return parse_protocol_2021(keys, [eval_dir])
    except (FileNotFoundError, ValueError) as exc:
        print(f"[WARNING] 2021 LA unavailable: {exc}")
        return []


@st.cache_data(show_spinner=False)
def get_samples_2021_df() -> List[Tuple[str, int]]:
    """Load and cache the full ASVspoof 2021 DF eval split (all 3 partitions)."""
    if _forced_demo():
        return []
    config    = load_config()
    cfg       = config.get("dataset_2021", {}).get("df", {})
    eval_dirs = cfg.get("eval_dirs", [])
    keys      = cfg.get("keys", "")
    try:
        return parse_protocol_2021(keys, eval_dirs)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[WARNING] 2021 DF unavailable: {exc}")
        return []


def corpus_available_2021_la() -> bool:
    return len(get_samples_2021_la()) > 0


def corpus_available_2021_df() -> bool:
    return len(get_samples_2021_df()) > 0


def corpus_configured_2021_la() -> bool:
    """Lightweight check (single stat call) — does NOT load audio index."""
    cfg = load_config().get("dataset_2021", {}).get("la", {})
    return os.path.isfile(cfg.get("keys", ""))


def corpus_configured_2021_df() -> bool:
    """Lightweight check (single stat call) — does NOT load audio index."""
    cfg = load_config().get("dataset_2021", {}).get("df", {})
    return os.path.isfile(cfg.get("keys", ""))


def split_by_label(
    samples: List[Tuple[str, int]],
) -> Tuple[List[str], List[str]]:
    bonafide = [p for p, e in samples if e == LABEL_BONAFIDE]
    spoof    = [p for p, e in samples if e == LABEL_SPOOF]
    return bonafide, spoof


def corpus_available() -> bool:
    return len(get_samples("train")) > 0

# ===========================================================================
# Public / CPU demo mode + pretrained model registry (Streamlit Cloud)
# ===========================================================================
# On the free public cloud there is no GPU and the multi-GB ASVspoof corpus is
# not on disk, so training / full-benchmark features cannot run. Instead the whole
# pretrained zoo — the two CNNs and the classic LR / SVM / XGBoost over every DSP
# front-end — is COMMITTED to the repo under models/ (the weights are small, a few
# MB total) and loaded directly from disk on CPU. Only the multi-GB DATASETS still
# stream from Hugging Face (HF_EVAL_DATASETS); the model weights do NOT.

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Pretrained weights live HERE, committed to the repo (the trainer also writes a
# legacy single checkpoint at the repo root, used by src/pipeline.py).
MODELS_DIR      = os.path.join(_REPO_ROOT, "models")
CHECKPOINT_PATH = os.path.join(_REPO_ROOT, "asvspoof_model_checkpoint.pth")

# ── Weight source ───────────────────────────────────────────────────────────
# Models are loaded straight from the committed models/ folder — no runtime
# download. Leave HF_BASE_URL EMPTY to keep it that way. Only set it to a Hugging
# Face "resolve/main" folder if you ever prefer to stream the weights instead of
# committing them (then each model's URL is derived from this one base). The
# DATASETS are unrelated and always come from HF (see HF_EVAL_DATASETS).
HF_BASE_URL = ""
MODEL_URL   = HF_BASE_URL


def _hf_url(file: str) -> str:
    """Derive a model's download URL from HF_BASE_URL (empty until it is set)."""
    base = HF_BASE_URL.rstrip("/")
    if "TU_ENLACE" in base or not base.startswith(("http://", "https://")):
        return ""
    return f"{base}/{file}"


# Build the registry of every servable detector. Fields:
#   kind  : "cnn" (torch .pth) or "classic" (joblib-dumped sklearn/xgb estimator)
#   clf   : classifier name (classic only) for get_classic_model / row matching
#   feat  : FeatureExtractor option key for classic models (CNNs read the STFT
#           spectrogram directly, so feat is None for them)
#   front : human-readable front-end, for the comparison table
_CLF_DEFS = [
    ("lr",  "logistic_regression", "Logistic Regression"),
    ("svm", "svm_lineal",          "SVM (RBF)"),
    ("xgb", "xgboost",             "XGBoost"),
]
# (key suffix, FeatureExtractor option, label) — every DSP front-end.
_FEAT_DEFS = [
    ("rms",  "1", "RMS"),
    ("mfcc", "2", "MFCC"),
    ("lfcc", "3", "LFCC"),
    ("dwt",  "4", "DWT"),
    ("cqcc", "6", "CQCC"),
]

PRETRAINED_REGISTRY: List[Dict] = [
    # Deep spectrogram models. "arch" is the canonical key consumed by
    # src.models.model_for_arch (single source of truth for class dispatch);
    # it is also what gets stored in each checkpoint's "arch" field.
    {"key": "cnn5", "name": "5-Block CNN", "kind": "cnn", "arch": "cnn",
     "clf": None, "feat": None, "front": "STFT-dB spectrogram",
     "file": "cnn5.pth", "url": _hf_url("cnn5.pth")},
    {"key": "cnn5_se", "name": "5-Block CNN + SE", "kind": "cnn", "arch": "cnn_se",
     "clf": None, "feat": None, "front": "STFT-dB spectrogram",
     "file": "cnn5_se.pth", "url": _hf_url("cnn5_se.pth")},
    {"key": "resnet", "name": "ResNet + SE", "kind": "cnn", "arch": "resnet",
     "clf": None, "feat": None, "front": "STFT-dB spectrogram",
     "file": "resnet.pth", "url": _hf_url("resnet.pth")},
    {"key": "resnext", "name": "ResNeXt + SE", "kind": "cnn", "arch": "resnext",
     "clf": None, "feat": None, "front": "STFT-dB spectrogram",
     "file": "resnext.pth", "url": _hf_url("resnext.pth")},
    {"key": "crnn", "name": "CRNN", "kind": "cnn", "arch": "crnn",
     "clf": None, "feat": None, "front": "STFT-dB spectrogram",
     "file": "crnn.pth", "url": _hf_url("crnn.pth")},
    # Self-supervised raw-waveform detector (fine-tuned wav2vec 2.0 base + linear
    # head). "raw" kind: no DSP front-end, no spectrogram — it eats the 16 kHz
    # waveform directly. Inference-only (it is evaluated, never trained, by the
    # Full-comparison sweep). On the web demo it is too large to commit to GitHub
    # (≈469 MB), so it is fetched from a PUBLIC Hugging Face model repo on demand
    # (hf_repo / hf_file) — no local heavy file, no HF_BASE_URL needed.
    {"key": "wav2vec2", "name": "wav2vec 2.0 (SSL)", "kind": "raw", "clf": None,
     "feat": None, "front": "Self-supervised raw-waveform",
     "file": "wav2vec2.pth", "url": _hf_url("wav2vec2.pth"),
     "hf_repo": "Sara1708/deepfake-audio-wav2vec2", "hf_file": "stage2_best.pt"},
]
for _ck, _cname, _clabel in _CLF_DEFS:
    for _fk, _fopt, _flabel in _FEAT_DEFS:
        _file = f"{_ck}_{_fk}.joblib"
        PRETRAINED_REGISTRY.append({
            "key":   f"{_ck}_{_fk}",
            "name":  f"{_clabel} · {_flabel}",
            "kind":  "classic",
            "clf":   _cname,
            "feat":  _fopt,
            "front": _flabel,
            "file":  _file,
            "url":   _hf_url(_file),
        })


def running_on_gpu() -> bool:
    """True when a CUDA GPU is available (local workstation), False on the
    CPU-only public cloud."""
    import torch
    return torch.cuda.is_available()


def demo_mode() -> bool:
    """Public-demo mode: the heavy ASVspoof corpus is NOT on disk (the case on
    Streamlit Community Cloud). Corpus-dependent sections degrade to notices;
    the pretrained multi-model file analysis remains fully usable.

    Honours DEEPFAKE_FORCE_DEMO=1 transparently (the loaders report empty, so
    corpus_available() is False) — handy to preview the cloud UI locally."""
    return not corpus_available()


def _url_set(url: Optional[str]) -> bool:
    """Whether a download URL has been filled in (not a placeholder)."""
    return (isinstance(url, str) and "TU_ENLACE" not in url
            and url.startswith(("http://", "https://")))


def _model_path(entry: Dict) -> str:
    return os.path.join(MODELS_DIR, entry["file"])


def _hf_cached(repo: str, fname: str) -> bool:
    """True if the file from a Hugging Face repo is already in the local HF cache
    (a cheap, offline-only probe — never hits the network)."""
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id=repo, filename=fname, local_files_only=True)
        return True
    except Exception:                       # not cached / hub unavailable
        return False


def model_available(entry: Dict) -> bool:
    """Servable if the weights are on disk, a direct URL is set, or the model has a
    public Hugging Face source (hf_repo) it can stream on demand."""
    return (os.path.isfile(_model_path(entry)) or _url_set(entry.get("url"))
            or bool(entry.get("hf_repo")))


def available_pretrained_models() -> List[Dict]:
    """Every registry entry whose weights are present or downloadable."""
    return [e for e in PRETRAINED_REGISTRY if model_available(e)]


def pretrained_available() -> bool:
    """True when at least one pretrained model can be served."""
    return len(available_pretrained_models()) > 0


def model_downloaded(entry: Dict) -> bool:
    """True when a model's weights are already cached locally — on disk, or (for an
    HF-sourced model like wav2vec2) in the Hugging Face cache."""
    if os.path.isfile(_model_path(entry)):
        return True
    if entry.get("hf_repo"):
        return _hf_cached(entry["hf_repo"], entry["hf_file"])
    return False


def models_trained() -> bool:
    """True once EVERY registry model has been trained and saved to disk — used
    to switch the Benchmark from 'train everything' to 'evaluate the saved zoo'."""
    return bool(PRETRAINED_REGISTRY) and all(
        model_downloaded(e) for e in PRETRAINED_REGISTRY)

# ── Bundled sample clips (so Signal Explorer always has audio to show) ─────── #
# A handful of real clips per corpus/subset are committed under
# samples/<key>/<subset>/ so the explorer works even without the multi-GB
# datasets (e.g. on the cloud). For whatever folder is still empty they are
# auto-populated from the live corpus the first time it is browsed locally.
# The label is encoded in the filename prefix (spoof__* / bonafide__*).
SAMPLES_DIR  = os.path.join(_REPO_ROOT, "samples")
_SAMPLE_KEYS = {"2019 LA": "2019_la", "2021 LA": "2021_la", "2021 DF": "2021_df"}


def _sample_dir(corpus: str, subset: Optional[str] = None) -> str:
    base = os.path.join(SAMPLES_DIR, _SAMPLE_KEYS.get(corpus, corpus))
    return os.path.join(base, subset) if subset else base


def _label_for(fn: str) -> int:
    return LABEL_SPOOF if fn.startswith("spoof") else LABEL_BONAFIDE


def _scan_clips(d: str) -> List[Tuple[str, int]]:
    if not os.path.isdir(d):
        return []
    return [(os.path.join(d, fn), _label_for(fn))
            for fn in sorted(os.listdir(d))
            if fn.lower().endswith((".flac", ".wav"))]


def bundled_samples(corpus: str, subset: Optional[str] = None
                    ) -> List[Tuple[str, int]]:
    """(path, label) list of committed clips for a corpus/subset.

    Reads samples/<corpus>/<subset>/ when a subset is given; falls back to the
    flat samples/<corpus>/ folder (legacy layout) so older bundles keep working.
    Empty when nothing has been bundled.
    """
    if subset:
        clips = _scan_clips(_sample_dir(corpus, subset))
        if clips:
            return clips
    return _scan_clips(_sample_dir(corpus))


def bundle_samples(corpus: str, samples: List[Tuple[str, int]],
                   subset: Optional[str] = None, n_per_class: int = 10) -> None:
    """Copy a few bonafide + spoof clips from the live corpus into
    samples/<corpus>/<subset>/ (once). Idempotent: does nothing if that folder
    already holds clips."""
    import shutil
    d = _sample_dir(corpus, subset)
    if _scan_clips(d):
        return
    bona  = [p for p, e in samples if e == LABEL_BONAFIDE][:n_per_class]
    spoof = [p for p, e in samples if e == LABEL_SPOOF][:n_per_class]
    if not bona and not spoof:
        return
    os.makedirs(d, exist_ok=True)
    for tag, paths in (("bonafide", bona), ("spoof", spoof)):
        for i, p in enumerate(paths):
            try:
                shutil.copy(p, os.path.join(d, f"{tag}__{i}_{os.path.basename(p)}"))
            except OSError:
                pass

# ── Eval clips streamed from public Hugging Face datasets ──────────────────── #
# On the corpus-less web demo the EVAL splits are far too large to commit, so we
# pull a small, balanced sample on demand from public HF datasets (the dev/train
# splits keep using the committed samples/ tree above). We use the lightweight
# datasets-server /rows API — no `datasets` dependency and no multi-GB parquet
# download — read each row's label + presigned audio URL, and cache a capped
# number of clips locally so browsing and scoring reuse them.
HF_EVAL_DATASETS: Dict[str, Dict[str, str]] = {
    "2019 LA": {"id": "Bisher/ASVspoof_2019_LA",
                "config": "default", "split": "test", "label_col": "key"},
    "2021 LA": {"id": "SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA",
                "config": "default", "split": "test", "label_col": "label"},
    "2021 DF": {"id": "SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF",
                "config": "default", "split": "test", "label_col": "label"},
}
HF_EVAL_PER_CLASS = 50          # clips PER CLASS to cache (web-demo friendly default)
HF_PAGE_CEILING   = 80          # max /rows pages (×100 rows) scanned per request
_HF_CACHE_DIR     = os.path.join(SAMPLES_DIR, "_hf_cache")
_DS_ROWS_URL      = "https://datasets-server.huggingface.co/rows"


def _hf_headers() -> Dict:
    """Build request headers, injecting HF_TOKEN when available (higher rate limits)."""
    h = {"User-Agent": "tfg-deepfake-demo"}
    token = os.environ.get("HF_TOKEN", "")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _hf_get_json(url: str) -> Dict:
    import json
    import urllib.request
    req = urllib.request.Request(url, headers=_hf_headers())
    with urllib.request.urlopen(req, timeout=30) as r:   # noqa: S310 — fixed host
        return json.load(r)


def _hf_pages_for(per_class: int) -> int:
    """How many /rows pages to scan to gather ``per_class`` clips of EACH class.
    ASVspoof is ~90 % spoof, so bonafide is the bottleneck: ~10 bonafide per
    100-row page. Scale to the request, clamped to a sane ceiling."""
    return max(4, min(HF_PAGE_CEILING, per_class // 10 + 4))


def _hf_scan_pages(base: str, offsets, ingest, done, max_workers: int = 8) -> None:
    """Fetch the /rows index pages at ``offsets`` in PARALLEL batches, calling
    ``ingest(payload)`` on each result, and stop once ``done()`` returns True.
    Per-page failures are skipped. Parallel batching is what keeps the higher
    clip caps responsive (10+ sequential 30 s-timeout calls were the bottleneck)
    while still honouring early-stop between batches."""
    import concurrent.futures as cf
    for i in range(0, len(offsets), max_workers):
        if done():
            return
        batch = offsets[i:i + max_workers]
        with cf.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = [pool.submit(_hf_get_json, base + f"&offset={off}&length=100")
                    for off in batch]
            for fut in cf.as_completed(futs):
                try:
                    ingest(fut.result())
                except Exception:                        # noqa: BLE001 — skip bad page
                    continue


def _hf_download(src_dst):
    import urllib.request
    src, dst, label = src_dst
    if not os.path.isfile(dst):
        try:
            req = urllib.request.Request(src, headers=_hf_headers())
            with urllib.request.urlopen(req, timeout=60) as r:   # noqa: S310
                data = r.read()
            with open(dst, "wb") as fh:
                fh.write(data)
        except Exception:                                # noqa: BLE001 — skip bad clip
            return None
    return (dst, label)


def _hf_eval_impl(corpus: str, n_per_class: int) -> List[Tuple[str, int]]:
    import concurrent.futures as cf
    import random
    from urllib.parse import quote

    spec = HF_EVAL_DATASETS.get(corpus)
    if not spec:
        return []

    cache_dir = os.path.join(_HF_CACHE_DIR, _SAMPLE_KEYS.get(corpus, corpus), "eval")
    cached = _scan_clips(cache_dir)
    if len(cached) >= 2 * n_per_class:                   # already populated → reuse
        return cached

    base = (f"{_DS_ROWS_URL}?dataset={quote(spec['id'], safe='')}"
            f"&config={spec['config']}&split={spec['split']}")

    # Read rows across random windows: ASVspoof is ~90% spoof, so sequential
    # pages can be all-spoof. Random offsets give both classes quickly.
    collected: Dict[int, List[str]] = {LABEL_BONAFIDE: [], LABEL_SPOOF: []}
    try:
        first = _hf_get_json(base + "&offset=0&length=100")
    except Exception:                                    # noqa: BLE001 — HF unreachable
        return cached
    total = int(first.get("num_rows_total", 100))
    rng = random.Random(42)
    pages = _hf_pages_for(n_per_class)
    offsets = [rng.randint(0, max(0, total - 100)) for _ in range(pages - 1)]

    def _ingest(payload):
        for row in payload.get("rows", []):
            r   = row.get("row", {})
            lab = r.get(spec["label_col"])
            if lab not in (LABEL_BONAFIDE, LABEL_SPOOF):
                continue
            if len(collected[lab]) >= n_per_class:
                continue
            audio = r.get("audio")
            src   = (audio[0].get("src") if isinstance(audio, list) and audio
                     else None)
            if src:
                collected[lab].append(src)

    _ingest(first)
    _hf_scan_pages(base, offsets, _ingest,
                   lambda: all(len(collected[c]) >= n_per_class for c in collected))

    os.makedirs(cache_dir, exist_ok=True)
    ckey = _SAMPLE_KEYS.get(corpus, corpus)
    tasks = []
    for lab, srcs in collected.items():
        tag = "bonafide" if lab == LABEL_BONAFIDE else "spoof"
        for i, src in enumerate(srcs):
            # Embed the corpus in the filename: the DSP/spectrogram caches key on
            # the file stem, so a bare bonafide__0.flac would collide across
            # corpora. Keep the label prefix so _label_for() still works.
            fname = f"{tag}__{ckey}__{i}.flac"
            tasks.append((src, os.path.join(cache_dir, fname), lab))

    out: List[Tuple[str, int]] = []
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        for res in pool.map(_hf_download, tasks):
            if res:
                out.append(res)
    return out or cached


@st.cache_data(show_spinner=False)
def hf_eval_samples(corpus: str,
                    n_per_class: int = HF_EVAL_PER_CLASS) -> List[Tuple[str, int]]:
    """A small, balanced eval sample streamed from the public HF dataset for
    ``corpus`` and cached under samples/_hf_cache/. Returns [(local_path, label)]
    (empty if the corpus has no HF mapping or HF is unreachable and nothing is
    cached). Cached per session by Streamlit and on disk across reruns."""
    return _hf_eval_impl(corpus, n_per_class)


# ── Browseable HF index (list MANY clips, download only the chosen one) ─────── #
# hf_eval_samples downloads its whole balanced set up front — fine for scoring,
# but it means the Signal Explorer only ever shows ~50 clips per class and pays
# the download cost immediately. For browsing we instead pull a large INDEX of
# rows (label + presigned audio URL, NO audio) and fetch a single clip lazily
# when the user actually selects it. Much faster, far more files to pick from.
HF_BROWSE_PER_CLASS = 500        # how many clips per class to list for browsing


def _hf_listing_impl(corpus: str, max_per_class: int):
    import random
    from urllib.parse import quote

    spec = HF_EVAL_DATASETS.get(corpus)
    if not spec:
        return []
    base = (f"{_DS_ROWS_URL}?dataset={quote(spec['id'], safe='')}"
            f"&config={spec['config']}&split={spec['split']}")
    try:
        first = _hf_get_json(base + "&offset=0&length=100")
    except Exception:                                    # noqa: BLE001 — HF unreachable
        return []
    total = int(first.get("num_rows_total", 100))
    rng = random.Random(7)
    pages = _hf_pages_for(max_per_class)
    offsets = [rng.randint(0, max(0, total - 100)) for _ in range(pages - 1)]

    collected: Dict[int, List[Tuple[int, str, str]]] = {
        LABEL_BONAFIDE: [], LABEL_SPOOF: []}
    seen = set()

    def _ingest(payload):
        for row in payload.get("rows", []):
            r   = row.get("row", {})
            lab = r.get(spec["label_col"])
            if lab not in (LABEL_BONAFIDE, LABEL_SPOOF):
                continue
            if len(collected[lab]) >= max_per_class:
                continue
            audio = r.get("audio")
            src   = (audio[0].get("src") if isinstance(audio, list) and audio
                     else None)
            if not src or src in seen:
                continue
            seen.add(src)
            # The presigned audio URLs all share the same basename (e.g.
            # "audio.wav"), so key the display name on the absolute row index:
            # this keeps every listed clip distinct (and its download path
            # unique in hf_fetch_clip) instead of collapsing to one file.
            ridx  = row.get("row_idx")
            tail  = os.path.basename(src.split("?")[0]) or "clip"
            stem  = f"{ridx}__{tail}" if ridx is not None else f"{len(seen)}__{tail}"
            collected[lab].append((lab, src, stem))

    _ingest(first)
    _hf_scan_pages(base, offsets, _ingest,
                   lambda: all(len(collected[c]) >= max_per_class for c in collected))
    return collected[LABEL_BONAFIDE] + collected[LABEL_SPOOF]


@st.cache_data(show_spinner=False)
def hf_eval_listing(corpus: str,
                    max_per_class: int = HF_BROWSE_PER_CLASS
                    ) -> List[Tuple[int, str, str]]:
    """A large browseable index of eval clips for ``corpus``: [(label, src_url,
    fname)] read from the HF datasets-server WITHOUT downloading any audio. The
    Signal Explorer lists these; hf_fetch_clip() fetches only the selected clip.
    Empty if the corpus has no HF mapping or HF is unreachable."""
    return _hf_listing_impl(corpus, max_per_class)


def hf_fetch_clip(corpus: str, src: str, fname: str, label: int) -> Optional[str]:
    """Download ONE listed clip into the browse cache and return its local path
    (or None on failure). Idempotent: reuses the file if already fetched."""
    import hashlib
    ckey      = _SAMPLE_KEYS.get(corpus, corpus)
    cache_dir = os.path.join(_HF_CACHE_DIR, ckey, "browse")
    os.makedirs(cache_dir, exist_ok=True)
    tag  = "bonafide" if label == LABEL_BONAFIDE else "spoof"
    # Key the cached filename on a hash of the full presigned URL: the URL
    # basenames are all identical, so hashing the unique URL is what guarantees
    # distinct clips never overwrite/alias one another on disk.
    digest = hashlib.sha1(src.encode("utf-8")).hexdigest()[:16]
    ext    = ".wav" if src.split("?")[0].lower().endswith(".wav") else ".flac"
    dst    = os.path.join(cache_dir, f"{tag}__{ckey}__{digest}{ext}")
    res = _hf_download((src, dst, label))
    return res[0] if res else None

def _download_if_missing(url: str, path: str, label: str) -> None:
    if os.path.isfile(path):
        return
    if not _url_set(url):
        raise FileNotFoundError(
            f"{label}: weights not found in models/. Run Benchmark → Full "
            "comparison → Train all locally to generate every model file, then "
            "commit models/ to the repo (or set HF_BASE_URL to stream them)."
        )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import urllib.request
    with st.spinner(f"Calibrating the kyber crystals — fetching {label} (first run only)…"):
        urllib.request.urlretrieve(url, path)


@st.cache_resource(show_spinner=False)
def load_pretrained_torch(file: str, url: str, name: str):
    """Load a torch CNN checkpoint on CPU (downloading it first if needed).
    Returns (model_in_eval_mode, checkpoint_meta). Cached per (file)."""
    import torch

    from src.models import model_for_arch

    path = file if os.path.isabs(file) else os.path.join(MODELS_DIR, file)
    _download_if_missing(url, path, name)
    with st.spinner(f"Consulting the Jedi Archives — loading {name} on CPU…"):
        ckpt = torch.load(path, map_location=torch.device("cpu"))
        model = model_for_arch(ckpt.get("arch", "cnn"),
                               dropout=float(ckpt.get("dropout", 0.3)))
        model.load_state_dict(ckpt["state_dict"])
        model.to("cpu").eval()
    return model, ckpt


@st.cache_resource(show_spinner=False)
def load_pretrained_classic(file: str, url: str, name: str):
    """Load a joblib-dumped classic estimator (downloading it first if needed)."""
    import warnings
    import joblib

    path = os.path.join(MODELS_DIR, file)
    _download_if_missing(url, path, name)
    with st.spinner(f"Consulting the Jedi Archives — loading {name}…"):
        # The committed XGBoost .joblib were pickled with an older xgboost; the
        # newer runtime prints a cosmetic "save_model from that version" warning
        # on unpickle. The model loads and scores identically — mute the noise.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*save_model.*")
            return joblib.load(path)


@st.cache_resource(show_spinner=False)
def load_pretrained_raw(file: str, url: str, name: str,
                        hf_repo: str = "", hf_file: str = ""):
    """Load a raw-waveform torch model (wav2vec 2.0 SSL detector) on CPU.

    Weight resolution order: a local ``models/`` file (used on the GPU machine),
    then an explicit direct ``url``, then a PUBLIC Hugging Face repo (``hf_repo`` /
    ``hf_file``) streamed via ``hf_hub_download`` — that last path is what makes the
    cloud demo self-contained without committing the 469 MB checkpoint to GitHub.

    The checkpoint stores the weights under ``model_state_dict`` (a training
    checkpoint, not the plain ``state_dict`` the CNNs use). ``transformers`` is only
    needed here; if it is missing the ImportError propagates and the caller skips
    this model (same as a missing weight file)."""
    import torch

    from src.models import Wav2Vec2Classifier

    path = file if os.path.isabs(file) else os.path.join(MODELS_DIR, file)
    if not os.path.isfile(path):
        if _url_set(url):
            _download_if_missing(url, path, name)
        elif hf_repo:
            from huggingface_hub import hf_hub_download
            with st.spinner(f"Calibrating the kyber crystals — fetching {name} "
                            "from Hugging Face (first run only)…"):
                path = hf_hub_download(repo_id=hf_repo, filename=hf_file)
        else:
            _download_if_missing(url, path, name)     # raises the helpful error
    with st.spinner(f"Consulting the Jedi Archives — loading {name} on CPU…"):
        ckpt = torch.load(path, map_location=torch.device("cpu"))
        state = ckpt.get("model_state_dict", ckpt)
        model = Wav2Vec2Classifier()
        model.load_state_dict(state)
        model.to("cpu").eval()
    return model


def load_pretrained_model(entry: Dict):
    """Load one registry entry (torch CNN, raw-waveform model or classic estimator) on CPU."""
    if entry["kind"] == "cnn":
        model, _ = load_pretrained_torch(entry["file"], entry["url"], entry["name"])
        return model
    if entry["kind"] == "raw":
        return load_pretrained_raw(entry["file"], entry.get("url", ""), entry["name"],
                                   entry.get("hf_repo", ""), entry.get("hf_file", ""))
    return load_pretrained_classic(entry["file"], entry["url"], entry["name"])


def load_pretrained_cnn():
    """Backward-compatible helper: load the first available CNN entry (or the
    legacy single checkpoint) and return (model, meta)."""
    for entry in available_pretrained_models():
        if entry["kind"] == "cnn":
            return load_pretrained_torch(entry["file"], entry["url"], entry["name"])
    if os.path.isfile(CHECKPOINT_PATH):
        return load_pretrained_torch(CHECKPOINT_PATH, "", "Pretrained CNN")
    raise FileNotFoundError("No pretrained CNN available.")

