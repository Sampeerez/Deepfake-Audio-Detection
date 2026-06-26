# -*- coding: utf-8 -*-
"""
src/pipeline.py — Experiment orchestration: feature extraction, classic
                  classifiers, and CNN training + evaluation.

Public entry points (all used by the web app):
    extract_feature_matrix(...) — vectorise a subset of audio with one extractor
    run_classic_models(...)     — DSP features + sklearn/XGBoost classifiers
    train_and_evaluate_cnn(...) — STFT-dB spectrogram + 2-D CNN (model + history)

The classic / CNN runners return a list of result dicts whose keys match the
COL_* constants in src.reporting, which the GUI reads to build its table.
"""

import copy
import hashlib
import json
import math
import os
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data_loader import (
    LABEL_BONAFIDE, LABEL_SPOOF, ASVspoofRawWaveDataset, ASVspoofTorchDataset,
)
from src.features import FeatureExtractor
from src.metrics import calculate_eer, calculate_min_dcf
from src.models import AudioDeepfakeCNN, ResNetCNN, get_classic_model
from src.reporting import (
    COL_ACCURACY, COL_EER, COL_FEATURES, COL_INFER_TIME,
    COL_MIN_DCF, COL_MODEL, COL_TRAIN_TIME,
)

# On-disk cache directory for DSP feature vectors.
CACHE_DIR = "cache"

# Maps the classic-classifier option key -> tuple of classic model identifiers.
MODEL_OPTIONS: Dict[str, Tuple[str, ...]] = {
    "1": ("logistic_regression",),
    "2": ("svm_lineal",),
    "3": ("xgboost",),
    "4": ("logistic_regression", "svm_lineal", "xgboost"),
    # The CNN is a separate pipeline branch (train_and_evaluate_cnn), driven by
    # the CNN Learning page — it is not one of these classic-model options.
}

# Human-readable display names for classic classifiers.
MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "logistic_regression": "Logistic Regression (L2)",
    "svm_lineal":          "SVM RBF (Platt)",
    "xgboost":             "XGBoost (L1/L2 + hist)",
}


# ===========================================================================
# Pipeline A: bulk feature extraction + classic classifiers (CPU)
# ===========================================================================

def _hash_config(extractor: FeatureExtractor, option: str) -> str:
    """8-character MD5 hash of the DSP hyperparameters relevant to 'option'.

    The hash changes whenever the relevant values in config.yaml change,
    automatically invalidating cached vectors for that configuration.
    """
    params: Dict = {
        "sr": extractor.sample_rate,
        "n_fft": extractor.n_fft,
        "hop": extractor.hop_length,
        "feat": option,
    }
    if option in ("2", "5"):
        params.update({"n_mfcc": extractor.n_mfcc, "n_mels": extractor.n_mels})
    if option in ("3", "5"):
        params.update({"n_lfcc": extractor.n_lfcc,
                       "n_lin": extractor.n_linear_filters})
    if option in ("4", "5"):
        params["wavelet"] = extractor.wavelet_mother
    if option in ("5", "6"):
        params.update({"n_cqcc": extractor.n_cqcc,
                       "n_bins": extractor.cqcc_n_bins})
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]


def _extract_with_cache(
    path: str,
    option: str,
    extractor: FeatureExtractor,
    config_hash: str,
    use_cache: bool,
) -> np.ndarray:
    """Return the 1-D feature vector for one audio file, using cache if enabled.

    Cache invalidation is automatic: the file name encodes the config hash,
    so changing any relevant hyperparameter in config.yaml produces a
    different hash and forces re-extraction.
    """
    if use_cache:
        filename   = f"{pathlib.Path(path).stem}_{option}_{config_hash}.npy"
        cache_path = pathlib.Path(CACHE_DIR) / option / filename
        if cache_path.exists():
            return np.load(str(cache_path))

    signal = extractor.load_audio(path)
    vector = extractor.get_flat_vector(signal, option)

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(cache_path), vector)

    return vector


def extract_feature_matrix(
    samples: List[Tuple[str, int]],
    extractor: FeatureExtractor,
    feature_option: str,
    subset_name: str,
    n_workers: int = 1,
    use_cache: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Vectorise a full subset: returns (X, y, ms_extraction_per_audio).

    Supports parallel extraction (ThreadPoolExecutor when n_workers > 1)
    and on-disk caching (.npy per audio).  Cache is invalidated automatically
    when DSP hyperparameters in config.yaml change.
    """
    n           = len(samples)
    labels      = [lbl for _, lbl in samples]
    config_hash = _hash_config(extractor, feature_option)

    mode_tags = ([f"×{n_workers} threads"] if n_workers > 1 else []) + \
                (["cache"] if use_cache else [])
    suffix = f" [{', '.join(mode_tags)}]" if mode_tags else ""
    print(f"[DSP] Extracting '{FeatureExtractor.OPTION_NAMES[feature_option]}'"
          f" from subset '{subset_name}' ({n} audio files){suffix}...")

    start   = time.perf_counter()
    vectors: List[Optional[np.ndarray]] = [None] * n

    if n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [
                pool.submit(_extract_with_cache,
                            path, feature_option, extractor,
                            config_hash, use_cache)
                for path, _ in samples
            ]
            for pos, future in enumerate(futures, start=1):
                vectors[pos - 1] = future.result()
                if pos % 500 == 0 or pos == n:
                    print(f"    {pos}/{n} audio files "
                          f"({time.perf_counter() - start:.1f}s elapsed)")
    else:
        for pos, (path, _) in enumerate(samples, start=1):
            vectors[pos - 1] = _extract_with_cache(
                path, feature_option, extractor, config_hash, use_cache
            )
            if pos % 500 == 0 or pos == n:
                print(f"    {pos}/{n} audio files "
                      f"({time.perf_counter() - start:.1f}s elapsed)")

    elapsed  = time.perf_counter() - start
    matrix   = np.vstack(vectors).astype(np.float32)
    y        = np.array(labels, dtype=np.int64)
    ms_audio = 1000.0 * elapsed / max(n, 1)
    print(f"[DSP] Matrix {subset_name}: {matrix.shape} | "
          f"avg extraction: {ms_audio:.2f} ms/audio")
    return matrix, y, ms_audio


def run_classic_models(
    model_names: Sequence[str],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    feature_label: str,
    ms_dsp_dev: float,
    seed: int,
    x_eval: Optional[np.ndarray] = None,
    y_eval: Optional[np.ndarray] = None,
    eval_sets: Optional[List[Tuple[str, np.ndarray, np.ndarray]]] = None,
    model_sink: Optional[Callable[[str, object], None]] = None,
) -> List[Dict[str, str]]:
    """Train and evaluate each classic classifier on the dev set.

    Eval scoring: pass either a single (x_eval, y_eval) or, to score several
    corpora, ``eval_sets`` as a list of (corpus_label, X, y). Each produces an
    extra row with the '[EVAL]' suffix and a 'Corpus' tag (so different corpora
    stay distinct in the leaderboard instead of colliding on the same eval row).

    model_sink: optional callback invoked as ``model_sink(name, fitted_model)``
    right after each classifier is fit — used by the Full-comparison export to
    persist the trained models to disk (e.g. joblib) without re-fitting.
    """
    # Normalise to a list of (corpus_label, X, y).
    if eval_sets is None:
        eval_sets = ([("", x_eval, y_eval)]
                     if x_eval is not None and y_eval is not None else [])

    results: List[Dict[str, str]] = []
    n_bonafide = int(np.sum(y_train == LABEL_BONAFIDE))
    n_spoof    = int(np.sum(y_train == LABEL_SPOOF))
    imbalance  = n_bonafide / max(n_spoof, 1)

    for name in model_names:
        print(f"\n[MODEL] Training {MODEL_DISPLAY_NAMES[name]} on CPU...")
        model = get_classic_model(name, seed=seed, scale_pos_weight=imbalance)

        t0 = time.perf_counter()
        model.fit(x_train, y_train)
        train_time = time.perf_counter() - t0

        if model_sink is not None:
            model_sink(name, model)          # persist the fitted estimator

        t0 = time.perf_counter()
        scores = model.predict_proba(x_dev)[:, 1]
        infer_time = time.perf_counter() - t0
        ms_infer   = 1000.0 * infer_time / len(x_dev)

        preds    = (scores >= 0.5).astype(np.int64)
        accuracy = float(np.mean(preds == y_dev))
        eer, threshold = calculate_eer(scores.tolist(), y_dev.tolist())
        min_dcf        = calculate_min_dcf(scores.tolist(), y_dev.tolist())

        print(f"[MODEL] {MODEL_DISPLAY_NAMES[name]} | "
              f"Accuracy={accuracy:.4f} | EER={100*eer:.2f}% "
              f"(threshold*={threshold:.4f}) | minDCF={min_dcf:.4f} | "
              f"train={train_time:.2f}s | "
              f"inference={ms_infer:.3f} ms/audio "
              f"(+{ms_dsp_dev:.2f} ms/audio DSP)")

        results.append({
            COL_FEATURES:   feature_label,
            COL_MODEL:      f"{MODEL_DISPLAY_NAMES[name]} [CPU]",
            COL_ACCURACY:   f"{accuracy:.4f}",
            COL_EER:        f"{100 * eer:.2f}",
            COL_MIN_DCF:    f"{min_dcf:.4f}",
            COL_TRAIN_TIME: f"{train_time:.2f}",
            COL_INFER_TIME: f"{ms_infer:.3f}",
        })

        # Optional evaluation on one or more held-out eval corpora (no refit —
        # the model is already trained, we only predict on each set).
        for corpus_label, x_ev, y_ev in eval_sets:
            if x_ev is None or y_ev is None:
                continue
            scores_ev   = model.predict_proba(x_ev)[:, 1]
            preds_ev    = (scores_ev >= 0.5).astype(np.int64)
            acc_ev      = float(np.mean(preds_ev == y_ev))
            eer_ev, _   = calculate_eer(scores_ev.tolist(), y_ev.tolist())
            dcf_ev      = calculate_min_dcf(scores_ev.tolist(), y_ev.tolist())
            _ctag = f" @ {corpus_label}" if corpus_label else ""
            print(f"[EVAL{_ctag}] {MODEL_DISPLAY_NAMES[name]} | "
                  f"Accuracy={acc_ev:.4f} | EER={100*eer_ev:.2f}% | "
                  f"minDCF={dcf_ev:.4f}")
            results.append({
                COL_FEATURES:   f"{feature_label} [EVAL]",
                COL_MODEL:      f"{MODEL_DISPLAY_NAMES[name]} [CPU][EVAL]",
                COL_ACCURACY:   f"{acc_ev:.4f}",
                COL_EER:        f"{100 * eer_ev:.2f}",
                COL_MIN_DCF:    f"{dcf_ev:.4f}",
                COL_TRAIN_TIME: "—",
                COL_INFER_TIME: "—",
                "Corpus":       corpus_label,
            })

    return results


def score_fitted_classic(
    fitted_models: Dict[str, object],
    x_set: np.ndarray,
    y_set: np.ndarray,
    feature_label: str,
    corpus_label: str = "",
    suffix: str = "[EVAL]",
) -> List[Dict[str, str]]:
    """Score pre-fitted sklearn estimators on a feature matrix without retraining.
    Returns one result-dict per model (same COL_* format as run_classic_models).
    Used by the Evaluate button in Classic mode.
    """
    results: List[Dict[str, str]] = []
    for name, model in fitted_models.items():
        scores = model.predict_proba(x_set)[:, 1]
        preds  = (scores >= 0.5).astype(np.int64)
        acc    = float(np.mean(preds == y_set))
        eer, _ = calculate_eer(scores.tolist(), y_set.tolist())
        dcf    = calculate_min_dcf(scores.tolist(), y_set.tolist())
        results.append({
            COL_FEATURES:   f"{feature_label}{suffix}",
            COL_MODEL:      f"{MODEL_DISPLAY_NAMES.get(name, name)} [CPU]{suffix}",
            COL_ACCURACY:   f"{acc:.4f}",
            COL_EER:        f"{100 * eer:.2f}",
            COL_MIN_DCF:    f"{dcf:.4f}",
            COL_TRAIN_TIME: "—",
            COL_INFER_TIME: "—",
            "Corpus":       corpus_label,
        })
    return results


def evaluate_cnn_on_set(
    model: "AudioDeepfakeCNN",
    samples: List[Tuple[str, int]],
    extractor: "FeatureExtractor",
    params: Dict,
    corpus_label: str = "",
    arch_label: str = "CNN",
    suffix: str = "[EVAL]",
) -> List[Dict[str, str]]:
    """Score an already-trained CNN on one sample set without retraining.
    Returns a list with one result dict (COL_* keys). Used by the Evaluate
    button in CNN mode to score a session-trained or HF-pretrained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _spec_tag = (f"{int(extractor.sample_rate)}_{int(extractor.n_fft)}_"
                 f"{int(extractor.hop_length)}_"
                 f"{int(extractor.freq_bins)}x{int(extractor.time_frames)}")
    n_dl_w = int(params.get("num_workers", 0))
    batch  = int(params.get("batch_size", 32))
    pin    = device.type == "cuda"
    extra: Dict = {}
    if n_dl_w > 0:
        extra = {"persistent_workers": True, "prefetch_factor": 4}
    loader = DataLoader(
        ASVspoofTorchDataset(
            samples, extractor.get_spectrogram_matrix,
            extractor.sample_rate, augment=False,
            cache_tag=_spec_tag, cache_dir=CACHE_DIR,
        ),
        batch_size=batch, shuffle=False,
        num_workers=n_dl_w, pin_memory=pin, **extra,
    )
    model = model.to(device)
    scores, labels, ms = _evaluate_cnn(model, loader, device)
    device_tag = "CUDA" if device.type == "cuda" else "CPU"
    preds    = [1 if s >= 0.5 else 0 for s in scores]
    accuracy = sum(p == e for p, e in zip(preds, labels)) / len(labels)
    eer, _   = calculate_eer(scores, labels)
    min_dcf  = calculate_min_dcf(scores, labels)
    return [{
        COL_FEATURES:   (
            f"STFT-dB Spectrogram ({extractor.freq_bins}×"
            f"{extractor.time_frames}){suffix}"
        ),
        COL_MODEL:      f"{arch_label} PyTorch [{device_tag}]{suffix}",
        COL_ACCURACY:   f"{accuracy:.4f}",
        COL_EER:        f"{100 * eer:.2f}",
        COL_MIN_DCF:    f"{min_dcf:.4f}",
        COL_TRAIN_TIME: "—",
        COL_INFER_TIME: f"{ms:.3f}",
        "Corpus":       corpus_label,
    }]


def evaluate_raw_on_set(
    model: "torch.nn.Module",
    samples: List[Tuple[str, int]],
    sample_rate: int,
    params: Dict,
    corpus_label: str = "",
    arch_label: str = "wav2vec 2.0 (SSL)",
    suffix: str = "[EVAL]",
) -> List[Dict[str, str]]:
    """Score a raw-waveform model (wav2vec 2.0) on one sample set, inference-only.

    Mirrors :func:`evaluate_cnn_on_set` (same COL_* result-dict shape so the
    leaderboard merge consumes it unchanged), but feeds the model the raw 16 kHz
    waveform instead of a spectrogram. p(spoof) comes from the model's own
    ``prob_spoof`` (softmax over its 2 logits), not a sigmoid. Batches are kept
    small and the CUDA cache is freed between them so the 6 GB GPU never saturates.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_samples = int(params.get("raw_max_samples", 4 * sample_rate))   # 4 s window
    batch       = int(params.get("raw_batch_size", 8))
    n_dl_w      = int(params.get("num_workers", 0))
    pin         = device.type == "cuda"
    extra: Dict = {"persistent_workers": True, "prefetch_factor": 4} if n_dl_w > 0 else {}
    loader = DataLoader(
        ASVspoofRawWaveDataset(samples, sample_rate, max_samples),
        batch_size=batch, shuffle=False,
        num_workers=n_dl_w, pin_memory=pin, **extra,
    )
    model = model.to(device).eval()
    scores: List[float] = []
    labels: List[int]   = []
    forward_time = 0.0
    with torch.no_grad():
        for waves, lbls in loader:
            waves = waves.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            probs = model.prob_spoof(waves)
            if device.type == "cuda":
                torch.cuda.synchronize()
            forward_time += time.perf_counter() - t0
            scores.extend(probs.float().cpu().tolist())
            labels.extend(lbls.long().tolist())
            if device.type == "cuda":
                torch.cuda.empty_cache()
    ms = 1000.0 * forward_time / max(len(labels), 1)
    device_tag = "CUDA" if device.type == "cuda" else "CPU"
    preds    = [1 if s >= 0.5 else 0 for s in scores]
    accuracy = sum(p == e for p, e in zip(preds, labels)) / max(len(labels), 1)
    eer, _   = calculate_eer(scores, labels)
    min_dcf  = calculate_min_dcf(scores, labels)
    return [{
        COL_FEATURES:   f"Raw waveform (16 kHz){suffix}",
        COL_MODEL:      f"{arch_label} [{device_tag}]{suffix}",
        COL_ACCURACY:   f"{accuracy:.4f}",
        COL_EER:        f"{100 * eer:.2f}",
        COL_MIN_DCF:    f"{min_dcf:.4f}",
        COL_TRAIN_TIME: "—",
        COL_INFER_TIME: f"{ms:.3f}",
        "Corpus":       corpus_label,
    }]


# ===========================================================================
# Pipeline B: 2-D CNN (CPU or GPU-CUDA)
# ===========================================================================

def _train_cnn(
    model: AudioDeepfakeCNN,
    loader_train: DataLoader,
    loader_dev: DataLoader,
    criterion: nn.Module,
    optimizer: "torch.optim.Optimizer",
    device: "torch.device",
    epochs: int,
    patience: int,
    epoch_callback: Optional[Callable[[Dict], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    batch_callback: Optional[Callable[[int, int, float], None]] = None,
) -> Tuple[float, int, List[Dict]]:
    """Training loop with per-epoch validation, LR scheduler, and early stopping.

    ReduceLROnPlateau halves the learning rate if val loss does not improve
    for patience//2 epochs; early stopping halts training if there is no
    improvement for patience full epochs.  Both share the same val loss
    computed with torch.no_grad() at the end of each epoch.

    If ``epoch_callback`` is given, it is invoked at the end of each epoch
    with the latest history record (useful for live UI plotting).

    Returns:
        (training_seconds, last_epoch, history) where history is a list of
        ``{"epoch", "train_loss", "val_loss", "lr"}`` records.
    """
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, patience // 2)
    )
    # Mixed precision on GPU (~1.5–2× faster, no meaningful accuracy change). The
    # autograd/scaler API moved from torch.cuda.amp to torch.amp in torch 2.x.
    #
    # PREFER bfloat16 when the GPU supports it: bf16 shares fp32's 8-bit exponent,
    # so the activations of a deep net (ResNet+SE) cannot overflow to inf/NaN the
    # way fp16 does. That overflow was the silent killer — it made the validation
    # loss non-finite every epoch, so the best checkpoint never updated and the
    # loop returned the *random initialisation* (≈50% EER). bf16 needs no loss
    # scaling, so the GradScaler is only enabled on the fp16 fallback path.
    use_amp = (device.type == "cuda")
    _bf16 = bool(use_amp and getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    _amp_dtype = torch.bfloat16 if _bf16 else torch.float16
    _use_scaler = use_amp and not _bf16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=_use_scaler)
        def _autocast():
            return torch.amp.autocast("cuda", dtype=_amp_dtype, enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=_use_scaler)
        def _autocast():
            return torch.cuda.amp.autocast(enabled=use_amp)

    best_val_loss   = float("inf")
    best_state_dict = copy.deepcopy(model.state_dict())
    ever_saved      = False     # did any epoch yield a finite, improving val loss?
    epochs_no_impr  = 0
    history: List[Dict] = []
    start           = time.perf_counter()

    _stop = should_stop if should_stop is not None else (lambda: False)
    for epoch in range(1, epochs + 1):
        if _stop():                                      # cooperative cancel
            break
        # --- Training pass -------------------------------------------- #
        model.train()
        train_loss, n_batches = 0.0, 0
        _cancelled = False
        for tensors, labels in loader_train:
            if _stop():                                  # cancel mid-epoch (near-instant)
                _cancelled = True
                break
            tensors = tensors.to(device, non_blocking=True)
            labels  = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with _autocast():
                loss = criterion(model(tensors), labels)
            scaler.scale(loss).backward()
            # Unscale before clipping so the norm is measured on real gradients;
            # gradient-norm clipping keeps the deeper ResNet+SE from taking a
            # destabilising step early in training (a second guard against the
            # divergence that produced the untrained ResNet).
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.item())
            n_batches  += 1
            if batch_callback is not None and n_batches % 50 == 0:
                batch_callback(epoch, n_batches, train_loss / n_batches)
        if _cancelled:
            break

        # --- Validation pass ------------------------------------------ #
        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for tensors, labels in loader_dev:
                tensors  = tensors.to(device, non_blocking=True)
                labels   = labels.to(device, non_blocking=True)
                with _autocast():
                    val_loss += float(criterion(model(tensors), labels).item())
                n_val    += 1
        val_loss /= max(n_val, 1)
        scheduler.step(val_loss)

        record = {
            "epoch":      epoch,
            "train_loss": train_loss / max(n_batches, 1),
            "val_loss":   val_loss,
            "lr":         optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(f"[CNN] Epoch {epoch:02d}/{epochs} | "
              f"loss_train={record['train_loss']:.4f} | "
              f"loss_val={val_loss:.4f} | lr={record['lr']:.2e}")
        if epoch_callback is not None:
            epoch_callback(record)

        # Only a FINITE, improving val loss is allowed to checkpoint — a NaN/inf
        # loss must never count as "best", or the model that gets restored is the
        # random initialisation.
        if math.isfinite(val_loss) and val_loss < best_val_loss - 1e-4:
            best_val_loss   = val_loss
            epochs_no_impr  = 0
            best_state_dict = copy.deepcopy(model.state_dict())
            ever_saved      = True
        else:
            if not math.isfinite(val_loss):
                print(f"[CNN] Warning: non-finite val loss at epoch {epoch} "
                      f"(loss={val_loss}); skipping checkpoint for this epoch.")
            epochs_no_impr += 1
            if epochs_no_impr >= patience:
                print(f"[CNN] Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} consecutive epochs).")
                if device.type == "cuda":
                    torch.cuda.synchronize()
                if ever_saved:
                    model.load_state_dict(best_state_dict)
                return time.perf_counter() - start, epoch, history

    if device.type == "cuda":
        torch.cuda.synchronize()
    # Restore the best finite checkpoint. If NO epoch ever improved (e.g. training
    # diverged), keep the LAST trained weights rather than overwriting them with
    # the random init — the caller still gets a real, scoreable model.
    if ever_saved:
        model.load_state_dict(best_state_dict)
    else:
        print("[CNN] Warning: no epoch produced a finite improving val loss; "
              "returning the last-epoch weights instead of the initialisation.")
    return time.perf_counter() - start, epochs, history


def _evaluate_cnn(
    model: AudioDeepfakeCNN,
    loader: DataLoader,
    device: "torch.device",
) -> Tuple[List[float], List[int], float]:
    """Full inference pass; returns (spoof_scores, labels, ms_per_audio)."""
    model.eval()
    scores:     List[float] = []
    out_labels: List[int]   = []
    forward_time = 0.0

    with torch.no_grad():
        for tensors, labels in loader:
            tensors = tensors.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = model(tensors)
            if device.type == "cuda":
                torch.cuda.synchronize()
            forward_time += time.perf_counter() - t0
            scores.extend(torch.sigmoid(logits).cpu().tolist())
            out_labels.extend(labels.long().tolist())

    ms_per_audio = 1000.0 * forward_time / max(len(out_labels), 1)
    return scores, out_labels, ms_per_audio


def train_and_evaluate_cnn(
    train_samples: List[Tuple[str, int]],
    dev_samples:   List[Tuple[str, int]],
    extractor:     FeatureExtractor,
    params:        Dict,
    eval_samples:  Optional[List[Tuple[str, int]]] = None,
    eval_sets:     Optional[List[Tuple[str, List[Tuple[str, int]]]]] = None,
    epoch_callback: Optional[Callable[[Dict], None]] = None,
    should_stop:   Optional[Callable[[], bool]] = None,
    checkpoint_path: Optional[str] = None,
    batch_callback: Optional[Callable[[int, int, float], None]] = None,
) -> Tuple[AudioDeepfakeCNN, List[Dict], List[Dict[str, str]]]:
    """Full CNN orchestration returning the model, training history and result rows.

    Args:
        train_samples: List of (path, label) for training.
        dev_samples:   List of (path, label) for validation / dev evaluation.
        extractor:     FeatureExtractor instance (provides get_spectrogram_matrix).
        params:        Dict from config.yaml['train_params'] plus 'semilla'.
        eval_samples:  Optional eval subset; if provided, an extra [EVAL] row
                       is appended to the results.
        epoch_callback: Optional per-epoch callback for live progress reporting.

    Returns:
        (trained_model, history, result_rows) where result_rows use the COL_*
        keys consumed by the GUI results table.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[CNN] Compute device: {device}")
    torch.manual_seed(int(params["semilla"]))
    # Inputs are a fixed 128×300 size → cuDNN can pick the fastest conv algorithms.
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    pin    = device.type == "cuda"
    n_dl_w = int(params["num_workers"])
    batch  = int(params["batch_size"])
    patience = int(params.get("early_stopping_patience", 3))

    use_augment = bool(params.get("augment", True))

    # Cache tag encoding the spectrogram config: the BASE spectrogram of each
    # file is computed ONCE, cached to disk and reused every epoch / arch / run.
    # This is the key speed-up — the STFT recompute was the per-epoch bottleneck.
    _spec_tag = (f"{int(extractor.sample_rate)}_{int(extractor.n_fft)}_"
                 f"{int(extractor.hop_length)}_"
                 f"{int(extractor.freq_bins)}x{int(extractor.time_frames)}")

    def _make_loader(samples, shuffle: bool, augment: bool = False) -> DataLoader:
        extra = {}
        if n_dl_w > 0:
            # Keep workers alive across epochs and prefetch ahead → less idle GPU.
            extra = {"persistent_workers": True, "prefetch_factor": 4}
        return DataLoader(
            ASVspoofTorchDataset(
                samples,
                extractor.get_spectrogram_matrix,
                extractor.sample_rate,
                augment=augment,
                cache_tag=_spec_tag,
                cache_dir=CACHE_DIR,
            ),
            batch_size=batch,
            shuffle=shuffle,
            num_workers=n_dl_w,
            pin_memory=pin,
            **extra,
        )

    loader_train = _make_loader(train_samples, shuffle=True,  augment=use_augment)
    loader_dev   = _make_loader(dev_samples,   shuffle=False, augment=False)

    n_bonafide = sum(1 for _, e in train_samples if e == LABEL_BONAFIDE)
    n_spoof    = sum(1 for _, e in train_samples if e == LABEL_SPOOF)
    pos_weight = torch.tensor(
        [n_bonafide / max(n_spoof, 1)], dtype=torch.float32, device=device
    )

    arch = params.get("arch", "cnn")
    if arch == "resnet":
        model      = ResNetCNN(dropout=float(params["dropout"])).to(device)
        arch_label = "ResNet+SE CNN"
    else:
        model      = AudioDeepfakeCNN(dropout=float(params["dropout"])).to(device)
        arch_label = "2D CNN"
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params["lr"]))

    train_time, n_epochs, history = _train_cnn(
        model, loader_train, loader_dev, criterion, optimizer,
        device, int(params["epochs"]), patience,
        epoch_callback=epoch_callback, should_stop=should_stop,
        batch_callback=batch_callback,
    )

    # Cancelled → skip the (potentially slow) dev/eval scoring and return now.
    if should_stop is not None and should_stop():
        return model, history, []

    # Persist the best (early-stopping-restored) weights so the model can be
    # reused later without retraining (the registry .pth files the demo serves).
    if checkpoint_path:
        try:
            torch.save({
                "arch":        arch,
                "dropout":     float(params["dropout"]),
                "freq_bins":   int(extractor.freq_bins),
                "time_frames": int(extractor.time_frames),
                "sample_rate": int(extractor.sample_rate),
                "state_dict":  model.state_dict(),
            }, checkpoint_path)
            print(f"[CNN] Saved checkpoint → {checkpoint_path}")
        except Exception as exc:                    # noqa: BLE001 — non-fatal
            print(f"[CNN] Could not save checkpoint: {exc}")

    device_tag = "CUDA" if device.type == "cuda" else "CPU"

    def _build_result(scores, labels, suffix: str = "", corpus: str = "") -> Dict[str, str]:
        preds    = [1 if s >= 0.5 else 0 for s in scores]
        accuracy = sum(p == e for p, e in zip(preds, labels)) / len(labels)
        eer, threshold = calculate_eer(scores, labels)
        min_dcf        = calculate_min_dcf(scores, labels)
        print(f"[CNN{suffix}{(' @ ' + corpus) if corpus else ''}] "
              f"Accuracy={accuracy:.4f} | EER={100*eer:.2f}% "
              f"(threshold*={threshold:.4f}) | minDCF={min_dcf:.4f}")
        return {
            COL_FEATURES:   (
                f"STFT-dB Spectrogram ({extractor.freq_bins}×"
                f"{extractor.time_frames}){suffix}"
            ),
            COL_MODEL:      f"{arch_label} PyTorch [{device_tag}]{suffix}",
            COL_ACCURACY:   f"{accuracy:.4f}",
            COL_EER:        f"{100 * eer:.2f}",
            COL_MIN_DCF:    f"{min_dcf:.4f}",
            COL_TRAIN_TIME: f"{train_time:.2f}" if not suffix else "—",
            COL_INFER_TIME: f"{ms:.3f}",
            "Corpus":       corpus,
        }

    scores_dev, labels_dev, ms = _evaluate_cnn(model, loader_dev, device)
    print(f"[CNN] train={train_time:.2f}s ({n_epochs} epochs) | "
          f"dev inference={ms:.3f} ms/audio")

    results = [_build_result(scores_dev, labels_dev)]

    # Eval scoring on one or more corpora (eval_sets); a bare eval_samples is
    # treated as one unlabelled corpus for backward compatibility.
    if eval_sets is None:
        eval_sets = [("", eval_samples)] if eval_samples is not None else []
    for corpus_label, samples in eval_sets:
        if not samples:
            continue
        scores_ev, labels_ev, ms = _evaluate_cnn(
            model, _make_loader(samples, shuffle=False), device
        )
        results.append(_build_result(scores_ev, labels_ev, suffix="[EVAL]",
                                      corpus=corpus_label))

    return model, history, results
