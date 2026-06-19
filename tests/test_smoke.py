# -*- coding: utf-8 -*-
"""
Smoke tests: synthetic 1-second signal and controlled data.
No ASVspoof corpus required.  Run with:
    pytest tests/
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import calculate_eer, calculate_min_dcf
from src.features import FeatureExtractor
from src.models import get_classic_model

CONFIG_PATH = str(Path(__file__).parent.parent / "config" / "config.yaml")
SAMPLE_RATE = 16_000  # Hz, matches config.yaml


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def extractor():
    return FeatureExtractor(CONFIG_PATH)


@pytest.fixture(scope="module")
def signal_1s():
    """440 Hz sine wave, 1 second at 16 kHz."""
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# EER — pure mathematical implementation
# ---------------------------------------------------------------------------

def test_eer_perfect_separation():
    """Perfectly discriminative scores -> EER must be 0."""
    scores = [0.1] * 50 + [0.9] * 50
    labels = [0]   * 50 + [1]   * 50
    eer, _ = calculate_eer(scores, labels)
    assert eer == pytest.approx(0.0, abs=1e-6)


def test_eer_random_scores():
    """Non-informative scores -> EER should be near 0.5."""
    rng    = np.random.default_rng(42)
    scores = rng.random(200).tolist()
    labels = [0] * 100 + [1] * 100
    eer, _ = calculate_eer(scores, labels)
    assert 0.3 < eer < 0.7


def test_eer_error_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        calculate_eer([0.5, 0.5], [0])


def test_eer_error_single_class():
    with pytest.raises(ValueError, match="BOTH classes"):
        calculate_eer([0.1, 0.9], [1, 1])


# ---------------------------------------------------------------------------
# minDCF — official ASVspoof 2019 metric
# ---------------------------------------------------------------------------

def test_min_dcf_perfect_separation():
    """Perfectly separable scores -> minDCF must be 0."""
    scores = [0.1] * 50 + [0.9] * 50
    labels = [0]   * 50 + [1]   * 50
    dcf = calculate_min_dcf(scores, labels)
    assert dcf == pytest.approx(0.0, abs=1e-6)


def test_min_dcf_upper_bound():
    """Normalised minDCF must always be <= 1 for any informative scores."""
    rng    = np.random.default_rng(0)
    scores = rng.random(200).tolist()
    labels = [0] * 100 + [1] * 100
    dcf = calculate_min_dcf(scores, labels)
    assert 0.0 <= dcf <= 1.0


def test_min_dcf_error_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        calculate_min_dcf([0.5], [0, 1])


def test_min_dcf_error_single_class():
    with pytest.raises(ValueError, match="BOTH classes"):
        calculate_min_dcf([0.5, 0.5], [0, 0])


# ---------------------------------------------------------------------------
# Extractors — output shapes and finite values
# ---------------------------------------------------------------------------

def test_rms_shape(extractor, signal_1s):
    v = extractor.extract_rms(signal_1s)
    assert v.shape == (2,)              # mean + variance of 1 band
    assert np.isfinite(v).all()


def test_mfcc_shape(extractor, signal_1s):
    v = extractor.extract_mfcc(signal_1s)
    assert v.shape == (40,)             # 2 × n_mfcc=20
    assert np.isfinite(v).all()


def test_lfcc_shape(extractor, signal_1s):
    v = extractor.extract_lfcc(signal_1s)
    assert v.shape == (40,)             # 2 × n_lfcc=20
    assert np.isfinite(v).all()


def test_dwt_shape(extractor, signal_1s):
    v = extractor.extract_wavelet_energy(signal_1s)
    assert v.shape == (4,)              # mean + variance of 2 sub-bands
    assert np.isfinite(v).all()


def test_cqcc_shape(extractor, signal_1s):
    v = extractor.extract_cqcc(signal_1s)
    assert v.shape == (26,)             # 2 × n_cqcc=13
    assert np.isfinite(v).all()


def test_fusion_shape(extractor, signal_1s):
    v = extractor.get_flat_vector(signal_1s, "5")
    assert v.shape == (112,)            # 2+40+40+4+26


def test_spectrogram_shape(extractor, signal_1s):
    m = extractor.get_spectrogram_matrix(signal_1s)
    assert m.shape == (extractor.freq_bins, extractor.time_frames)
    assert np.isfinite(m).all()


def test_invalid_extractor(extractor, signal_1s):
    with pytest.raises(ValueError, match="not valid"):
        extractor.get_flat_vector(signal_1s, "9")


def test_no_nan_on_silence(extractor):
    """Absolute silence must not produce NaN or inf values."""
    silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
    v = extractor.get_flat_vector(silence, "5")
    assert np.isfinite(v).all()


# ---------------------------------------------------------------------------
# Padding — signals shorter than n_fft
# ---------------------------------------------------------------------------

def test_load_audio_pads_short_signal(extractor, tmp_path):
    """load_audio must zero-pad signals shorter than n_fft."""
    short = np.zeros(100, dtype=np.float32)
    path  = str(tmp_path / "short.wav")
    sf.write(path, short, extractor.sample_rate)
    loaded = extractor.load_audio(path)
    assert len(loaded) >= extractor.n_fft


def test_load_audio_normal_length_unchanged(extractor, tmp_path):
    """A 1-second signal must not be altered by padding."""
    t      = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    signal = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path   = str(tmp_path / "normal.wav")
    sf.write(path, signal, extractor.sample_rate)
    loaded = extractor.load_audio(path)
    assert len(loaded) == SAMPLE_RATE


# ---------------------------------------------------------------------------
# Classic models — fit + predict_proba
# ---------------------------------------------------------------------------

def _make_xy(n=100, dim=40, seed=42):
    rng = np.random.default_rng(seed)
    X   = rng.standard_normal((n, dim)).astype(np.float32)
    y   = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y


def test_logistic_regression_probabilities():
    X, y = _make_xy()
    m    = get_classic_model("logistic_regression", seed=42)
    m.fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (100, 2)
    assert np.allclose(p.sum(axis=1), 1.0)


def test_svm_probabilities():
    X, y = _make_xy(n=60)   # fewer samples: internal calibration CV is faster
    m    = get_classic_model("svm_lineal", seed=42)
    m.fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (60, 2)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)


def test_xgboost_probabilities():
    X, y = _make_xy()
    m    = get_classic_model("xgboost", seed=42)
    m.fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (100, 2)
    assert np.allclose(p.sum(axis=1), 1.0)


def test_unknown_model():
    with pytest.raises(ValueError, match="Unknown"):
        get_classic_model("magic_neural_network")
